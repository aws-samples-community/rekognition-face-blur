import json
import os
import urllib

import boto3
import botocore
import cv2
import numpy as np

from aws_lambda_powertools import Logger
from aws_lambda_powertools.utilities.typing import LambdaContext
from aws_lambda_powertools.utilities.data_classes import event_source, S3Event

logger = Logger()

rekognition = boto3.client('rekognition')
s3 = boto3.client('s3')

output_bucket = os.environ['OUTPUT_BUCKET']


def anonymize_face_simple(image, factor=3.0):
    """
    Adrian Rosebrock, Blur and anonymize faces with OpenCV and Python, PyImageSearch,
    https://www.pyimagesearch.com/2020/04/06/blur-and-anonymize-faces-with-opencv-and-python/,
    accessed on 6 January 2021

    Anonymizes faces with a Gaussian blur and OpenCV

    Args:
        image (ndarray): The image to be modified
        factor (float): The blurring kernel scale factor. Increasing the factor will increase the amount of blur applied
            to the face (default is 3.0)

    Returns:
        image (ndarray): The modified image
    """

    # automatically determine the size of the blurring kernel based
    # on the spatial dimensions of the input image
    (h, w) = image.shape[:2]
    kW = int(w / factor)
    kH = int(h / factor)

    # ensure the width of the kernel is odd
    if kW % 2 == 0:
        kW -= 1

    # ensure the height of the kernel is odd
    if kH % 2 == 0:
        kH -= 1

    # apply a Gaussian blur to the input image using our computed
    # kernel size
    return cv2.GaussianBlur(image, (kW, kH), 0)


def anonymize_face_pixelate(image, blocks=10):
    """
    Adrian Rosebrock, Blur and anonymize faces with OpenCV and Python, PyImageSearch,
    https://www.pyimagesearch.com/2020/04/06/blur-and-anonymize-faces-with-opencv-and-python/,
    accessed on 6 January 2021

    Creates a pixelated face blur with OpenCV

    Args:
        image (ndarray): The image to be modified
        blocks (int): Number of pixel blocks (default is 10)

    Returns:
        image (ndarray): The modified image
    """

    # divide the input image into NxN blocks
    (h, w) = image.shape[:2]
    xSteps = np.linspace(0, w, blocks + 1, dtype="int")
    ySteps = np.linspace(0, h, blocks + 1, dtype="int")

    # loop over the blocks in both the x and y direction
    for i in range(1, len(ySteps)):
        for j in range(1, len(xSteps)):
            # compute the starting and ending (x, y)-coordinates
            # for the current block
            startX = xSteps[j - 1]
            startY = ySteps[i - 1]
            endX = xSteps[j]
            endY = ySteps[i]

            # extract the ROI using NumPy array slicing, compute the
            # mean of the ROI, and then draw a rectangle with the
            # mean RGB values over the ROI in the original image
            roi = image[startY:endY, startX:endX]
            (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
            cv2.rectangle(image, (startX, startY), (endX, endY),
                          (B, G, R), -1)

    # return the pixelated blurred image
    return image


@logger.inject_lambda_context(log_event=True)
@event_source(data_class=S3Event)
def lambda_handler(event: S3Event, context: LambdaContext) -> dict:
    logger.debug("S3 event: {}", event=event)

    for record in event.records:
        try:
            bucket: str = record.s3.bucket.name
            key: str = urllib.parse.unquote_plus(record.s3.get_object.key)
            filename: str = key.split('/')[-1]
            local_filename: str = f"/tmp/{filename}"
        except KeyError:
            logger.error("Unable to retrieve S3 object metadata")

        if local_filename.split('.')[-1] not in ['png', 'jpeg', 'jpg']:
            logger.error("File extension is not supported")

        # download file locally to /tmp retrieve metadata
        try:
            s3.download_file(bucket, key, local_filename)
        except botocore.exceptions.ClientError:
            logger.error("Lack of permissions to download file from S3")

        image = cv2.imread(local_filename)
        image_height: int
        image_width: int
        image_height, image_width, _ = image.shape

        try:
            response = rekognition.detect_faces(Image={"S3Object": {"Bucket": bucket, "Name": key}})
        except rekognition.exceptions.AccessDeniedException:
            logger.error("Lack of permissions to access Amazon Rekognition")
        except rekognition.exceptions.InvalidS3ObjectException:
            logger.error("S3 object does not exist")

        for detected_face in response['FaceDetails']:
            x1 = int(detected_face['BoundingBox']['Left'] * image_width)
            x2 = x1 + int(detected_face['BoundingBox']['Width'] * image_width)
            y1 = int(detected_face['BoundingBox']['Top'] * image_height)
            y2 = y1 + int(detected_face['BoundingBox']['Height'] * image_height)

            # extract the face ROI
            face_roi = image[y1:y2, x1:x2]

            # anonymize/blur faces
            if os.environ['BLUR_TYPE'] == 'pixelate':
                face = anonymize_face_pixelate(face_roi, blocks=10)
            else:
                face = anonymize_face_simple(face_roi, factor=3.0)

            # store the blurred face in the output image
            image[y1:y2, x1:x2] = face

        # overwrite local image file with blurred faces
        cv2.imwrite(local_filename, image)

        # uploaded modified image to Amazon S3 bucket
        try:
            s3.upload_file(local_filename, output_bucket, key)
        except boto3.exceptions.S3UploadFailedError:
            logger.error("Lack of permissions to upload file to S3")

        # clean up /tmp
        if os.path.exists(local_filename):
            os.remove(local_filename)

    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "cv2_version": cv2.__version__,
                "message": "blur_faces lambda function executed successfully!"
            }
        )
    }
