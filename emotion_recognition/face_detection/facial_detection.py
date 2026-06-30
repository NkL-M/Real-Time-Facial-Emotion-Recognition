"""
Computer Vision Face Detection Module

This module provides a real-time face detection system using Google's Mediapipe library.
It includes a `FaceDetector` class for detecting faces live camera feeds,
returning bounding boxes, detection scores, and unique identifiers for each detected face.

The module also includes utility functions for visualizing detection results, such as
drawing bounding boxes and displaying FPS counters.

Functions:

`main()`: Runs a real-time face detection demo using the default camera,
while printing in the terminal the bounding boxes, detection scores,
and unique identifiers for each detected face.
"""
import cv2
import numpy as np
import mediapipe as mp


class FaceDetector():
    def __init__(self, detect_conf: float = 0.5):
        self.detect_conf=detect_conf
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(min_detection_confidence=self.detect_conf)
        self.frame_count = 0

    def detect_faces(self, img: np.ndarray) -> tuple[np.ndarray, tuple]:
        """
        Detects faces in an image and returns their IDs, bounding boxes, and detection scores.

        Returns
        ----
        (detected_ids, bounding_boxes, detection_scores) : tuple
            - detected_ids (set): Set of unique identifiers for detected faces.
            - bounding_boxes (list[tuple]): List of bounding box coordinates as (x, y, width, height).
            - detection_scores (list[float]): List of confidence scores for each detected face.

        Notes
        ----
        If no faces are detected, the image is annotated with "No face detected".

        Bounding boxes are converted from relative to absolute pixel coordinates.
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Change OpenCV's BGR colors to RGB
        self.results = self.faceDetection.process(imgRGB)
        self.frame_count += 1 # Necessary for choosing on which frame to do inference
        detected_ids = set()
        bounding_boxes = []
        detection_scores = []

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bounding_box = detection.location_data.relative_bounding_box

                h, w, c = img.shape
                coordinates = (int(bounding_box.xmin * w),
                               int(bounding_box.ymin * h),
                               int(bounding_box.width * w),
                               int(bounding_box.height * h))
                bounding_boxes.append(coordinates)
                detection_scores.append(detection.score[0]) # Mediapipe face detection certainty
                detected_ids.add(id)

        return (detected_ids, bounding_boxes, detection_scores)
