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

import numpy as np
import time
import cv2
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
        self.frame_count += 1
        coordinates = None # default value when there's no face coordinates, avoid crash when no faces
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

        if coordinates==None:
            cv2.putText(img,
                        text='No face detected',
                        org=(1450,70),
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=3,
                        color=(0, 0, 255),
                        thickness=4)

        return (detected_ids, bounding_boxes, detection_scores)



def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = FaceDetector(detect_conf=0.5)

    while True:
        success, img = cap.read()
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img=img,
                    text=f"{str(int(fps))} FPS",
                    org=(10,70),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=3,
                    color=(0, 255, 0),
                    thickness=4)

        ids, bboxes, scores = detector.detect_faces(img)
        for id, bbox, score in zip(ids, bboxes, scores):
            print(f"Face's id: {id} - Bbox coords: {bbox} - Score: {round(score*100, 2)}%")

        cv2.imshow("Real-Time FER", img)

        key = cv2.waitKey(30)

        if key == ord("q"): # Press 'Q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
