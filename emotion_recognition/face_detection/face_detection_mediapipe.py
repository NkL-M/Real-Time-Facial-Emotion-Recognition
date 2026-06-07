"""
Module for real time face detection using mediapipe.
"""

import tensorflow as tf
from keras import Model
import numpy as np
import time
import cv2
import mediapipe as mp
from emotion_recognition.src.registry import load_model


class FaceDetector():
    def __init__(self, detect_conf: float = 0.5):
        self.detect_conf=detect_conf
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(min_detection_confidence=self.detect_conf)

    def detect_faces(self,
                     img,
                     draw: bool = True,
                     fer_model: Model = None):
        """
        Function that draw bounding boxes on faces detected on an image.

        arg
        ----
        draw : bool
            - Choose wether to draw bounding boxes around faces.

        fer_model : Model
            - Use deep learning model for predicting facial emotion recognition (FER).

        returns
        ----
        img, bounding_boxes : tuple
            - Bounding boxes pixel coordinates and image.
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bounding_boxes = []

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                boundingBox = detection.location_data.relative_bounding_box
                h, w, c = img.shape
                bbox = int(boundingBox.xmin * w), int(boundingBox.ymin * h), int(boundingBox.width * w), int(boundingBox.height * h)
                # bounding_boxes.append([id, bbox, detection.score])
                bounding_boxes.append(bbox)

                if draw:
                    img = self.draw_boundingbox(img, bbox, length=30, thickness=8)
                    cv2.putText(img=img,
                                text=f"Visage {id+1} - {int(detection.score[0]*100)} %",
                                org=(bbox[0],bbox[1]-10),
                                fontFace=cv2.FONT_HERSHEY_PLAIN,
                                fontScale=2,
                                color=(255, 255, 255),
                                thickness=1)

                if fer_model:
                    pred = self.get_facial_emotion(img=img,
                                                   bbox=bbox,
                                                   model=fer_model)
                    cv2.putText(img=img,
                                text=f"Emotion: {pred}",
                                org=(bbox[0], bbox[1] + bbox[3] + 50),
                                fontFace=cv2.FONT_HERSHEY_PLAIN,
                                fontScale=2,
                                color=(255, 255, 255),
                                thickness=1)

        return img, bbox # TODO or bounding_boxes

    def draw_boundingbox(self,
                         img,
                         bbox: tuple,
                         length: int = 30,
                         thickness: int = 10):

        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        cv2.rectangle(img, bbox, (255, 255, 255), 2)

        # Top left corner
        cv2.line(img, (x, y), (x+length, y), (255, 0, 255), thickness)
        cv2.line(img, (x, y), (x, y+length), (255, 0, 255), thickness)

        # Top right corner
        cv2.line(img, (x1, y), (x1-length, y), (255, 0, 255), thickness)
        cv2.line(img, (x1, y), (x1, y+length), (255, 0, 255), thickness)

        # Bottom left corner
        cv2.line(img, (x, y1), (x, y1-length), (255, 0, 255), thickness)
        cv2.line(img, (x, y1), (x+length, y1), (255, 0, 255), thickness)

        # Bottom right corner
        cv2.line(img, (x1, y1), (x1, y1-length), (255, 0, 255), thickness)
        cv2.line(img, (x1, y1), (x1-length, y1), (255, 0, 255), thickness)

        return img

    def get_facial_emotion(self,
                           img,
                           bbox,
                           model: Model,
                           pad: int = 0):

        emotions_dict = {0 : 'Neutre',
                         1 : 'Joie',
                         2 : 'Colere',
                         3 : 'Tristesse',
                         4 : 'Peur',
                         5 : 'Degout',
                         6 : 'Surprise'}

        x1, y1, width, height = bbox
        x2, y2 = x1 + width, y1 + height

        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(img.shape[1], x2 + pad)
        y2 = min(img.shape[0], y2 + pad)

        # Inputs
        face_crop = img[y1:y2, x1:x2]
        face_crop = cv2.resize(face_crop, (48, 48)) # Resize image to 48x48 pixels
        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY) # Change from RGB to Grayscale
        face_crop = np.expand_dims(face_crop, axis=0)  # (1, 48, 48)

        # Probabilities outputs
        outputs = model(face_crop, training=False)

        # Prediction
        prediction_index = tf.argmax(outputs[0]).numpy()
        prediction = emotions_dict[prediction_index]

        return prediction

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = FaceDetector(detect_conf=0.5)
    model = load_model(model_name='custom_fer_model_01') # TODO Change Model name

    while True:
        success, img = cap.read()
        img, bbox = detector.detect_faces(img=img,
                                          draw=True,
                                          fer_model=model)

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

        cv2.imshow("Image", img)

        key = cv2.waitKey(30)

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
