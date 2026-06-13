"""
Module for real time face detection using mediapipe.
"""

import tensorflow as tf
from keras import Model
import numpy as np
import time
import cv2
import mediapipe as mp
from collections import deque, defaultdict
from emotion_recognition.src.registry import load_model
from emotion_recognition.params import EMOTION_DICT, FRAME_PRED_STRIDE, WINDOW_LENGHT, FER_MODEL, IMAGE_SIZE


class PredictionSmoother():
    def __init__(self, windown_lenght: int = 10):
        self.probs_window = deque(maxlen=windown_lenght)

    def update(self, probs: np.ndarray) -> None:
        """
        Append to a rolling list a list of probabilities.

        returns
        -----
        None
        """
        self.probs_window.append(probs)

    def smoothed_prediction(self) -> tuple[str, float]:
        """
        Compute a smoothed prediction class of a multiclass classification task
        and the probability associated to the class.

        returns
        ----
        pred_class, pred_prob : tuple[str, float]
        """
        smoothed_probs = np.mean(self.probs_window, axis=0)
        pred_prob = np.amax(smoothed_probs)
        pred_idx = np.argmax(smoothed_probs)
        pred_class = EMOTION_DICT[pred_idx]
        return pred_class, pred_prob


class FaceDetector():
    def __init__(self, detect_conf: float = 0.5):
        self.detect_conf=detect_conf
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(min_detection_confidence=self.detect_conf)
        self.prediction_smoother = defaultdict(lambda: PredictionSmoother(windown_lenght=WINDOW_LENGHT))
        self.frame_count = 0
        self.faces_predictions = {}

    def detect_faces(self,
                     img: np.ndarray,
                     model: Model = None,
                     draw: bool = True
        ) -> tuple[np.ndarray, tuple]:
        """
        Function that draw bounding boxes on faces detected on an image.

        arg
        ----
        draw : bool
            - Choose wether to draw bounding boxes around faces.

        returns
        ----
        img, bounding_boxes : tuple
            - Bounding boxes pixel coordinates and image.
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Change OpenCV's BGR cap to RGB
        self.results = self.faceDetection.process(imgRGB)
        self.frame_count += 1
        id, coordinates = None, None # default value when no detected face
        bounding_boxes = []
        face_crops = []
        faces_indexes = []
        inference = (self.frame_count < FRAME_PRED_STRIDE or  # Predict every 3 frame to reduce compute load
                     self.frame_count % FRAME_PRED_STRIDE==0) # Predict frame 1 and 2 as the first latest preds

        if self.results.detections:
            detected_ids = set()
            for id, detection in enumerate(self.results.detections):
                bounding_box = detection.location_data.relative_bounding_box

                h, w, c = img.shape
                coordinates = (int(bounding_box.xmin * w),
                               int(bounding_box.ymin * h),
                               int(bounding_box.width * w),
                               int(bounding_box.height * h))
                bounding_boxes.append(coordinates)
                detected_ids.add(id)

                if draw:
                    img = self.draw_boundingbox(img, coordinates, length=30, thickness=5)

                    cv2.putText(img=img,
                                text=f"Visage {id+1} - {int(detection.score[0]*100)} %",
                                org=(coordinates[0],coordinates[1] - 25),
                                fontFace=cv2.FONT_HERSHEY_PLAIN,
                                fontScale=2,
                                color=(255, 255, 255),
                                thickness=1)

                if model and inference:
                        face_crop = self.preprocess(img, coordinates, pad=30)
                        face_crops.append(face_crop)
                        faces_indexes.append(id)

            if model and inference and face_crops:
                prediction_batch = tf.stack(face_crops, axis=0) # Stack faces to predict entire batch in 1 forward pass, adds batch dim
                probs = model(prediction_batch, training=False) # Probas outputs

                for id, face_idx in enumerate(faces_indexes):
                    self.prediction_smoother[face_idx].update(probs[id])
                    pred_class, pred_prob = self.prediction_smoother[face_idx].smoothed_prediction()
                    self.faces_predictions[face_idx] = (pred_class, pred_prob)

            if model:
                for id, coords in enumerate(bounding_boxes):
                    if id in self.faces_predictions:
                        pred_class, pred_prob = self.faces_predictions[id]

                        if pred_prob >= 0.3:
                            text_color=(255, 255, 255)
                        else:
                            text_color=(0, 0, 255) # If model has low prob, prediction shown in red

                        cv2.putText(img=img,
                                    text=f"Emotion: {pred_class} - {int(pred_prob*100)}%",
                                    org=(coords[0], coords[1] + coords[3] + 50),
                                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                                    fontScale=2,
                                    color=text_color,
                                    thickness=1)

            # Clean up when face leaves frame
            lost_ids = set(self.faces_predictions.keys()) - detected_ids
            for lost_id in lost_ids:
                del self.faces_predictions[lost_id]
                self.prediction_smoother[lost_id].probs_window.clear()

        return img, id, self.faces_predictions


    def draw_boundingbox(self,
                         img: np.ndarray,
                         bbox: tuple,
                         length: int = 30,
                         thickness: int = 5
        ) -> np.ndarray:
        """
        Allows to draw precises bounding box around detected faces.
        """
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

    def preprocess(self,
                   img: np.ndarray,
                   bbox: tuple,
                   pad: int = 0
        ) -> np.ndarray:
        """
        Preprocess an inputed image to the right input format for model inference.

        arg
        ----
        pad : int
            - Quantity of pixel padding aroung bounding box.

        returns
        ----
        face_crop : np.ndarray
        """
        x1, y1, width, height = bbox
        x2, y2 = x1 + width, y1 + height

        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(img.shape[1], x2 + pad)
        y2 = min(img.shape[0], y2 + pad)

        face_crop = img[y1:y2, x1:x2]
        face_crop = cv2.resize(face_crop, IMAGE_SIZE) # Resize img to 48x48 pixels
        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY) # Change from rgb to grayscale

        return face_crop


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = FaceDetector(detect_conf=0.5)
    model = load_model(model_name=FER_MODEL)

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

        img, id, predictions = detector.detect_faces(img, model)

        if id==None:
            cv2.putText(img,
                        text='No face detected',
                        org=(1450,70),
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=3,
                        color=(0, 0, 200),
                        thickness=4)

        cv2.imshow("Real-Time FER", img)

        key = cv2.waitKey(30)

        if key == ord("q"): # Press 'Q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
