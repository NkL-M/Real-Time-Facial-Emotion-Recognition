"""
Facial Emotion Recognition (FER) Pipeline Module.

This module provides a real-time pipeline for detecting faces, performing emotion
recognition, and visualizing results. It includes classes for smoothing predictions
over time and managing the end-to-end workflow of face detection, preprocessing,
inference, and visualization.
"""

import time
import numpy as np
from collections import deque, defaultdict
from emotion_recognition.face_detection.facial_detection import FaceDetector
from emotion_recognition.face_detection.inference import FERModel
from emotion_recognition.face_detection.visuals import draw_boundingbox, draw_facial_emotion_detection
from emotion_recognition.params import EMOTIONS_CLASSES, ONNX_MODEL_PATH, WINDOW_LENGHT, FRAME_PRED_STRIDE


class PredictionSmoother():
    """
    A class for smoothing predictions over a rolling window of frames.

    This class maintains a rolling window of probability predictions and computes
    smoothed predictions by averaging probabilities over the window. This helps
    reduce noise and improve stability in real-time emotion recognition.

    Attributes
    ----
    probs_window : deque
        - A fixed-length deque to store recent probability predictions.
    """
    def __init__(self, window_lenght: int = 10):
        self.probs_window = deque(maxlen=window_lenght)

    def update(self, probs: np.ndarray) -> None:
        """
        Appends a new array of probabilities to the rolling window.

        Args
        ----
        probs : np.ndarray
            - Probability array for each emotion class.
        """
        self.probs_window.append(probs)

    def smoothed_prediction(self) -> tuple[str, float]:
        """
        Computes a smoothed prediction by averaging probabilities over the rolling window.

        Returns
        ----
        (classes_prediction, predicted_class, predicted_prob) : tuple
            - probs_dict (dict): Dictionary of smoothed probabilities for each emotion class.
            - predicted_class (str): The emotion class with the highest smoothed probability.
            - predicted_prob (float): The smoothed probability of the predicted class.
        """
        smoothed_probs = np.mean(self.probs_window, axis=0)
        classes_prediction = {label.capitalize(): float(score) for label, score in zip(EMOTIONS_CLASSES, smoothed_probs)}
        predicted_class = max(classes_prediction, key=classes_prediction.get)
        predicted_prob = classes_prediction[predicted_class]
        return (classes_prediction, predicted_class, predicted_prob)


class FERPipeline():
    """
    A class for managing the end-to-end Facial Emotion Recognition (FER) pipeline.

    This class integrates face detection, emotion inference, and visualization into a
    cohesive workflow. It supports real-time processing of video frames, with optional
    drawing of detection and prediction results.
    """
    def __init__(self, draw: bool = True):
        """
        Initializes the FERPipeline with face detection, inference, and visualization components.

        Args
        ----
        draw : bool (optional)
            - Flag to enable/disable drawing of results. Defaults to `True`.
        """
        self.detector = FaceDetector()
        self.fer_model = FERModel(ONNX_MODEL_PATH)
        self.prediction_smoother = defaultdict(lambda: PredictionSmoother(window_lenght=WINDOW_LENGHT))
        self.draw = draw
        self.results = {}
        self.frame_count = 0

    def pipeline_flow(self, img: np.ndarray) -> tuple:
        """
        Processes an input image through the FER pipeline, including face detection,
        emotion inference, and visualization.

        Args
        ----
        img : np.ndarray
            - Input image as a NumPy array.

        Returns
        ----
        (img, self.results) : tuple
        img : np.ndarray
            - Processed image with optional drawings.

        results : dict
            - Dictionary of inference results for each detected face.
        """
        inputs = []
        self.frame_count += 1
        inference = (self.frame_count < FRAME_PRED_STRIDE or  # Predict frame 1 and 2 as the first preds
                     self.frame_count % FRAME_PRED_STRIDE==0) # Predict every 3 frame to reduce compute load

        detected_ids, bounding_boxes, detection_scores = self.detector.detect_faces(img)

        if inference:
            for bbox in bounding_boxes:
                input = self.fer_model.preprocess(img, bbox, pad=30).astype(np.float32)
                inputs.append(input)

        if inference and inputs:
            inference_start = time.time()
            inputs_batch = np.stack(inputs, axis=0, dtype=np.float32) # Stack faces to predict entire batch in 1 forward pass
            inputs_batch = np.expand_dims(inputs_batch, axis=3)
            probs_batch = self.fer_model.onnx_predict(inputs_batch)
            inference_time = (time.time() - inference_start) * 1000

        for face_idx, bounding_box, detection_score in zip(detected_ids, bounding_boxes, detection_scores):
            if inference and inputs:
                self.prediction_smoother[face_idx].update(probs_batch[face_idx])
                probs_dict, pred_class, pred_prob = self.prediction_smoother[face_idx].smoothed_prediction()
                self.results[face_idx] = dict(face_detection_score=round(detection_score, 2),
                                              bounding_box=bounding_box,
                                              proba_by_class=probs_dict,
                                              predicted_class=pred_class,
                                              predicted_prob=pred_prob,
                                              inference_time_ms=round(inference_time, 2))

        for face_idx, bbox in enumerate(bounding_boxes):
            if self.draw:
                img = draw_boundingbox(img, bbox)
                if face_idx in self.results.keys():
                    img = draw_facial_emotion_detection(img,
                                                        face_idx,
                                                        bbox,
                                                        self.results[face_idx],
                                                        font_scale=2,
                                                        font_thickness=1)

        # Remove IDs no longer detected
        active_ids = set(detected_ids)
        stale_ids = set(self.results.keys()) - active_ids
        for stale_id in stale_ids:
            del self.results[stale_id]
            if stale_id in self.prediction_smoother:
                del self.prediction_smoother[stale_id]

        return img, self.results
