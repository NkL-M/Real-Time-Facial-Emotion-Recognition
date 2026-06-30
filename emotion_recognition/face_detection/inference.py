"""
Model Inference Module for Facial Emotion Recognition (FER).

This module provides a class and utilities for performing inference using a pre-trained
Facial Emotion Recognition (FER) model. It supports ONNX runtime for efficient inference
and includes preprocessing and prediction methods.
"""

import cv2
import numpy as np
import onnxruntime as ort

from keras import Model
from pathlib import Path
from emotion_recognition.params import IMAGE_SIZE, ONNX_MODEL_PATH, TF_MODEL
from emotion_recognition.src.registry import load_tf_model

class FERModel():
    """
    This class initializes an ONNX runtime session for inference and provides methods
    for preprocessing input images and predicting emotions.

    Attributes
    ----
    session = ort.InferenceSession
        - ONNX runtime session for model inference.

    input_name : str
        - Name of the model's input layer.

    output_name : str
        - Name of the model's output layer.
    """
    def __init__(self, model_path: Path = ONNX_MODEL_PATH):
        """
        Arg
        ----
        model_path (Path, optional): Path to the ONNX model file. Defaults
        to `ONNX_MODEL_PATH` variable from `params.py`.
        """
        # Tensorflow model instantiation
        self.tf_model = load_tf_model(TF_MODEL)

        # ONNX model instantiation
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(
        self,
        img: np.ndarray,
        bbox: tuple,
        pad: int = 0
        ) -> np.ndarray:
        """
        Preprocesses an input image to the correct format for model inference.

        Args
        ----
        img : np.ndarray
            - Input image as a NumPy array.

        bbox : tuple
            - Bounding box coordinates in the format (x, y, width, height).

        pad : int (optional)
            - Number of pixels to pad around the bounding box. Defaults to 0.

        Returns
        ----
        input : np.ndarray
            - Preprocessed image (grayscale and resized to `IMAGE_SIZE`).
        """
        x1, y1, width, height = bbox
        x2, y2 = x1 + width, y1 + height

        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(img.shape[1], x2 + pad)
        y2 = min(img.shape[0], y2 + pad)

        face_crop = img[y1:y2, x1:x2]
        face_crop = cv2.resize(face_crop, IMAGE_SIZE) # Resize face crop to 48x48 pixels
        input = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY) # Change from rgb to grayscale

        return input

    def tf_predict(self, inputs: np.ndarray) -> tuple:
        """

        returns
        ----
        (probs_array, probs_dict, predicted_class, predicted_prob) : tuple

        probs_array : np.ndarray

        probs_dict : dict

        predicted_class : str

        predicted_prob : float
        """
        outputs = self.tf_model(inputs, training=False) # Probas outputs
        return outputs

    def onnx_predict(self, inputs: np.ndarray) -> tuple:
        """
        Predicts the probability of each emotion class for the input image with
        an ONNX-based inference.

        Args
        ----
        inputs : np.ndarray
            - Preprocessed input image(s) as a NumPy array.

        Returns
        ----
        outputs[0] : np.ndarray
            - Probability scores for each emotion class.
        """
        outputs = self.session.run([self.output_name], {self.input_name: inputs})
        return outputs[0]
