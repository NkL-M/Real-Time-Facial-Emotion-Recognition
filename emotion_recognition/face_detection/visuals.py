"""
Computer Vision Visuals Utility Module

This module provides utility functions for drawing annotations on images,
including FPS counters, bounding boxes, and facial emotion detection results.
All functions use OpenCV (cv2) for rendering and operate on NumPy arrays.
"""
import numpy as np
import cv2

from emotion_recognition.params import CONFIDENCE_THRESHOLD


def draw_boundingbox(
    img: np.ndarray,
    bounding_box: tuple,
    length: int = 30,
    thickness: int = 5
    ) -> np.ndarray:
    """
    Draws a bounding box with corner markers around a detected region in an image.

    Args
    ----
    img : np.ndarray
        - Input image as a NumPy array.

    bounding_box : Tuple
        - (x, y, w, h) representing the top-left coordinates, width, and height of the bounding box.

    length : int
        - Length of the corner markers (default: 30).

    thickness : int
        - Thickness of the corner markers (default: 5).

    Returns
    ----
    img : np.ndarray
        - The image with the bounding box and corner markers drawn on it.
    """
    x, y, w, h = bounding_box
    x1, y1 = x + w, y + h

    cv2.rectangle(img, bounding_box, (255, 255, 255), 2)

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


def draw_facial_emotion_detection(
    img: np.ndarray,
    id: int,
    bbox: np.ndarray,
    results: dict,
    font_scale: int = 2,
    font_thickness: int = 1
    ) -> np.ndarray:
    """
    Draws facial detection and emotion prediction results on an image.

    Args
    ----
    img : np.ndarray
        - Input image as a NumPy array.

    id : int
        - Unique identifier for the detected face.

    bbox : np.ndarray
        - Bounding box coordinates of the detected face as [x, y, width, height].

    results : dict
        - 'face_detection_score' (float - confidence score for face detection).
        - 'predicted_class' (str - predicted emotion class).
        - 'predicted_prob': (float - confidence score for the predicted emotion).

    font_scale : int
        - Font scale for the text (default: 2).

    font_thickness : int
        - Thickness of the text (default: 1).

    Returns
    ----
    img : np.ndarray
        - The image with facial detection and emotion prediction results drawn on it.
    """
    detection_score=results['face_detection_score']
    pred_class=results['predicted_class']
    pred_prob=results['predicted_prob']

    if detection_score >= CONFIDENCE_THRESHOLD - 1.0:
        text_color_mp=(255, 255, 255)
    else:
        text_color_mp=(0, 0, 255)

    cv2.putText(
        img=img,
        text=f"Visage {id+1} - {int(detection_score*100)} %",
        org=(bbox[0], bbox[1] - 25),
        fontFace=cv2.FONT_HERSHEY_PLAIN,
        fontScale=font_scale,
        color=text_color_mp,
        thickness=font_thickness
        )

    if pred_prob >= CONFIDENCE_THRESHOLD:
        text_color=(255, 255, 255)
    else:
        text_color=(0, 0, 255)

    cv2.putText(
        img=img,
        text=f"Emotion: {pred_class} - {int(pred_prob*100)}%",
        org=(bbox[0], bbox[1] + bbox[3] + 50),
        fontFace=cv2.FONT_HERSHEY_PLAIN,
        fontScale=font_scale,
        color=text_color,
        thickness=font_thickness)

    return img

def draw_fps(
    img: np.ndarray,
    fps: str,
    font_scale: int = 2,
    font_thickness: int = 3
    ) -> np.ndarray:
    """
    Draws the FPS value on an image using OpenCV.

    Args
    ----
    img : np.ndarray
        - Input image as a NumPy array.

    fps : float
        - FPS value to display.

    font_scale : int
        - Font scale for the text (default: 3).

    font_thickness : int
        - Thickness of the text (default: 4).

    Returns
    ----
    img : np.ndarray
        - The image with the FPS value drawn on it.
    """
    cv2.rectangle(img,
                  (20, 20),
                  (20 + 210, 85),
                  color=(80, 180, 0),
                  thickness=-1)

    cv2.putText(img=img,
                text=f"{str(int(fps))} FPS",
                org=(40,70),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=font_scale,
                color=(220, 220, 220),
                thickness=2)

    return img
