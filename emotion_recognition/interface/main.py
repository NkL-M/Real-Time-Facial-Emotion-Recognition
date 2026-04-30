from emotion_recognition.params import *
from emotion_recognition.src.data import load_data
from emotion_recognition.src.model import initialize_model, compile_model, train_model#, evaluate_model, prediction
from emotion_recognition.src.registry import save_model, load_model

from emotion_recognition.face_detection.face_detection_mediapipe import FaceDetector
# from emotion_recognition.face_detection import face_detection_opencv
from emotion_recognition.face_detection.face_landmarks import FacialTracker


def get_data():
    """

    """
    pass

def train():
    """

    """
    train_data, val_data = load_data(dataset_type='train',
                                     batch_size=BATCH_SIZE,
                                     input_size=IMAGE_SIZE,
                                     fetch_ratio=0.2)

    test_data = load_data(dataset_type='test',
                          batch_size=BATCH_SIZE,
                          input_size=IMAGE_SIZE,
                          fetch_ratio=0.2)
    pass


def evaluate():
    """

    """
    pass

def predict():
    """

    """
    pass

def main():
    pass

if __name__ == '__main__':
    main()
