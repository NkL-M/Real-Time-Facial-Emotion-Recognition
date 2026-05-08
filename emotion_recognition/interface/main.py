import tensorflow as tf
from emotion_recognition.params import *
from emotion_recognition.src.data import load_data
from emotion_recognition.src.model import *
from emotion_recognition.src.registry import save_model, save_results, load_model

from emotion_recognition.face_detection.face_detection_mediapipe import FaceDetector
# from emotion_recognition.face_detection import face_detection_opencv
from emotion_recognition.face_detection.face_landmarks import FacialTracker


def train(data_ratio=0.2,
          lr=0.01,
          epochs=50,
          patience=10,
          checkpoint=True,
          registry=True,
          save_name='default_model01'
        ) -> tuple[tf.keras.Model, dict]:
    """
    Load data, train model on training set.

    args
    ----
    data_ratio : float

    Ration of data to load.

    lr : float

    Learning rate (hyper-parameter).

    patience : int

    Number of times validation metric not-improving before stopping training.
    Also correspond to the warm-up.

    checkpoint : bool

    Save weigths during training.

    registry : bool

    Save parameters, last/best validation metric value, and entire model.

    save_name : str

    Name to which the registered model's data will be saved as.

    returns
    ----
    (model, history) : tuple
    """
    train_data, val_data = load_data(
        dataset_type='train',
        batch_size=BATCH_SIZE,
        input_size=IMAGE_SIZE,
        fetch_ratio=data_ratio
    )

    model = initialize_baseline_model(             # Baseline model architecture
        input_shape=INPUT_SHAPE
    )

    # model = initialize_model(                             # Model architecture
    #     input_shape=MODEL_INPUT_SHAPE
    # )

    # model = initialize_transfer_learning_model(         # Tranfer Learning Model
    #     model_name='vgg16',
    #     input_shape=INPUT_SHAPE
    # )

    model = compile_model(
        model=model,
        learning_rate=lr
    )

    model, history = train_model(
        model=model,
        train_dataset=train_data,
        val_dataset=val_data,
        epochs=epochs,
        es_patience=patience,
        checkpoint=checkpoint,
        save_name=save_name
    )

    if registry:
        params = dict(
            model_save_name=save_name,
            learning_rate=lr,
            nb_epochs_set=epochs,
            early_stopping_patience=patience,
            batch_size=BATCH_SIZE,
            split_ratio=data_ratio
        )

        val_metric = history.history['val_accuracy'][-1]

        save_results(
            params=params,
            metrics=val_metric,
            model_name=save_name
        )

        save_model(
            model=model,
            model_name=save_name
        )

    print(f'Validation metric= {val_metric}')

    return model, history


def evaluate(model, test_data_ratio=0.2) -> float:
    """
    Function that evaluates model performance

    returns
    ----
    eval_score : float
    """
    test_data = load_data(
        dataset_type='test',
        batch_size=BATCH_SIZE,
        input_size=IMAGE_SIZE,
        fetch_ratio=test_data_ratio
    )

    eval_score = evaluate_model(
        model=model,
        test_dataset=test_data
    )

    return eval_score

def predict(model: tf.keras.Model, X_new):
    """

    """
    X_new = tf.convert_to_tensor()
    prediction = model.predict(X_new)
    return prediction

def main():
    pass

if __name__ == '__main__':
    train(ratio=0.2)
    # evaluate()
    main()
