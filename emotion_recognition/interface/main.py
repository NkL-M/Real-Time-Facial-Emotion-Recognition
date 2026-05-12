import numpy as np
import tensorflow as tf
from emotion_recognition.params import *
from emotion_recognition.src.data import *
from emotion_recognition.src.model import *
from emotion_recognition.src.registry import save_model, save_results, load_model

from emotion_recognition.face_detection.face_detection_mediapipe import FaceDetector
# from emotion_recognition.face_detection import face_detection_opencv
from emotion_recognition.face_detection.face_landmarks import FacialTracker

def train(model_name: str = 'resnet50',
          data_ratio: float = 0.2,
          lr: float = 0.01,
          epochs: int = 50,
          patience: int = 10,
          checkpoint: bool = True,
          registry: bool =True,
          save_name: str ='default_model01'
        ) -> tuple[Model, dict]:
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
    train_data = load_data(
        dataset_type='train',
        batch_size=64,
        image_size=IMAGE_SIZE,
        fetch_ratio=data_ratio
    )

    val_data = load_data(
        dataset_type='val',
        batch_size=64,
        image_size=IMAGE_SIZE,
        fetch_ratio=data_ratio
    )

    # train_data, val_data = load_data_val_split(
    #     dataset_type='train',
    #     batch_size=BATCH_SIZE,
    #     image_size=IMAGE_SIZE,
    #     fetch_ratio=data_ratio
    # )

    # model = initialize_baseline_model(             # Baseline model architecture
    #     input_shape=INPUT_SHAPE
    # )

    # model = initialize_model(                               # Model architecture
    #     input_shape=INPUT_SHAPE
    # )

    model = initialize_transfer_learning_model(         # Tranfer Learning Model
        model_name=model_name,
        input_shape=INPUT_SHAPE
    )

    model = compile_model(
        model=model,
        # learning_rate=lr,
        trainset=train_data
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

        best_val_metric = np.max(history.history['val_accuracy']) # history.history['val_accuracy'][-1]

        save_results(
            params=params,
            metrics=best_val_metric,
            model_name=save_name
        )

        save_model(
            model=model,
            model_name=save_name
        )

    print(f'Validation metric= {best_val_metric}')

    return model, history

def fine_tuning(model_name : str,
                latest_model: bool = False,
                unfroze_layers: float = 15,
                data_ratio: float = 0.2,
                lr: float = 0.01,
                epochs: int = 50,
                patience: int = 10,
                checkpoint: bool = True,
                registry: bool = True,
                save_name: str = 'fine_tuning_model01'
            ) -> tuple[Model, dict]:
    """
    Load data, train model on training set.

    args
    ----
    model_name : str

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
    train_data = load_data(
        dataset_type='train',
        batch_size=64,
        image_size=IMAGE_SIZE,
        fetch_ratio=data_ratio
    )

    val_data = load_data(
        dataset_type='val',
        batch_size=64,
        image_size=IMAGE_SIZE,
        fetch_ratio=data_ratio
    )

    model = initialize_finetuning_model(model_name=model_name,
                                        latest_model=latest_model,
                                        unfroze_layers=unfroze_layers)

    model = compile_model(
        model=model,
        # learning_rate=lr,
        trainset=train_data
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

        best_val_metric = np.max(history.history['val_accuracy']) # history.history['val_accuracy'][-1]

        save_results(
            params=params,
            metrics=best_val_metric,
            model_name=save_name
        )

        save_model(
            model=model,
            model_name=save_name
        )

    print(f'Validation metric= {best_val_metric}')

    return model, history

def evaluate_model(
        model: Model,
        test_data_ratio: tf.data.Dataset = 0.2
    ) -> float:
    """
    Evaluates model's performance on test dataset.

    arg
    ----
    test_data_ratio : float
    Ration of the test data to evaluate model on.

    returns
    ----
    eval_score : float
    """
    test_data = load_data(
        dataset_type='test',
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        fetch_ratio=test_data_ratio
    )

    eval_score = model.evaluate(test_data)

    return eval_score

def predict(X_new: Path,
            model_name: str = 'vgg16_model_01',
            latest_model: bool = False
        ) -> tuple:
    """
    TODO Doc
    Generates output predictions for the input samples.

    returns
    ----
    prediction : np.array
    """
    model = load_model(model_name=model_name,
                       latest_model=latest_model)

    # X_new = tf.
    X_new = tf.convert_to_tensor(X_new)

    prediction = model.predict(X_new)

    print(f"Probability: {prediction[0]}")

    return prediction[1]

def main():
    pass

if __name__ == '__main__':
    main()
