import cv2
import numpy as np
import tensorflow as tf

from emotion_recognition.params import *
from emotion_recognition.src.data import *
from emotion_recognition.src.model import *
from emotion_recognition.src.registry import save_model, save_results, load_model
from emotion_recognition.params import EMOTION_DICT
from emotion_recognition.face_detection.face_detection import FaceDetector

def training(data_ratio: float = 0.2,
             learning_rate: float = 0.01,
             epochs: int = 50,
             patience: int = 10,
             checkpoint: bool = True,
             reduce_lr: bool = True,
             registry: bool = True,
             data_aug: bool = True,
             class_weights: bool = True,
             save_name: str = 'default_model01'
    ) -> tuple[Model, dict]:
    """
    Load data, train model on training set.

    args
    ----
    data_ratio : float
        - Ratio of data to load.

    learning_rate : float
        - Learning rate (hyper-parameter).

    patience : int
        - Number of times validation metric not-improving before stopping training.
        Also correspond to the warm-up period.

    checkpoint : bool
        - Save weigths during training.

    reduce_lr : bool
        - Wether the learning will reduce when loss doesn't decrease

    registry : bool
        - Save parameters, last/best validation metric value, and entire model.

    data_aug : bool

    class_weights : bool
        - Class weight added to each image for handling class imbalance

    save_name : str
        - Name to which the registered model's data will be saved as.

    returns
    ----
    (model, history) : tuple
    """
    if class_weights or data_aug:
        train_data, val_data = class_weight_and_augment(data_ratio=data_ratio,
                                                   data_aug=data_aug,
                                                   class_weights=class_weights)

    else:
        train_data, val_data = load_data(dataset_type='train',
                                         batch_size=BATCH_SIZE,
                                         image_size=IMAGE_SIZE,
                                         fetch_ratio=data_ratio
                                        )

    # model = initialize_baseline_model(             # Baseline model architecture
    #     input_shape=INPUT_SHAPE
    # )
    # model = initialize_transfer_learning_model(         # Tranfer Learning Model
    #     model_name=model_name,
    #     input_shape=INPUT_SHAPE
    # )

    model = initialize_custom_model(INPUT_SHAPE)
    model = compile_model(model=model, learning_rate=learning_rate)

    model, history = train_model(
        model=model,
        train_dataset=train_data,
        val_dataset=val_data,
        epochs=epochs,
        patience=patience,
        checkpoint=checkpoint,
        reduce_lr=reduce_lr,
        save_name=save_name
    )

    if registry:
        params = dict(
            model_save_name=save_name,
            learning_rate=learning_rate,
            nb_epochs_set=epochs,
            early_stopping_patience=patience,
            batch_size=BATCH_SIZE,
            split_ratio=data_ratio
        )

        best_val_metric = np.max(history.history['val_f1_macro'])

        save_results(
            params=params,
            metrics=best_val_metric,
            model_name=save_name
        )

        save_model(
            model=model,
            model_name=save_name
        )

    print(f'Validation score= {round(best_val_metric, 3)}')

    return model, history

def finetuning(model_name : str,
               latest_model: bool = False,
               unfroze_layers: float = 15,
               data_ratio: float = 0.2,
               learning_rate: float = 0.01,
               epochs: int = 50,
               patience: int = 10,
               checkpoint: bool = True,
               reduce_lr: bool = True,
               registry: bool = True,
               save_name: str = 'fine_tuning_model01'
            ) -> tuple[Model, dict]:
    """
    Load data, train model on training set.

    args
    ----
    model_name : str
        - Name of the model to load.

    data_ratio : float
        - Ratio of data to load.

    lr : float
        - Learning rate (hyper-parameter).

    patience : int
        - Number of times validation metric not-improving before stopping training.
        Also correspond to the warm-up period.

    checkpoint : bool
        - Save weigths during training.

    registry : bool
        - Save parameters, last/best validation metric value, and entire model.

    save_name : str
        - Name to which the registered model's parameters, weights, metrics will be saved as.

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

    model = load_model_for_finetuning(model_name=model_name,
                                      latest_model=latest_model,
                                      unfroze_layers=unfroze_layers)

    model.summary()

    model = compile_model(
        model=model,
        learning_rate=learning_rate
    )

    model, history = train_model(
        model=model,
        train_dataset=train_data,
        val_dataset=val_data,
        epochs=epochs,
        patience=patience,
        checkpoint=checkpoint,
        reduce_lr=reduce_lr,
        save_name=save_name
    )

    if registry:
        params = dict(
            model_save_name=save_name,
            learning_rate=learning_rate,
            nb_epochs_set=epochs,
            early_stopping_patience=patience,
            batch_size=BATCH_SIZE,
            split_ratio=data_ratio
        )

        best_val_metric = np.max(history.history['val_accuracy'])

        save_results(
            params=params,
            metrics=best_val_metric,
            model_name=save_name
        )

        save_model(
            model=model,
            model_name=save_name
        )

    print(f'Validation score= {round(best_val_metric, 3)}')

    return model, history

def evaluate_model(model: Model,
                   ratio: float = 0.2
    ) -> float:
    """
    Evaluates model's performance on test dataset.

    arg
    ----
    ratio : float
        - Ratio of the test data to evaluate model on.

    returns
    ----
    eval_score : float
    """
    test_data = load_data(
        dataset_type='test',
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        fetch_ratio=ratio,
        validation_split=0.2
    )

    eval_score = model.evaluate(test_data)

    return eval_score

def predict_image(image_path: Path,
                  model: Model
    ) -> str:
    """
    Predict images by inputing paths.

    arg
    ----
    image_size : tuple


    returns
    ----
    prediction : str

    Expected prediction time for an image (%timeit):
    30.8 ms ± 298 μs per loop (mean ± std. dev. of 7 runs, 10 loops each)
    """
    model = load_tf_model(FER_MODEL)

    img = tf.io.read_file(str(image_path))       # img = Image.open(path).convert("RGB")
    img = tf.image.decode_image(img, channels=NB_CHANNELS)
    img = tf.image.resize(img, size=INPUT_SHAPE)

    x_preprocessed = tf.expand_dims(input=img, axis=0) # Adding batch dimension to shape = (1, 48, 48, 1)
    outputs = model(x_preprocessed)
    prediction_index = tf.argmax(outputs[0]).numpy()
    prediction = EMOTION_DICT[prediction_index]
    return prediction

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = FaceDetector(detect_conf=0.5)
    model = load_model(FER_MODEL)

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

        img, id, preds = detector.detect_faces(img, model)

        if id==None:
            cv2.putText(img,
                        text='No face detected',
                        org=(1450,70),
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=3,
                        color=(0, 0, 200),
                        thickness=4)

        cv2.imshow("Image", img)

        key = cv2.waitKey(30)

        if key == ord("q"): # Press 'Q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
