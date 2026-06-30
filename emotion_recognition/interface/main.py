import cv2
import numpy as np
import tensorflow as tf
import onnxruntime as ort

from emotion_recognition.params import *
from emotion_recognition.src.data import *
from emotion_recognition.src.model import *
from emotion_recognition.src.registry import save_model, save_results
from emotion_recognition.interface.pipeline import FERPipeline
from emotion_recognition.face_detection.visuals import draw_fps

def training(
    data_ratio: float = 0.2,
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

        best_val_metric = np.max(history.history['val_f1_macro']) # TODO verify

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


def evaluate_model(
    model: Model,
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
        fetch_ratio=ratio
    )

    eval_score = model.evaluate(test_data)

    return eval_score


def predict_images_list(images_paths: list) -> dict:
    """
    Predict an image by inputing a list of paths.

    returns
    ----
    resutls : dict
        - probabilities_dict : dict
        - prediction_class : str
        - prediction_prob : float
    """
    def preprocess_img(path):
        input = tf.io.read_file(str(path)) # img = Image.open(path).convert("RGB")
        input = tf.image.decode_image(input, channels=NB_CHANNELS)
        input = tf.image.resize(input, size=IMAGE_SIZE)
        return input

    preprocessed_imgs = []

    for path in images_paths:
        preprocessed_imgs.append(preprocess_img(path))

    batch_input = np.stack(preprocessed_imgs)

    # Inference ONNX
    session = ort.InferenceSession(ONNX_MODEL_PATH)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    outputs = session.run([output_name], {input_name: batch_input})[0]

    results = {}

    for path, output in zip(images_paths, outputs):
        probs_dict = {label.capitalize(): score for label, score in zip(EMOTIONS_CLASSES, output)}
        pred_class = max(probs_dict, key=probs_dict.get)
        pred_prob = probs_dict[pred_class]
        results[Path(path).name] = dict(probabilities_dict=probs_dict,
                             prediction_class=pred_class,
                             prediction_prob=pred_prob)

        print(f"{Path(path).name}: Predicted class ({pred_class}) - Confidence ({pred_prob:.2%})")

    return results


def main():
    """
    Runs a real-time Facial Emotion Recognition (FER) demo using the default camera.

    This function initializes a video capture device, processes each frame through
    the FER pipeline, and displays the results in a window. Press 'Q' to stop.
    """
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FPS, CAP_FPS)
    detector = FERPipeline(draw=True)

    while True:
        success, img = cap.read()
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        img = draw_fps(img, fps, font_scale=3, font_thickness=4)

        img, results = detector.pipeline_flow(img)

        if not results:
            cv2.putText(img,
                        text="No face detected",
                        org=(1450,70),
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=3,
                        color=(0, 0, 220),
                        thickness=3)

        cv2.putText(img,
                    text="Press 'Q' to quit",
                    org=(30,1050),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=2,
                    color=(220, 220, 220),
                    thickness=2)

        cv2.imshow("Real-Time FER", img)

        key = cv2.waitKey(30)

        if key == ord("q"): # Press 'Q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
