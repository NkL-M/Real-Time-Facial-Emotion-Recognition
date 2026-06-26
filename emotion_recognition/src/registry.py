import time
import glob
import pickle
import onnx
import tensorflow as tf
import tf2onnx
from colorama import Fore, Style
from keras import models, Model

from emotion_recognition.params import MODELS_REGISTRY_DIR


def save_model(model: Model = None,
               model_name: str = 'default-model01'
    ) -> None:
    """
    Persist trained model locally at: MODELS_REGISTRY_DIR/saved_models/timestamp_model_name.keras
    """
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

    model_path = MODELS_REGISTRY_DIR/'saved_models'/f'{timestamp}_{model_name}.keras'

    # Save model locally
    model.save(model_path)
    print(Fore.BLUE + f"\nModel saved locally as '{timestamp}_{model_name}.keras'" + Style.RESET_ALL)


def save_results(params: dict,
                 metrics: dict,
                 model_name: str = 'default-model01'
    ) -> None:
    """
    Persist params & metrics locally on the hard drive at:
        MODELS_REGISTRY_DIR/saved_params/params_timestamp_model_name.pickle
        MODELS_REGISTRY_DIR/saved_metrics/metrics_timestamp_model_name.pickle
    """
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

    # Save params locally
    if params is not None:
        params_path = MODELS_REGISTRY_DIR/'saved_params'/f'params_{timestamp}_{model_name}.pickle'
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # Save metrics locally
    if metrics is not None:
        metrics_path = MODELS_REGISTRY_DIR/'saved_metrics'/f'metrics_{timestamp}_{model_name}.pickle'
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    print(Fore.BLUE + f"\nResults saved locally" + Style.RESET_ALL)


def load_tf_model(model_name: str = 'default-model01',
               latest_model: bool = True
    ) -> Model:
    """
    Load a saved model stored on disk

    arg
    ----
    latest_model : bool
        - Load latest one according to timestamp.
        If `False`, specify model full name in `model_name` with timestamp `Years-Months-Days_Hours-Minutes-Seconds`.

        Exemple: `model_name='2026-01-31_08-37-58_default-model01'`

    returns
    ----
    model: Model
    """
    models_paths_match = str(MODELS_REGISTRY_DIR / 'saved_models' / f'*{model_name}*.keras')
    models_list = sorted(glob.glob(models_paths_match))

    print(Fore.WHITE + f"\nLoad model from local registry..." + Style.RESET_ALL)

    if not models_list:
        print(Fore.RED + f"\nNo model found on local registry" + Style.RESET_ALL)
        return None

    if latest_model == True:
        model_path = models_list[-1]
    else:
        model_path = str(MODELS_REGISTRY_DIR/'saved_models'/ f'*{model_name}*.keras')

    model = models.load_model(model_path)

    print(Fore.GREEN + f"\nModel loaded from local disk, from the following path: {model_path}" + Style.RESET_ALL)

    return model


def export_tf_to_onnx(model_name: str) -> None:
    """
    Export a Tensorflow model to the ONNX format for faster inference.
    """
    model = load_tf_model(model_name)

    spec = (
        tf.TensorSpec(
            model.inputs[0].shape,
            model.inputs[0].dtype,
            name="input",
        ),
    )

    onnx_model_keras, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        opset=13
    )

    onnx_model_path = MODELS_REGISTRY_DIR / 'saved_models' / f'{model_name}_v2.onnx'

    with open(onnx_model_path, "wb") as f:
        f.write(onnx_model_keras.SerializeToString())

    onnx.checker.check_model(onnx.load(onnx_model_path))

    print(Fore.GREEN + f"\nONNX Model exported on local disk, at the following path: {onnx_model_path}" + Style.RESET_ALL)
