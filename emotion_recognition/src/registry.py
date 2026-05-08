from pathlib import Path
import time
import glob
import pickle
from colorama import Fore, Style
import keras
from emotion_recognition.params import *


def save_model(
        model: keras.Model = None,
        model_name: str = 'default-model01'
    ) -> None:
    """
    Persist trained model locally at:

        {MODELS_REGISTRY_DIR}/saved_models/{model_name}_{timestamp}.keras
    """
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

    model_path = MODELS_REGISTRY_DIR/'saved_models'/f'{timestamp}_{model_name}.keras'

    # Save model locally
    model.save(model_path)
    print(Fore.BLUE + f"\nModel saved locally as '{timestamp}_{model_name}.keras'" + Style.RESET_ALL)


def save_results(
        params: dict,
        metrics: dict,
        model_name: str = 'default-model01'):
    """
    Persist params & metrics locally on the hard drive at:

        {MODELS_REGISTRY_DIR}/saved_params/params_{timestamp}_{model_name}.pickle
        {MODELS_REGISTRY_DIR}/saved_metrics/metrics_{timestamp}_{model_name}.pickle
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


def load_model(
        model_name: str = 'default-model01',
        latest_model=True
    ) -> keras.Model:
    """
    Return a saved model stored on disk

    arg
    ----
    latest_model : bool
        - Load latest one according to timestamp.
        If `False`, specify model full name in `model_name` with timestamp `Years-Months-Days_Hours-Minutes-Seconds`.

        Exemple: `model_name`= `'default-model01_2026-01-31_08-37-58'`

    Return None (but do not Raise) if no model is found.
    """
    models_paths_match = str(MODELS_REGISTRY_DIR / 'saved_models' / f'{model_name}_*.keras')
    models_list = sorted(glob.glob(models_paths_match))

    print(Fore.WHITE + f"\nLoad model from local registry..." + Style.RESET_ALL)

    if not models_list:
        print(Fore.RED + f"\nNo model found on local registry" + Style.RESET_ALL)
        return None

    if latest_model == True:
        model_path = models_list[-1]
    else:
        model_path = str(MODELS_REGISTRY_DIR/'saved_models'/ f'{model_name}.keras')

    model = keras.models.load_model(model_path)

    print(Fore.GREEN + f"\nModel loaded from local disk" + Style.RESET_ALL)

    return model
