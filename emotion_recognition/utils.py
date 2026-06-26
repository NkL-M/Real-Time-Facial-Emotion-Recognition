""" Module for various functions for plotting. """

import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import onnxruntime as ort
from keras import Model, models
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

from emotion_recognition.src.data import load_data
from emotion_recognition.src.registry import load_tf_model
from emotion_recognition.params import EMOTIONS_CLASSES


def plot_loss_accuracy(history: dict):
    """
    Function to plot the loss and accuracy.
    """
    fig, ax = plt.subplots(1,2, figsize=(20,7))

    # Loss
    ax[0].plot(history.history['loss'])
    ax[0].plot(history.history['val_loss'])

    ax[0].set_title('Model loss')
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].legend(['Train', 'Val'], loc='best')

    ax[0].grid(axis="x",linewidth=0.5)
    ax[0].grid(axis="y",linewidth=0.5)

    ax[0].set_ylim((0,3))

    # Metric (accuracy)
    ax[1].plot(history.history['accuracy'])
    ax[1].plot(history.history['val_accuracy'])

    ax[1].set_title('Model Accuracy')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(['Train', 'Val'], loc='best')

    ax[1].grid(axis="x",linewidth=0.5)
    ax[1].grid(axis="y",linewidth=0.5)

    ax[1].set_ylim((0,1))


def classif_report(model: Model) -> None:
    """
    Print a model's classification report.
    """
    dataset = load_data(dataset_type='test', fetch_ratio=1.0)

    y_true = []
    y_pred = []

    for images, labels in dataset:
        logits = model(images, training=False)
        y_true.extend(tf.argmax(labels, axis=1).numpy())
        y_pred.extend(tf.argmax(logits, axis=1).numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print(classification_report(y_true, y_pred, target_names=EMOTIONS_CLASSES))


def conf_matrix(model: Model, normalize: bool = True) -> None :
    """
    Print a model's confusion matrix.

    arg
    ----
    normalize : bool
        If `normalize=False`, counts values.
    """
    dataset = load_data(dataset_type='test', fetch_ratio=1.0)

    y_true = []
    y_pred = []

    for images, labels in dataset:
        logits = model(images)
        y_true.extend(tf.argmax(labels, axis=1).numpy())
        y_pred.extend(tf.argmax(logits, axis=1).numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Confusion matrix
    conf_mat = confusion_matrix(y_true, y_pred)

    # Normalize
    if normalize:
        conf_mat = conf_mat.astype("float") / conf_mat.sum(axis=1, keepdims=True)

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(data=conf_mat,
                annot=True,
                fmt='.2f' if normalize else 'd',
                cmap='RdPu',
                xticklabels=EMOTIONS_CLASSES,
                yticklabels=EMOTIONS_CLASSES,
                linewidths=0.5,
                linecolor='white',
                square=True,
                cbar_kws={'shrink': 0.8}
                )
    plt.title('Confusion Matrix', fontsize=18, pad=20)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def tf_model_avg_inference_time(model_path: Path,
                                model_format: str = 'TF'):
    """
    Measures and prints the average inference time of a model (TensorFlow or ONNX).

    This function loads a pre-trained model in the specified format, performs a warmup phase
    to stabilize performance, and then measures the average inference time over 100 runs
    using a dummy input of shape (1, 48, 48, 1).

    Args
    ----
    model_path : Path)
        - Path to the model file.

    model_format : str (optional)
        - Format of the model, either 'TF' (TensorFlow) or 'ONNX'. Defaults to 'TF'.

    Returns:
        None: Prints the average inference time in milliseconds to the console.

    Raises:
        ValueError: If an unsupported `model_format` is provided.

    Notes:
        - The warmup phase consists of 10 inference runs to avoid cold-start bias.
        - The dummy input is a random array of shape (1, 48, 48, 1) with dtype float32.
        - For ONNX models, the `CPUExecutionProvider` is used for inference.
    """
    model_format = model_format.upper()
    formats = ['TF', 'ONNX']

    if model_format in formats:
        if model_format=='TF':
            model = models.load_model(model_path)

        else:
            session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"]) # Load ONNX model
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name

        dummy = np.random.rand(1, 48, 48, 1).astype("float32") # Dummy input

        # Warmup
        for i in range(10):
            if model_format=='TF':
                model.predict(dummy)
            else:
                session.run([output_name], {input_name: dummy})

        # Timed inference
        n = 100
        start = time.time()
        for i in range(n):
            if model_format=='TF':
                model.predict(dummy)
            else:
                session.run([output_name], {input_name: dummy})
        end = time.time()

        avg_inference_time_ms = (end - start) / n * 1000

        print(f'The {model_format} model average inference time: {round(avg_inference_time_ms, 2)} ms')

    else:
        raise ValueError(f"Wrong model format '{model_format}'")
