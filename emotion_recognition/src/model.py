"""
Module for models training and testing
"""

import time
import tensorflow as tf
from keras.preprocessing import image
from keras import Model, Sequential, layers, Input, regularizers, optimizers, models
from keras import callbacks

# Transfer learning models
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.efficientnet import EfficientNetB4

# Project's packages
# from emotion_recognition.src.registry import save_model,
from emotion_recognition.params import *
from emotion_recognition.src.registry import *

# TODO: Put baseline architecture


def initialize_model(input_shape : tuple) -> tf.keras.Model: # TODO Use global var or even put it only in the code npt args
    """
    Initialize a tensorflow model

    arg
    ----
    input_shape : tuple
        Tensor shape

    returns
    ----
    model : tf.keras.Model
        Tensorflow model initialized
    """
    # instantiate and input layer
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(layers.Rescaling(1/255))

    # Hidden Conv Layer 1
    model.add(layers.Conv2D(32, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Dropout(rate=0.2))

    # Hidden Conv Layer 2
    model.add(layers.Conv2D(64, kernel_size=(2,2), padding='same', activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Dropout(rate=0.2))

    # Flatten Layer
    model.add(layers.Flatten()) # TODO Global Average Pooling ??

    # Final Dense Layers
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.Dense(7, activation='softmax'))

    print(Fore.GREEN + f"Model successfully initialized" + Style.RESET_ALL)

    print(model.summary())

    return model

def compile_model(model, learning_rate=0.01) -> tf.keras.Model:
    """
    Compile model

    arg
    ----
    learning_rate : float

    returns
    -----
    model : tf.keras.Model
    """
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 'f1'] # TODO: Choose which metrics
    )

    print(Fore.GREEN + f"Model successfully compiled" + Style.RESET_ALL)
    return model

def train_model(
        model : tf.keras.Model,
        train_dataset,
        val_dataset,
        epochs=10,
        es_patience=5,
        checkpoint=True,
        save_name='default_model01'
    ) -> tuple[Model, dict]:
    """
    Train model and save it locally

    args
    ----
    epochs : int

    patience : int

    checkpoint : bool

    saved_name : str


    returns
    ----
    model, history : tuple (tf.keras.Model, dict)
    """
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S") # TODO CHECK if needed here

    if checkpoint:
        model_cbs = [
            callbacks.EarlyStopping(
                patience=es_patience,
                restore_best_weights=True,
                start_from_epoch=es_patience # Allows warm-up period before early stopping
            ),
            callbacks.ModelCheckpoint(
                filepath="checkpoint.keras",
                # filepath=SAVE_MODELS_DIR,
                monitor="val_accuracy", # TODO: Choose monitor
                save_best_only=True,
                save_weights_only=True,
                mode='max',
                verbose=1
            )
        ]
    else:
        model_cbs = [
            callbacks.EarlyStopping(
                patience=es_patience,
                restore_best_weights=True,
                start_from_epoch=es_patience
            )
        ]

    history = model.fit(
        train_dataset,
        epochs=epochs,
        callbacks=model_cbs,
        validation_data=val_dataset,
        verbose=1
    )

    if checkpoint:
        save_model(               # TODO: verify every checkpoint function works
            model=model,
            model_name=save_name
        )

        save_results(
            params=1,
            metrics=2,
            model_name=save_name
        )

    print(Fore.GREEN + f"\nModel sucessfully trained" + Style.RESET_ALL) # TODO add ?? -> + f"Validation metric: {round(history['val_accuracy'][-1], 2)}")

    return model, history


#-------------------#
# Data Augmentation # TODO
#-------------------#


#-------------------#
# Transfer Learning # TODO
#-------------------#
