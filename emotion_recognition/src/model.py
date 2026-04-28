"""
Module for models training and testing
"""

import time
import tensorflow as tf
from keras.preprocessing import image
from keras import Sequential, layers, Input, regularizers, optimizers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Transfer learning models
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.efficientnet import EfficientNetB4

# Project's packages
# from emotion_recognition.src.preprocessing import ...
# from emotion_recognition.src.registry import ...

# TODO: Put baseline architecture


def initialize_model(input_shape : tuple) -> tf.keras.Model:
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
    model.add(layers.Flatten())

    # Final Dense Layers
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.Dense(7, activation='softmax'))

    print('Model successfully initialized')

    print(model.summary())

    return model

def compile_model(model,
                  learning_rate=0.01) -> tf.keras.Model:
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

    print("Model successfully compiled")
    return model

def train_model(model : tf.keras.Model,
                train_dataset,
                val_dataset,
                epochs=10,
                patience=5,
                checkpoint=True,
                saved_name='default_model'):
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
    model, history : tf.keras.Model, dict
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    if checkpoint:
        es = [
            EarlyStopping(
                patience=patience,
                restore_best_weights=True),
            ModelCheckpoint(
                filepath="checkpoint.keras",
                monitor="val_accuracy", # TODO: Choose monitor
                save_best_only=True,
                mode='max',
                verbose=0
            )
        ]
    else:
        es = [
            EarlyStopping(
                patience=patience,
                restore_best_weights=True
            )
        ]

    history = model.fit(
        train_dataset,
        epochs=epochs,
        callbacks=es,
        validation_data=val_dataset,
        verbose=1
    )

    print("Model sucessfully trained")

    if checkpoint==True:
        # store_model( # TODO: Create store model
        #     bucket_name="pixel-truth-bucket",
        #     source_file_name="checkpoint.keras",
        #     destination_blob_name=(f"saved-model-weights/{saved_name}-{timestamp}.keras")
        # )

        print(f"Model successfully saved locally as '{saved_name}' at timestamp: {timestamp}")

    return model, history


#-------------------#
# Data Augmentation # TODO
#-------------------#


#-------------------#
# Transfer Learning # TODO
#-------------------#
