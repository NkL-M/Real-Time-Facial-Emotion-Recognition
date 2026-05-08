"""
Module for training, evaluating and predicting models.
"""

import time
import tensorflow as tf
from keras.preprocessing import image
from keras import Model, Sequential, layers, Input, regularizers, optimizers, models, metrics
from keras import callbacks

# Transfer learning models
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.efficientnet import EfficientNetB4

# Project's packages
from emotion_recognition.params import *
from emotion_recognition.src.registry import *

#---------------------#
#   Baseline Model    # TODO
#---------------------#
def initialize_baseline_model(input_shape : tuple) -> tf.keras.Model: # TODO: Put baseline architecture
    """
    Initialize a baseline tensorflow model

    arg
    ----
    input_shape : tuple
        Tensor shape

    returns
    ----
    model : tf.keras.Model
        Tensorflow model initialized
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(layers.Rescaling(1/255))

    # Hidden Conv Layer
    model.add(layers.Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))

    # Flatten Layer
    model.add(layers.Flatten()) # TODO Global Average Pooling ??

    # Final Dense Layers
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(7, activation='softmax'))

    print(Fore.GREEN + f"Model successfully initialized" + Style.RESET_ALL)

    print(model.summary())

    return model


#---------------------#
#    Custom Model     # TODO
#---------------------#
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
    model.add(layers.Conv2D(16, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    # model.add(layers.Dropout(rate=0.1))

    # Hidden Conv Layer 1
    model.add(layers.Conv2D(16, kernel_size=(2,2), padding='same', activation='relu'))

    # Flatten Layer
    model.add(layers.Flatten()) # TODO Global Average Pooling ??

    # Final Dense Layers
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(rate=0.3))
    model.add(layers.Dense(7, activation='softmax'))

    print(Fore.GREEN + f"Model successfully initialized" + Style.RESET_ALL)

    print(model.summary())

    return model

def compile_model(
    model: tf.keras.Model,
    learning_rate=0.01
    ) -> tf.keras.Model: # TODO Choose Learning_rate
    """
    Compile model

    arg
    ----
    learning_rate : float

    returns
    -----
    model : tf.keras.Model
    """
    # optimizer = optimizers.Adam(learning_rate=learning_rate)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    metric = metrics.F1Score(average='weighted', name='f1_weighted')
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', metric] # TODO: Choose which metrics
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
        model_cbks = [
            callbacks.EarlyStopping(
                patience=es_patience,
                restore_best_weights=True,
                start_from_epoch=es_patience # Allows warm-up period before early stopping
            ),
            callbacks.ModelCheckpoint(
                # filepath=str(MODELS_REGISTRY_DIR / 'saved_weights' / f'{timestamp}_{save_name}.h5'),
                filepath = str(MODELS_REGISTRY_DIR / 'saved_weights' /
                            f'{timestamp}_{save_name}_epoch{{epoch:02d}}_val_f1w{{val_accuracy:.1f}}.weights.h5'), # TODO Change metric val
                monitor='val_accuracy', # TODO: Which monitor metric
                # monitor='val_f1_weighted',
                save_best_only=True,
                save_weights_only=True,
                mode='max',
                verbose=1
            )
        ]
        print(Fore.BLUE + f"\nModel saved locally as '{timestamp}_{save_name}.h5'" + Style.RESET_ALL)
    else:
        model_cbks = [
            callbacks.EarlyStopping(
                patience=es_patience,
                restore_best_weights=True,
                start_from_epoch=es_patience
            )
        ]
    # TODO Add tensorboard for monitoring
    history = model.fit( # TODO Add batch size ???
        train_dataset,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=model_cbks,
        validation_data=val_dataset,
        verbose=1
    )

    print(Fore.GREEN + f"\nModel sucessfully trained" + Style.RESET_ALL) # TODO add ?? -> + f"Validation metric: {round(history['val_accuracy'][-1], 2)}")

    return model, history

def evaluate_model(
    model: tf.keras.Model,
    test_dataset: tf.data.Dataset
    ) -> float:
    """
    Evaluates model's performance on test dataset.

    returns
    ----
    eval_score : float
    """
    eval_score = model.evaluate(test_dataset)
    return eval_score


#---------------------#
#  Data Augmentation  # TODO
#---------------------#



#---------------------#
#  Transfer Learning  # TODO
#---------------------#
def load_transfer_learning_model( # TODO Add and select Tranfer Learning models
        model_name='vgg16',
        input_shape=(48, 48, 1)
    ) -> tf.keras.Model:
    """
    Load an already built and trained model from Tensorflow.

    Arg
    ----
    tl_model_name : str
        - 'vgg16'
        - 'resnet'
        - 'efficientnet'

    returns
    ----
    model : tf.keras.Model
        Load pre-trained model.
    """
    if model_name == 'vgg16':
        model = VGG16(weights='imagenet',
                      include_top=False,
                      input_tensor=input_shape)
        print(Fore.GREEN + f"\nVGG16 model loaded" + Style.RESET_ALL)

    elif model_name == 'resnet':
        model = ResNet50(weights='imagenet',
                         include_top=False,
                         input_shape=input_shape)
        print(Fore.GREEN + f"\nResNet50 model loaded" + Style.RESET_ALL)

    elif model_name == 'efficientnet':
        model = EfficientNetB4(weights='imagenet',
                               include_top=False,
                               input_shape=input_shape)
        print(Fore.GREEN + f"\nEfficientNetB4 model loaded" + Style.RESET_ALL)

    else:
        print(Fore.RED + f"\nUnknown model name: '{model_name}'" + Style.RESET_ALL)

    return model

def initialize_transfer_learning_model(model_name='vgg16',
                                       input_shape=(48, 48, 1)) -> tf.keras.Model:
    """
    Load a transfer learning model, set its parameters as non-trainable,
    and add additional trainable dense layers at the end.

    Arg 'tl_model_name':
        - 'vgg16'
        - 'resnet'
        _ 'efficientnet'


    returns
    ----
    tl_model : tf.keras.Model
    Return a fully structured transfer learning model with frozen weights on its
    deep layers and with trainable top layers.
    """
    model = load_transfer_learning_model(model_name=model_name,
                                         input_shape=input_shape)

    model.trainable = True  # set layers to be untrainable

    print(Fore.BLUE + f"\nModel's weights set to 'untrainable'" + Style.RESET_ALL)

    # Trainable top layers
    flatten_layer = layers.Flatten()
    dense_layer_1 = layers.Dense(128, activation='relu')
    dropout_layer_1 = layers.Dropout(rate=0.2)
    dense_layer_2 = layers.Dense(64, activation='relu')
    dropout_layer_2 = layers.Dropout(rate=0.2)
    dense_layer_3 = layers.Dense(32, activation='relu')
    dropout_layer_3 = layers.Dropout(rate=0.2)
    output_layer = layers.Dense(7, activation='softmax')

    # Model Architecture
    tl_model = Sequential(
        [model,
         flatten_layer,
         dense_layer_1,
         dropout_layer_1,
         dense_layer_2,
         dropout_layer_2,
         dense_layer_3,
         dropout_layer_3,
         output_layer
        ]
    )

    print(tl_model.summary())

    return tl_model


#---------------------#
#     Fine Tuning     # TODO
#---------------------#
