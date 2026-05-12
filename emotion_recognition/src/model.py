"""
Module for training, evaluating and predicting models.
"""

import time
import tensorflow as tf
from colorama import Fore, Style
from keras.preprocessing import image # TODO remove
from keras import Model, Sequential, layers, Input, regularizers, optimizers, models, metrics
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Transfer learning models
from keras.applications.resnet import ResNet50
from keras.applications import MobileNetV3Large, MobileNetV3Small
from keras.applications.efficientnet import EfficientNetB0, EfficientNetB3

# Project's packages
from emotion_recognition.params import *
from emotion_recognition.src.registry import load_model

#---------------------------#
#   Models Architectures    #
#---------------------------#
def initialize_baseline_model(input_shape : tuple) -> Model: # TODO: Put baseline architecture
    """
    Initialize a baseline tensorflow model

    arg
    ----
    input_shape : tuple
        Tensor shape

    returns
    ----
    model : Model
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
    model.add(layers.Dense(8, activation='softmax'))

    print(Fore.GREEN + f"Model successfully initialized" + Style.RESET_ALL)

    print(model.summary())

    return model

def initialize_custom_model(input_shape : tuple) -> Model:
    """
    Initialize a tensorflow model

    arg
    ----
    input_shape : tuple
        Tensor shape

    returns
    ----
    model : Model
        Tensorflow model initialized
    """
    # instantiate and input layer
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(layers.Rescaling(1/255))

    # Hidden Conv Layer 1
    model.add(layers.Conv2D(32, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))

    # Hidden Conv Layer 2
    model.add(layers.Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))

    # Hidden Conv Layer 3
    model.add(layers.Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))

    # Flatten Layer
    model.add(layers.Flatten()) # TODO Global Average Pooling ??

    # Final Dense Layers
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(8, activation='softmax'))

    print(Fore.GREEN + f"Model successfully initialized" + Style.RESET_ALL)

    print(model.summary())

    return model


#---------------------#
#  Transfer Learning  #
#---------------------#
def load_transfer_learning_model( # TODO Add and select Tranfer Learning models
        model_name='resnet50',
        input_shape=(112, 112, 3)
    ) -> Model:
    """
    Load an already built and trained model from Tensorflow.

    Arg
    ----
    tl_model_name : str
        - 'vgg16'
        - 'resnet'
        - 'efficientnet'
        - 'resnet50_vggface'

    returns
    ----
    model : Model
        Load pre-trained model.
    """
    if model_name == 'resnet50':
        model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        print(Fore.GREEN + f"\nResNet50 model loaded" + Style.RESET_ALL)

    elif model_name == 'mobilenet_v3_large':
        model = MobileNetV3Large(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        print(Fore.GREEN + f"\MobileNetV3Large model loaded" + Style.RESET_ALL)

    elif model_name == 'mobilenet_v3_small':
        model = MobileNetV3Small(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        print(Fore.GREEN + f"\MobileNetV3Small model loaded" + Style.RESET_ALL)

    elif model_name == 'efficientnetb3':
        model = EfficientNetB3(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        print(Fore.GREEN + f"\nEfficientNetB3 model loaded" + Style.RESET_ALL)

    elif model_name == 'resnet50_vggface_weights':
        model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape,
            pooling=None
        )
        weights_path = MODELS_REGISTRY_DIR/'saved_weights_vggface'/'vggface2_resnet50_notop.h5'
        model.load_weights(weights_path, by_name=True, skip_mismatch=False)

        print(Fore.GREEN + f"\nResNet50 model loaded with VGGFace2 weights" + Style.RESET_ALL)

    else:
        print(Fore.RED + f"\nUnknown model name: '{model_name}'" + Style.RESET_ALL)

    return model

def initialize_transfer_learning_model(
        model_name='resnet50',
        input_shape=(112, 112, 3)
    ) -> Model:
    """
    Load a transfer learning model, set its parameters as non-trainable,
    and add additional trainable dense layers at the end.

    Arg 'tl_model_name':
        - 'vgg16'
        - 'resnet'
        _ 'efficientnet'


    returns
    ----
    tl_model : Model
    Return a fully structured transfer learning model with frozen weights on its
    deep layers and with trainable top layers.
    """
    model = load_transfer_learning_model(model_name=model_name,
                                         input_shape=input_shape)

    model.trainable = False  # set layers to be untrainable

    print(Fore.BLUE + f"\nModel's weights set to 'untrainable'" + Style.RESET_ALL)

    # # Trainable top layers
    # flatten_layer = layers.Flatten()
    # dense_layer_1 = layers.Dense(128, activation='relu')
    # dropout_layer_1 = layers.Dropout(rate=0.2)
    # dense_layer_2 = layers.Dense(64, activation='relu')
    # dropout_layer_2 = layers.Dropout(rate=0.2)
    # dense_layer_3 = layers.Dense(32, activation='relu')
    # dropout_layer_3 = layers.Dropout(rate=0.5)
    # output_layer = layers.Dense(8, activation='softmax')

    # # Model Architecture
    # tl_model = Sequential(
    #     [model,
    #      flatten_layer,
    #      dense_layer_1,
    #      dropout_layer_1,
    #      dense_layer_2,
    #      dropout_layer_2,
    #      dense_layer_3,
    #      dropout_layer_3,
    #      output_layer
    #     ]
    # )

    # Replace Flatten with GAP as discussed
    gap_layer         = layers.GlobalAveragePooling2D()
    dense_layer_1     = layers.Dense(512, kernel_regularizer=regularizers.l2(1e-4))
    activation_1      = layers.Activation('gelu')
    bn_1              = layers.BatchNormalization()
    dropout_layer_1   = layers.Dropout(rate=0.5)
    dense_layer_2     = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4))
    activation_2      = layers.Activation('gelu')
    bn_2              = layers.BatchNormalization()
    dropout_layer_2   = layers.Dropout(rate=0.5)
    output_layer      = layers.Dense(8, activation='softmax')   # TODO raw logits, no softmax

    # Model Architecture
    tl_model = Sequential(
        [model,
         gap_layer,
         dense_layer_1,
         activation_1,
         bn_1,
         dropout_layer_1,
         dense_layer_2,
         activation_2,
         bn_2,
         dropout_layer_2,
         output_layer
        ]
    )

    print(tl_model.summary())

    return tl_model


#---------------------#
#     Fine Tuning     # TODO
#---------------------#
def initialize_finetuning_model(
        model_name='resnet50',
        latest_model=True,
        unfroze_layers=15
    ) -> Model:
    """
    Load a model saved locally and unfrozen a selected number of layers.

    args
    ----
    model_name : str

    unfroze_layers : int

    returns
    ----
    model : Model
    """
    model = load_model(model_name=model_name, latest_model=latest_model)

    model_backbone = model.layers[0]

    layers_to_unfroze = len(model_backbone.layers) - unfroze_layers
    for layer in model_backbone.layers[layers_to_unfroze:]:
        layer.trainable = True

    print(Fore.GREEN + f"\nModel's {unfroze_layers} last layers unfroze (out of the {len(model_backbone.layers)} layers)" + Style.RESET_ALL)

    return model

def train_ft_model(
        model : Model,
        train_dataset,
        val_dataset,
        epochs=10,
        es_patience=5,
        checkpoint=True,
        save_name='fine_tuning_model01'
    ) -> tuple[Model, dict]:
    """
    Train / Fine tuning model and save it locally.

    args
    ----
    epochs : int

    patience : int

    checkpoint : bool

    saved_name : str


    returns
    ----
    model, history : tuple (Model, dict)
    """
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

    if checkpoint:
        model_cbks = [
            EarlyStopping(
                patience=es_patience,
                restore_best_weights=True,
                start_from_epoch=es_patience # Allows warm-up period before early stopping
            ),

            ModelCheckpoint(
                filepath = str(MODELS_REGISTRY_DIR / 'saved_weights' /
                            f'{timestamp}_{save_name}_epoch{{epoch:02d}}_val_accuracy{{val_accuracy:.2f}}.weights.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=True,
                mode='max',
                verbose=1
            ),

            ReduceLROnPlateau(
                factor=0.5, patience=4,
                min_lr=1e-5
            )
        ]

        print(Fore.BLUE + f"\nModel's first checkpoint saved locally as '{timestamp}_{save_name}.h5'" + Style.RESET_ALL)

    else:
        model_cbks = [
            EarlyStopping(
                patience=es_patience,
                restore_best_weights=True,
                start_from_epoch=es_patience
            ),

            ReduceLROnPlateau(
                factor=0.5, patience=4,
                min_lr=1e-5
            )
        ]

    history = model.fit(
        train_dataset,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=model_cbks,
        validation_data=val_dataset,
        verbose=1
    )

    print(Fore.GREEN + f"\nModel sucessfully fine-tuned" + Style.RESET_ALL) # TODO add ?? -> + f"Validation metric: {round(history['val_accuracy'][-1], 2)}")

    return model, history


#---------------------#
#   Model Function    #
#---------------------#
def compile_model(
    model: Model,
    trainset: tf.data.Dataset#,
    # learning_rate=0.01
    ) -> Model: # TODO Choose Learning_rate
    """
    Compile model

    arg
    ----
    learning_rate : float

    returns
    -----
    model : Model
    """
    lr_scheduler = optimizers.schedules.CosineDecay(initial_learning_rate=0.01,
                                                          decay_steps=50 * len((trainset)),
                                                          alpha=1e-5)
    # lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    #     monitor='val_accuracy',
    #     factor=0.5,
    #     patience=3,
    #     verbose=1
    #     )
    optimizer = optimizers.SGD(learning_rate=lr_scheduler,
                                     momentum=0.9,
                                     nesterov=True,
                                     weight_decay=5e-4)

    # optimizer = optimizers.AdamW(
    #     # learning_rate=learning_rate
    #     learning_rate=learning_rate,    # 10× lower than Phase 1
    #     weight_decay=learning_rate
    #     )
    f1_weighted = metrics.F1Score(average='weighted', name='f1_weighted')
    f1_macro = metrics.F1Score(average='macro', name='f1_macro')
    metric = [f1_weighted, f1_macro]

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),#'categorical_crossentropy',
        metrics=['accuracy'] # TODO: Choose which metrics
    )

    print(Fore.GREEN + f"Model successfully compiled" + Style.RESET_ALL)
    return model

def train_model(
        model : Model,
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
    model, history : tuple (Model, dict)
    """
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S") # TODO CHECK if needed here

    if checkpoint:
        model_cbks = [
            EarlyStopping(
                patience=es_patience,
                restore_best_weights=True,
                start_from_epoch=es_patience # Allows warm-up period before early stopping
            ),

            ModelCheckpoint(
                filepath = str(MODELS_REGISTRY_DIR / 'saved_weights' /
                            f'{timestamp}_{save_name}_epoch{{epoch:02d}}_val_accuracy{{val_accuracy:.2f}}.weights.h5'), # TODO Change metric val
                monitor='val_accuracy', # TODO: Which monitor metric
                # monitor='val_f1_weighted',
                save_best_only=True,
                save_weights_only=True,
                mode='max',
                verbose=1
            )
        ]

        print(Fore.BLUE + f"\nModel's first checkpoint saved locally as '{timestamp}_{save_name}.h5'" + Style.RESET_ALL)

    else:
        model_cbks = [
            EarlyStopping(
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


#---------------------#
#  Data Augmentation  # TODO
#---------------------#
