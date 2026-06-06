"""
Module for training, evaluating and make predictions with models.
"""

import time
from colorama import Fore, Style
import tensorflow as tf
from keras import Model, Sequential, layers, Input, regularizers, optimizers, losses, metrics
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
def initialize_baseline_model(input_shape: tuple) -> Model:
    """
    Initialize a baseline model architecture.

    arg
    ----
    input_shape : tuple
        - Tensor shape

    returns
    ----
    model : Model
        - Build tensorflow model.
    """
    input = Input(shape=input_shape)

    x = layers.Rescaling(1/255)(input)
    x = layers.Conv2D(16, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation='relu')(x)

    output = layers.Dense(NB_OUTPUTS, activation='softmax')

    model = Model(input, output, name='fer_baseline_cnn')

    print(Fore.GREEN + f"Baseline model successfully initialized" + Style.RESET_ALL)
    print(model.summary())

    return model

def initialize_custom_model(input_shape: tuple) -> tf.keras.Model:
    """
    Initialize a tensorflow model.

    arg
    ----
    input_shape : tuple
        - Tensor shape

    returns
    ----
    model : tf.keras.Model
        - Build model architecture
    """
    reg = regularizers.l2(1e-4)
    input = Input(shape=input_shape)
    x = layers.Rescaling(1/255)(input)

    # Conv Layer 1
    x = layers.Conv2D(64, kernel_size=3, padding='same', kernel_regularizer=reg, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Conv Layer 2
    x = layers.Conv2D(64, kernel_size=3, padding='same', kernel_regularizer=reg, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x) # 48 pixels to 24 pixels images
    x = layers.Dropout(0.25)(x) # layers.SpatialDropout2D(0.25)(x)

    # Conv Layer 3
    x = layers.Conv2D(128, kernel_size=3, padding='same', kernel_regularizer=reg, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Conv Layer 4
    x = layers.Conv2D(128, kernel_size=3, padding='same', kernel_regularizer=reg, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)          # 24 to 12
    x = layers.Dropout(0.25)(x) # layers.SpatialDropout2D(0.25)(x)

    # Conv Layer 5
    x = layers.Conv2D(256, kernel_size=3, padding='same', kernel_regularizer=reg, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Conv Layer 6
    x = layers.Conv2D(256, kernel_size=3, padding='same', kernel_regularizer=reg, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Conv Layer 7
    x = layers.Conv2D(512, kernel_size=3, padding='same', kernel_regularizer=reg, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)          # 12 to 6
    x = layers.Dropout(0.25)(x) # layers.SpatialDropout2D(0.25)(x)

    # Conv Layer 8
    x = layers.Conv2D(512, kernel_size=3, padding='same', kernel_regularizer=reg, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Conv Layer 9
    x = layers.Conv2D(256, kernel_size=3, padding='same', kernel_regularizer=reg, activation='relu')(x) # OLD 512
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x) # layers.SpatialDropout2D(0.25)(x)

    # GAP Layer
    x = layers.GlobalAveragePooling2D()(x)  # (B, 512)

    # Dense Top Layers
    x = layers.Dense(512, kernel_regularizer=reg, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(256, kernel_regularizer=reg, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    output = layers.Dense(NB_OUTPUTS, activation='softmax')(x)

    model = Model(input, output, name='fer_custom_cnn')

    print(Fore.GREEN + f"Model successfully initialized" + Style.RESET_ALL)
    print(model.summary())

    return model

#---------------------------------#
#  Transfer Learning Architecture #
#---------------------------------#
def load_transfer_learning_model(model_name: str = 'resnet50',
                                 input_shape: tuple = (112, 112, 3)
    ) -> Model:
    """
    Load an already built and trained model from Tensorflow.

    args
    ----
    tl_model_name : str
        - 'vgg16'
        - 'resnet'
        - 'efficientnet'
        - 'resnet50_vggface'

    input_shape : tuple

    returns
    ----
    model : Model
        - Load pre-trained model.
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

    elif model_name == 'efficientnet_b0':
        model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        print(Fore.GREEN + f"\nEfficientNetB0 model loaded" + Style.RESET_ALL)

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

def initialize_transfer_learning_model(model_name: str = 'resnet50',
                                       input_shape: tuple = (112, 112, 3)
    ) -> Model:
    """
    Load a transfer learning model, set its parameters as non-trainable,
    and add additional trainable dense layers at the end.

    args
    ----
    model_name : str
        - 'vgg16'
        - 'resnet'
        _ 'efficientnet'

    input_shape : tuple


    returns
    ----
    model : Model
        - A fully structured transfer learning model with frozen weights on its
          deep layers and with trainable top layers.
    """
    tl_model = load_transfer_learning_model(model_name=model_name,
                                            input_shape=input_shape)

    tl_model.trainable = False  # set layers to be untrainable

    print(Fore.BLUE + f"\nModel's weights set to 'untrainable'" + Style.RESET_ALL)

    # # Trainable top layers
    gap_layer         = layers.GlobalAveragePooling2D()
    dense_layer_1     = layers.Dense(512, kernel_regularizer=regularizers.l2(1e-4), activation='gelu')
    bn_1              = layers.BatchNormalization()
    dropout_layer_1   = layers.Dropout(rate=0.5)
    dense_layer_2     = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4), activation='gelu')
    bn_2              = layers.BatchNormalization()
    dropout_layer_2   = layers.Dropout(rate=0.5)
    output_layer      = layers.Dense(NB_OUTPUTS, activation='softmax')   # TODO raw logits, no softmax

    # Model Architecture
    model = Sequential(
        [tl_model,
         gap_layer,
         dense_layer_1,
         bn_1,
         dropout_layer_1,
         dense_layer_2,
         bn_2,
         dropout_layer_2,
         output_layer
        ]
    )

    print(model.summary())

    return model


#---------------------#
#     Fine Tuning     #
#---------------------#
def load_model_for_finetuning(model_name: str = 'efficientnet_b0',
                              latest_model: bool = True,
                              unfroze_layers: int = 15
    ) -> Model:
    """
    Load a model saved locally and unfrozen a selected number of layers.

    args
    ----
    model_name : str
        - Model's name to load from disk.

    unfroze_layers : int
        - Number of the last layers to unfroze for training.

    returns
    ----
    model : Model
        - Model with unfrozen layers.
    """
    model = load_model(model_name=model_name, latest_model=latest_model)

    model_backbone = model.layers[0]

    layers_to_unfroze = len(model_backbone.layers) - unfroze_layers
    for layer in model_backbone.layers[layers_to_unfroze:]:
        layer.trainable = True

    print(Fore.GREEN + f"\nModel's {unfroze_layers} last layers unfroze (out of the {len(model_backbone.layers)} layers)" + Style.RESET_ALL)

    return model


#---------------------#
#   Model Function    #
#---------------------#
def compile_model(model: Model,
                  learning_rate: float = 0.01
    ) -> Model:
    """
    Compile model

    arg
    ----
    learning_rate : float

    returns
    -----
    model : Model
    """
    # lr_scheduler = optimizers.schedules.CosineDecay(initial_learning_rate=learning_rate,
    #                                                 decay_steps=50 * len((trainset)),
    #                                                 alpha=1e-5)

    # optimizer = optimizers.AdamW(
    #     learning_rate=learning_rate,    # 10× lower than Phase 1
    #     weight_decay=learning_rate * 4
    #     )

    lr_scheduler = optimizers.schedules.CosineDecayRestarts(initial_learning_rate=learning_rate,
                                                            first_decay_steps=20,
                                                            t_mul=2.0,
                                                            m_mul=0.9)

    optimizer = optimizers.SGD(learning_rate=lr_scheduler,
                               momentum=0.9,
                               nesterov=True,
                               weight_decay=5e-4)

    loss = losses.CategoricalCrossentropy(label_smoothing=0.1)

    acc = metrics.CategoricalAccuracy(name='accuracy')
    # f1_weighted = metrics.F1Score(average='weighted', name='f1_weighted')
    # f1_macro = metrics.F1Score(average='macro', name='f1_macro')
    # metric = [f1_weighted, f1_macro] # TODO: Choose which metrics

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=[acc]
    )

    print(Fore.GREEN + f"Model successfully compiled" + Style.RESET_ALL)

    return model

def train_model(
        model : Model,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        epochs: int = 10,
        patience: int = 5,
        checkpoint: bool = True,
        reduce_lr: bool = True,
        save_name: str = 'default_model01'
    ) -> tuple[Model, dict]:
    """
    Train model and save it locally

    args
    ----
    epochs : int
        - Number of times the entire dataset passes through the model during training.

    patience : int
        - Number of times the loss doesn't decrease before stopping training.

    checkpoint : bool
        - Wether saving model's weights when loss decrease.

    saved_name : str
        - Name with which the model's weights, parameters, metrics will be saved.

    returns
    ----
    model, history : tuple
    """
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

    early_stopping = EarlyStopping(
                patience=patience,
                restore_best_weights=True,
                start_from_epoch=patience # Warm-up period before early stopping
            )

    model_checkpoints = ModelCheckpoint(
                filepath = str(MODELS_REGISTRY_DIR / 'saved_weights' /
                    f'{timestamp}_{save_name}_epoch{{epoch:02d}}_val_accuracy{{val_accuracy:.2f}}.weights.h5'),
                monitor='val_accuracy', # monitor='val_f1_weighted', TODO: Which monitor metric
                save_best_only=True,
                save_weights_only=True,
                mode='max',
                verbose=1
            )

    reduce_lr_on_plateau = ReduceLROnPlateau(
                factor=0.5,
                patience=5,
                verbose=1,
                min_lr=1e-5
            )

    model_callbacks = []

    if patience:
        model_callbacks.append(early_stopping)

    if checkpoint:
        model_callbacks.append(model_checkpoints)
        print(Fore.BLUE + f"\nModel's weights will be saved locally as '{timestamp}_{save_name}.weights.h5'" + Style.RESET_ALL)

    if reduce_lr:
        model_callbacks.append(reduce_lr_on_plateau)

    history = model.fit(   # TODO Add tensorboard for monitoring ??
        train_dataset,
        epochs=epochs,
        # batch_size=BATCH_SIZE, TODO Remove ?? (already in load_data)
        # class_weight=class_weights, TODO Dict for fixing class imbalance
        callbacks=model_callbacks,
        validation_data=val_dataset,
        verbose=1
    )

    print(Fore.GREEN + f"\nModel sucessfully trained" + Style.RESET_ALL)

    return model, history
