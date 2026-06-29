"""
Module for training, evaluating and make predictions with models.

This module provides functions for loading model achitectures, compiling, training,
evaluating and inference.
"""

import time
from colorama import Fore, Style
import tensorflow as tf
from keras import Model, layers, Input, regularizers, optimizers, losses, metrics
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from emotion_recognition.params import MODELS_REGISTRY_DIR, NB_OUTPUTS

# --------------------------- #
#    Models Architectures     #
# --------------------------- #
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
        - A tensor for data shape inputed in model.

    returns
    ----
    model : tf.keras.Model
        - Build custom model architecture.
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


# ----------------- #
#   Model Training  #
# ----------------- #
def compile_model(
    model: Model,
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

    accuracy = metrics.CategoricalAccuracy(name='accuracy')
    f1_macro = metrics.F1Score(average='macro', name='f1_macro')

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=[accuracy],
                  weighted_metrics=[f1_macro]
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
                    f'{timestamp}_{save_name}_epoch{{epoch:02d}}_val_f1_macro{{val_f1_macro:.2f}}.weights.h5'),
                monitor='val_f1_macro', # 'val_accuracy', TODO: Which monitor metric
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
        # class_weight=class_weights, TODO Dict for fixing class imbalance
        callbacks=model_callbacks,
        validation_data=val_dataset,
        verbose=1
    )

    print(Fore.GREEN + f"\nModel sucessfully trained" + Style.RESET_ALL)

    return model, history
