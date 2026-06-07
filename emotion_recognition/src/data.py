"""
Module for loading image datasets from the `data` directory and transforming into
a tensorflow dataset ready to be fed into tf models.
"""

import tensorflow as tf
from keras import layers, Sequential
from keras.utils import image_dataset_from_directory
from colorama import Fore, Style
from emotion_recognition.params import SEED, DATA_DIR, EMOTIONS_CLASSES


def batch_ratio(dataset: tf.data.Dataset,
                ratio: int = 0.2
    ) -> tuple[tf.data.Dataset, int]:
    """
    Allow to keep a ratio of the batches of a tensorflow dataset.

    arg
    ----
    ratio : float
        Ratio of batch between 0 and 1.

    returns
    ----
    (ratio_dataset, batch_number) : tuple
    """
    if ratio <= 1.0 and ratio > 0.0:
        batch_number = dataset.cardinality().numpy() # Computes number of batches
        ratio_dataset = dataset.take(int(batch_number * ratio)) # Takes a ratio of the batches
        ratio_dataset = ratio_dataset.prefetch(tf.data.AUTOTUNE) # Turn back into a prefetch dataset for performance
        return ratio_dataset, batch_number

    else:
        raise ValueError(f'Out of range ratio inputed: {ratio}')

def load_data(dataset_type: str = 'train',
              batch_size: int = 64,
              image_size: tuple = (48, 48),
              fetch_ratio: float = 0.2,
              validation_split: float = 0.2
    ) -> tf.data.Dataset:
    """
    Load the dataset from local directory and transform it in a tensorflow dataset.

    args
    ----
    data_type : str
        Wether to load the train/validation datasets or the test dataset.
        - 'train'
        - 'test'

    batch_size : int
        Number of images per batch.

    input_size : tuple
        Width and heigth of input images.

    fetching_ratio : float
        Ratio of the dataset to fetch (between 0.0 and 1.0).

    validation_split : float
        Ratio of the training dataset used for validation (between 0.0 and 1.0).

    returns
    ----
    dataset : tf.PrefetchDataset
        Return a dataset for `data_type='test'` and a list of 2 datasets for
        `data_type='train'`.
    """
    train_path=DATA_DIR/'train'
    test_path=DATA_DIR/'test'

    if dataset_type=='train':
        dataset = image_dataset_from_directory(
            directory=train_path,
            labels="inferred",
            label_mode='categorical',
            class_names=EMOTIONS_CLASSES,
            color_mode='grayscale', # TODO Change if neccesary to 'rgb'
            batch_size=batch_size,
            image_size=image_size,
            shuffle=True,
            validation_split=validation_split,
            subset='both',
            seed=SEED
        )

        train_ds, train_batch_nb = batch_ratio(
            dataset=dataset[0],
            ratio=fetch_ratio
        )

        val_ds, val_batch_nb = batch_ratio(
            dataset=dataset[1],
            ratio=fetch_ratio
        )

        print(Fore.GREEN + f"Sucessfully loaded {int(fetch_ratio*100)}% of the training and validation datasets." + Style.RESET_ALL)
        print(Fore.WHITE + f"Training dataset: {int(train_batch_nb * fetch_ratio)} batches of {batch_size} images." + Style.RESET_ALL)
        print(Fore.WHITE + f"Validation dataset: {int(val_batch_nb * fetch_ratio)} batches of {batch_size} images." + Style.RESET_ALL)

        dataset = (train_ds, val_ds)


    elif dataset_type=='test':
        dataset = image_dataset_from_directory(
            directory=test_path,
            labels="inferred",
            label_mode='categorical',
            class_names=EMOTIONS_CLASSES,
            color_mode='grayscale', # TODO Change if neccesary
            batch_size=batch_size,
            image_size=image_size,
            shuffle=True,
            seed=SEED
        )

        dataset, test_batch_nb = batch_ratio(
            dataset=dataset,
            ratio=fetch_ratio
        )

        print(Fore.GREEN + f"Sucessfully loaded {int(fetch_ratio*100)}% of the test dataset." + Style.RESET_ALL)
        print(Fore.WHITE + f"Test dataset: {int(test_batch_nb * fetch_ratio)} batches of {batch_size} images." + Style.RESET_ALL)

    else:
        raise ValueError(f"dataset_type not found: '{dataset_type}'")

    return dataset

def data_augmentation(train_dataset: tf.data.Dataset,
                      val_dataset: tf.data.Dataset
    ) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Apply data augmentation to training dataset.

    returns
    ----
    train_ds, val_ds : tuple
    """
    data_augment = Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05, fill_mode='reflect'), # ± 10° maximum on 48px
            layers.RandomTranslation(0.05, 0.05, fill_mode='reflect'),
            layers.RandomZoom(height_factor=(-0.05, 0.1),fill_mode='reflect'),  # small zoom since 48px is already small
            layers.RandomBrightness(0.2), # ± 20% brightness
            layers.RandomContrast(0.2), # ± 20% contrast
        ], name="augmentation"
    )

    def preprocess(image, label):
        image = tf.cast(image, tf.float32)
        return image, label

    def augment_and_preprocess(image, label):
        image = tf.cast(image, tf.float32)
        image = data_augment(image, training=True)
        return image, label

    # New augmented datasets
    train_ds = (train_dataset
        .map(augment_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (val_dataset
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds
