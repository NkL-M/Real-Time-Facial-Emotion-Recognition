"""
Module for loading image datasets from the `data` directory and transforming into
a tensorflow dataset ready to be fed into tf models.
"""
from colorama import Fore, Style
import tensorflow as tf
from keras.utils import image_dataset_from_directory
from emotion_recognition.params import SEED, DATA_DIR


def batch_ratio(
        tf_dataset,
        ratio: int = 0.2
    ) -> tuple[tf.data.Dataset, int]:
    """
    Function allowing to keep a ratio of initial tensorflow dataset batches.

    arg
    ----
    ratio : float
        Ratio of batch between 0 and 1.

    returns
    ----
    (ratio_dataset, batch_number) : tuple
        ratio_dataset : tf.data.Dataset
        batch_number : int
    """
    if ratio <= 1.0 and ratio > 0.0:
        batch_number = tf_dataset.cardinality().numpy() # Computes number of batches
        ratio_dataset = tf_dataset.take(int(batch_number * ratio)) # Takes a ratio of the batches
        ratio_dataset = ratio_dataset.prefetch(tf.data.AUTOTUNE) # Turn back into a prefetch datset for performance
        return ratio_dataset, batch_number

    else:
        raise ValueError(f'Out of range ratio inputed: {ratio}')

def load_data_val_split(
        dataset_type: str = 'train',
        batch_size: int = 32,
        image_size: tuple = (48, 48),
        fetch_ratio: float = 0.2
    ) -> tf.data.Dataset:
    """
    Load the dataset from local directory and transform it in a tensorflow dataset.
    Allows to load data from a train with a fraction of 0.2 kept for the validation.

    args
    ----
    data_type : str
        Wether to load the train/validation datasets or the test dataset.
        - 'train'
        - 'test'

    batch_size : int
        Number of images per batch.

    input_size : tuple
        The width and heigth of input images.

    fetching_ratio : float
        The ratio of the dataset to fetch (between 0 and 1).

    returns
    ----
    dataset : tf.PrefetchDataset
        Return a dataset for `data_type='test'` and a list of 2 datasets for
        `data_type='train'`.
    """
    train_path=DATA_DIR/'train'
    test_path=DATA_DIR/'test'

    emotions_classes = [
        'neutral',
        'happy',
        'angry',
        'sad',
        'fear',
        'disgust',
        'surprise'
    ] # Categorical output ordered same as this list

    if dataset_type=='train':
        dataset = image_dataset_from_directory(
            directory=train_path,
            labels="inferred",
            label_mode='categorical',
            class_names=emotions_classes,
            color_mode='rgb', # TODO Change if neccesary to 'grayscale'
            batch_size=batch_size,
            image_size=image_size,
            shuffle=True,
            validation_split=0.2,
            subset='both',
            seed=SEED
        )

        train_ds, train_batch_nb = batch_ratio(
            tf_dataset=dataset[0],
            ratio=fetch_ratio
        )

        val_ds, val_batch_nb = batch_ratio(
            tf_dataset=dataset[1],
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
            class_names=emotions_classes,
            color_mode='rgb', # TODO Change if neccesary
            batch_size=batch_size,
            image_size=image_size,
            shuffle=True,
            seed=SEED
        )

        dataset, test_batch_nb = batch_ratio(
            tf_dataset=dataset,
            ratio=fetch_ratio
        )

        print(Fore.GREEN + f"Sucessfully loaded {int(fetch_ratio*100)}% of the test dataset." + Style.RESET_ALL)
        print(Fore.WHITE + f"Test dataset: {int(test_batch_nb * fetch_ratio)} batches of {batch_size} images." + Style.RESET_ALL)

    else:
        raise ValueError(f"dataset_type not found: '{dataset_type}'")

    return dataset

def load_data(
        dataset_type: str = 'train',
        batch_size: int = 32,
        image_size: tuple = (112, 112),
        fetch_ratio: float = 0.2
    ) -> tf.data.Dataset:
    """
    Load the dataset from local directory and transform it in a tensorflow dataset.

    args
    ----
    data_type : str
        Wether to load the train, val or the test dataset.
        - 'train'
        - 'val'
        - 'test'

    batch_size : int
        Number of images per batch.

    input_size : tuple
        The width and heigth of input images.

    fetching_ratio : float
        The ratio of the dataset to fetch (between 0 and 1).

    returns
    ----
    dataset : tf.PrefetchDataset
        Return a dataset of the given type.
    """
    emotions_classes = [
        'neutral',
        'happy',
        'anger', # 'angry' for FER-2013
        'sad',
        'fear',
        'disgust',
        'contempt', # Not in FER-2013
        'surprise'
    ] # Categorical output ordered same as this list

    dataset_types = ['train', 'val', 'test']

    if dataset_type not in dataset_types:
        raise ValueError(f"dataset_type not found: '{dataset_type}'")

    else:
        dataset = image_dataset_from_directory(
            directory=DATA_DIR/dataset_type,
            labels="inferred",
            label_mode='categorical',
            class_names=emotions_classes,
            color_mode='rgb',
            batch_size=batch_size,
            image_size=image_size,
            shuffle=True,
            seed=SEED
        )

        dataset,dataset_batch_nb = batch_ratio(
            tf_dataset=dataset,
            ratio=fetch_ratio
        )

        print(Fore.GREEN + f"Sucessfully loaded {int(fetch_ratio*100)}% of the {dataset_type} dataset." + Style.RESET_ALL)
        print(Fore.WHITE + f"Training dataset: {int(dataset_batch_nb * fetch_ratio)} batches of {batch_size} images." + Style.RESET_ALL)

        return dataset
