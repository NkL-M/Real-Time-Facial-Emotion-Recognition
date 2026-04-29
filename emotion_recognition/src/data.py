"""
Module for creating dataset
"""
import tensorflow as tf
from keras.utils import image_dataset_from_directory
from emotion_recognition.params import *


def batch_ratio(tf_dataset,
                ratio=0.2):
    """
    Function allowing to keep a ratio of initial tensorflow dataset batches.

    arg
    ----
    ratio : float
        Ratio of batch between 0 and 1.

    returns
    ----
    ratio_dataset : tensorflow.python.data.ops.prefetch_op._PrefetchDataset
    """
    if ratio <= 1.0 and ratio > 0.0:
        batch_number = tf_dataset.cardinality().numpy() # Computes number of batches
        ratio_dataset = tf_dataset.take(int(batch_number * ratio)) # Takes a ratio of the batches
        ratio_dataset = ratio_dataset.prefetch(tf.data.AUTOTUNE) # Turn back into a prefetch datset for performance
        print(f'Sucessfully loaded dataset of {int(batch_number * ratio)} ({int(ratio*100)}%) batches of {BATCH_SIZE} images).')
        return ratio_dataset

    else:
        raise ValueError(f'Out of range ratio inputed: {ratio}')

def load_data(dataset_type='train',
              batch_size=BATCH_SIZE,
              input_size=IMAGE_SIZE,
              fetching_ratio=0.2):
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
            color_mode='grayscale',
            batch_size=batch_size,
            image_size=input_size,
            shuffle=True,
            validation_split=0.2,
            subset='both',
            seed=SEED
        )

        dataset_train = batch_ratio(
            tf_dataset=dataset[0],
            ratio=fetching_ratio
        )

        dataset_val = batch_ratio(
            tf_dataset=dataset[1],
            ratio=fetching_ratio
        )

        dataset = (dataset_train, dataset_val)


    elif dataset_type=='test':
        dataset = image_dataset_from_directory(
            directory=test_path,
            labels="inferred",
            label_mode='categorical',
            class_names=emotions_classes,
            color_mode='grayscale',
            batch_size=batch_size,
            image_size=input_size,
            shuffle=True,
            seed=SEED
        )

        dataset = batch_ratio(
            tf_dataset=dataset,
            ratio=fetching_ratio
        )

    else:
        raise ValueError(f"dataset_type not found: '{dataset_type}'")

    return dataset
