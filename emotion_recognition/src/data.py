"""
Module for creating dataset
"""

from keras.utils import image_dataset_from_directory

#-------------#
#  Load Data  #
#-------------#

def load_data(dataset_type='train', batch_size=32):
    """
    Load the dataset from local directory and transform it in a tf dataset with label

    args
    ----
    data_type : str

    batch_size : int


    returns
    ----
    _PrefetchDataset (2 for data_type='train')
    """
    train_path = '../data/train'
    test_path = '../data/test'
    emotions_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    if dataset_type=='train':
        dataset = image_dataset_from_directory(directory=train_path,
                                               labels="inferred",
                                               label_mode='categorical',
                                               class_names=emotions_classes,
                                               color_mode='grayscale',
                                               batch_size=batch_size,
                                               image_size=(48, 48),
                                               shuffle=True,
                                               validation_split=0.2,
                                               subset='both',
                                               seed=42
                                               )

    elif dataset_type=='test':
        dataset = image_dataset_from_directory(directory=test_path,
                                               labels="inferred",
                                               label_mode='categorical',
                                               class_names=emotions_classes,
                                               color_mode='grayscale',
                                               batch_size=batch_size,
                                               image_size=(48, 48),
                                               shuffle=True,
                                               seed=42
                                               )

    else:
        raise ValueError(f"dataset_type not found: '{dataset_type}'")

    return dataset


#---------------------#
#  Data preprocessing #
#---------------------#

def data_augmentation():
    pass
