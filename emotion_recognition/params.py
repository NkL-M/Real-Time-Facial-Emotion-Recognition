from pathlib import Path

# Dir paths
PROJECT_ROOT=Path(__file__).resolve().parent.parent
DATA_DIR=PROJECT_ROOT/'data'
MODELS_REGISTRY_DIR=PROJECT_ROOT/'models_registry'
TRAINED_MODEL=MODELS_REGISTRY_DIR/'saved_models'/'trained_model_fer'

# Training args
NB_CHANNELS=1
IMAGE_SIZE=(48, 48)
INPUT_SHAPE=(IMAGE_SIZE[0], IMAGE_SIZE[1], NB_CHANNELS)
BATCH_SIZE=64
SEED=42
EMOTIONS_CLASSES=['neutral',
                  'happy',
                  'angry',
                  'sad',
                  'fear',
                  'disgust',
                  'surprise']
