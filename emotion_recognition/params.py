from pathlib import Path

# Dir paths
PROJECT_ROOT=Path(__file__).resolve().parent.parent
DATA_DIR=PROJECT_ROOT/'data'
MODELS_REGISTRY_DIR=PROJECT_ROOT/'models_registry'

# Model args
NB_CHANNELS=3
IMAGE_SIZE=(112, 112)
INPUT_SHAPE=(IMAGE_SIZE[0], IMAGE_SIZE[1], NB_CHANNELS)
BATCH_SIZE=64
SEED=42
