from pathlib import Path

# Directories paths
PROJECT_ROOT=Path(__file__).resolve().parent.parent
DATA_DIR=PROJECT_ROOT/'data'
SAVE_MODELS_DIR=PROJECT_ROOT/'saved_models'

# Models arguments
IMAGE_SIZE=(48, 48)
BATCH_SIZE=32
SEED=42
