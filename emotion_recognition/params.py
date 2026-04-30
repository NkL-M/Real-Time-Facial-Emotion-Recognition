from pathlib import Path

# Dir paths
PROJECT_ROOT=Path(__file__).resolve().parent.parent
DATA_DIR=PROJECT_ROOT/'data'
MODELS_REGISTRY_DIR=PROJECT_ROOT/'models_registry'

# Models args
IMAGE_SIZE=(48, 48)
BATCH_SIZE=32
SEED=42
