from pathlib import Path

# --- Paths --- #
PROJECT_ROOT=Path(__file__).resolve().parent.parent
DATA_DIR=PROJECT_ROOT/'data'
MODELS_REGISTRY_DIR=PROJECT_ROOT/'models_registry'
FER_MODEL='custom_fer_model'
FER_MODEL_PATH=MODELS_REGISTRY_DIR/'saved_models'/f'*{FER_MODEL}*.keras'

# --- Model Training variables --- #
NB_CHANNELS=1
IMAGE_SIZE=(48, 48)
INPUT_SHAPE=(IMAGE_SIZE[0], IMAGE_SIZE[1], NB_CHANNELS)
BATCH_SIZE=64
SEED=42
EMOTION_DICT = {0 : 'Neutre',
                1 : 'Joie',
                2 : 'Colere',
                3 : 'Tristesse',
                4 : 'Peur',
                5 : 'Degout',
                6 : 'Surprise'}
EMOTIONS_CLASSES=['neutral',
                  'happy',
                  'angry',
                  'sad',
                  'fear',
                  'disgust',
                  'surprise']
NB_OUTPUTS=len(EMOTIONS_CLASSES)

# --- Facial Emotion Detection --- #
WINDOW_LENGHT=5
FRAME_PRED_STRIDE=3
