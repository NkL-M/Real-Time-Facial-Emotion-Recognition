from pathlib import Path

# --- Paths --- #
PROJECT_ROOT=Path(__file__).resolve().parent.parent
DATA_DIR=PROJECT_ROOT/'data'
MODELS_REGISTRY_DIR=PROJECT_ROOT/'models_registry'
TF_MODEL='custom_fer_model'
TF_MODEL_PATH=MODELS_REGISTRY_DIR/'saved_models'/f'*{TF_MODEL}*.keras'
ONNX_MODEL_PATH=MODELS_REGISTRY_DIR/'saved_models'/f'{TF_MODEL}.onnx'

# --- Model Training --- #
NB_CHANNELS=1
IMAGE_SIZE=(48, 48)
INPUT_SHAPE=(IMAGE_SIZE[0], IMAGE_SIZE[1], NB_CHANNELS)
BATCH_SIZE=64
SEED=42
EMOTION_DICT = {0 : 'Neutral',
                1 : 'Happy',
                2 : 'Angry',
                3 : 'Sad',
                4 : 'Fear',
                5 : 'Disgust',
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
CAP_RESOLUTION=(1920, 1080)
CAP_FPS=30
WINDOW_LENGHT=5
FRAME_PRED_STRIDE=3
CONFIDENCE_THRESHOLD=0.3
