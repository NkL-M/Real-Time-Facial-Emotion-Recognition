import cv2
import numpy as np
import tensorflow as tf
# from io import BytesIO
# from PIL import Image
from colorama import Fore, Style
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from emotion_recognition.interface.main import predict_image
from emotion_recognition.src.registry import load_model
from emotion_recognition.params import MODEL_PATH, NB_CHANNELS, INPUT_SHAPE, FER_MODEL


app = FastAPI()
app.state.model = None
# app.state.face_detector = FaceDetector()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_model():
    """
    Load model only on first request, then reuse it for the next requests.
    """
    if app.state.model is None:
        try:
            app.state.model = load_model(FER_MODEL)
            print(Fore.GREEN + "\nModel loaded successfully" + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"\nError loading model: {e}" + Style.RESET_ALL)
            raise HTTPException(status_code=500, detail="Model could not be loaded")

    return app.state.model


@app.get("/")
def root():
    return {"status": "Hello, the API is alive!"}


@app.get("/predict")
async def predict(file: UploadFile = File(...)):
    model = get_model()
    emotions_dict = {0 : 'Neutral',
                     1 : 'Happy',
                     2 : 'Angry',
                     3 : 'Sad',
                     4 : 'Fear',
                     5 : 'Disgust',
                     6 : 'Surprise'}

    try:
        content = await file.read()
        # img = Image.open(BytesIO(content))
        # img.verify()
        # img = Image.open(BytesIO(content)).convert("RGB")

        img = tf.io.read_file(file)
        img = tf.image.decode_image(img, channels=NB_CHANNELS)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Image corrupted or unreadable."
        )

    img = tf.image.resize(img, size=INPUT_SHAPE)
    x_preprocessed = tf.expand_dims(input=img, axis=0)

    #oldddd
    face_crop = cv2.resize(face_crop, (48, 48)) # Resize img to 48x48 pixels
    face_crop = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY) # Change from rgb to grayscale
    # face_crop = tf.expand_dims(face_crop, axis=0)  # (1, 48, 48) TODO remove it since, there is stacking later


    try:
        outputs = model(x_preprocessed)
        prediction_index = tf.argmax(outputs[0]).numpy()
        prediction = emotions_dict[prediction_index]
    except Exception as e:
        print(Fore.RED + f"\nPrediction error: {e}" + Style.RESET_ALL)
        raise HTTPException(status_code=500, detail="Prediction failed")

    return {
        "Emotion": prediction
    }

def predict(img,
            bbox,
            model,
            pad=0):

    # emotions_dict = {0 : 'Neutre',
    #                  1 : 'Joie',
    #                  2 : 'Colere',
    #                  3 : 'Tristesse',
    #                  4 : 'Peur',
    #                  5 : 'Degout',
    #                  6 : 'Surprise'}

    # x1, y1, width, height = bbox
    # x2, y2 = x1 + width, y1 + height

    # x1 = max(0, x1 - pad)
    # y1 = max(0, y1 - pad)
    # x2 = min(img.shape[1], x2 + pad)
    # y2 = min(img.shape[0], y2 + pad)

    # # Inputs
    # face_crop = img[y1:y2, x1:x2]
    # face_crop = cv2.resize(face_crop, (48, 48)) # Resize image to 48x48 pixels
    # face_crop = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY) # Change from RGB to Grayscale
    # face_crop = np.expand_dims(face_crop, axis=0)  # (1, 48, 48)

    # # Probabilities outputs
    # outputs = model(face_crop, training=False)

    # # Prediction
    # prediction_index = tf.argmax(outputs[0]).numpy()
    # prediction = emotions_dict[prediction_index]

    # return prediction
    pass
