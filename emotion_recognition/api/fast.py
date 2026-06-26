import cv2
import numpy as np
import tensorflow as tf
from io import BytesIO
from PIL import Image
from colorama import Fore, Style
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from emotion_recognition.interface.main import predict_image
from emotion_recognition.src.registry import load_tf_model
from emotion_recognition.params import NB_CHANNELS, INPUT_SHAPE, FER_MODEL, EMOTION_DICT
from emotion_recognition.face_detection.face_detection import FaceDetector



app = FastAPI()
app.state.model = None
app.state.face_detector = FaceDetector()

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
            app.state.model = load_tf_model(FER_MODEL)
            print(Fore.GREEN + "\nModel loaded successfully" + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"\nError loading model: {e}" + Style.RESET_ALL)
            raise HTTPException(status_code=500, detail="Model could not be loaded")

    return app.state.model


@app.get("/")
def root():
    return {"status": "Hello, the API is alive!"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    model = get_model()

    try:
        content = await file.read()
        img = Image.open(BytesIO(content))
        img.verify()
        img = Image.open(BytesIO(content)).convert("RGB")

        # img = tf.io.read_file(file)
        img = tf.image.decode_image(img, channels=NB_CHANNELS)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Image corrupted or unreadable."
        )

    img = tf.image.resize(img, size=INPUT_SHAPE)
    x_preprocessed = tf.expand_dims(input=img, axis=0)


    try:
        outputs = model(x_preprocessed)
        prediction_index = tf.argmax(outputs[0]).numpy()
        prediction = EMOTION_DICT[prediction_index]
    except Exception as e:
        print(Fore.RED + f"\nPrediction error: {e}" + Style.RESET_ALL)
        raise HTTPException(status_code=500, detail="Prediction failed")

    return {
        "Emotion": prediction
    }


from pydantic import BaseModel
from typing import List

class EmotionScore(BaseModel):
    label: str
    confidence: float

class FaceResult(BaseModel):
    bounding_box: dict       # {x, y, width, height}
    predicted_emotion: str
    confidence: float
    all_scores: List[EmotionScore]
    is_confident: bool       # see point 3
    inference_time_ms: float

class AnalyzeResponse(BaseModel):
    faces_detected: int
    results: List[FaceResult]
    image_width: int
    image_height: int

# @app.post("/analyze", response_model=AnalyzeResponse)
# async def analyze(image: UploadFile = File(...)):
#     """
#     Detects faces and predicts emotions for each detected face.

#     Returns bounding boxes, per-class confidence scores, and
#     a confidence flag indicating whether the prediction should
#     be trusted.
#     """
#     img_array = await read_image(image)
#     # faces = detect_faces(img_array)          # MediaPipe
#     faces = app.state.face_detector.detect_faces(img_array)

#     results = []
#     for face in faces:
#         cropped = preprocess(face, img_array)
#         start = time.time()
#         scores = run_inference(cropped)       # ONNX session
#         inference_time = (time.time() - start) * 1000

#         top_label, top_conf = get_top_prediction(scores)

#         results.append(FaceResult(
#             bounding_box=face.bbox,
#             predicted_emotion=top_label,
#             confidence=top_conf,
#             all_scores=[EmotionScore(label=l, confidence=c) for l, c in scores.items()],
#             is_confident=top_conf >= CONFIDENCE_THRESHOLD,
#             inference_time_ms=round(inference_time, 2)
#         ))

#     return AnalyzeResponse(
#         faces_detected=len(faces),
#         results=results,
#         image_width=img_array.shape[1],
#         image_height=img_array.shape[0]
#     )
