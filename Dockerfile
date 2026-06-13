FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY emotion_recognition/ emotion_recognition/

CMD uvicorn emotion_recognition.api.fast:app --host 0.0.0.0 --port $PORT
