<div align="center">


<img src="https://img.shields.io/badge/Python-3.10.6-2496ED?style=flat-square&logo=python&logoColor=2496ED" />
<img src="https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=flat-square&logo=tensorflow&logoColor=FF6F00" />
<img src="https://img.shields.io/badge/Mediapipe-0.10.9-008080?style=flat-square&logo=mediapipe&logoColor=008080" />
<img src="https://img.shields.io/badge/OpenCV-4.7.0.72-red?style=flat-square&logo=opencv&logoColor=red" />
<img src="https://img.shields.io/badge/FastAPI-0.135.1-009688?style=flat-square&logo=fastapi&logoColor=009688" />
<img src="https://img.shields.io/badge/Docker-Ready-4285F4?style=flat-square&logo=docker&logoColor=2496ED" />


# Real Time Facial Emotion Recognition


</div>

### Table of Contents

1. [About this Project](#about-this-project)
2. [App Visuals](#app-visuals)
3. [Installation and Usage](#installation-and-usage)
4. [Project's Structure](#projects-structure)
5. [Dataset Used](#dataset-used)
6. [Stack](#stack)
7. [Methodological Approach](#methodological-approach)
8. [General Informations](#general-informations)


## About this Project
This project is a computer vision project that use deep learning to detect emotions on faces
in real time.

It allows to detect multiple faces at the same time.


## App Visuals

[Images to put here]

## Installation and Usage

#### 1. Repository clonning
```bash
mkdir ~/<your_path>/Real-Time-Facial-Emotion-Recognition && cd "$_"
git clone git@github.com:NkL-M/Real-Time-Facial-Emotion-Recognition.git
```

#### 2. Project navigation
```bash
cd Real-Time-Facial-Emotion-Recognition
```

#### 3. Virtual environment creation and activation
```bash
pyenv virtualenv 3.10.6 emotion-recognition
pyenv local emotion-recognition
```

#### 4. Dependencies installation
```bash
pip install -r requirements.txt
pip install -e .
```

#### 5. API
```bash
make run_api
```

## Project's Structure

```bash
.
│
├── emotion_recognition
│   ├── api
│   │   └── fast.py
│   ├── face_detection
│   │   └── face_detection.py
│   ├── interface
│   │   └── main.py
│   ├── src
│   │   ├── data.py
│   │   ├── model.py
│   │   └── registry.py
│   ├── params.py
│   └── utils.py
├── images
│   ├── app_demo.png
│   ├── api.png
│   └── video_main.png
├── models_registry
│   ├── saved_metrics
│   ├── saved_models
│   ├── saved_params
│   └── saved_weights
├── notebooks
│   ├── data_exploration.ipynb
│   └── train_model.ipynb
├── .gitignore
├── Makefile
├── Dockerfile
├── README.md
├── requirements.txt
└── setup.py
```

## Dataset Used
The dataset used for this project is [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013), which for this project was downloaded from Kaggle. It is made of 35 887 grayscale images of faces, all in a 48x48 pixels format.

Split in 7 facial emotion classes:
- Neutral (6198 images)
- Happy (8989 images)
- Angry (4953 images)
- Sad (6077 images)
- Fear (5121 images)
- Disgust (547 images)
- Surprise (4002 images)

For the training and testing, the dataset was split as followed:
- 22968 images for the training set
- 5741 images for the validation set
- 7178 images for the test set

## Stack
* Language: Python (3.10.6)
* DL Framework: TensorFlow / Keras
* Face Detection: Mediapipe
* API Framework: FastAPI
* Containerization: Docker
* Front: Streamlit


## Methodological Approach

### 1 - Preprocessing
The images from the dataset had vastly different sizes/resolutions, thus for the preprocessing, we opted for an offline preprocessing with 2 different techniques:
  * A _resize_ of the images in 48 by 48 pixels

In the end, we settled with the resize preprocessing since it performed slightly better than the random crop while avoiding some unfortunate cases where the crop could randomly land in a location where the image would be less be prone to show patterns of possible AI generation.

### 2 - Data Augmentation
Class imbalance

### 2 - Evaluation Metric
We chose **accuracy** as the metric for our models since we wanted to have as little _False Positive_ and _False Negative_ as possible.

### 3 - Results
We tested multiple deep learning models architectures to compare them and select the ones that yielded the best results.
3 models showed promising results in our evaluation:


| Model              |  Test Score        |
|--------------------|--------------------|
| Baseline CNN       |  XX.XX% accuracy   |
| EfficientNetB4     |  XX.XX% accuracy   |
| MobileNet V3 Large |  XX.XX% accuracy   |
| ResNet50           |  XX.XX% accuracy   |
| Custom CNN         |  63.90% accuracy   |


## General Informations

    Author : Nicolas Marechal
    Mail : marechal.n@hotmail.com
