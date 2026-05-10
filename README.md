# SENTIMENT-TRACKER - Real Time Emotion Recognition

An app that use deep-learning and computer vision to detect Emotions shown on a face in real-time.


### 🚀 Table of Contents

1. [About this Project](#-about-this-project)
2. [App Visuals](#-app-visuals)
3. [Installation and Usage](#-installation-and-usage)
4. [Project's Structure](#-projects-structure)
5. [Dataset Used](#-dataset-used)
6. [Methodological Approach](#-methodological-approach)
7. [Environnement and Tools](#-environnement-and-tools)
8. [General Informations](#-general-informations)


## 📋 About this Project
This app is a computer vision project that use deep learning to detect emotions on faces
on real time.


## 📷 App Visuals

[Images to put here]

## 🧰 Installation and Usage

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

## 📁 Project's Structure

```bash
.
├── data
│   ├── train
│   ├── test
│   └── videos
├── emotion_recognition
│   ├── api
│   │   └── fast.py
│   ├── interface
│   │   └── main.py
│   ├── src
│   │   ├── data.py
│   │   ├── model.py
│   │   └── registry.py
│   ├── params.py
│   └── utils.py
├── .env.sample
├── .envrc
├── .gitignore
├── Makefile
├── Dockerfile
├── README.md
├── requirements.txt
└── setup.py
```

## 📊 Dataset Used
Our model was trained using the [Balanced AffectNet dataset](https://www.kaggle.com/datasets) from Kaggle.
This dataset aims to balance the label issues from AffectNet by ...

The dataset included XX XXX images of faces with different gender, age, etc.
The dataset has been separated in:
- XX XXX images for the training dataset
- XX XXX images for the validation dataset
- XX XXX images for the test dataset



## 🔎 Methodological Approach

### 1 - Data Preprocessing
The images from the dataset had vastly different sizes/resolutions, thus for the preprocessing, we opted for an offline preprocessing with 2 different techniques:
  * A _resize_ of the images in 256 by 256 pixels

In the end, we settled with the resize preprocessing since it performed slightly better than the random crop while avoiding some unfortunate cases where the crop could randomly land in a location where the image would be less be prone to show patterns of possible AI generation.

### 2 - Model Selection
We tested multiple deep learning models architectures to compare them and select the ones that yielded the best results.
* Baseline CNN (3 layers)
* Custom Complex CNN (10 layers)
* VGG16
* EfficientNetB4
* ResNet50

### 3 - Evaluation Metric
We chose **accuracy** as the metric for our models since we wanted to have as little _False Positive_ and _False Negative_ as possible.

### 4 - Results
3 models showed promising results in our evaluation:

* Baseline CNN: XX.XX% accuracy
* EfficientNetB4: XX.XX% accuracy
* ResNet50: XX.XX% accuracy
* ResNet50 with fine tuning: XX.XX% accuracy


## 🌱 Environnement and Tools

* Programming Language : Python 3.10.6
* Versioning : Git / GitHub
* DL Framework : TensorFlow
* API Framework : FastAPI
* Containerization : Docker
* Frontend : Streamlit


## 📑 General Informations

    Author : Nicolas Marechal

    Mail : marechal.n@hotmail.com

    Last update: 2026-05-20
