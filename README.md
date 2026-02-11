# Plant Disease Detection using CNN

This repository contains an implementation of plant disease detection using a Convolutional Neural Network (CNN) built from scratch. The repository also contains a web app allowing users to upload an image of a plant and the CNN model predicts whether the plant is healthy or if it is affected by powdery mildew or rust.

---

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Model](#model)
- [Dataset](#dataset)
- [Live Demo](#live-demo)
- [License](#license)

## Introduction

Plant diseases can have a detrimental impact on crop yield. Early detection and intervention are crucial to prevent the spread of diseases and minimize damage. The Streamlit web application provides an easy-to-use interface for users to upload plant images, which are then processed using a CNN model to predict the health status of the plant.

## Prerequisites

Before running the application, make sure you have the following dependencies installed:

- Python 3.8+
- TensorFlow 2.15
- Streamlit
- NumPy
- Pillow (PIL)

You can install the required packages using:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install tensorflow==2.15.0 streamlit numpy pillow
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/RajaAIStarter/plant-disease-prediction.git
cd plant-disease-prediction
```

### 2. Download Model Weights

The trained model weights (97% validation accuracy) are not included in this repository due to file size limitations.

**Download from Google Drive:** [plant_disease_classifier.h5](https://drive.google.com/file/d/your-file-id/view?usp=sharing)  
*(Replace with actual download link)*

After downloading, place the file in the `weights/` folder:

```
plant-disease-prediction/
├── weights/
│   └── plant_disease_classifier.h5    <-- Place downloaded file here
├── app.py
└── ...
```

> **Note:** The `weights/` folder exists but is empty by default (contains `.gitkeep`).

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

The web app will open in your default browser at `http://localhost:8501`, allowing you to upload plant images for disease classification.

## File Structure

```
plant-disease-prediction/
├── LICENSE
├── README.md                    # Project documentation
├── app.py                       # Streamlit web application source code
├── requirements.txt             # Python dependencies
│
├── assets/                      # Static assets for web app
│   ├── bg.png                   # Background image
│   ├── logo.png                 # Logo image
│   └── sample/                  # Sample images for testing
│       ├── healthy_sample.jpg
│       ├── powdery_mildew_sample.jpg
│       └── rust_sample.jpg
│
├── notebook/                    # Training notebooks
│   └── Plant-Disease-Detection-using-CNN.ipynb
│
└── weights/                     # Model weights directory
    ├── .gitkeep                # Keeps folder in git
    └── plant_disease_classifier.h5  # Download and place here (not in repo)
```

## Model

The CNN model was trained from scratch with the following architecture:

| Layer | Details |
|-------|---------|
| Input | 256 × 256 × 3 RGB images |
| Conv2D | 32 filters, (3,3), ReLU, BatchNorm, MaxPool, Dropout(0.2) |
| Conv2D | 64 filters, (5,5), ReLU, BatchNorm, MaxPool, Dropout(0.2) |
| Conv2D | 128 filters, (3,3), ReLU, BatchNorm, MaxPool, Dropout(0.3) |
| Conv2D | 256 filters, (5,5), ReLU, BatchNorm, MaxPool, Dropout(0.3) |
| Conv2D | 512 filters, (3,3), ReLU, BatchNorm, MaxPool, Dropout(0.3) |
| Flatten | - |
| Dense | 2048 units, ReLU, Dropout(0.5) |
| Output | 3 classes (Softmax) |

### Performance:
- **Validation Accuracy:** 97%
- **Validation Loss:** 0.29

### Classes:
1. **Healthy** – No disease detected
2. **Powdery Mildew** – Fungal infection (white powdery spots)
3. **Rust** – Rust pathogen (orange/yellow spots)

## Dataset

### Source
**Plant Disease Recognition Dataset** from Kaggle  
[https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset](https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset)

Derived from the original PlantVillage dataset with 3 classes focused on common agricultural diseases.

### Classes Distribution:
- **Healthy:** Apple healthy leaves
- **Powdery Mildew:** Cherry powdery mildew  
- **Rust:** Corn common rust

### Preprocessing:
- Resized to 256 × 256 pixels
- Normalized to [0, 1] range
- Data Augmentation: Random rotation, brightness, contrast adjustments
- Split: 70% Train | 15% Validation | 15% Test

## Live Demo

Try the deployed application on Hugging Face Spaces:

🔗 [https://huggingface.co/spaces/YOUR_USERNAME/plant-disease-prediction](https://huggingface.co/spaces/YOUR_USERNAME/plant-disease-prediction)

*(Replace with your actual HF Space URL after deployment)*

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**Developed by:** Team 4-KIET II | 2025  
**Institution:** Kakinada Institute of Engineering And Technology II