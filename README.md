# Fish Classification and Detection Project

## Project Overview

This project demonstrates the process of training a model to classify fish diseases using the **VIT_16** model and then detecting and visualizing fish conditions in videos. The project is divided into two main parts:

1. **Fish Classification (Training)**
2. **Fish Detection (Inference/Visualization)**

## Table of Contents

- [Dataset](#dataset)
- [Model](#model)
- [Installation](#installation)
- [Training](#training)
- [Fish Detection](#fish-detection)
- [Usage](#usage)
- [Notes](#notes)
- [License](#license)

## Dataset

### Source

The dataset used for training the fish disease classification model is available on Kaggle:  
[Fish Disease Dataset](https://www.kaggle.com/datasets/alaamahmoud2010/fish-disease)

### Setup

1. Download the dataset from Kaggle.
2. Extract the files and place them in the appropriate folder as described in the Training section.

## Model

The model used for training is **VIT_16** (Vision Transformer with 16x16 patches), known for its strong performance in image classification tasks.

## Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. Create and Activate a Virtual Environment

It is recommended to use a virtual environment for dependency management:

```bash
python -m venv .env
source .env/bin/activate  # On Windows: .env\Scripts\activate
```

### 3. Install Dependencies

Use the `requirements.txt` file to install the required packages:

```bash
pip install -r requirements.txt
```

## Training

### Steps for Training:

#### 1. Download the Dataset

- Navigate to the `fish_classification/data` folder.
- Download the dataset and place it in the `data` folder.

#### 2. Configure Training Settings

- Open `train.py` in the `fish_classification` folder.
- Set the dataset folder path.
- Adjust hyperparameters (learning rate, batch size, number of epochs, etc.).
- Specify where the trained model should be saved.

#### 3. Run the Training Script

```bash
python train.py
```

The trained model will be saved in the specified directory.

## Fish Detection

### Steps for Detection:

#### 1. Configure Detection Settings

- Navigate to the `fish_detection` folder.
- Open the detection script (or configuration file) and set the model directory.
- Ensure all required data paths are correct.

#### 2. Run the Detection Script

```bash
python <detection.py>
```

### Output Details

- **Green boundaries** indicate healthy fish.
- **Other boundaries (or indicators)** highlight infected fish.

## Usage

### Training

- Configure dataset and hyperparameters in `train.py`.
- Run the script to train and save the model.

### Detection

- Adjust model and data paths in the detection script.
- Run the script to visualize the results.

### Customization

- Both training and detection scripts are modular.
- Modify settings, file paths, and hyperparameters as needed.

## Notes

- Ensure all dependencies are installed before running scripts.
- The dataset should follow the expected directory structure.
- If using a GPU, ensure CUDA and relevant libraries are configured.