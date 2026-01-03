# Image Classification Model Project

## Project Overview
This project builds an end-to-end image classification machine learning pipeline. It covers data preparation, model development, training, evaluation, and deployment considerations. We implement both traditional ML concepts and deep learning approaches (Transfer Learning and Custom CNNs).

## Project Objectives
- Build an end-to-end image classification pipeline.
- Implement Transfer Learning (ResNet/VGG/MobileNet) and Custom CNNs.
- Achieve high accuracy while avoiding overfitting.
- Create a deployable model with proper documentation.

## Directory Structure
```
project/
├── data/
│   ├── raw/             # Original dataset
│   ├── processed/       # Preprocessed data (arrays, etc.)
│   └── augmented/       # Augmented images (if saved)
├── notebooks/           # Jupyter notebooks for experimentation
├── src/                 # Source code
│   ├── data_loader.py   # Data loading scripts
│   ├── preprocessing.py # Image preprocessing functions
│   ├── model.py         # Model architecture definitions
│   ├── train.py         # Training loop
│   ├── evaluate.py      # Evaluation scripts
│   └── predict.py       # Inference scripts
├── models/              # Saved models (.h5)
├── config.py            # Configuration parameters
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
1. **Data Preparation**: Ensure your dataset is in `data/raw` or use the built-in CIFAR-10 downloader options in `src/data_loader.py`.
2. **Training**:
   ```bash
   python src/train.py
   ```
3. **Evaluation**:
   ```bash
   python src/evaluate.py
   ```
4. **Prediction**:
   ```bash
   python src/predict.py --image_path path/to/image.jpg
   ```

## Dataset
This project supports loading data from a directory structure or standard datasets (CIFAR-10).
Expected custom data structure:
```
data/raw/
├── train/
│   ├── class1/
│   └── class2/
└── test/
    ├── class1/
    └── class2/
```

## Models
- **Transfer Learning**: Uses ResNet50V2 / MobileNetV2 pre-trained on ImageNet.
- **Custom CNN**: A standard convolutional neural network trained from scratch.
