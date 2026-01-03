import os

# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Data Parameters
IMG_HEIGHT = 128  # Reduce to 32 for CIFAR-10 if not resizing
IMG_WIDTH = 128
CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)
BATCH_SIZE = 32
NUM_CLASSES = 10 # Default for CIFAR-10, update for custom data

# Training Parameters
EPOCHS = 20
LEARNING_RATE = 0.001
VAL_SPLIT = 0.2
TEST_SPLIT = 0.1
SEED = 42

# Model Parameters
MODEL_TYPE = 'transfer_learning' # 'custom_cnn' or 'transfer_learning'
BASE_MODEL = 'MobileNetV2' # Options: 'ResNet50', 'MobileNetV2', 'VGG16'

# Paths
BEST_MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.h5')
