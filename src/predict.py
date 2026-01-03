import argparse
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
import numpy as np
import os
from config import BEST_MODEL_PATH, IMG_HEIGHT, IMG_WIDTH

def load_and_prep_image(filename):
    """
    Reads an image from filename, turns it into a tensor and reshapes it
    to (IMG_HEIGHT, IMG_WIDTH, 3).
    """
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = img / 255.
    return img

def predict_image(image_path):
    if not os.path.exists(BEST_MODEL_PATH):
        print("Model not found. Please train the model first.")
        return

    print(f"Loading model from {BEST_MODEL_PATH}...")
    model = tf.keras.models.load_model(BEST_MODEL_PATH)
    
    print(f"Processing image: {image_path}")
    img = load_and_prep_image(image_path)
    img_expanded = tf.expand_dims(img, axis=0) # Add batch dimension
    
    pred_prob = model.predict(img_expanded)
    pred_class = np.argmax(pred_prob)
    confidence = np.max(pred_prob)
    
    # CIFAR-10 Mapping (Standard)
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # For custom datasets, you would load class names from a file (e.g., class_names.txt saved during training)
    
    if pred_class < len(class_names):
        pred_label = class_names[pred_class]
    else:
        pred_label = str(pred_class)
        
    print(f"\nPrediction: {pred_label}")
    print(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict image class")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image file")
    args = parser.parse_args()
    
    predict_image(args.image_path)
