import sys
import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import BEST_MODEL_PATH, BATCH_SIZE
from src.data_loader import get_data
from src.preprocessing import preprocess_numpy_data, prepare_dataset

def main():
    print("Loading data for evaluation...")
    data, source_type = get_data()
    
    if source_type == 'cifar10':
        (x_train, y_train), (x_test, y_test) = data
        _, test_ds = preprocess_numpy_data(x_train, y_train, x_test, y_test)
    else:
        # For custom dataset, data dispatcher returns train, val, test
        _, _, test_ds = data
        if test_ds is None:
            print("No separate test set found in custom dataset.")
            return

    # Apply preprocessing pipeline (resizing, scaling)
    # augment=False for evaluation
    test_ds = prepare_dataset(test_ds, augment=False, batch_size=BATCH_SIZE)

    if not os.path.exists(BEST_MODEL_PATH):
        print(f"Model file not found at {BEST_MODEL_PATH}")
        return

    print(f"Loading model from {BEST_MODEL_PATH}...")
    model = tf.keras.models.load_model(BEST_MODEL_PATH)

    print("Evaluating model...")
    loss, acc = model.evaluate(test_ds)
    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test Accuracy: {acc:.4f}")

    print("\nGenerating Classification Report...")
    
    # Get predictions
    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Get true labels
    # Iterate over dataset to extract labels in order
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    
    # If using CIFAR-10, y_true might be one-hot or sparse integers depending on loading, 
    # but preprocess_numpy_data keeps them as is (integers for CIFAR-10 from keras.datasets).
    # tf.keras.datasets.cifar10.load_data() returns integer labels.
    # However, if using image_dataset_from_directory with 'categorical' (default is 'int'?), 
    # we need to be careful.
    # checking data_loader: for custom it uses image_dataset_from_directory. Default label_mode is 'int'.
    # checking preprocessing: preprocess_numpy_data uses from_tensor_slices
    
    # Flatten y_true if it has shape (N, 1)
    if y_true.ndim > 1:
        y_true = y_true.flatten()

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    if source_type != 'cifar10':
        # Ideally we load class names from the directory structure or a saved file
        # For now, we'll use generic placeholders or skip specific names
        class_names = [str(i) for i in range(len(np.unique(y_true)))]

    print(classification_report(y_true, y_pred, target_names=class_names[:len(np.unique(y_true))]))

if __name__ == "__main__":
    main()
