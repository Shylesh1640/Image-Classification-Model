import tensorflow as tf
import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from config import BEST_MODEL_PATH, BATCH_SIZE
from src.data_loader import get_data
from src.preprocessing import preprocess_numpy_data, prepare_dataset

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def evaluate():
    if not os.path.exists(BEST_MODEL_PATH):
        print("Model not found. Train first.")
        return

    print("Loading best model...")
    model = tf.keras.models.load_model(BEST_MODEL_PATH)
    
    # Load Data (Test set)
    data, source_type = get_data()
    
    if source_type == 'cifar10':
        _, (x_test, y_test) = data
        test_ds, _ = preprocess_numpy_data(x_test, y_test, x_test, y_test) # Dummy val
        y_true = y_test.flatten()
    else:
        # Assuming we have a test set
        train_ds, val_ds, test_ds = data
        if test_ds is None:
             print("No test set found, using validation set.")
             test_ds = val_ds
        
        # We need to extract labels from tf.data.Dataset for sklearn metrics
        # This can be slow for large datasets
        y_true = np.concatenate([y for x, y in test_ds], axis=0)

    test_ds = prepare_dataset(test_ds, augment=False, batch_size=BATCH_SIZE)

    print("Evaluating model...")
    loss, acc = model.evaluate(test_ds)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {acc:.4f}")

    # Predictions
    print("Generating predictions...")
    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Metrics
    print("\nClassification Report:")
    # CIFAR-10 Classes
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # If custom, we might need to load class names from directory
    
    if len(np.unique(y_true)) == 10:
         print(classification_report(y_true, y_pred, target_names=class_names))
         plot_confusion_matrix(y_true, y_pred, class_names)
    else:
         print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    import os
    evaluate()
