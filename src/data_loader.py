import os
import tensorflow as tf
import numpy as np
from config import RAW_DATA_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, SEED

def load_cifar10():
    """Loads and preprocesses the CIFAR-10 dataset."""
    print("Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # We will split training into train and val later or during training
    return (x_train, y_train), (x_test, y_test)

def load_custom_dataset(directory=RAW_DATA_DIR):
    """
    Loads dataset from directory. 
    Expects structure:
    directory/
        class_a/
        class_b/
    """
    if not os.path.exists(directory) or not os.listdir(directory):
        print(f"Directory {directory} is empty or does not exist.")
        return None

    print(f"Loading data from {directory}...")
    
    # Check if 'train' subfolder exists, otherwise assume structure is class folders directly
    train_dir = os.path.join(directory, 'train')
    if os.path.exists(train_dir):
        print("Found train/test structure.")
        # Load train
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            validation_split=None,
            seed=SEED,
            image_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE
        )
        # Load val if exists
        val_dir = os.path.join(directory, 'validation')
        val_ds = None
        if os.path.exists(val_dir):
             val_ds = tf.keras.utils.image_dataset_from_directory(
                val_dir,
                seed=SEED,
                image_size=(IMG_HEIGHT, IMG_WIDTH),
                batch_size=BATCH_SIZE
            )
        
        # Load test if exists
        test_dir = os.path.join(directory, 'test')
        test_ds = None
        if os.path.exists(test_dir):
             test_ds = tf.keras.utils.image_dataset_from_directory(
                test_dir,
                seed=SEED,
                image_size=(IMG_HEIGHT, IMG_WIDTH),
                batch_size=BATCH_SIZE
            )
            
        return train_ds, val_ds, test_ds
    else:
        print("Found flat structure. using validation split.")
        train_ds = tf.keras.utils.image_dataset_from_directory(
            directory,
            validation_split=0.2,
            subset="training",
            seed=SEED,
            image_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            directory,
            validation_split=0.2,
            subset="validation",
            seed=SEED,
            image_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE
        )
        return train_ds, val_ds, None

def get_data():
    """Dispatcher function to determine which data source to use."""
    # Check if we have data in RAW_DATA_DIR
    custom_data = load_custom_dataset()
    if custom_data is not None:
        return custom_data, 'custom'
    else:
        return load_cifar10(), 'cifar10'
