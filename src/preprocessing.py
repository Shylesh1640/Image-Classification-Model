import tensorflow as tf
from tensorflow.keras import layers
from config import IMG_HEIGHT, IMG_WIDTH

def get_augmentation_layer():
    """Returns a sequential model of augmentation layers."""
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])
    return data_augmentation

def preprocess_image(image, label):
    """
    Basic preprocessing: Rescaling.
    Applied to tf.data.Dataset
    """
    image = tf.image.resize(image, (IMG_HEIGHT, IMG_WIDTH))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def prepare_dataset(ds, augment=False, batch_size=32):
    """
    Prepares a tf.data.Dataset for high performance.
    """
    # Resize and Rescale
    ds = ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        aug_layer = get_augmentation_layer()
        # Apply augmentation
        ds = ds.map(lambda x, y: (aug_layer(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    # ds = ds.cache() # Disabled to prevent OOM with large image sizes
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

def preprocess_numpy_data(x_train, y_train, x_test, y_test):
    """
    Converts numpy data (CIFAR-10) to tf.data.Dataset and applies preprocessing.
    """
    # Convert to tf.data.Dataset
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    
    # Shuffle train
    train_ds = train_ds.shuffle(buffer_size=1000)
    
    # Batch
    from config import BATCH_SIZE
    train_ds = train_ds.batch(BATCH_SIZE)
    test_ds = test_ds.batch(BATCH_SIZE)
    
    return train_ds, test_ds
