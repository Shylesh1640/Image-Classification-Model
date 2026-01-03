import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
from config import EPOCHS, BATCH_SIZE, MODEL_TYPE, BASE_MODEL, BEST_MODEL_PATH
from src.data_loader import get_data
from src.preprocessing import preprocess_numpy_data, prepare_dataset
from src.model import build_custom_cnn, build_transfer_model

def main():
    # 1. Load Data
    data, source_type = get_data()
    
    if source_type == 'cifar10':
        (x_train, y_train), (x_test, y_test) = data
        train_ds, val_ds = preprocess_numpy_data(x_train, y_train, x_test, y_test)
        # Using test as val for CIFAR-10 demo simplicity, usually split x_train further
    else:
        train_ds, val_ds, test_ds = data
    
    # 2. Preprocess Pipelines
    train_ds = prepare_dataset(train_ds, augment=True, batch_size=BATCH_SIZE)
    val_ds = prepare_dataset(val_ds, augment=False, batch_size=BATCH_SIZE)

    # 3. Build Model
    print(f"Building model: {MODEL_TYPE}")
    if MODEL_TYPE == 'custom_cnn':
        model = build_custom_cnn()
    else:
        model = build_transfer_model(base_model_name=BASE_MODEL)
    
    model.summary()

    # 4. Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(BEST_MODEL_PATH, save_best_only=True, monitor='val_loss'),
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    ]
    
    if not os.path.exists(os.path.dirname(BEST_MODEL_PATH)):
        os.makedirs(os.path.dirname(BEST_MODEL_PATH))

    # 5. Train
    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    print("Training complete.")

if __name__ == "__main__":
    main()
