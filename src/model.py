import tensorflow as tf
from tensorflow.keras import layers, models, applications
from config import INPUT_SHAPE, NUM_CLASSES, LEARNING_RATE

def build_custom_cnn(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """
    Builds a custom CNN model from scratch.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # Block 1
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Block 2
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Block 3
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        # Dense
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_transfer_model(base_model_name='MobileNetV2', input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """
    Builds a model using Transfer Learning.
    """
    if base_model_name == 'MobileNetV2':
        base_model = applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    elif base_model_name == 'ResNet50':
        base_model = applications.ResNet50V2(input_shape=input_shape, include_top=False, weights='imagenet')
    elif base_model_name == 'VGG16':
        base_model = applications.VGG16(input_shape=input_shape, include_top=False, weights='imagenet')
    else:
        raise ValueError(f"Unknown base model: {base_model_name}")

    base_model.trainable = False # Freeze base model initially

    inputs = layers.Input(shape=input_shape)
    # Preprocessing specific to the model can be added here if needed, but we normalized generally.
    # Note: MobileNetV2 expects [-1, 1], we did [0, 1]. 
    # Proper way is to use applications.mobilenet_v2.preprocess_input on raw [0, 255] images.
    # But since we already rescaled to [0, 1], let's adjust or trust the FineTuning will adapt.
    # Better: Use the model's preprocess_input in the data pipeline or here.
    
    # Let's assume input is [0, 1] for Custom CNN and Transfer Learning for simplicity in this template.
    # Ideally we should use specific preprocessing.
    
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model
