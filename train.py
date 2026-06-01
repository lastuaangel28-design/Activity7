import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import time

# 1. Dataset Preparation
TRAIN_DIR = 'dataset/train'
TEST_DIR = 'dataset/test'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print("Loading Training Data...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE
)

# IMPORTANT: Check which class is 0 and which is 1
class_names = train_ds.class_names
print(f"Class Indices found: {class_names} (0: {class_names[0]}, 1: {class_names[1]})")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE
)

# Normalize pixel values (0-255 -> 0-1)
normalization_layer = tf.keras.layers.Rescaling(1./255.)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# 2. Function to build models
def build_model(base_model_class, name):
    # Load base model with ImageNet weights, exclude top
    base = base_model_class(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    base.trainable = False  # Freeze base layers
    
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    # Binary classification output (1 neuron, sigmoid)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base.input, outputs=predictions)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 3. Train and Evaluate Loop
results = []
architectures = [
    (MobileNetV2, 'MobileNetV2'),
    (ResNet50, 'ResNet50'),
    (EfficientNetB0, 'EfficientNetB0')
]

for arch, name in architectures:
    print(f"\n--- Training {name} ---")
    model = build_model(arch, name)
    
    start_time = time.time()
    history = model.fit(train_ds, epochs=5, validation_data=test_ds, verbose=1)
    end_time = time.time()
    
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    duration = end_time - start_time
    
    results.append({
        'Model': name,
        'Train Acc': f"{train_acc:.2%}",
        'Val Acc': f"{val_acc:.2%}",
        'Time (s)': f"{duration:.1f}"
    })
    
    # Save the best model for Streamlit
    if name == 'EfficientNetB0': # Assuming this wins
        model.save('best_model.h5')

print("\nTraining Complete.")
