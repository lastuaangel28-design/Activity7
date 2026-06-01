import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
import time
import pandas as pd
import os
import numpy as np

# --- Configuration ---
DATA_DIR = 'dataset'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 5  # Increase this for better accuracy (e.g., 10-20)

# --- Prepare Data ---
print("Loading Training Data...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

print("Loading Test Data...")
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
print(f"Classes found: {class_names}")

# Performance optimization
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- Model Building Function ---
def build_model(base_model_class, model_name):
    print(f"\n--- Building {model_name} ---")
    
    # Load Pre-trained Base
    base_model = base_model_class(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze the base

    inputs = tf.keras.Input(shape=(224, 224, 3))
    # Preprocessing layer (Rescaling to 0-1 or specific inputs needed by model)
    # Standard rescaling 1./255 works generally well for transfer learning fine-tuning
    x = layers.Rescaling(1./255)(inputs) 
    
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x) # Regularization
    # Binary Classification: 1 neuron with sigmoid activation
    outputs = layers.Dense(1, activation='sigmoid')(x) 
    
    model = models.Model(inputs, outputs, name=model_name)
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# --- Training Loop ---
results = []

models_to_train = [
    (MobileNetV2, "MobileNetV2"),
    (ResNet50, "ResNet50"),
    (EfficientNetB0, "EfficientNetB0")
]

best_model = None
best_val_acc = 0

for ModelClass, name in models_to_train:
    model = build_model(ModelClass, name)
    
    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )
    end_time = time.time()
    
    training_time = end_time - start_time
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    
    # Evaluate on Test Set
    print(f"Evaluating {name} on Test Set...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    
    results.append({
        "Model": name,
        "Training Accuracy": f"{final_train_acc*100:.2f}%",
        "Validation Accuracy": f"{final_val_acc*100:.2f}%",
        "Test Accuracy": f"{test_acc*100:.2f}%",
        "Training Time (s)": f"{training_time:.2f}"
    })

    # Check if this is the best model so far
    if final_val_acc > best_val_acc:
        best_val_acc = final_val_acc
        best_model = model
        print(f"*** New best model found: {name} ***")

# --- Output Comparison ---
print("\n" + "="*50)
print("PERFORMANCE COMPARISON TABLE")
print("="*50)
df_results = pd.DataFrame(results)
print(df_results)

# --- Save the Best Model for the Web App ---
# We save it as 'flower_model.keras' to be loaded by app.py
if best_model:
    best_model.save('flower_model.keras')
    print(f"\nBest model saved as 'flower_model.keras'")
