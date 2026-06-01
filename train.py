import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
import time
import pandas as pd
import os

# --- Configuration ---
DATA_DIR = 'dataset'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS_INITIAL = 10  # Train longer
EPOCHS_FINE_TUNE = 5 # Fine-tune phase

# --- Data Augmentation (Crucial for better accuracy) ---
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal"),
  layers.RandomRotation(0.2),
  layers.RandomZoom(0.2),
])

# --- Prepare Data ---
print("Loading Training Data...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, validation_split=0.2, subset="training", seed=123, image_size=IMG_SIZE, batch_size=BATCH_SIZE
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, validation_split=0.2, subset="validation", seed=123, image_size=IMG_SIZE, batch_size=BATCH_SIZE
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
)

class_names = train_ds.class_names
print(f"Classes: {class_names}")

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
    base_model.trainable = False  # Freeze initially

    inputs = tf.keras.Input(shape=(224, 224, 3))
    # 1. Apply Data Augmentation
    x = data_augmentation(inputs)
    # 2. Preprocessing (Normalize to -1 to 1 for MobileNet/EfficientNet compatibility)
    x = layers.Rescaling(1./127.5, offset=-1)(x)
    
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x) # Increased Dropout
    outputs = layers.Dense(1, activation='sigmoid')(x) 
    
    model = models.Model(inputs, outputs, name=model_name)
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model, base_model

# --- Training Loop with Fine-Tuning ---
results = []
models_to_train = [
    (MobileNetV2, "MobileNetV2"),
    (ResNet50, "ResNet50"),
    (EfficientNetB0, "EfficientNetB0")
]

best_model = None
best_val_acc = 0

for ModelClass, name in models_to_train:
    model, base_model = build_model(ModelClass, name)
    
    print(f"Phase 1: Training head of {name}...")
    start_time = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_INITIAL)
    
    # --- FINE TUNING PHASE ---
    print(f"Phase 2: Fine-tuning {name}...")
    base_model.trainable = True
    # Freeze bottom layers, keep top layers trainable
    # Example: Keep last 20 layers unfrozen
    for layer in base_model.layers[:-20]:
        layer.trainable = False
        
    # Recompile with very low learning rate
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), 
                  loss='binary_crossentropy', metrics=['accuracy'])
    
    history_fine = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE_TUNE)
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # Get best metrics
    final_val_acc = max(history_fine.history['val_accuracy'])
    
    # Evaluate on Test Set
    print(f"Evaluating {name}...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    
    results.append({
        "Model": name,
        "Val Accuracy": f"{final_val_acc*100:.2f}%",
        "Test Accuracy": f"{test_acc*100:.2f}%",
        "Time (s)": f"{training_time:.2f}"
    })

    if final_val_acc > best_val_acc:
        best_val_acc = final_val_acc
        best_model = model
        print(f"*** New best model: {name} ***")

# --- Output ---
print("\n" + "="*50)
df_results = pd.DataFrame(results)
print(df_results)

if best_model:
    best_model.save('flower_model.keras')
    print(f"\nBest model saved as 'flower_model.keras'")
