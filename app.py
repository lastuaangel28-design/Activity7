import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Flower Classifier", layout="centered")

@st.cache_resource
def load_model():
    try:
        # Load the model trained with the new script
        model = tf.keras.models.load_model("flower_model.keras")
        return model
    except:
        return None

model = load_model()

st.title("Sunflower vs. Dandelion Classifier")
st.write("Upload an image to classify.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button("Predict"):
        if model:
            # 1. Resize
            image = image.resize((224, 224))
            img_array = np.array(image)
            
            # 2. Preprocessing MUST match training
            # Training used: layers.Rescaling(1./127.5, offset=-1)
            # This converts 0-255 -> -1 to 1
            img_array = img_array.astype("float32")
            img_array = (img_array / 127.5) - 1.0
            
            # 3. Expand dimensions for batch size
            img_array = np.expand_dims(img_array, axis=0) 
            
            # 4. Predict
            prediction = model.predict(img_array)
            score = float(tf.nn.sigmoid(prediction[0][0]))
            
            # 5. Interpret Result
            # 0 = Dandelion, 1 = Sunflower (Alphabetical order usually)
            if score < 0.5:
                label = "Dandelion"
                confidence = 1.0 - score
            else:
                label = "Sunflower"
                confidence = score
                
            st.success(f"Predicted Class: {label}")
            st.write(f"Confidence: {confidence * 100:.2f}%")
        else:
            st.error("Model not found. Please run train.py first.")
