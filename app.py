import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Flower Classifier", layout="centered")

# Load the model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("flower_model.keras")
        return model
    except:
        return None

model = load_model()

st.title("Sunflower vs. Dandelion Classifier")
st.write("Upload an image of a sunflower or a dandelion to classify it.")

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button("Predict"):
        if model:
            # Preprocess image
            image = image.resize((224, 224))
            img_array = np.array(image)
            # Expand dimensions to match batch size (1, 224, 224, 3)
            img_array = np.expand_dims(img_array, axis=0) 
            
            # Predict
            prediction = model.predict(img_array)
            score = float(tf.nn.sigmoid(prediction[0][0])) # Get sigmoid probability
            
            # Interpret result
            # Assuming Sunflower is class 0 and Dandelion is class 1 (or vice versa)
            # Check train.py output for class_names mapping if unsure, 
            # usually alphabetical: 0 = dandelion, 1 = sunflower
            
            label = "Dandelion" if score < 0.5 else "Sunflower"
            confidence = 1 - score if score < 0.5 else score
            
            st.success(f"Predicted Class: {label}")
            st.write(f"Confidence: {confidence * 100:.2f}%")
        else:
            st.error("Model not found. Please run train.py first.")