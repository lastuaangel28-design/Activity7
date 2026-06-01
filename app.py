
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Iris Classifier", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("iris_model.pkl")

model = load_model()

st.title("Iris Flower Classification")
st.write("Enter the flower measurements and predict the Iris species.")

sepal_length = st.number_input("Sepal Length (cm)", 0.0, 10.0, 5.1)
sepal_width = st.number_input("Sepal Width (cm)", 0.0, 10.0, 3.5)
petal_length = st.number_input("Petal Length (cm)", 0.0, 10.0, 1.4)
petal_width = st.number_input("Petal Width (cm)", 0.0, 10.0, 0.2)

if st.button("Predict"):
    X = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    pred = model.predict(X)[0]
    st.success(f"Predicted Species: {pred}")
