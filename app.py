import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained VGG16 model
model = load_model(r"C:\Users\Asus\OneDrive\Desktop\AICTE\pneumonia_model.h5")  # Ensure this file exists in Colab

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to model's input size
    image = image.convert("RGB")  # Ensure it's in RGB format
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


# Streamlit UI
st.title("Pneumonia Detection using VGG16")

uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    if st.button("Analyze"):
        processed_image = preprocess_image(image)
        prediction = float(model.predict(processed_image)[0])

        if prediction > 0.5:
            st.error("Pneumonia Detected ❌")
        else:
            st.success("No Pneumonia Detected ✅")

