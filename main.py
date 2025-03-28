# -*- coding: utf-8 -*-
"""AICTE.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1OlfMK72iRPrTaZzfmlDeNphaOFBrCzQN
"""

!pip install tensorflow keras numpy pandas matplotlib opencv-python flask flask-cors scikit-learn pillow

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

DATASET_PATH = "/content/drive/MyDrive/chest_xray"  # Update 'MyDrive' with your Drive folder

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import os

# Set dataset path
dataset_path = r"/content/drive/MyDrive/chest_xray"

# Image parameters
img_size = (224, 224)
batch_size = 32

# ✅ Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# ✅ Load Data
train_generator = train_datagen.flow_from_directory(
    os.path.join(dataset_path, 'train'),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

val_generator = test_datagen.flow_from_directory(
    os.path.join(dataset_path, 'val'),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(dataset_path, 'test'),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# ✅ Load Pretrained VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base model layers (except last few)
for layer in base_model.layers[:-4]:
    layer.trainable = False

# ✅ Add Custom Layers
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(1, activation='sigmoid')(x)  # Binary Classification

# ✅ Create Model
model = Model(inputs=base_model.input, outputs=output_layer)

# ✅ Compile Model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# ✅ Train Model
epochs = 30
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs
)

# ✅ Save the Trained Model
model.save("pneumonia_model.h5")
print("✅ Model saved as pneumonia_model.h5")

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ✅ Load the trained model
model = tf.keras.models.load_model("pneumonia_model.h5")

# ✅ Load test data (same as used during training)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    r"/content/drive/MyDrive/chest_xray/test",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# ✅ Evaluate the model
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"✅ Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"✅ Test Loss: {test_loss:.4f}")

# ✅ Get Predictions
y_true = test_generator.classes
y_pred_probs = model.predict(test_generator)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

# ✅ Generate Classification Report
print("📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Normal', 'Bacterial']))

# ✅ Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# ✅ Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=['Normal', 'Bacterial'], yticklabels=['Normal', 'Bacterial'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt

def check_xray_quality(image_path):
    """
    Analyzes X-ray quality based on blurriness, noise, and brightness.
    Returns a quality report.
    """
    # Load image in grayscale
    img = cv2.imread(r"/content/drive/MyDrive/chest_xray/test/PNEUMONIA/person94_bacteria_456.jpeg", cv2.IMREAD_GRAYSCALE)

    # ✅ Blurriness Detection (Laplacian Variance Method)
    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
    blur_status = "Blurry" if laplacian_var < 100 else "Clear"

    # ✅ Noise Detection (Histogram Analysis)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    noise_level = "High Noise" if np.std(hist) > 50 else "Low Noise"

    # ✅ Brightness Check (Mean Pixel Intensity)
    brightness = np.mean(img)
    brightness_status = "Too Dark" if brightness < 50 else "Too Bright" if brightness > 200 else "Good"

    # ✅ Display Image & Histogram
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("X-ray Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.plot(hist)
    plt.title("Histogram Analysis")

    plt.show()

    # ✅ Return Quality Report
    quality_report = {
        "Blur Status": blur_status,
        "Noise Level": noise_level,
        "Brightness Status": brightness_status
    }
    return quality_report

# Example Usage
image_path = "sample_xray.jpg"  # Change to actual image path
quality_report = check_xray_quality(image_path)
print("📷 X-ray Quality Report:", quality_report)

import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model

def get_gradcam_heatmap(model, img_array, layer_name="block5_conv3"):
    """
    Generates Grad-CAM heatmap for a given image.
    """
    # ✅ Create model that maps input image to last convolutional layer & predictions
    grad_model = Model(inputs=model.input, outputs=[model.get_layer(layer_name).output, model.output])

    # ✅ Compute gradients
    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_array)
        class_index = tf.argmax(preds[0])
        loss = preds[:, class_index]

    # ✅ Compute gradients w.r.t feature maps
    grads = tape.gradient(loss, conv_output)

    # ✅ Compute pooled gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # ✅ Convert tensors to NumPy (to allow modification)
    conv_output = conv_output.numpy()[0]  # Convert to NumPy
    pooled_grads = pooled_grads.numpy()  # Convert to NumPy

    # ✅ Modify feature maps with gradient importance
    for i in range(pooled_grads.shape[-1]):
        conv_output[:, :, i] *= pooled_grads[i]

    # ✅ Compute heatmap
    heatmap = np.mean(conv_output, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap

# ✅ Use this function inside the `apply_heatmap()` function as before.



def apply_heatmap(image_path, model, layer_name="block5_conv3"):
    """
    Applies Grad-CAM to an X-ray image.
    """
    # ✅ Load & preprocess image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # ✅ Generate Grad-CAM heatmap
    heatmap = get_gradcam_heatmap(model, img_array, layer_name)

    # ✅ Load original image (for overlay)
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))

    # ✅ Convert heatmap to color
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # ✅ Overlay heatmap on image
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # ✅ Display images
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original X-ray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.title("Grad-CAM Heatmap")
    plt.axis("off")

    plt.show()

# ✅ Load trained model
model = tf.keras.models.load_model("pneumonia_model.h5")

# ✅ Test Grad-CAM on an X-ray
image_path = r"/content/drive/MyDrive/chest_xray/test/PNEUMONIA/person101_bacteria_484.jpeg"  # Change this to an actual X-ray image
apply_heatmap(image_path, model)

pip install reportlab

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import datetime

def generate_report(image_path, prediction, confidence, quality_report, output_path="medical_report.pdf"):
    """
    Generates an AI-powered medical report as a PDF using ReportLab.
    """

    # ✅ Create a new PDF
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter

    # ✅ Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(200, height - 50, "🩺 Pneumonia Detection Report")

    # ✅ Date & Time
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 80, f"📅 Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ✅ Add X-ray Image (if available)
    if image_path:
        try:
            img = ImageReader(image_path)
            c.drawImage(img, 50, height - 350, width=200, height=200)
            c.drawString(50, height - 360, "🖼️ X-ray Image:")
        except Exception as e:
            print(f"⚠️ Could not load image: {e}")

    # ✅ X-ray Quality Report
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 400, "📷 X-ray Quality Assessment:")
    c.setFont("Helvetica", 12)
    y_position = height - 420
    for key, value in quality_report.items():
        c.drawString(60, y_position, f"🔹 {key}: {value}")
        y_position -= 20

    # ✅ Diagnosis & Confidence Score
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_position - 20, "🔍 Diagnosis:")
    c.setFont("Helvetica", 12)
    c.drawString(60, y_position - 40, f"✔️ Prediction: {prediction}")
    c.drawString(60, y_position - 60, f"📊 Confidence Score: {confidence:.2f}%")

    # ✅ Severity Estimation
    severity = "Mild" if confidence < 70 else "Moderate" if confidence < 90 else "Severe"
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_position - 90, "🔥 Estimated Severity Level:")
    c.setFont("Helvetica", 12)
    c.drawString(60, y_position - 110, f"⚠️ Severity: {severity}")

    # ✅ Save PDF
    c.save()
    print(f"✅ Report saved as: {output_path}")

# ✅ Example usage
image_path = "/content/drive/MyDrive/chest_xray/test/PNEUMONIA/person94_bacteria_456.jpeg"  # Change as needed
prediction = "Bacterial Pneumonia"
confidence = 92.5
quality_report = {
    "Blur Level": "Low (Clear Image)",
    "Contrast Level": "Good",
    "Brightness Status": "Optimal"
}

generate_report(image_path, prediction, confidence, quality_report, "medical_report.pdf")

import os
os.listdir()

from google.colab import files
files.download("medical_report.pdf")

!pip install flask flask-ngrok tensorflow pillow opencv-python reportlab

# Commented out IPython magic to ensure Python compatibility.
# %%writefile app.py
# from flask import Flask, render_template, request, send_file
# import tensorflow as tf
# import numpy as np
# import cv2
# import os
# from werkzeug.utils import secure_filename
# from reportlab.lib.pagesizes import letter
# from reportlab.pdfgen import canvas
# from reportlab.lib.utils import ImageReader
# import datetime
# from flask_ngrok import run_with_ngrok
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# import numpy as np
# from PIL import Image
# 
# app = Flask(__name__)
# run_with_ngrok(app)  # Enable public URL via ngrok
# 
# UPLOAD_FOLDER = "static/uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# 
# # Load trained model
# model = tf.keras.models.load_model("pneumonia_model.h5")
# 
# # Preprocess X-ray image
# def preprocess_image(image_path):
#     img = Image.open(image_path)
# 
#     # Convert grayscale images to RGB (3 channels)
#     if img.mode != 'RGB':
#         img = img.convert('RGB')
# 
#     img = img.resize((224, 224))  # Resize for VGG16
#     img = img_to_array(img)
#     img = np.expand_dims(img, axis=0)  # Add batch dimension
#     img = img / 255.0  # Normalize
# 
#     return img
# 
# # Generate AI report (PDF)
# def generate_report(image_path, prediction, confidence, output_path="report.pdf"):
#     c = canvas.Canvas(output_path, pagesize=letter)
#     width, height = letter
#     c.setFont("Helvetica-Bold", 16)
#     c.drawString(200, height - 50, "🩺 Pneumonia Detection Report")
#     c.setFont("Helvetica", 12)
#     c.drawString(50, height - 80, f"📅 Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# 
#     if image_path:
#         img = ImageReader(image_path)
#         c.drawImage(img, 50, height - 350, width=200, height=200)
#         c.drawString(50, height - 360, "🖼️ X-ray Image:")
# 
#     c.setFont("Helvetica-Bold", 14)
#     c.drawString(50, height - 400, "🔍 Diagnosis:")
#     c.setFont("Helvetica", 12)
#     c.drawString(60, height - 420, f"✔️ Prediction: {prediction}")
#     c.drawString(60, height - 440, f"📊 Confidence Score: {confidence:.2f}%")
# 
#     severity = "Mild" if confidence < 70 else "Moderate" if confidence < 90 else "Severe"
#     c.setFont("Helvetica-Bold", 14)
#     c.drawString(50, height - 470, "🔥 Estimated Severity Level:")
#     c.setFont("Helvetica", 12)
#     c.drawString(60, height - 490, f"⚠️ Severity: {severity}")
# 
#     c.save()
#     return output_path
# 
# @app.route("/", methods=["GET", "POST"])
# def index():
#     if request.method == "POST":
#         file = request.files["file"]
#         if file:
#             filename = secure_filename(file.filename)
#             file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
#             file.save(file_path)
# 
#             img = preprocess_image(file_path)
#             prediction = model.predict(img)[0][0]
#             label = "Bacterial Pneumonia" if prediction > 0.5 else "Normal"
#             confidence = float(prediction) * 100 if prediction > 0.5 else (1 - float(prediction)) * 100
# 
#             report_path = generate_report(file_path, label, confidence)
# 
#             return f"""
#             <h2>📊 AI Analysis Results</h2>
#             <img src="{file_path}" width="300"><br>
#             <p><strong>Prediction:</strong> {label}</p>
#             <p><strong>Confidence:</strong> {confidence:.2f}%</p>
#             <a href='/download_report'>📄 Download Report</a>
#             <br><a href='/'>🔄 Analyze Another</a>
#             """
# 
#     return """
#     <h2>🩺 Pneumonia Detection AI</h2>
#     <form action="/" method="post" enctype="multipart/form-data">
#         <input type="file" name="file" required>
#         <button type="submit">🔍 Analyze</button>
#     </form>
#     """
# 
# @app.route("/download_report")
# def download_report():
#     return send_file("report.pdf", as_attachment=True)
# 
# if __name__ == "__main__":
#     app.run()
#

!pip install streamlit opencv-python-headless pyngrok

!pip install streamlit
!pip install pyngrok

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained VGG16 model
model = load_model("pneumonia_model.h5")  # Ensure this file exists in Colab

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize image to match model input
    image = np.array(image)
    if image.shape[-1] == 4:
        image = image[:, :, :3]  # Convert RGBA to RGB
    image = image / 255.0  # Normalize
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

from pyngrok import ngrok

# Start Streamlit in the background
!streamlit run app.py &

# Create ngrok tunnel
public_url = ngrok.connect(port="8501")
print("Access your Streamlit app here:", public_url)

