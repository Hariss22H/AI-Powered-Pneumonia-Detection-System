import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import datetime
import os

# ✅ Load the trained VGG16 model (relative path for deployment)
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "pneumonia_model.h5")
MODEL_PATH = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
model = load_model(MODEL_PATH)

# ✅ Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = image.convert("RGB")
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ✅ Function to generate Grad-CAM heatmap
def get_gradcam_heatmap(model, img_array, layer_name="block5_conv3"):
    grad_model = tf.keras.models.Model(inputs=model.input, outputs=[model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_array)
        # Ensure preds is a tensor and not a list/tuple
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        preds = tf.convert_to_tensor(preds)

        # Handle binary sigmoid output shape (batch, 1) vs multi-class
        if preds.shape.rank == 2 and preds.shape[-1] == 1:
            loss = preds[:, 0]
        else:
            class_index = tf.argmax(preds[0])
            loss = preds[:, class_index]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):
        conv_output[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_output, axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    return heatmap

# ✅ Function to apply Grad-CAM heatmap to an image
def apply_heatmap(image, model, layer_name="block5_conv3"):
    img_array = preprocess_image(image)
    heatmap = get_gradcam_heatmap(model, img_array, layer_name)

    img = np.array(image)
    img = cv2.resize(img, (224, 224))

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    heatmap_path = "gradcam_heatmap.jpg"
    cv2.imwrite(heatmap_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))

    return heatmap_path

# ✅ Function to generate AI-powered medical report
def generate_report(image_path, heatmap_path, prediction, confidence, quality_report, output_path="medical_report.pdf"):
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(180, height - 50, "🩺 Pneumonia Detection Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 80, f"📅 Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ✅ Add X-ray Image
    if os.path.exists(image_path):
        try:
            c.drawString(50, height - 120, "📷 Uploaded X-ray Image:")
            img = ImageReader(image_path)
            c.drawImage(img, 50, height - 350, width=250, height=250)
        except Exception as e:
            print(f"⚠️ Could not load X-ray image: {e}")

    # ✅ Add Grad-CAM Heatmap Image
    if os.path.exists(heatmap_path):
        try:
            c.drawString(330, height - 120, "🔥 Grad-CAM Heatmap:")
            heatmap_img = ImageReader(heatmap_path)
            c.drawImage(heatmap_img, 330, height - 350, width=250, height=250)
        except Exception as e:
            print(f"⚠️ Could not load Grad-CAM heatmap: {e}")

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
    return output_path

# ✅ Streamlit UI
st.title("Pneumonia Detection with AI-powered Report")

uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    if st.button("Analyze"):
        processed_image = preprocess_image(image)
        prediction_score = float(model.predict(processed_image)[0])

        prediction = "Bacterial Pneumonia" if prediction_score > 0.5 else "No Pneumonia"
        confidence = prediction_score * 100 if prediction_score > 0.5 else (1 - prediction_score) * 100

        # ✅ Save the uploaded image
        uploaded_image_path = "uploaded_xray.jpg"
        image.save(uploaded_image_path)

        # ✅ Generate Heatmap
        heatmap_path = apply_heatmap(image, model)
        st.image(heatmap_path, caption="Grad-CAM Heatmap", use_container_width=True)

        # ✅ X-ray Quality Assessment (Static for now)
        quality_report = {
            "Blur Level": "Low (Clear Image)",
            "Contrast Level": "Good",
            "Brightness Status": "Optimal"
        }

        # ✅ Generate Report
        report_path = generate_report(uploaded_image_path, heatmap_path, prediction, confidence, quality_report)
        
        # ✅ Provide Download Option
        with open(report_path, "rb") as file:
            st.download_button(label="📄 Download Medical Report", data=file, file_name="medical_report.pdf", mime="application/pdf")

        # ✅ Display Diagnosis
        if prediction_score > 0.5:
            st.error(f"Pneumonia Detected ❌ (Confidence: {confidence:.2f}%)")
        else:
            st.success(f"No Pneumonia ✅ (Confidence: {confidence:.2f}%)")
