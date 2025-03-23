# AI-Powered Pneumonia Detection using Deep Learning (VGG16)
 

## 📌 Project Overview
This project implements a **deep learning-based pneumonia detection system** using **VGG16**. It classifies chest X-ray images into **normal** or **pneumonia** cases with an accuracy of **92%**. Additionally, it incorporates **Grad-CAM heatmaps** for explainability and aims to introduce **AI-powered automated report generation**.

## 🚀 Features
- **Pneumonia Classification:** Identifies pneumonia from chest X-rays.
- **Explainability with Grad-CAM:** Highlights affected lung regions.
- **X-ray Quality Check:** Ensures reliable input data.
- **AI-Powered Report Generation:** (Upcoming) Automatically generates diagnostic reports.
- **Severity Prediction:** (Planned) Classifies pneumonia into mild, moderate, or severe.
- **Streamlit Deployment:** A doctor-friendly UI for easy access.

## 📂 Dataset
The dataset used is the **Chest X-Ray Images (Pneumonia) dataset** from **Kaggle**:  
[🔗 Download Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

It consists of:  
- **Normal X-rays** 🫁 
- **Pneumonia-infected X-rays** 🦠 (Viral & Bacterial)

## 🏗 Model Architecture
The model is built using **VGG16**, a pre-trained CNN, and fine-tuned for pneumonia classification.

- **Input:** Chest X-ray images (224x224)
- **Backbone:** VGG16 (pre-trained on ImageNet)
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Metrics:** Accuracy, Precision, Recall

## 📊 Performance
- **Accuracy:** 92%
- **Precision:** High
- **Recall:** High (minimizing false negatives)

## 🛠 Installation & Setup
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/pneumonia-detection.git
cd pneumonia-detection
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Run the Model
```bash
python main.py
```
### 4️⃣ Deploy with Streamlit
```bash
streamlit run app.py
```

## 📌 Deployment
The model is deployed using **Streamlit**, making it accessible via a web interface.

## 🛠 Future Work
- **Severity classification (Mild, Moderate, Severe)**
- **Integrate AI-powered report generation**
- **Optimize for real-time performance using TensorFlow Lite**
- **Expand dataset for better generalization**

## 📝 Contributors
- **SHAIK HARRISS RAZVI** - Deep Learning & Deployment

## 💡 Acknowledgments
- Dataset by **Paul Mooney (Kaggle)**
- VGG16 from **Keras Applications**
- Grad-CAM implementation based on research papers


---
Feel free to ⭐ **star this repo** if you find it helpful!
