# 🪖 Helmet vs No-Helmet Classifier (ResNet18)

This project implements a **Convolutional Neural Network (ResNet18)** to classify whether a person is **wearing a safety helmet** or **not wearing a helmet**.  
It is part of my **AI Engineering portfolio project** to showcase end-to-end computer vision skills: model training, conversion, and deployment.  

---

## 📊 Model Details
- **Architecture:** ResNet18 (transfer learning)  
- **Framework:** PyTorch & TorchVision  
- **Export:** ONNX and Safetensors formats  
- **Deployment:** Hugging Face Spaces (Gradio demo)  

---

## 📈 Results
- Dataset split: **70% train / 15% validation / 15% test** (49 images total)  
- **Best validation accuracy:** 100%  
- **Test accuracy:** 90%  

Confusion Matrix (test set):  
![Confusion Matrix](https://huggingface.co/TristanF04/helmet-detector/resolve/main/confusion_matrix.png)

---

## 🚀 Quickstart (Local)

### 1. Setup environment
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
