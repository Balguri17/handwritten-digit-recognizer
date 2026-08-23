# Handwritten Digit Recognizer 🔢

A CNN-based handwritten digit recognition web application built using TensorFlow, Keras, Python, and Streamlit.

## 📌 Project Overview

This project recognizes handwritten digits from 0 to 9 using a Convolutional Neural Network (CNN).

The model is trained on the MNIST handwritten digit dataset and integrated into a Streamlit web application where users can upload an image and receive a predicted digit with a confidence score.

## 🚀 Features

- Upload handwritten digit images
- Image preprocessing and normalization
- CNN-based digit classification
- Prediction of digits from 0–9
- Confidence score for predictions
- Interactive Streamlit web interface

## 🧠 Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pillow
- Streamlit

## 📊 Model Performance

The CNN achieved approximately **98% test accuracy** on the MNIST test dataset.

Example:

**Handwritten 7 → Predicted 7**  
**Confidence: 87.47%**

## 📁 Project Structure

```text
Handwritten_Digit_Recognizer/
│
├── app.py
├── train_model.py
├── digit_model.keras
├── requirements.txt
├── README.md
└── .gitignore