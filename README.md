# 🧠 Double-Digit Classifier using CNN

A Convolutional Neural Network (CNN) built with Keras and TensorFlow to classify **double-digit numbers (00–99)** based on scanned single-digit images. The model is trained on the MNIST dataset, combining digits to simulate a 100-class classification task.

---
double-digit-classifier/
├── double_digit_classifier.ipynb       # Main notebook or script
├── README.md                           # Project documentation
├── example_digits/                     # Folder for sample images
│   └── digit_sample.png                # Example digit image
└── LICENSE                             # MIT License file

---

## 🚀 Working

- Trains a CNN model on the MNIST dataset  
- Transforms single digits into double-digit labels (00 to 99)  
- Classifies grayscale input images into double-digit numbers  
- Includes preprocessing pipeline for scanned images  
- Trained with ~60,000 samples and validated on test data  

---

## 🛠 Tech Stack

- Python  
- TensorFlow / Keras  
- NumPy  
- Pillow (PIL)  
- OpenCV (cv2)  
- Google Colab (for development)

---


## 📸 How to Use

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ApoorvaTiwari26/double-digit-classifier.git
   cd double-digit-classifier
