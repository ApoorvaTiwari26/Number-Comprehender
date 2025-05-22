# 🧠 Double-Digit Classifier using CNN

A Convolutional Neural Network (CNN) built with Keras and TensorFlow to classify **double-digit numbers (00–99)** based on scanned single-digit images. The model is trained on the MNIST dataset, combining digits to simulate a 100-class classification task.

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


## 🧪 How to Use
1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/double-digit-classifier.git
   cd double-digit-classifier
   ```

2. **Install Dependencies**

   Make sure Python is installed, then install the required libraries:

   ```bash
   pip install tensorflow numpy pillow opencv-python
   ```

3. **Run the Classifier**

   ```bash
   python double_digit_classifier.py
   ```

4. **Predict from Image**

   When prompted, enter the path to a scanned digit image.

   The image should:
   - Be **grayscale**
   - Be sized **28×28 pixels**
   - Contain a **single digit (0–9)**

   The model will predict the corresponding **double-digit** class (00–99).

---

Built by Apoorva Tiwari (https://github.com/ApoorvaTiwari26)
