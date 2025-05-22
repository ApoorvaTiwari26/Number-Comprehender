import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from PIL import Image
import cv2

# Load and preprocess the MNIST data (for model training and structure)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

# Double-digit labels: Combine two digits into a single label
y_train_double = y_train // 10 * 10 + y_train % 10
y_test_double = y_test // 10 * 10 + y_test % 10
y_train_double = to_categorical(y_train_double, num_classes=100) # One-hot encode labels
y_test_double = to_categorical(y_test_double, num_classes=100)

# Build the CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax')) # Output layer with 100 units for double digits

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train_double, batch_size=128, epochs=10, validation_split=0.1)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test_double)
print("Test Accuracy:", accuracy)

# Function to preprocess a single scanned image
def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = img.resize((28, 28))  # Resize to 28x28
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)  # Reshape for model input
        return img_array
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Function to predict the double digit from a preprocessed image
def predict_digit(model, preprocessed_image):
    if preprocessed_image is not None:
        predictions = model.predict(preprocessed_image)
        predicted_class = np.argmax(predictions)
        digit1 = predicted_class // 10
        digit2 = predicted_class % 10
        return f"{digit1}{digit2}"
    else:
        return None

if __name__ == "__main__":
    image_path = input("Enter the path to your scanned double-digit image: ")
    preprocessed_image = preprocess_image(image_path)

    if preprocessed_image is not None:
        predicted_double_digit = predict_digit(model, preprocessed_image)
        if predicted_double_digit is not None:
            print(f"The predicted double digit is: {predicted_double_digit}")