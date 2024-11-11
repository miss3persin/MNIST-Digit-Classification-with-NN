
# MNIST Digit Classification using Neural Network

This project is a basic implementation of a Neural Network (NN) for classifying handwritten digits from the popular MNIST dataset. The goal is to predict the digit (0-9) present in an image of a handwritten digit.

## Dataset
The dataset used is the [MNIST Handwritten Digits Dataset](http://yann.lecun.com/exdb/mnist/), which contains 60,000 training images and 10,000 testing images of handwritten digits, each sized 28x28 pixels.

## Model Architecture
The Neural Network was built using TensorFlow's Keras API with the following architecture:

```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), # Input Layer (Flatten 28x28 images to a 1D vector)
    keras.layers.Dense(50, activation='relu'), # Hidden Layer 1 with 50 neurons
    keras.layers.Dense(50, activation='relu'), # Hidden Layer 2 with 50 neurons
    keras.layers.Dense(10, activation='sigmoid') # Output Layer with 10 neurons (for each digit 0-9)
])
```

### Model Compilation
The model was compiled with:
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy (used for multi-class classification)
- **Metrics**: Accuracy

## Model Performance

- **Training Accuracy**: 98.99%
- **Training Loss**: 0.0337
- **Test Accuracy**: 97.01%
- **Test Loss**: 0.01188

## How to Use

This model can be used to classify any new handwritten digit image. Follow the usage example below:

### Example Usage

```python
input_img_path = input('Enter the image path: ')

# Reading the image using OpenCV
input_img = cv2.imread(input_img_path)

# Displaying the image
cv2_imshow(input_img)

# Converting to grayscale
grayscale = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

# Resizing to 28x28 pixels
grayscale_resized = cv2.resize(grayscale, (28,28))

# Standardizing the pixel values
grayscale_std = grayscale_resized / 255

# Reshaping to match model input shape
image_reshaped = grayscale_std.reshape(1,28,28)

# Making the prediction
predicted_probs = model.predict(image_reshaped)
predicted_label = np.argmax(predicted_probs)

print(f'The handwritten digit is recognized as: {predicted_label}')
```

### Notes:
- Make sure to install the required libraries before running the code:
  ```bash
  pip install tensorflow opencv-python-headless numpy
  ```
- The input image should be clear and preferably centered for best results.

## Conclusion
This project demonstrates how to build a simple yet effective Neural Network for handwritten digit classification. With a training accuracy of 98.99% and a test accuracy of 97.01%, the model performs well in recognizing handwritten digits from the MNIST dataset.
