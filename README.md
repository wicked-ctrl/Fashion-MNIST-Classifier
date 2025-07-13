# 🧠 Fashion MNIST Classifier with Dropout (Keras)

This project builds and evaluates a neural network to classify images of clothing using the Fashion MNIST dataset. It uses a simple fully connected architecture with ReLU activations and Dropout layers to prevent overfitting.

---

## 📂 Dataset

- **Fashion MNIST** (built-in from `tensorflow.keras.datasets`)
- 60,000 training samples, 10,000 test samples
- 10 classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

---

## 🏗 Model Architecture

```python
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='linear')  # Softmax applied via loss function
])
