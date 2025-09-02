import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

def load_and_preprocess_data(num_classes=10):
    """Load and preprocess MNIST dataset."""
    print("Loading MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    print("MNIST loaded successfully")


    # Normalize pixel values
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # Reshape for CNN input
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    # One-hot encode labels
    y_train = to_categorical(y_train, num_classes).astype("float32")
    y_test = to_categorical(y_test, num_classes).astype("float32")

    return (X_train, y_train), (X_test, y_test)
