import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def load_data(num_classes=10, val_size=0.2, random_state=42):
    """Load MNIST dataset, create validation set, use one-hot encodeing on labels"""
    print("Loading MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    print("MNIST loaded successfully")

    # Create validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state
    )

    # One-hot encode labels
    y_train = to_categorical(y_train, num_classes).astype("float32")
    y_test = to_categorical(y_test, num_classes).astype("float32")
    y_val = to_categorical(y_val, num_classes).astype("float32")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def preprocess_data_CNN(X_train, y_train, X_val, y_val, X_test, y_test):
    """Preprocess data for CNN"""

    # Normalize pixel values
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    X_val = X_val.astype("float32") / 255.0

    # Reshape for CNN input
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    X_val = X_val.reshape(-1, 28, 28, 1)

    print(f"CNN preprocessing done: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def preprocess_data_dense(X_train, y_train, X_val, y_val, X_test, y_test):
    # Normalize pixel values
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    X_val = X_val.astype("float32") / 255.0

    # Reshape for CNN input
    X_train = X_train.reshape(-1, 28*28)
    X_test = X_test.reshape(-1, 28*28)
    X_val = X_val.reshape(-1, 28*28)

    print(f"Dense preprocessing done: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
