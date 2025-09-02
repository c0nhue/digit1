from tensorflow import keras
from tensorflow.keras import layers, models

def build_model_A(num_classes=10):
    """Define deeper CNN model for MNIST classification."""
    model_A = models.Sequential([
        keras.Input(shape=(28,28,1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])
    model_A.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model_A


def build_model_B(num_classes=10):
    """Define simpler CNN model for MNIST classification."""
    model_B = models.Sequential([
        keras.Input(shape=(28,28,1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])
    model_B.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model_B


def build_model_C(num_classes=10):
    """Define simple model for MNIST classification"""
    model_C = models.Sequential([
        keras.Input(shape=(28,28,1)),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])

    model_C.compile(optimizer="adam",
                    loss="categorical_crossentropy",
                    metrics=["accuracy"])
    return model_C