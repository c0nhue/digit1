import matplotlib.pyplot as plt
import numpy as np

def plot_training_history(history, out_path="outputs/history.png"):
    """Plot accuracy and loss curves."""
    plt.figure(figsize=(12, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title("Model accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Model loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_sample_predictions(model, X_test, y_test, out_path="outputs/sample_predictions.png"):
    """Plot sample test predictions."""
    preds = model.predict(X_test[:9])
    preds_classes = np.argmax(preds, axis=1)
    true_classes = np.argmax(y_test[:9], axis=1)

    plt.figure(figsize=(6, 6))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(X_test[i].reshape(28, 28), cmap="gray")
        plt.title(f"Pred: {preds_classes[i]}, True: {true_classes[i]}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
