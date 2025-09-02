import os
from src.data_preprocessing import load_data, preprocess_data_CNN
from src.model import build_model_A, build_model_B, build_model_C
from src.train import plot_training_history, plot_sample_predictions

def main():
    os.makedirs("outputs", exist_ok=True)

    # Load data
    (X_train, y_train),(X_val, y_val), (X_test, y_test) = load_data()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocess_data_CNN(X_train, y_train, X_val, y_val, X_test, y_test)

    # Build model
    model = build_model_A()

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=5,
        batch_size=128,
        verbose=2
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    # Save plots
    plot_training_history(history)
    plot_sample_predictions(model, X_test, y_test)

    # Save model
    model.save("outputs/mnist_cnn.keras")

if __name__ == "__main__":
    main()
