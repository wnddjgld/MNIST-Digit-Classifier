import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

SEED = int(os.environ.get("SEED", "42"))
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def build_model():
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax'),
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def plot_history(history, out_path="history.png"):
    hist = history.history
    epochs = range(1, len(hist['loss']) + 1)

    plt.figure(figsize=(10, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist['loss'], 'b-', label='Train Loss')
    plt.plot(epochs, hist['val_loss'], 'r-', label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training vs Validation Loss')

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, hist['accuracy'], 'b-', label='Train Acc')
    plt.plot(epochs, hist['val_accuracy'], 'r-', label='Val Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training vs Validation Accuracy')

    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Training history plot saved to {out_path}")
    plt.show()

def main():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    model = build_model()

    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "best.weights.h5")

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            mode='max',
            patience=3,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        x_train, y_train,
        epochs=20,
        batch_size=128,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=2
    )

    plot_history(history, "history.png")

    last_loss, last_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"[Last epoch model] Test accuracy: {last_acc:.4f}, loss: {last_loss:.4f}")

    model.load_weights(ckpt_path)
    best_loss, best_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"[Best checkpoint] Test accuracy: {best_acc:.4f}, loss: {best_loss:.4f}")

    os.makedirs("model", exist_ok=True)
    model.save("model.weights.h5")  # SavedModel 형식으로 ./model 에 저장
    print("Best model (weights) saved to ./model (SavedModel format)")

if __name__ == "__main__":
    main()
