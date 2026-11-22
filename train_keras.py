"""Train the Keras bike-demand network."""

from pathlib import Path
from time import perf_counter

import keras
import matplotlib.pyplot as plt
import pandas as pd

from bikes_data import make_rentals
from keras_model import build_keras_model
from prepare import SEED, prepare_data

REPORTS_DIRECTORY = Path("reports")
MODELS_DIRECTORY = Path("models")

HIDDEN_SIZES = (64, 32)
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 80


def make_callbacks(checkpoint_path, epochs):
    """Create callbacks that preserve the best validation model."""

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=epochs,
        restore_best_weights=True,
    )

    model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    )

    return [early_stopping, model_checkpoint]


def train_keras_model(
    prepared_data,
    hidden_sizes=HIDDEN_SIZES,
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    seed=SEED,
    checkpoint_path=MODELS_DIRECTORY / "best_keras_model.keras",
    verbose=2,
):
    """Train Keras using the arrays prepared for the PyTorch version."""

    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)

    model = build_keras_model(
        input_size=prepared_data["train_features"].shape[1],
        hidden_sizes=hidden_sizes,
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate,
        ),
        loss="mse",
        metrics=[
            keras.metrics.MeanAbsoluteError(name="mae"),
        ],
    )

    callbacks = make_callbacks(
        checkpoint_path=checkpoint_path,
        epochs=epochs,
    )

    start_time = perf_counter()

    keras_history = model.fit(
        prepared_data["train_features"],
        prepared_data["train_targets"],
        validation_data=(
            prepared_data["validation_features"],
            prepared_data["validation_targets"],
        ),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=verbose,
    )

    seconds = perf_counter() - start_time

    history = pd.DataFrame(
        {
            "epoch": range(1, len(keras_history.history["loss"]) + 1),
            "training_loss": keras_history.history["loss"],
            "training_mae": keras_history.history["mae"],
            "validation_loss": keras_history.history["val_loss"],
            "validation_mae": keras_history.history["val_mae"],
        }
    )

    best_index = history["validation_loss"].idxmin()
    best_epoch = int(history.loc[best_index, "epoch"])

    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "seconds": seconds,
    }


def plot_learning_curve(history, best_epoch, output_path):
    """Save the Keras training and validation loss curves."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(
        history["epoch"],
        history["training_loss"],
        label="Training loss",
    )
    plt.plot(
        history["epoch"],
        history["validation_loss"],
        label="Validation loss",
    )
    plt.axvline(
        best_epoch,
        color="black",
        linestyle="--",
        label=f"Best epoch: {best_epoch}",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Mean squared error")
    plt.title("Keras learning curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    """Train the main Keras configuration and recreate its reports."""

    REPORTS_DIRECTORY.mkdir(exist_ok=True)

    frame = make_rentals()
    prepared_data = prepare_data(frame, seed=SEED)

    result = train_keras_model(
        prepared_data=prepared_data,
    )

    history_path = REPORTS_DIRECTORY / "history_keras.csv"
    curve_path = REPORTS_DIRECTORY / "learning_curves_keras.png"

    result["history"].to_csv(
        history_path,
        index=False,
    )

    plot_learning_curve(
        history=result["history"],
        best_epoch=result["best_epoch"],
        output_path=curve_path,
    )

    best_row = (
        result["history"]
        .loc[result["history"]["epoch"] == result["best_epoch"]]
        .iloc[0]
    )

    print("\nKeras training complete:")
    print(f"Best epoch: {result['best_epoch']}")
    print(f"Best validation loss: {best_row['validation_loss']:.4f}")
    print(f"Best validation MAE: {best_row['validation_mae']:.4f}")
    print(f"Training seconds: {result['seconds']:.2f}")
    print(f"History saved to: {history_path}")
    print(f"Learning curve saved to: {curve_path}")


if __name__ == "__main__":
    main()
