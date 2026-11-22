"""Compare the PyTorch and Keras versions over several seeds."""

from pathlib import Path
from time import perf_counter

import pandas as pd
import torch
from torch import nn

from bikes_data import make_rentals
from model import BikeDemandNetwork
from prepare import SEED, make_loaders, prepare_data
from train import run_epoch, train_model
from train_keras import (
    BATCH_SIZE,
    EPOCHS,
    HIDDEN_SIZES,
    LEARNING_RATE,
    train_keras_model,
)

REPORT_PATH = Path("reports/comparison.csv")
COMPARISON_SEEDS = [2026, 2027, 2028]


def run_pytorch_once(prepared_data, seed):
    """Train and score one PyTorch model."""

    torch.manual_seed(seed)

    training_loader, validation_loader, test_loader = make_loaders(
        prepared_data,
        batch_size=BATCH_SIZE,
        seed=seed,
    )

    model = BikeDemandNetwork(
        input_size=prepared_data["train_features"].shape[1],
        hidden_sizes=HIDDEN_SIZES,
    )

    start_time = perf_counter()

    _, best_epoch = train_model(
        model=model,
        training_loader=training_loader,
        validation_loader=validation_loader,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
    )

    seconds = perf_counter() - start_time

    loss_function = nn.MSELoss()

    _, validation_mae = run_epoch(
        model,
        validation_loader,
        loss_function,
    )
    _, test_mae = run_epoch(
        model,
        test_loader,
        loss_function,
    )

    return {
        "framework": "PyTorch",
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_mae": round(validation_mae, 4),
        "test_mae": round(test_mae, 4),
        "seconds": round(seconds, 2),
    }


def run_keras_once(prepared_data, seed):
    """Train and score one Keras model."""

    result = train_keras_model(
        prepared_data=prepared_data,
        hidden_sizes=HIDDEN_SIZES,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        seed=seed,
        checkpoint_path=Path(f"models/comparison_keras_seed_{seed}.keras"),
        verbose=0,
    )

    validation_metrics = result["model"].evaluate(
        prepared_data["validation_features"],
        prepared_data["validation_targets"],
        batch_size=BATCH_SIZE,
        verbose=0,
        return_dict=True,
    )

    test_metrics = result["model"].evaluate(
        prepared_data["test_features"],
        prepared_data["test_targets"],
        batch_size=BATCH_SIZE,
        verbose=0,
        return_dict=True,
    )

    return {
        "framework": "Keras",
        "seed": seed,
        "best_epoch": result["best_epoch"],
        "validation_mae": round(validation_metrics["mae"], 4),
        "test_mae": round(test_metrics["mae"], 4),
        "seconds": round(result["seconds"], 2),
    }


def run_comparison(
    prepared_data,
    seeds=COMPARISON_SEEDS,
    output_path=REPORT_PATH,
):
    """Run both frameworks for every seed and save the results."""

    rows = []

    for seed in seeds:
        print(f"\nRunning PyTorch with seed {seed}")
        rows.append(
            run_pytorch_once(
                prepared_data=prepared_data,
                seed=seed,
            )
        )

        print(f"\nRunning Keras with seed {seed}")
        rows.append(
            run_keras_once(
                prepared_data=prepared_data,
                seed=seed,
            )
        )

    comparison = pd.DataFrame(
        rows,
        columns=[
            "framework",
            "seed",
            "best_epoch",
            "validation_mae",
            "test_mae",
            "seconds",
        ],
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)

    return comparison


def main():
    """Prepare one dataset and run the complete comparison."""

    frame = make_rentals()
    prepared_data = prepare_data(frame, seed=SEED)

    comparison = run_comparison(
        prepared_data=prepared_data,
    )

    print("\nFramework comparison:")
    print(comparison.to_string(index=False))
    print(f"\nComparison saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
