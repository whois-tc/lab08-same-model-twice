"""Train, compare and report bike-demand models."""

import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from torch import nn

from bikes_data import make_rentals
from model import BikeDemandNetwork
from prepare import SEED, make_loaders, prepare_data

REPORTS_DIRECTORY = Path("reports")


def run_epoch(model, loader, loss_function, optimizer=None):
    """Train for one epoch, or measure when no optimizer is supplied."""

    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_absolute_error = 0.0
    total_examples = 0

    for features, targets in loader:
        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            predictions = model(features)
            loss = loss_function(predictions, targets)

        if is_training:
            loss.backward()
            optimizer.step()

        batch_size = len(targets)
        total_loss += loss.item() * batch_size
        total_absolute_error += torch.abs(predictions - targets).sum().item()
        total_examples += batch_size

    mean_loss = total_loss / total_examples
    mean_absolute_error = total_absolute_error / total_examples

    return mean_loss, mean_absolute_error


def train_model(
    model,
    training_loader,
    validation_loader,
    epochs=80,
    learning_rate=0.001,
):
    """Train for several epochs and restore the best validation weights."""

    if epochs < 1:
        raise ValueError("Training requires at least one epoch.")

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    history = []
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        training_loss, training_mae = run_epoch(
            model,
            training_loader,
            loss_function,
            optimizer,
        )

        validation_loss, validation_mae = run_epoch(
            model,
            validation_loader,
            loss_function,
        )

        history.append(
            {
                "epoch": epoch,
                "training_loss": training_loss,
                "training_mae": training_mae,
                "validation_loss": validation_loss,
                "validation_mae": validation_mae,
            }
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch:03d} | "
            f"training loss: {training_loss:.2f} | "
            f"validation loss: {validation_loss:.2f} | "
            f"validation MAE: {validation_mae:.2f}"
        )

    model.load_state_dict(best_state)

    return history, best_epoch


def run_experiment(
    name,
    prepared_data,
    hidden_sizes,
    learning_rate,
    batch_size,
    epochs,
):
    """Train one network configuration and summarize its results."""

    torch.manual_seed(SEED)

    training_loader, validation_loader, _ = make_loaders(
        prepared_data,
        batch_size=batch_size,
        seed=SEED,
    )

    model = BikeDemandNetwork(
        input_size=prepared_data["train_features"].shape[1],
        hidden_sizes=hidden_sizes,
    )

    history, best_epoch = train_model(
        model,
        training_loader,
        validation_loader,
        epochs=epochs,
        learning_rate=learning_rate,
    )

    best_row = history[best_epoch - 1]
    final_row = history[-1]

    summary = {
        "run": name,
        "hidden_sizes": list(hidden_sizes),
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_validation_loss": best_row["validation_loss"],
        "best_validation_mae": best_row["validation_mae"],
        "final_training_loss": final_row["training_loss"],
        "final_validation_loss": final_row["validation_loss"],
    }

    return {
        "model": model,
        "history": history,
        "summary": summary,
    }


def calculate_baselines(prepared_data):
    """Calculate the training-mean and linear-regression baselines."""

    training_targets = prepared_data["train_targets"]
    validation_targets = prepared_data["validation_targets"]
    test_targets = prepared_data["test_targets"]

    training_mean = float(np.mean(training_targets))

    validation_mean_predictions = np.full(
        len(validation_targets),
        training_mean,
    )
    test_mean_predictions = np.full(
        len(test_targets),
        training_mean,
    )

    linear_model = LinearRegression()
    linear_model.fit(
        prepared_data["train_features"],
        training_targets,
    )

    validation_linear_predictions = linear_model.predict(
        prepared_data["validation_features"]
    )
    test_linear_predictions = linear_model.predict(prepared_data["test_features"])

    return {
        "training_mean": {
            "validation_mae": mean_absolute_error(
                validation_targets,
                validation_mean_predictions,
            ),
            "test_mae": mean_absolute_error(
                test_targets,
                test_mean_predictions,
            ),
        },
        "linear_regression": {
            "validation_mae": mean_absolute_error(
                validation_targets,
                validation_linear_predictions,
            ),
            "test_mae": mean_absolute_error(
                test_targets,
                test_linear_predictions,
            ),
        },
    }


def evaluate_network(model, prepared_data, batch_size):
    """Evaluate the selected network on validation and test rows."""

    _, validation_loader, test_loader = make_loaders(
        prepared_data,
        batch_size=batch_size,
        seed=SEED,
    )

    loss_function = nn.MSELoss()

    validation_loss, validation_mae = run_epoch(
        model,
        validation_loader,
        loss_function,
    )

    test_loss, test_mae = run_epoch(
        model,
        test_loader,
        loss_function,
    )

    return {
        "validation_loss": validation_loss,
        "validation_mae": validation_mae,
        "test_loss": test_loss,
        "test_mae": test_mae,
    }


def plot_learning_curves(history, best_epoch, title, output_path):
    """Save training and validation loss curves."""

    epochs = [row["epoch"] for row in history]
    training_losses = [row["training_loss"] for row in history]
    validation_losses = [row["validation_loss"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, training_losses, label="Training loss")
    plt.plot(epochs, validation_losses, label="Validation loss")
    plt.axvline(
        best_epoch,
        color="black",
        linestyle="--",
        label=f"Best epoch: {best_epoch}",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Mean squared error")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_rentals_by_hour(frame, output_path):
    """Plot hourly rentals to inspect whether the pattern is linear."""

    hourly_average = frame.groupby("hour", as_index=False)["rentals"].mean()

    plt.figure(figsize=(8, 5))
    plt.scatter(
        frame["hour"],
        frame["rentals"],
        alpha=0.05,
        s=10,
        label="Station-hour rows",
    )
    plt.plot(
        hourly_average["hour"],
        hourly_average["rentals"],
        color="black",
        linewidth=2,
        label="Average by hour",
    )
    plt.xlabel("Hour")
    plt.ylabel("Bike rentals")
    plt.title("Bike rentals by hour")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    """Run every experiment and recreate all report files."""

    REPORTS_DIRECTORY.mkdir(exist_ok=True)

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    frame = make_rentals()
    prepared_data = prepare_data(frame, seed=SEED)

    configurations = [
        {
            "name": "main",
            "hidden_sizes": (64, 32),
            "learning_rate": 0.001,
            "batch_size": 64,
            "epochs": 80,
        },
        {
            "name": "slow_learning",
            "hidden_sizes": (64, 32),
            "learning_rate": 0.00005,
            "batch_size": 64,
            "epochs": 80,
        },
        {
            "name": "fast_learning",
            "hidden_sizes": (64, 32),
            "learning_rate": 0.01,
            "batch_size": 64,
            "epochs": 80,
        },
    ]

    experiment_results = []

    for configuration in configurations:
        print(f"\nRunning: {configuration['name']}")

        result = run_experiment(
            prepared_data=prepared_data,
            **configuration,
        )
        experiment_results.append(result)

        if configuration["name"] == "main":
            curve_path = REPORTS_DIRECTORY / "learning_curves.png"
        else:
            curve_path = (
                REPORTS_DIRECTORY / f"learning_curves_{configuration['name']}.png"
            )

        plot_learning_curves(
            result["history"],
            result["summary"]["best_epoch"],
            f"Learning curves: {configuration['name']}",
            curve_path,
        )

    selected_result = min(
        experiment_results,
        key=lambda result: result["summary"]["best_validation_mae"],
    )

    selected_model = selected_result["model"]
    selected_summary = selected_result["summary"]

    baseline_metrics = calculate_baselines(prepared_data)
    network_metrics = evaluate_network(
        selected_model,
        prepared_data,
        batch_size=selected_summary["batch_size"],
    )

    metrics = {
        "selection_rule": "Lowest validation MAE",
        "selected_run": selected_summary["run"],
        "selected_settings": {
            "hidden_sizes": selected_summary["hidden_sizes"],
            "learning_rate": selected_summary["learning_rate"],
            "batch_size": selected_summary["batch_size"],
            "epochs": selected_summary["epochs"],
            "best_epoch": selected_summary["best_epoch"],
        },
        "results": {
            "Training mean baseline": {
                "validation_mae": round(
                    float(baseline_metrics["training_mean"]["validation_mae"]),
                    4,
                ),
                "test_mae": round(
                    float(baseline_metrics["training_mean"]["test_mae"]),
                    4,
                ),
            },
            "Linear regression": {
                "validation_mae": round(
                    float(baseline_metrics["linear_regression"]["validation_mae"]),
                    4,
                ),
                "test_mae": round(
                    float(baseline_metrics["linear_regression"]["test_mae"]),
                    4,
                ),
            },
            "Neural network": {
                "validation_mae": round(
                    float(network_metrics["validation_mae"]),
                    4,
                ),
                "test_mae": round(
                    float(network_metrics["test_mae"]),
                    4,
                ),
            },
        },
    }

    main_result = next(
        result for result in experiment_results if result["summary"]["run"] == "main"
    )

    pd.DataFrame(main_result["history"]).to_csv(
        REPORTS_DIRECTORY / "history.csv",
        index=False,
    )

    pd.DataFrame([result["summary"] for result in experiment_results]).to_csv(
        REPORTS_DIRECTORY / "experiments.csv",
        index=False,
    )

    (REPORTS_DIRECTORY / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    plot_rentals_by_hour(
        frame,
        REPORTS_DIRECTORY / "rentals_by_hour.png",
    )

    print("\nFinal model comparison:")
    print(json.dumps(metrics, indent=2))
    print(f"\nReports saved to: {REPORTS_DIRECTORY}")


if __name__ == "__main__":
    main()
