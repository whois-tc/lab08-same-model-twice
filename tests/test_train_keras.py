"""Tests for Keras training and shared prepared data."""

import keras
import numpy as np
import pandas as pd
import pytest

from keras_model import build_keras_model
from prepare import SEED
from train_keras import train_keras_model


@pytest.fixture(scope="module")
def repeated_short_runs(prepared_data, tmp_path_factory):
    """Run the same short training experiment twice."""

    temporary_directory = tmp_path_factory.mktemp("keras_runs")

    settings = {
        "prepared_data": prepared_data,
        "hidden_sizes": (16, 8),
        "learning_rate": 0.003,
        "batch_size": 512,
        "epochs": 3,
        "seed": SEED,
        "verbose": 0,
    }

    first_result = train_keras_model(
        checkpoint_path=temporary_directory / "first.keras",
        **settings,
    )
    second_result = train_keras_model(
        checkpoint_path=temporary_directory / "second.keras",
        **settings,
    )

    return first_result, second_result


def test_keras_and_pytorch_receive_same_prepared_arrays(prepared_data):
    for split_name in ["train", "validation", "test"]:
        numpy_features = prepared_data[f"{split_name}_features"]
        pytorch_features = prepared_data[f"{split_name}_dataset"].features.numpy()

        numpy_targets = prepared_data[f"{split_name}_targets"]
        pytorch_targets = prepared_data[f"{split_name}_dataset"].targets.numpy()

        np.testing.assert_array_equal(
            numpy_features,
            pytorch_features,
        )
        np.testing.assert_array_equal(
            numpy_targets,
            pytorch_targets,
        )


def test_keras_model_can_memorize_small_batch(prepared_data):
    keras.backend.clear_session()
    keras.utils.set_random_seed(SEED)

    features = prepared_data["train_features"][:16]
    targets = prepared_data["train_targets"][:16]

    model = build_keras_model(
        input_size=features.shape[1],
        hidden_sizes=(16, 8),
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss="mse",
    )

    initial_loss = model.evaluate(
        features,
        targets,
        verbose=0,
    )

    model.fit(
        features,
        targets,
        epochs=300,
        batch_size=16,
        shuffle=False,
        verbose=0,
    )

    final_loss = model.evaluate(
        features,
        targets,
        verbose=0,
    )

    assert final_loss < initial_loss * 0.05


def test_saved_keras_model_repeats_predictions(tmp_path):
    keras.backend.clear_session()
    keras.utils.set_random_seed(SEED)

    model = build_keras_model(
        input_size=8,
        hidden_sizes=(16, 8),
    )

    features = (
        np.random.default_rng(SEED)
        .normal(
            size=(12, 8),
        )
        .astype(np.float32)
    )

    original_predictions = model(
        features,
        training=False,
    ).numpy()

    model_path = tmp_path / "reloaded_model.keras"
    model.save(model_path)

    reloaded_model = keras.models.load_model(model_path)
    reloaded_predictions = reloaded_model(
        features,
        training=False,
    ).numpy()

    np.testing.assert_allclose(
        original_predictions,
        reloaded_predictions,
        rtol=1e-6,
        atol=1e-6,
    )


def test_best_validation_weights_are_restored(
    prepared_data,
    repeated_short_runs,
):
    first_result, _ = repeated_short_runs

    restored_metrics = first_result["model"].evaluate(
        prepared_data["validation_features"],
        prepared_data["validation_targets"],
        batch_size=512,
        verbose=0,
        return_dict=True,
    )

    best_recorded_loss = first_result["history"]["validation_loss"].min()

    assert restored_metrics["loss"] == pytest.approx(
        best_recorded_loss,
        rel=1e-5,
    )


def test_same_seed_repeats_short_training(repeated_short_runs):
    first_result, second_result = repeated_short_runs

    pd.testing.assert_frame_equal(
        first_result["history"],
        second_result["history"],
        check_exact=False,
        rtol=1e-5,
        atol=1e-5,
    )
