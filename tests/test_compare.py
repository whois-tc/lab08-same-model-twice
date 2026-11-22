"""Tests for the framework comparison table."""

import pandas as pd

import compare


def test_comparison_contains_both_frameworks_and_all_seeds(
    monkeypatch,
    tmp_path,
):
    def fake_pytorch_run(prepared_data, seed):
        return {
            "framework": "PyTorch",
            "seed": seed,
            "best_epoch": 10,
            "validation_mae": 3.5,
            "test_mae": 3.6,
            "seconds": 1.0,
        }

    def fake_keras_run(prepared_data, seed):
        return {
            "framework": "Keras",
            "seed": seed,
            "best_epoch": 12,
            "validation_mae": 3.4,
            "test_mae": 3.5,
            "seconds": 1.1,
        }

    monkeypatch.setattr(
        compare,
        "run_pytorch_once",
        fake_pytorch_run,
    )
    monkeypatch.setattr(
        compare,
        "run_keras_once",
        fake_keras_run,
    )

    output_path = tmp_path / "comparison.csv"

    result = compare.run_comparison(
        prepared_data={},
        seeds=[11, 22, 33],
        output_path=output_path,
    )

    assert len(result) == 6
    assert set(result["framework"]) == {"PyTorch", "Keras"}
    assert set(result["seed"]) == {11, 22, 33}
    assert output_path.exists()


def test_comparison_csv_has_required_columns(
    monkeypatch,
    tmp_path,
):
    def fake_run(prepared_data, seed):
        return {
            "framework": "Test",
            "seed": seed,
            "best_epoch": 1,
            "validation_mae": 2.0,
            "test_mae": 2.1,
            "seconds": 0.5,
        }

    monkeypatch.setattr(
        compare,
        "run_pytorch_once",
        fake_run,
    )
    monkeypatch.setattr(
        compare,
        "run_keras_once",
        fake_run,
    )

    output_path = tmp_path / "comparison.csv"

    compare.run_comparison(
        prepared_data={},
        seeds=[2026],
        output_path=output_path,
    )

    saved_comparison = pd.read_csv(output_path)

    assert list(saved_comparison.columns) == [
        "framework",
        "seed",
        "best_epoch",
        "validation_mae",
        "test_mae",
        "seconds",
    ]
