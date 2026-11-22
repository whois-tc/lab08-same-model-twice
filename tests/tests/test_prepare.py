"""Tests for data splitting, preparation and loading."""

import numpy as np
import torch

from prepare import make_loaders


def test_dataset_length_dtype_and_shapes(prepared_data):
    dataset = prepared_data["train_dataset"]
    features, target = dataset[0]

    assert len(dataset) == 3600
    assert features.shape == torch.Size([8])
    assert target.shape == torch.Size([])
    assert features.dtype == torch.float32
    assert target.dtype == torch.float32


def test_split_sizes_and_no_overlap(prepared_data):
    training_rows = prepared_data["train_rows"]
    validation_rows = prepared_data["validation_rows"]
    test_rows = prepared_data["test_rows"]

    assert len(training_rows) == 3600
    assert len(validation_rows) == 1200
    assert len(test_rows) == 1200
    assert len(training_rows) + len(validation_rows) + len(test_rows) == 6000

    training_indices = set(training_rows.index)
    validation_indices = set(validation_rows.index)
    test_indices = set(test_rows.index)

    assert training_indices.isdisjoint(validation_indices)
    assert training_indices.isdisjoint(test_indices)
    assert validation_indices.isdisjoint(test_indices)


def test_scaler_is_fitted_on_training_rows_only(prepared_data):
    training_numeric = prepared_data["train_features"][:, :5]
    validation_numeric = prepared_data["validation_features"][:, :5]

    np.testing.assert_allclose(
        training_numeric.mean(axis=0),
        np.zeros(5),
        atol=1e-6,
    )

    validation_means = np.abs(validation_numeric.mean(axis=0))
    assert validation_means.max() > 0.01


def test_training_loader_shuffles_repeatably(prepared_data):
    first_loader, _, _ = make_loaders(
        prepared_data,
        batch_size=64,
        seed=123,
    )
    second_loader, _, _ = make_loaders(
        prepared_data,
        batch_size=64,
        seed=123,
    )

    first_features, _ = next(iter(first_loader))
    repeated_features, _ = next(iter(second_loader))
    original_first_features = prepared_data["train_dataset"].features[:64]

    assert torch.equal(first_features, repeated_features)
    assert not torch.equal(first_features, original_first_features)
