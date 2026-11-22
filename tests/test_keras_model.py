"""Tests for the Keras bike-demand network."""

import numpy as np
import pytest
import torch

from keras_model import build_keras_model, copy_pytorch_weights
from model import BikeDemandNetwork
from prepare import SEED


@pytest.mark.parametrize(
    "hidden_sizes",
    [
        (16, 8),
        (64, 32),
        (128, 64, 32),
    ],
)
def test_keras_model_returns_one_value_per_row(hidden_sizes):
    model = build_keras_model(
        input_size=8,
        hidden_sizes=hidden_sizes,
    )

    predictions = model(np.zeros((7, 8), dtype=np.float32))

    assert predictions.shape == (7, 1)


def test_keras_model_requires_two_hidden_layers():
    with pytest.raises(
        ValueError,
        match="at least two hidden layers",
    ):
        build_keras_model(
            input_size=8,
            hidden_sizes=(16,),
        )


def test_keras_hidden_layers_match_requested_sizes():
    model = build_keras_model(
        input_size=8,
        hidden_sizes=(64, 32),
    )

    dense_layers = model.layers[1:]

    assert [layer.units for layer in dense_layers] == [64, 32, 1]


def test_keras_uses_relu_hidden_layers_and_linear_output():
    model = build_keras_model(
        input_size=8,
        hidden_sizes=(64, 32),
    )

    assert model.get_layer("hidden_1").activation.__name__ == "relu"
    assert model.get_layer("hidden_2").activation.__name__ == "relu"
    assert model.get_layer("output").activation.__name__ == "linear"


def test_copied_weights_produce_matching_predictions():
    torch.manual_seed(SEED)

    pytorch_model = BikeDemandNetwork(
        input_size=8,
        hidden_sizes=(64, 32),
    )
    keras_model = build_keras_model(
        input_size=8,
        hidden_sizes=(64, 32),
    )

    copy_pytorch_weights(pytorch_model, keras_model)

    features = (
        np.random.default_rng(SEED)
        .normal(
            size=(12, 8),
        )
        .astype(np.float32)
    )

    with torch.no_grad():
        pytorch_predictions = pytorch_model(
            torch.tensor(features),
        ).numpy()

    keras_predictions = (
        keras_model(
            features,
            training=False,
        )
        .numpy()
        .squeeze(1)
    )

    np.testing.assert_allclose(
        pytorch_predictions,
        keras_predictions,
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize(
    "hidden_sizes",
    [
        (16, 8),
        (64, 32),
        (128, 64, 32),
    ],
)
def test_keras_parameter_count_matches_pytorch(hidden_sizes):
    pytorch_model = BikeDemandNetwork(
        input_size=8,
        hidden_sizes=hidden_sizes,
    )
    keras_model = build_keras_model(
        input_size=8,
        hidden_sizes=hidden_sizes,
    )

    pytorch_parameter_count = sum(
        parameter.numel() for parameter in pytorch_model.parameters()
    )

    assert keras_model.count_params() == pytorch_parameter_count
