"""Keras version of the bike-demand neural network."""

import keras
from torch import nn


def build_keras_model(input_size, hidden_sizes=(64, 32)):
    """Build the same regression architecture used by PyTorch."""

    if len(hidden_sizes) < 2:
        raise ValueError("The network requires at least two hidden layers.")

    inputs = keras.Input(shape=(input_size,), name="features")
    values = inputs

    for layer_number, hidden_size in enumerate(hidden_sizes, start=1):
        values = keras.layers.Dense(
            hidden_size,
            activation="relu",
            name=f"hidden_{layer_number}",
        )(values)

    outputs = keras.layers.Dense(
        1,
        activation="linear",
        name="output",
    )(values)

    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="bike_demand_network",
    )


def copy_pytorch_weights(pytorch_model, keras_model):
    """Copy PyTorch linear-layer weights into matching Keras layers."""

    pytorch_layers = [
        layer for layer in pytorch_model.modules() if isinstance(layer, nn.Linear)
    ]

    keras_layers = [
        layer for layer in keras_model.layers if isinstance(layer, keras.layers.Dense)
    ]

    if len(pytorch_layers) != len(keras_layers):
        raise ValueError("The PyTorch and Keras models have different layer counts.")

    for pytorch_layer, keras_layer in zip(
        pytorch_layers,
        keras_layers,
        strict=True,
    ):
        weights = pytorch_layer.weight.detach().cpu().numpy().T
        biases = pytorch_layer.bias.detach().cpu().numpy()
        keras_layer.set_weights([weights, biases])
