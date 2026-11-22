"""Tests for the bike-demand neural network."""

import pytest
import torch
from torch import nn

from model import BikeDemandNetwork


@pytest.mark.parametrize(
    "hidden_sizes",
    [
        (16, 8),
        (64, 32),
        (128, 64, 32),
    ],
)
def test_network_returns_one_value_per_row(hidden_sizes):
    model = BikeDemandNetwork(
        input_size=8,
        hidden_sizes=hidden_sizes,
    )
    features = torch.randn(7, 8)

    predictions = model(features)

    assert predictions.shape == torch.Size([7])


def test_optimizer_step_changes_every_layer():
    torch.manual_seed(2026)

    model = BikeDemandNetwork(input_size=8)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_function = nn.MSELoss()

    features = torch.randn(32, 8)
    targets = torch.full((32,), 10.0)

    linear_layers = [layer for layer in model.modules() if isinstance(layer, nn.Linear)]
    original_weights = [layer.weight.detach().clone() for layer in linear_layers]

    optimizer.zero_grad()
    predictions = model(features)
    loss = loss_function(predictions, targets)
    loss.backward()
    optimizer.step()

    for original, layer in zip(original_weights, linear_layers, strict=True):
        assert not torch.equal(original, layer.weight)
