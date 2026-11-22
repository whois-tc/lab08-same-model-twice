"""Tests for the hand-written training loop."""

import pytest
import torch
from torch import nn

from model import BikeDemandNetwork
from prepare import SEED, make_loaders
from train import run_epoch, train_model


def test_network_can_memorize_small_batch(prepared_data):
    torch.manual_seed(SEED)

    features = prepared_data["train_dataset"].features[:16]
    targets = prepared_data["train_dataset"].targets[:16]

    model = BikeDemandNetwork(input_size=features.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_function = nn.MSELoss()

    initial_loss = loss_function(model(features), targets).item()

    for _ in range(600):
        optimizer.zero_grad()
        predictions = model(features)
        loss = loss_function(predictions, targets)
        loss.backward()
        optimizer.step()

    final_loss = loss_function(model(features), targets).item()

    assert final_loss < initial_loss * 0.05


def test_validation_epoch_leaves_weights_unchanged(prepared_data):
    torch.manual_seed(SEED)

    _, validation_loader, _ = make_loaders(
        prepared_data,
        batch_size=128,
        seed=SEED,
    )
    model = BikeDemandNetwork(input_size=8)

    original_parameters = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }

    run_epoch(
        model,
        validation_loader,
        nn.MSELoss(),
    )

    for name, parameter in model.named_parameters():
        assert torch.equal(original_parameters[name], parameter)


def test_training_history_and_best_weights(prepared_data):
    torch.manual_seed(SEED)

    training_loader, validation_loader, _ = make_loaders(
        prepared_data,
        batch_size=256,
        seed=SEED,
    )
    model = BikeDemandNetwork(input_size=8)

    history, best_epoch = train_model(
        model,
        training_loader,
        validation_loader,
        epochs=5,
        learning_rate=0.003,
    )

    required_keys = {
        "epoch",
        "training_loss",
        "training_mae",
        "validation_loss",
        "validation_mae",
    }

    assert len(history) == 5
    assert required_keys == set(history[0])
    assert 1 <= best_epoch <= 5

    restored_loss, _ = run_epoch(
        model,
        validation_loader,
        nn.MSELoss(),
    )
    best_recorded_loss = history[best_epoch - 1]["validation_loss"]

    assert restored_loss == pytest.approx(best_recorded_loss, rel=1e-6)
