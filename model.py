"""Neural network for predicting hourly bike rentals."""

from torch import nn


class BikeDemandNetwork(nn.Module):
    """A regression network with at least two hidden layers."""

    def __init__(self, input_size, hidden_sizes=(64, 32)):
        super().__init__()

        if len(hidden_sizes) < 2:
            raise ValueError("The network requires at least two hidden layers.")

        layers = []
        current_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU())
            current_size = hidden_size

        layers.append(nn.Linear(current_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features):
        """Return one predicted rental count per row."""

        return self.network(features).squeeze(1)


if __name__ == "__main__":
    from bikes_data import make_rentals
    from prepare import make_loaders, prepare_data

    prepared = prepare_data(make_rentals())
    training_loader, _, _ = make_loaders(prepared)

    batch_features, _ = next(iter(training_loader))

    model = BikeDemandNetwork(input_size=batch_features.shape[1])
    predictions = model(batch_features)

    print(f"Input shape: {batch_features.shape}")
    print(f"Prediction shape: {predictions.shape}")
    print(model)
