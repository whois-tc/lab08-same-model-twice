"""Prepare the bike-rental data for the neural network."""

import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from bikes_data import TARGET, make_rentals

NUMERIC_FEATURES = [
    "hour",
    "temperature_c",
    "rain_mm",
    "wind_kph",
    "is_weekend",
]

CATEGORICAL_FEATURES = ["station_type"]

SEED = 2026


class BikeDataset(Dataset):
    """Store prepared features and targets as float32 tensors."""

    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]


def split_data(frame, seed=SEED):
    """Split the rows into 60% training, 20% validation and 20% test."""

    train_rows, remaining_rows = train_test_split(
        frame,
        test_size=0.40,
        random_state=seed,
    )

    validation_rows, test_rows = train_test_split(
        remaining_rows,
        test_size=0.50,
        random_state=seed,
    )

    return train_rows, validation_rows, test_rows


def build_preprocessor():
    """Create the numeric scaler and categorical encoder."""

    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def prepare_data(frame, seed=SEED):
    """Split the data and fit preprocessing on training rows only."""

    train_rows, validation_rows, test_rows = split_data(frame, seed)

    preprocessor = build_preprocessor()

    train_features = np.asarray(
        preprocessor.fit_transform(train_rows),
        dtype=np.float32,
    )
    validation_features = np.asarray(
        preprocessor.transform(validation_rows),
        dtype=np.float32,
    )
    test_features = np.asarray(
        preprocessor.transform(test_rows),
        dtype=np.float32,
    )

    train_targets = train_rows[TARGET].to_numpy(dtype=np.float32)
    validation_targets = validation_rows[TARGET].to_numpy(dtype=np.float32)
    test_targets = test_rows[TARGET].to_numpy(dtype=np.float32)

    return {
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "test_rows": test_rows,
        "train_features": train_features,
        "validation_features": validation_features,
        "test_features": test_features,
        "train_targets": train_targets,
        "validation_targets": validation_targets,
        "test_targets": test_targets,
        "train_dataset": BikeDataset(train_features, train_targets),
        "validation_dataset": BikeDataset(
            validation_features,
            validation_targets,
        ),
        "test_dataset": BikeDataset(test_features, test_targets),
        "preprocessor": preprocessor,
    }


def make_loaders(prepared_data, batch_size=64, seed=SEED):
    """Create loaders that shuffle training batches only."""

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        prepared_data["train_dataset"],
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    validation_loader = DataLoader(
        prepared_data["validation_dataset"],
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        prepared_data["test_dataset"],
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, validation_loader, test_loader


if __name__ == "__main__":
    data = make_rentals()
    prepared = prepare_data(data)
    training_loader, validation_loader, testing_loader = make_loaders(prepared)

    batch_features, batch_targets = next(iter(training_loader))

    print(f"All rows: {len(data)}")
    print(f"Training rows: {len(prepared['train_rows'])}")
    print(f"Validation rows: {len(prepared['validation_rows'])}")
    print(f"Test rows: {len(prepared['test_rows'])}")
    print(f"Feature batch shape: {batch_features.shape}")
    print(f"Target batch shape: {batch_targets.shape}")
    print(f"Feature type: {batch_features.dtype}")
    print(f"Target type: {batch_targets.dtype}")
