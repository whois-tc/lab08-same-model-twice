"""Shared test data."""

import pytest

from bikes_data import make_rentals
from prepare import prepare_data


@pytest.fixture(scope="session")
def bike_frame():
    """Return the repeatable raw bike-rental table."""

    return make_rentals()


@pytest.fixture(scope="session")
def prepared_data(bike_frame):
    """Return one prepared version of the table."""

    return prepare_data(bike_frame)
