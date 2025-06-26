import numpy as np
import pytest


@pytest.fixture(scope="session")
def grid():
    return {
        "x": np.linspace(0, 1, 5),
        "y": np.linspace(-1, 1, 3),
        "z": np.linspace(-1, 1, 4),
    }
