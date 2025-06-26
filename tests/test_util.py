from math import isclose

import numpy as np
import pytest

from conditional_kde.util import DataWhitener, Interpolator


@pytest.fixture(scope="module")
def random_sample():
    mean = np.array([[1, 2, 3]], dtype=np.float32)
    std = np.array([[1, 5, 10]], dtype=np.float32)

    sample = np.random.normal(size=(100, 3))

    return mean + sample * std


@pytest.fixture(scope="module")
def values(grid):
    mesh = np.meshgrid(*grid.values(), indexing="ij")
    return np.sum(np.stack(mesh, axis=0), axis=0)


def test_whitener(random_sample):
    algorithms = [None, "center", "rescale", "PCA", "ZCA"]
    dw = {}
    for algo in algorithms:
        dw[algo] = DataWhitener(algo)
        dw[algo].fit(random_sample)

    with pytest.raises(ValueError):
        DataWhitener("bla").fit(random_sample)

    samples = [
        np.array([0.0, 0.0, 0.0]),
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
    ]
    for sample in samples:
        whiten_unwhiten_res = [
            dw[algo].unwhiten(dw[algo].whiten(sample)) for algo in algorithms
        ]
        for res in whiten_unwhiten_res:
            assert sample.shape == res.shape
            assert np.allclose(whiten_unwhiten_res[0], res)


def test_interpolator(values, grid):
    interp = Interpolator(list(grid.values()), values, method="linear")
    xi = np.array([0.5, -0.1, 0.7])
    result = interp(xi, return_aux=False)
    assert isclose(sum(xi), result.item())
