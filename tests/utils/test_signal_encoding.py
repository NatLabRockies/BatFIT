import numpy as np
import pytest
import torch

from batfit.utils.signal_encoding import split_end_time


def test_split_end_time():
    n_curves, n_points = 5, 32
    t_end = np.linspace(1000.0, 5000.0, n_curves).astype("float32")
    time = t_end[:, None] * np.linspace(0.0, 1.0, n_points, dtype="float32")
    voltage = np.random.uniform(3.0, 4.1, (n_curves, n_points))
    X = np.stack((time, voltage.astype("float32")), axis=1)

    V, T = split_end_time(X)
    assert V.shape == (n_curves, 1, n_points)
    assert T.shape == (n_curves, 1)
    assert np.allclose(V[:, 0, :], X[:, 1, :])
    assert np.allclose(T[:, 0], t_end)

    # Make sure we did not duplicate arrays
    assert np.shares_memory(V, X)

    # torch tensors are split the same way
    V_t, T_t = split_end_time(torch.from_numpy(X))
    assert isinstance(V_t, torch.Tensor)
    assert tuple(T_t.shape) == (n_curves, 1)

    # a grid not starting at 0 cannot be described by t_end alone
    X_shifted = X.copy()
    X_shifted[:, 0, :] += 10.0
    with pytest.raises(AssertionError):
        split_end_time(X_shifted)
    # a single-channel signal has no time channel
    with pytest.raises(AssertionError):
        split_end_time(X[:, 1:, :])
