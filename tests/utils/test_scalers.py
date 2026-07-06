import os
import pickle
import tempfile

import numpy as np
from sklearn.preprocessing import StandardScaler

from batfit.utils.scalers import (
    CustomScaler,
    scale_dataset_from_scaler,
    scale_input_from_scaler,
    scale_output_from_scaler,
    unscale_dataset_from_scaler,
    unscale_input_from_scaler,
    unscale_output_from_scaler,
    unscale_pred_from_scaler,
)


def test_CustomScaler():
    # means/stds have shape (1, channels, 1) — how scale_dataset_from_np creates them
    means = np.array([[[2.0], [10.0]]])  # (1, 2, 1)
    stds = np.array([[[1.0], [5.0]]])
    scaler = CustomScaler(means, stds)
    X = np.ones((8, 2, 50), dtype="float32") * np.array([[[3.0], [15.0]]])
    X_scaled = scaler.transform(X)
    # channel 0: (3 - 2) / 1 = 1.0; channel 1: (15 - 10) / 5 = 1.0
    assert np.allclose(X_scaled[:, 0, :], 1.0)
    assert np.allclose(X_scaled[:, 1, :], 1.0)
    X_back = scaler.inverse_transform(X_scaled)
    assert np.allclose(X_back, X)

    # fit(): reduces over the given axis with dims kept, matching the shape
    # __init__ expects, and round-trips through transform/inverse_transform
    N, n_chan, T = 20, 2, 50
    X_fit = np.random.randn(N, n_chan, T).astype("float32")
    fitted_scaler = CustomScaler.fit(X_fit, axis=(0, 2))
    assert fitted_scaler.means.shape == (1, n_chan, 1)
    assert fitted_scaler.stds.shape == (1, n_chan, 1)
    X_fit_scaled = fitted_scaler.transform(X_fit)
    assert np.allclose(X_fit_scaled.mean(axis=(0, 2)), 0.0, atol=1e-5)
    assert np.allclose(X_fit_scaled.std(axis=(0, 2)), 1.0, atol=1e-5)
    assert np.allclose(fitted_scaler.inverse_transform(X_fit_scaled), X_fit)


def test_scale_input_from_scaler():
    N, n_chan, T = 8, 2, 20
    means = np.array([[[1.0], [2.0]]])
    stds = np.array([[[2.0], [4.0]]])
    scaler = CustomScaler(means, stds)
    X = np.random.randn(N, n_chan, T).astype("float32")

    with tempfile.TemporaryDirectory() as tmp_dir:
        scaler_file = os.path.join(tmp_dir, "scaler_X.pkl")
        with open(scaler_file, "wb") as f:
            pickle.dump(scaler, f)
        X_scaled = scale_input_from_scaler(X, scaler_file)

    assert np.allclose(X_scaled, scaler.transform(X))


def test_scale_output_from_scaler():
    N, n_params = 30, 3
    Y = np.random.randn(N, n_params).astype("float32")
    scaler = StandardScaler().fit(Y)

    with tempfile.TemporaryDirectory() as tmp_dir:
        scaler_file = os.path.join(tmp_dir, "scaler_Y.pkl")
        with open(scaler_file, "wb") as f:
            pickle.dump(scaler, f)
        Y_scaled = scale_output_from_scaler(Y, scaler_file)

    assert np.allclose(Y_scaled, scaler.transform(Y))


def test_scale_dataset_from_scaler():
    N, n_chan, T, n_params = 8, 2, 20, 3
    means = np.array([[[0.0], [0.0]]])
    stds = np.array([[[1.0], [1.0]]])
    scaler_X = CustomScaler(means, stds)
    X = np.random.randn(N, n_chan, T).astype("float32")
    Y = np.random.randn(N, n_params).astype("float32")
    scaler_Y = StandardScaler().fit(Y)

    with tempfile.TemporaryDirectory() as tmp_dir:
        scaler_X_file = os.path.join(tmp_dir, "scaler_X.pkl")
        scaler_Y_file = os.path.join(tmp_dir, "scaler_Y.pkl")
        with open(scaler_X_file, "wb") as f:
            pickle.dump(scaler_X, f)
        with open(scaler_Y_file, "wb") as f:
            pickle.dump(scaler_Y, f)
        X_scaled, Y_scaled = scale_dataset_from_scaler(
            X, Y, scaler_X_file, scaler_Y_file
        )

    assert np.allclose(X_scaled, scaler_X.transform(X))
    assert np.allclose(Y_scaled, scaler_Y.transform(Y))


def test_unscale_input_from_scaler():
    N, n_chan, T = 8, 2, 20
    means = np.array([[[1.0], [2.0]]])
    stds = np.array([[[2.0], [4.0]]])
    scaler = CustomScaler(means, stds)
    X_scaled = np.random.randn(N, n_chan, T).astype("float32")

    with tempfile.TemporaryDirectory() as tmp_dir:
        scaler_file = os.path.join(tmp_dir, "scaler_X.pkl")
        with open(scaler_file, "wb") as f:
            pickle.dump(scaler, f)
        X_unscaled = unscale_input_from_scaler(X_scaled, scaler_file)
        # missing file: passthrough
        X_passthrough = unscale_input_from_scaler(
            X_scaled, os.path.join(tmp_dir, "does_not_exist.pkl")
        )

    assert np.allclose(X_unscaled, scaler.inverse_transform(X_scaled))
    assert np.allclose(X_passthrough, X_scaled)
    # None scaler file: passthrough
    assert np.allclose(
        unscale_input_from_scaler(X_scaled, None), X_scaled
    )


def test_unscale_output_from_scaler():
    N, n_params = 30, 3
    Y = np.random.randn(N, n_params).astype("float32")
    scaler = StandardScaler().fit(Y)
    Y_scaled = scaler.transform(Y).astype("float32")

    with tempfile.TemporaryDirectory() as tmp_dir:
        scaler_file = os.path.join(tmp_dir, "scaler_Y.pkl")
        with open(scaler_file, "wb") as f:
            pickle.dump(scaler, f)
        Y_unscaled = unscale_output_from_scaler(Y_scaled, scaler_file)

    assert np.allclose(Y_unscaled, Y, atol=1e-5)
    assert np.allclose(unscale_output_from_scaler(Y_scaled, None), Y_scaled)


def test_unscale_dataset_from_scaler():
    N, n_chan, T, n_params = 8, 2, 20, 3
    means = np.array([[[0.0], [0.0]]])
    stds = np.array([[[1.0], [1.0]]])
    scaler_X = CustomScaler(means, stds)
    X_scaled = np.random.randn(N, n_chan, T).astype("float32")
    Y = np.random.randn(N, n_params).astype("float32")
    scaler_Y = StandardScaler().fit(Y)
    Y_scaled = scaler_Y.transform(Y).astype("float32")

    with tempfile.TemporaryDirectory() as tmp_dir:
        scaler_X_file = os.path.join(tmp_dir, "scaler_X.pkl")
        scaler_Y_file = os.path.join(tmp_dir, "scaler_Y.pkl")
        with open(scaler_X_file, "wb") as f:
            pickle.dump(scaler_X, f)
        with open(scaler_Y_file, "wb") as f:
            pickle.dump(scaler_Y, f)
        X_unscaled, Y_unscaled = unscale_dataset_from_scaler(
            X_scaled, Y_scaled, scaler_X_file, scaler_Y_file
        )

    assert np.allclose(X_unscaled, scaler_X.inverse_transform(X_scaled))
    assert np.allclose(Y_unscaled, Y, atol=1e-5)


def test_unscale_pred_from_scaler():
    N, n_params = 30, 3
    Y = np.random.randn(N, n_params).astype("float32")
    scaler = StandardScaler().fit(Y)
    Y_scaled = scaler.transform(Y).astype("float32")

    with tempfile.TemporaryDirectory() as tmp_dir:
        scaler_file = os.path.join(tmp_dir, "scaler_Y.pkl")
        with open(scaler_file, "wb") as f:
            pickle.dump(scaler, f)
        Y_unscaled = unscale_pred_from_scaler(Y_scaled, scaler_file)

    assert np.allclose(Y_unscaled, Y, atol=1e-5)
    # default scaler_Y_file=None: passthrough
    assert np.allclose(unscale_pred_from_scaler(Y_scaled), Y_scaled)
