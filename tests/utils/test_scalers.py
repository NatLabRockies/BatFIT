import os
import pickle
import tempfile

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from batfit.utils.scalers import (
    BoundedScaler,
    CustomScaler,
    MarginSigmoid,
    ZScoreScaler,
    scale_dataset_from_scaler,
    scale_input_from_scaler,
    scale_output_from_scaler,
    scaling_to_dict,
    unscale_dataset_from_scaler,
    unscale_input_from_scaler,
    unscale_output_from_scaler,
    unscale_pred_from_scaler,
    unscale_pred_std_from_scaler,
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

    # type-agnostic: a torch tensor stays a torch tensor, matches the numpy
    # result, and propagates gradients (no numpy __array_wrap__ deprecation)
    X_t = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    X_t_scaled = scaler.transform(X_t)
    assert isinstance(X_t_scaled, torch.Tensor)
    assert np.allclose(X_t_scaled.detach().numpy(), scaler.transform(X))
    X_t_scaled.sum().backward()
    assert X_t.grad is not None
    assert torch.allclose(X_t.grad, 1.0 / torch.tensor(stds, dtype=X_t.dtype))
    X_t_back = scaler.inverse_transform(X_t_scaled.detach())
    assert isinstance(X_t_back, torch.Tensor)
    assert torch.allclose(X_t_back, X_t.detach())

    # single-channel fallback to channel-1 stats also works for tensors
    X1_t = torch.ones((4, 1, 10), dtype=torch.float32) * 15.0
    X1_scaled = scaler.transform(X1_t)
    assert isinstance(X1_scaled, torch.Tensor)
    assert torch.allclose(X1_scaled, torch.ones_like(X1_scaled))


def test_BoundedScaler():
    import pytest

    low = np.array([0.5, 10.0], dtype="float32")
    high = np.array([1.5, 30.0], dtype="float32")
    scaler = BoundedScaler(low, high)

    # bounds map onto [0, 1] and the midpoint onto 0.5
    Y = np.array([[0.5, 10.0], [1.5, 30.0], [1.0, 20.0]], dtype="float32")
    Y_scaled = scaler.transform(Y)
    assert isinstance(Y_scaled, np.ndarray)
    assert Y_scaled.dtype == np.float32
    assert np.allclose(Y_scaled, [[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
    assert np.allclose(scaler.inverse_transform(Y_scaled), Y)

    # in-place scaling overwrites the array and matches transform
    Y_inplace = Y.copy()
    assert scaler.transform_(Y_inplace) is Y_inplace
    assert np.allclose(Y_inplace, Y_scaled)

    # std only picks up the range, never the offset
    std_scaled = np.array([[0.1, 0.2]], dtype="float32")
    assert np.allclose(scaler.inverse_transform_std(std_scaled), [[0.1, 4.0]])

    # clipping to the physical bounds, numpy and torch
    Y_out = np.array([[0.0, 40.0]], dtype="float32")
    assert np.allclose(scaler.clip_physical(Y_out), [[0.5, 30.0]])
    assert torch.allclose(
        scaler.clip_physical(torch.from_numpy(Y_out)),
        torch.tensor([[0.5, 30.0]]),
    )

    # torch in, torch out, with gradients flowing through the input
    Y_t = torch.tensor(Y, requires_grad=True)
    Y_t_scaled = scaler.transform(Y_t)
    assert isinstance(Y_t_scaled, torch.Tensor)
    Y_t_scaled.sum().backward()
    assert torch.allclose(Y_t.grad, 1.0 / torch.from_numpy(high - low))

    # bounds are buffers, so they are saved in the state dict
    assert set(scaler.state_dict().keys()) == {"low", "high"}
    assert scaler.to_dict() == {"low": [0.5, 10.0], "high": [1.5, 30.0]}

    # construction from parsed experiment-config bounds
    sim_params = {
        "deg_param_names": ["i0_a", "ds_c"],
        "deg_i0_a_min": 0.5,
        "deg_i0_a_max": 1.5,
        "deg_ds_c_min": 10.0,
        "deg_ds_c_max": 30.0,
        "prot_param_names": ["amplitude"],
        "prot_amplitude_min": 0.0,
        "prot_amplitude_max": 10.0,
        "vmin": 3.0,
        "vmax": 4.2,
    }
    deg_scaler = BoundedScaler.from_sim_params(sim_params, kind="deg")
    assert np.allclose(deg_scaler.transform(Y), Y_scaled)
    prot_scaler = BoundedScaler.from_sim_params(sim_params, kind="prot")
    assert np.allclose(prot_scaler.transform(np.array([[5.0]])), 0.5)
    v_scaler = BoundedScaler.from_sim_params(sim_params, kind="voltage")
    assert np.allclose(
        v_scaler.transform(np.array([[3.0], [4.2]])), [[0], [1]]
    )

    # invalid bounds are rejected
    with pytest.raises(AssertionError):
        BoundedScaler([1.0], [1.0])


def test_ZScoreScaler():
    # matches CustomScaler on the same statistics, numpy and torch
    means = np.array([[[2.0], [10.0]]], dtype="float32")
    stds = np.array([[[1.0], [5.0]]], dtype="float32")
    scaler = ZScoreScaler(means, stds)
    scaler_ref = CustomScaler(means, stds)
    X = np.random.randn(8, 2, 50).astype("float32")
    X_scaled = scaler.transform(X)
    assert X_scaled.dtype == np.float32
    assert np.allclose(X_scaled, scaler_ref.transform(X))
    assert np.allclose(scaler.inverse_transform(X_scaled), X, atol=1e-5)

    X_t = torch.tensor(X, requires_grad=True)
    X_t_scaled = scaler.transform(X_t)
    assert isinstance(X_t_scaled, torch.Tensor)
    X_t_scaled.sum().backward()
    assert torch.allclose(
        X_t.grad, (1.0 / torch.from_numpy(stds)).expand_as(X_t)
    )

    # in-place scaling overwrites the array and matches transform
    X_inplace = X.copy()
    assert scaler.transform_(X_inplace) is X_inplace
    assert np.allclose(X_inplace, X_scaled)

    # single-channel input falls back to the channel-1 statistics
    X1 = np.ones((4, 1, 10), dtype="float32") * 15.0
    assert np.allclose(scaler.transform(X1), 1.0)
    assert np.allclose(scaler.transform_(X1.copy()), 1.0)

    # fit() gives zero-mean unit-std channels
    X_fit = np.random.randn(20, 2, 50).astype("float32") * 3.0 + 1.0
    fitted = ZScoreScaler.fit(X_fit, axis=(0, 2))
    assert tuple(fitted.means.shape) == (1, 2, 1)
    X_fit_scaled = fitted.transform(X_fit)
    assert np.allclose(X_fit_scaled.mean(axis=(0, 2)), 0.0, atol=1e-5)
    assert np.allclose(X_fit_scaled.std(axis=(0, 2)), 1.0, atol=1e-5)

    # statistics are buffers and exportable
    assert set(scaler.state_dict().keys()) == {"means", "stds"}
    assert scaler.to_dict() == {
        "means": [[[2.0], [10.0]]],
        "stds": [[[1.0], [5.0]]],
    }


def test_MarginSigmoid():
    head = MarginSigmoid(margin=0.05)
    z = torch.tensor([-1e4, 0.0, 1e4])
    out = head(z)
    # spans [-margin, 1 + margin], centred on 0.5
    assert torch.allclose(out, torch.tensor([-0.05, 0.5, 1.05]))

    # the bounds 0 and 1 are reached with finite logits
    z_bounds = torch.tensor(
        [-np.log(21.0), np.log(21.0)], dtype=torch.float32, requires_grad=True
    )
    out_bounds = head(z_bounds)
    assert torch.allclose(out_bounds, torch.tensor([0.0, 1.0]), atol=1e-6)
    out_bounds.sum().backward()
    assert torch.all(z_bounds.grad > 0.0)

    # margin 0 is a plain sigmoid
    assert torch.allclose(MarginSigmoid(margin=0.0)(z), torch.sigmoid(z))
    assert "margin" in head.state_dict()
    assert np.isclose(head.to_dict()["margin"], 0.05)


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
    assert np.allclose(unscale_input_from_scaler(X_scaled, None), X_scaled)


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


def test_unscale_pred_std_from_scaler():
    import pytest
    from sklearn.preprocessing import MinMaxScaler

    N, n_params = 30, 3
    # nonzero mean so a wrong inverse_transform would shift the std
    Y = (np.random.randn(N, n_params) + 5.0).astype("float32")
    scaler = StandardScaler().fit(Y)
    std_scaled = np.abs(np.random.randn(N, n_params)).astype("float32")

    with tempfile.TemporaryDirectory() as tmp_dir:
        scaler_file = os.path.join(tmp_dir, "scaler_Y.pkl")
        with open(scaler_file, "wb") as f:
            pickle.dump(scaler, f)
        std_unscaled = unscale_pred_std_from_scaler(std_scaled, scaler_file)

        # std transforms with the scale only, never the mean shift
        assert np.allclose(std_unscaled, std_scaled * scaler.scale_, atol=1e-5)
        assert std_unscaled.dtype == std_scaled.dtype

        # non (x - mu) / sigma scalers are not supported
        minmax_file = os.path.join(tmp_dir, "scaler_minmax.pkl")
        with open(minmax_file, "wb") as f:
            pickle.dump(MinMaxScaler().fit(Y), f)
        with pytest.raises(NotImplementedError):
            unscale_pred_std_from_scaler(std_scaled, minmax_file)

    # scaler_Y_file=None or missing file: passthrough
    assert np.allclose(unscale_pred_std_from_scaler(std_scaled), std_scaled)
    assert np.allclose(
        unscale_pred_std_from_scaler(std_scaled, "does_not_exist.pkl"),
        std_scaled,
    )


def test_scaling_to_dict():
    import json

    model = torch.nn.Module()
    model.scaler_Y = BoundedScaler([0.5], [1.5])
    model.scaler_X = ZScoreScaler(np.array([[1.0]]), np.array([[2.0]]))
    model.head = torch.nn.Sequential(torch.nn.Linear(2, 1), MarginSigmoid(0.1))
    model.sim_config = "spm_chirp.yaml"
    model.sim_params = {
        "deg_param_names": ["i0_a"],
        "prot_param_names": ["amplitude"],
    }

    summary = scaling_to_dict(model)
    # every scaler and margin head, under its attribute path
    assert set(summary["scalers"]) == {"scaler_Y", "scaler_X", "head.1"}
    assert summary["scalers"]["scaler_Y"] == {
        "type": "BoundedScaler",
        "low": [0.5],
        "high": [1.5],
    }
    assert summary["scalers"]["head.1"]["type"] == "MarginSigmoid"
    assert summary["deg_param_names"] == ["i0_a"]
    assert summary["prot_param_names"] == ["amplitude"]
    assert summary["sim_config"] == "spm_chirp.yaml"
    # JSON-serialisable
    json.dumps(summary)

    # a model without scalers gives an empty summary
    assert scaling_to_dict(torch.nn.Linear(2, 1)) == {"scalers": {}}
