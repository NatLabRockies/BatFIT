import numpy as np
import torch

from batfit.utils.scalers import (
    BoundedScaler,
    MarginSigmoid,
    ZScoreScaler,
    scaling_to_dict,
)


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
    # per-channel (X - means) / stds, numpy and torch
    means = np.array([[[2.0], [10.0]]], dtype="float32")
    stds = np.array([[[1.0], [5.0]]], dtype="float32")
    scaler = ZScoreScaler(means, stds)
    X = np.random.randn(8, 2, 50).astype("float32")
    X_scaled = scaler.transform(X)
    assert X_scaled.dtype == np.float32
    assert np.allclose(X_scaled, (X - means) / stds)
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

    # per-time-point fit: stds below min_std (coinciding curves) are clipped
    V = np.random.randn(30, 1, 16).astype("float32") * 0.1 + 3.5
    V[:, :, -1] = 4.1
    fitted_t = ZScoreScaler.fit(V, axis=0, min_std=1e-3)
    assert tuple(fitted_t.means.shape) == (1, 1, 16)
    assert np.isclose(float(fitted_t.stds[0, 0, -1]), 1e-3)
    assert np.allclose(
        fitted_t.stds[0, 0, :-1].numpy(), V[:, 0, :-1].std(axis=0), rtol=1e-5
    )
    V_scaled = fitted_t.transform(V)
    assert np.allclose(V_scaled[:, 0, :-1].mean(axis=0), 0.0, atol=1e-4)
    assert np.all(np.isfinite(V_scaled))

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
