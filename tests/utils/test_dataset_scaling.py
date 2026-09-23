import numpy as np

from batfit.utils.dataset_scaling import (
    build_scalers,
    build_surrogate_scalers,
    scale_splits,
    scale_surrogate_splits,
)
from batfit.utils.scalers import BoundedScaler, ZScoreScaler


def test_build_scalers():
    X_train = np.random.randn(20, 2, 30).astype("float32") * 2.0 + 1.0
    sim_params = {
        "deg_param_names": ["i0_a"],
        "deg_i0_a_min": 0.5,
        "deg_i0_a_max": 1.5,
        "prot_param_names": ["amplitude", "length"],
        "prot_amplitude_min": 0.0,
        "prot_amplitude_max": 10.0,
        "prot_length_min": 50.0,
        "prot_length_max": 200.0,
    }

    scalers = build_scalers(X_train, sim_params)
    assert set(scalers) == {"X", "Y"}
    assert isinstance(scalers["X"], ZScoreScaler)
    assert isinstance(scalers["Y"], BoundedScaler)
    # X is fitted on train, Y comes from the bounds
    X_scaled = scalers["X"].transform(X_train)
    assert np.allclose(X_scaled.mean(axis=(0, 2)), 0.0, atol=1e-5)
    assert scalers["Y"].to_dict() == {"low": [0.5], "high": [1.5]}

    scalers_p = build_scalers(X_train, sim_params, with_prot=True)
    assert set(scalers_p) == {"X", "Y", "P"}
    assert scalers_p["P"].to_dict() == {
        "low": [0.0, 50.0],
        "high": [10.0, 200.0],
    }

    # time-dependent z-score: per-time-point voltage stats, z-scored T
    V_train = np.random.randn(20, 1, 30).astype("float32") * 0.1 + 3.5
    V_train[:, :, 0] = 3.0  # identical start: std clipped to 1e-3
    T_train = np.random.uniform(1000.0, 5000.0, (20, 1)).astype("float32")
    scalers_td = build_scalers(
        V_train,
        sim_params,
        signal_scaling="time_dependent_zscore",
        T_train=T_train,
    )
    assert set(scalers_td) == {"X", "T", "Y"}
    assert tuple(scalers_td["X"].means.shape) == (1, 1, 30)
    assert np.isclose(float(scalers_td["X"].stds[0, 0, 0]), 1e-3)
    V_scaled = scalers_td["X"].transform(V_train)
    assert np.allclose(V_scaled.mean(axis=0), 0.0, atol=1e-4)
    T_scaled = scalers_td["T"].transform(T_train)
    assert np.isclose(T_scaled.mean(), 0.0, atol=1e-5)
    assert np.isclose(T_scaled.std(), 1.0, atol=1e-5)


def test_scale_splits():
    scalers = {
        "X": ZScoreScaler(
            np.array([[[1.0], [2.0]]]), np.array([[[2.0], [4.0]]])
        ),
        "Y": BoundedScaler([0.0, 10.0], [2.0, 30.0]),
    }
    X_train = np.random.randn(10, 2, 5).astype("float32")
    X_ref = scalers["X"].transform(X_train)
    splits = {
        "X_train": X_train,
        "Y_train": np.array([[0.0, 10.0], [2.0, 30.0]], dtype="float32"),
        "X_val": None,
        "P_train": np.ones((10, 1), dtype="float32"),
    }

    scaled = scale_splits(splits, scalers)
    # None arrays and quantities without a scaler are dropped
    assert set(scaled) == {"X_train", "Y_train"}
    assert np.allclose(scaled["Y_train"], [[0.0, 0.0], [1.0, 1.0]])
    assert np.allclose(scaled["X_train"], X_ref)
    # scaling is in place: no copy, the input array holds the scaled values
    assert scaled["X_train"] is X_train
    assert scaled["X_train"].dtype == np.float32
    # quantities without a scaler are left untouched
    assert np.allclose(splits["P_train"], 1.0)


def test_build_surrogate_scalers():
    rng = np.random.default_rng(0)
    # rows (time, deg_params...): time in seconds, params inside the bounds
    X_train = np.column_stack(
        [rng.uniform(0.0, 3600.0, 50), rng.uniform(0.5, 1.5, 50)]
    ).astype("float32")
    sim_params = {
        "deg_param_names": ["i0_a"],
        "deg_i0_a_min": 0.5,
        "deg_i0_a_max": 1.5,
        "vmin": 3.0,
        "vmax": 4.2,
    }

    scalers = build_surrogate_scalers(X_train, sim_params)
    assert set(scalers) == {"t", "Y", "V"}
    assert isinstance(scalers["t"], ZScoreScaler)
    # time z-score fitted on the time column only
    t_scaled = scalers["t"].transform(X_train[:, :1])
    assert np.allclose(t_scaled.mean(), 0.0, atol=1e-5)
    assert np.allclose(t_scaled.std(), 1.0, atol=1e-5)
    # parameters and voltage from the config bounds
    assert scalers["Y"].to_dict() == {"low": [0.5], "high": [1.5]}
    assert np.allclose(scalers["V"].to_dict()["low"], [3.0])
    assert np.allclose(scalers["V"].to_dict()["high"], [4.2])


def test_scale_surrogate_splits():
    scalers = {
        "t": ZScoreScaler(np.array([[100.0]]), np.array([[50.0]])),
        "Y": BoundedScaler([0.0, 10.0], [2.0, 30.0]),
        "V": BoundedScaler([3.0], [4.0]),
    }
    X_train = np.array(
        [[100.0, 0.0, 10.0], [150.0, 2.0, 30.0]], dtype="float32"
    )
    Y_train = np.array([[3.0], [3.5]], dtype="float32")
    splits = {"X_train": X_train, "Y_train": Y_train, "X_val": None}

    scaled = scale_surrogate_splits(splits, scalers)
    assert set(scaled) == {"X_train", "Y_train"}
    # time column z-scored, parameter columns and voltage scaled to [0, 1]
    assert np.allclose(scaled["X_train"][:, 0], [0.0, 1.0])
    assert np.allclose(scaled["X_train"][:, 1:], [[0.0, 0.0], [1.0, 1.0]])
    assert np.allclose(scaled["Y_train"], [[0.0], [0.5]])
    # scaling is in place: no copy of the dataset
    assert scaled["X_train"] is X_train
    assert scaled["Y_train"] is Y_train
