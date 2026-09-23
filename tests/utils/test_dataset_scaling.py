import os
import pickle
import tempfile

import numpy as np

from batfit.utils.dataset_scaling import (
    build_scalers,
    scale_dataset_from_np,
    scale_protocol_dataset_from_np,
    scale_splits,
    scale_surrogate_dataset_from_np,
)
from batfit.utils.scalers import BoundedScaler, CustomScaler, ZScoreScaler


def test_scale_dataset_from_np():
    N, n_chan, T, n_params = 40, 2, 30, 3
    X_train = np.random.randn(N, n_chan, T).astype("float32")
    X_test = np.random.randn(10, n_chan, T).astype("float32")
    X_val = np.random.randn(8, n_chan, T).astype("float32")
    Y_train = np.random.randn(N, n_params).astype("float32")
    Y_test = np.random.randn(10, n_params).astype("float32")
    Y_val = np.random.randn(8, n_params).astype("float32")

    with tempfile.TemporaryDirectory() as tmp_dir:
        X_tr_sc, Y_tr_sc, X_te_sc, Y_te_sc, X_va_sc, Y_va_sc = (
            scale_dataset_from_np(
                X_train,
                X_test,
                Y_train,
                Y_test,
                X_val=X_val,
                Y_val=Y_val,
                save_path=tmp_dir,
            )
        )
        assert os.path.isfile(os.path.join(tmp_dir, "scaler_X.pkl"))
        assert os.path.isfile(os.path.join(tmp_dir, "data_scaled.npz"))
        # validation slice is saved and transform-only (same scaler as train)
        assert "X_val" in np.load(os.path.join(tmp_dir, "data_scaled.npz"))
        assert X_va_sc.shape == X_val.shape
        with open(os.path.join(tmp_dir, "scaler_X.pkl"), "rb") as f:
            scaler_X = pickle.load(f)
        assert np.allclose(X_va_sc, scaler_X.transform(X_val))
        # Y is unscaled by default
        assert np.allclose(Y_tr_sc, Y_train)
        assert np.allclose(Y_va_sc, Y_val)
        # X is z-scored per channel: mean ~0, std ~1 over (N, T)
        assert np.allclose(X_tr_sc.mean(axis=(0, 2)), 0.0, atol=1e-5)
        assert np.allclose(X_tr_sc.std(axis=(0, 2)), 1.0, atol=1e-5)

        # cache-hit: second call returns identical arrays including val
        X_tr_sc2, _, _, _, X_va_sc2, _ = scale_dataset_from_np(
            X_train,
            X_test,
            Y_train,
            Y_test,
            X_val=X_val,
            Y_val=Y_val,
            save_path=tmp_dir,
        )
        assert np.allclose(X_tr_sc, X_tr_sc2)
        assert np.allclose(X_va_sc, X_va_sc2)

    # X-scaler reuse: a pre-existing scaler_X.pkl is reused rather than re-fit
    with tempfile.TemporaryDirectory() as tmp_dir:
        wrong_scaler = CustomScaler(
            means=np.zeros((1, n_chan, 1), dtype="float32"),
            stds=np.ones((1, n_chan, 1), dtype="float32"),
        )
        with open(os.path.join(tmp_dir, "scaler_X.pkl"), "wb") as f:
            pickle.dump(wrong_scaler, f)
        X_tr_sc, _, _, _, _, _ = scale_dataset_from_np(
            X_train, X_test, Y_train, Y_test, save_path=tmp_dir
        )
        assert np.allclose(X_tr_sc, wrong_scaler.transform(X_train))

    # scale_y=True: separate cache file, Y is z-scored
    with tempfile.TemporaryDirectory() as tmp_dir:
        X_tr_sc, Y_tr_sc, X_te_sc, Y_te_sc, _, _ = scale_dataset_from_np(
            X_train, X_test, Y_train, Y_test, save_path=tmp_dir, scale_y=True
        )
        assert os.path.isfile(os.path.join(tmp_dir, "scaler_Y.pkl"))
        assert os.path.isfile(os.path.join(tmp_dir, "data_scaled_y.npz"))
        assert not os.path.isfile(os.path.join(tmp_dir, "data_scaled.npz"))
        assert np.allclose(Y_tr_sc.mean(axis=0), 0.0, atol=1e-5)
        assert np.allclose(Y_tr_sc.std(axis=0), 1.0, atol=1e-5)


def test_scale_protocol_dataset_from_np():
    N, n_chan, T, n_prot, n_params = 80, 2, 50, 3, 6
    X_train = np.random.randn(N, n_chan, T).astype("float32")
    X_test = np.random.randn(20, n_chan, T).astype("float32")
    X_val = np.random.randn(10, n_chan, T).astype("float32")
    # Protocol params with known range [0, 1] so MinMax scaling is identity
    P_train = np.random.rand(N, n_prot).astype("float32")
    P_test = np.random.rand(20, n_prot).astype("float32")
    P_val = np.random.rand(10, n_prot).astype("float32")
    Y_train = np.random.randn(N, n_params).astype("float32")
    Y_test = np.random.randn(20, n_params).astype("float32")
    Y_val = np.random.randn(10, n_params).astype("float32")

    with tempfile.TemporaryDirectory() as tmp_dir:
        (
            X_tr_sc,
            P_tr_sc,
            Y_tr_sc,
            X_te_sc,
            P_te_sc,
            Y_te_sc,
            X_va_sc,
            P_va_sc,
            Y_va_sc,
        ) = scale_protocol_dataset_from_np(
            X_train,
            P_train,
            X_test,
            P_test,
            Y_train,
            Y_test,
            X_val=X_val,
            P_val=P_val,
            Y_val=Y_val,
            save_path=tmp_dir,
        )
        # cache-hit: second call loads from data_scaled.npz
        cached = scale_protocol_dataset_from_np(
            X_train,
            P_train,
            X_test,
            P_test,
            Y_train,
            Y_test,
            X_val=X_val,
            P_val=P_val,
            Y_val=Y_val,
            save_path=tmp_dir,
        )

    assert X_tr_sc.shape == X_train.shape
    assert P_tr_sc.shape == P_train.shape
    # validation slice present and correctly shaped
    assert X_va_sc.shape == X_val.shape
    assert P_va_sc.shape == P_val.shape
    assert Y_va_sc.shape == Y_val.shape
    # Y is unscaled when scale_y=False
    assert np.allclose(Y_tr_sc, Y_train)
    assert np.allclose(Y_va_sc, Y_val)
    # scaler fit on P_train only: training values must be in [0, 1]
    assert P_tr_sc.min() >= -1e-5
    assert P_tr_sc.max() <= 1.0 + 1e-5
    # cache-hit returns identical val arrays (index 6 == X_val)
    assert np.allclose(X_tr_sc, cached[0])
    assert np.allclose(X_va_sc, cached[6])

    # scale_y=True: Y should be z-scored and scaler_Y.pkl should be saved
    with tempfile.TemporaryDirectory() as tmp_dir:
        result = scale_protocol_dataset_from_np(
            X_train,
            P_train,
            X_test,
            P_test,
            Y_train,
            Y_test,
            save_path=tmp_dir,
            scale_y=True,
        )
        Y_tr_sc = result[2]
        assert Y_tr_sc.shape == Y_train.shape
        assert np.allclose(Y_tr_sc.mean(axis=0), 0.0, atol=1e-5)
        assert np.allclose(Y_tr_sc.std(axis=0), 1.0, atol=1e-5)
        assert os.path.isfile(os.path.join(tmp_dir, "scaler_Y.pkl"))
        assert os.path.isfile(os.path.join(tmp_dir, "data_scaled_y.npz"))
        assert not os.path.isfile(os.path.join(tmp_dir, "data_scaled.npz"))


def test_scale_surrogate_dataset_from_np():
    N, n_features, n_params = 200, 5, 1
    X_train = np.random.randn(N, n_features).astype("float32")
    X_test = np.random.randn(50, n_features).astype("float32")
    X_val = np.random.randn(30, n_features).astype("float32")
    Y_train = np.random.randn(N, n_params).astype("float32")
    Y_test = np.random.randn(50, n_params).astype("float32")
    Y_val = np.random.randn(30, n_params).astype("float32")

    with tempfile.TemporaryDirectory() as tmp_dir:
        X_tr_sc, Y_tr_sc, X_te_sc, Y_te_sc, X_va_sc, Y_va_sc = (
            scale_surrogate_dataset_from_np(
                X_train,
                X_test,
                Y_train,
                Y_test,
                X_val=X_val,
                Y_val=Y_val,
                save_path=tmp_dir,
            )
        )
        assert os.path.isfile(os.path.join(tmp_dir, "scaler_surrogate_X.pkl"))
        assert os.path.isfile(
            os.path.join(tmp_dir, "data_surrogate_scaled.npz")
        )
        assert np.allclose(X_tr_sc.mean(axis=0), 0.0, atol=1e-5)
        assert np.allclose(X_tr_sc.std(axis=0), 1.0, atol=1e-5)
        assert np.allclose(Y_tr_sc, Y_train)
        assert X_va_sc.shape == X_val.shape

        # cache-hit: second call returns identical arrays including val
        X_tr_sc2, _, _, _, X_va_sc2, _ = scale_surrogate_dataset_from_np(
            X_train,
            X_test,
            Y_train,
            Y_test,
            X_val=X_val,
            Y_val=Y_val,
            save_path=tmp_dir,
        )
        assert np.allclose(X_tr_sc, X_tr_sc2)
        assert np.allclose(X_va_sc, X_va_sc2)

        # scale_y=True recomputes and reuses the SAME npz filename
        X_tr_sc_y, Y_tr_sc_y, _, _, _, _ = scale_surrogate_dataset_from_np(
            X_train, X_test, Y_train, Y_test, save_path=tmp_dir, scale_y=True
        )
        assert os.path.isfile(os.path.join(tmp_dir, "scaler_surrogate_Y.pkl"))

    assert np.allclose(Y_tr_sc_y.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(Y_tr_sc_y.std(axis=0), 1.0, atol=1e-5)


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
