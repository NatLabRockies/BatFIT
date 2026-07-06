import os
import pickle
import tempfile

import numpy as np

from batfit.utils.dataset_scaling import (
    scale_dataset_from_np,
    scale_protocol_dataset_from_np,
    scale_surrogate_dataset_from_np,
)
from batfit.utils.scalers import CustomScaler


def test_scale_dataset_from_np():
    N, n_chan, T, n_params = 40, 2, 30, 3
    X_train = np.random.randn(N, n_chan, T).astype("float32")
    X_test = np.random.randn(10, n_chan, T).astype("float32")
    Y_train = np.random.randn(N, n_params).astype("float32")
    Y_test = np.random.randn(10, n_params).astype("float32")

    with tempfile.TemporaryDirectory() as tmp_dir:
        X_tr_sc, Y_tr_sc, X_te_sc, Y_te_sc = scale_dataset_from_np(
            X_train, X_test, Y_train, Y_test, save_path=tmp_dir
        )
        assert os.path.isfile(os.path.join(tmp_dir, "scaler_X.pkl"))
        assert os.path.isfile(os.path.join(tmp_dir, "data_scaled.npz"))
        # Y is unscaled by default
        assert np.allclose(Y_tr_sc, Y_train)
        # X is z-scored per channel: mean ~0, std ~1 over (N, T)
        assert np.allclose(X_tr_sc.mean(axis=(0, 2)), 0.0, atol=1e-5)
        assert np.allclose(X_tr_sc.std(axis=(0, 2)), 1.0, atol=1e-5)

        # cache-hit: second call returns identical arrays
        X_tr_sc2, _, _, _ = scale_dataset_from_np(
            X_train, X_test, Y_train, Y_test, save_path=tmp_dir
        )
        assert np.allclose(X_tr_sc, X_tr_sc2)

    # X-scaler reuse: a pre-existing scaler_X.pkl is reused rather than re-fit
    with tempfile.TemporaryDirectory() as tmp_dir:
        wrong_scaler = CustomScaler(
            means=np.zeros((1, n_chan, 1), dtype="float32"),
            stds=np.ones((1, n_chan, 1), dtype="float32"),
        )
        with open(os.path.join(tmp_dir, "scaler_X.pkl"), "wb") as f:
            pickle.dump(wrong_scaler, f)
        X_tr_sc, _, _, _ = scale_dataset_from_np(
            X_train, X_test, Y_train, Y_test, save_path=tmp_dir
        )
        assert np.allclose(X_tr_sc, wrong_scaler.transform(X_train))

    # scale_y=True: separate cache file, Y is z-scored
    with tempfile.TemporaryDirectory() as tmp_dir:
        X_tr_sc, Y_tr_sc, X_te_sc, Y_te_sc = scale_dataset_from_np(
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
    # Protocol params with known range [0, 1] so MinMax scaling is identity
    P_train = np.random.rand(N, n_prot).astype("float32")
    P_test = np.random.rand(20, n_prot).astype("float32")
    Y_train = np.random.randn(N, n_params).astype("float32")
    Y_test = np.random.randn(20, n_params).astype("float32")

    with tempfile.TemporaryDirectory() as tmp_dir:
        X_tr_sc, P_tr_sc, Y_tr_sc, X_te_sc, P_te_sc, Y_te_sc = (
            scale_protocol_dataset_from_np(
                X_train,
                P_train,
                X_test,
                P_test,
                Y_train,
                Y_test,
                save_path=tmp_dir,
            )
        )
        # cache-hit: second call loads from data_scaled.npz
        X_tr_sc2, P_tr_sc2, Y_tr_sc2, X_te_sc2, P_te_sc2, Y_te_sc2 = (
            scale_protocol_dataset_from_np(
                X_train,
                P_train,
                X_test,
                P_test,
                Y_train,
                Y_test,
                save_path=tmp_dir,
            )
        )

    assert X_tr_sc.shape == X_train.shape
    assert P_tr_sc.shape == P_train.shape
    # Y is unscaled when scale_y=False
    assert np.allclose(Y_tr_sc, Y_train)
    assert np.allclose(Y_te_sc, Y_test)
    # scaler fit on P_train only: training values must be in [0, 1]
    assert P_tr_sc.min() >= -1e-5
    assert P_tr_sc.max() <= 1.0 + 1e-5
    # cache-hit returns identical arrays
    assert np.allclose(X_tr_sc, X_tr_sc2)
    assert np.allclose(P_tr_sc, P_tr_sc2)

    # scale_y=True: Y should be z-scored and scaler_Y.pkl should be saved
    with tempfile.TemporaryDirectory() as tmp_dir:
        X_tr_sc, P_tr_sc, Y_tr_sc, X_te_sc, P_te_sc, Y_te_sc = (
            scale_protocol_dataset_from_np(
                X_train,
                P_train,
                X_test,
                P_test,
                Y_train,
                Y_test,
                save_path=tmp_dir,
                scale_y=True,
            )
        )
        assert Y_tr_sc.shape == Y_train.shape
        assert np.allclose(Y_tr_sc.mean(axis=0), 0.0, atol=1e-5)
        assert np.allclose(Y_tr_sc.std(axis=0), 1.0, atol=1e-5)
        assert os.path.isfile(os.path.join(tmp_dir, "scaler_Y.pkl"))
        assert os.path.isfile(os.path.join(tmp_dir, "data_scaled_y.npz"))
        # scale_y=False cache is separate: data_scaled.npz should not exist
        assert not os.path.isfile(os.path.join(tmp_dir, "data_scaled.npz"))
        # cache-hit for scale_y=True
        X_tr_sc2, P_tr_sc2, Y_tr_sc2, X_te_sc2, P_te_sc2, Y_te_sc2 = (
            scale_protocol_dataset_from_np(
                X_train,
                P_train,
                X_test,
                P_test,
                Y_train,
                Y_test,
                save_path=tmp_dir,
                scale_y=True,
            )
        )
        assert np.allclose(Y_tr_sc, Y_tr_sc2)


def test_scale_surrogate_dataset_from_np():
    N, n_features, n_params = 200, 5, 1
    X_train = np.random.randn(N, n_features).astype("float32")
    X_test = np.random.randn(50, n_features).astype("float32")
    Y_train = np.random.randn(N, n_params).astype("float32")
    Y_test = np.random.randn(50, n_params).astype("float32")

    with tempfile.TemporaryDirectory() as tmp_dir:
        X_tr_sc, Y_tr_sc, X_te_sc, Y_te_sc = scale_surrogate_dataset_from_np(
            X_train, X_test, Y_train, Y_test, save_path=tmp_dir
        )
        assert os.path.isfile(os.path.join(tmp_dir, "scaler_surrogate_X.pkl"))
        assert os.path.isfile(
            os.path.join(tmp_dir, "data_surrogate_scaled.npz")
        )
        assert np.allclose(X_tr_sc.mean(axis=0), 0.0, atol=1e-5)
        assert np.allclose(X_tr_sc.std(axis=0), 1.0, atol=1e-5)
        assert np.allclose(Y_tr_sc, Y_train)

        # cache-hit: second call returns identical arrays
        X_tr_sc2, Y_tr_sc2, _, _ = scale_surrogate_dataset_from_np(
            X_train, X_test, Y_train, Y_test, save_path=tmp_dir
        )
        assert np.allclose(X_tr_sc, X_tr_sc2)

        # scale_y=True: not a cache-hit (scaler_surrogate_Y.pkl missing), so
        # it recomputes and reuses the SAME npz filename (unlike
        # scale_dataset_from_np, there is no data_surrogate_scaled_y.npz)
        X_tr_sc_y, Y_tr_sc_y, _, _ = scale_surrogate_dataset_from_np(
            X_train, X_test, Y_train, Y_test, save_path=tmp_dir, scale_y=True
        )
        assert os.path.isfile(os.path.join(tmp_dir, "scaler_surrogate_Y.pkl"))

    assert np.allclose(Y_tr_sc_y.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(Y_tr_sc_y.std(axis=0), 1.0, atol=1e-5)
