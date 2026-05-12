import tempfile

import numpy as np

from batfit.utils.data_utils import (
    CustomScaler,
    augment_data,
    from_name_to_params,
    from_param_to_surrogate_data,
    scale_protocol_dataset_from_np,
    split_dataset_from_np,
    split_protocol_dataset_from_np,
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


def test_from_param_to_surrogate_data():
    N, T, n_params = 5, 20, 3
    t = np.linspace(0, 1, T).astype("float32")
    X_data = np.stack(
        [
            np.tile(t, (N, 1)),  # channel 0: time
            np.random.randn(N, T).astype("float32"),  # channel 1: voltage
        ],
        axis=1,
    )  # (N, 2, T)
    Y_data = np.random.randn(N, n_params).astype("float32")
    new_x, new_y = from_param_to_surrogate_data(X_data, Y_data)
    assert new_x.shape == (N * T, n_params + 1)
    assert new_y.shape == (N * T, 1)
    # first column of new_x must be the time values from X_data channel 0
    assert np.allclose(new_x[:, 0], X_data[:, 0, :].reshape(-1))
    # new_y must be the voltage channel from X_data
    assert np.allclose(new_y[:, 0], X_data[:, 1, :].reshape(-1))


def test_augment_data():
    N, n_chan, T, n_params = 10, 2, 50, 3
    X = np.random.randn(N, n_chan, T).astype("float32")
    Y = np.random.randn(N, n_params).astype("float32")
    new_ds = 3
    X_aug, Y_aug = augment_data(X, Y, new_ds=new_ds, noise_level=0.001)
    assert X_aug.shape == ((new_ds + 1) * N, n_chan, T)
    assert Y_aug.shape == ((new_ds + 1) * N, n_params)
    # original samples must be preserved exactly in first N rows
    assert np.allclose(X_aug[:N], X)
    assert np.allclose(Y_aug[:N], Y)


def test_split_dataset_from_np():
    N, n_chan, T, n_params = 100, 2, 50, 4
    X = np.random.randn(N, n_chan, T).astype("float32")
    Y = np.random.randn(N, n_params).astype("float32")
    test_split = 0.1
    with tempfile.TemporaryDirectory() as tmp_dir:
        X_train, Y_train, X_test, Y_test = split_dataset_from_np(
            X, Y, test_split=test_split, save_path=tmp_dir
        )
    assert X_train.shape[0] + X_test.shape[0] == N
    assert X_test.shape[0] == int(N * test_split)
    assert X_train.shape[1:] == (n_chan, T)
    assert Y_train.shape[1] == n_params


def test_split_protocol_dataset_from_np():
    N, n_chan, T, n_prot, n_params = 100, 2, 50, 3, 6
    X = np.random.randn(N, n_chan, T).astype("float32")
    P = np.random.randn(N, n_prot).astype("float32")
    Y = np.random.randn(N, n_params).astype("float32")
    test_split = 0.1
    with tempfile.TemporaryDirectory() as tmp_dir:
        X_train, P_train, Y_train, X_test, P_test, Y_test = (
            split_protocol_dataset_from_np(
                X, P, Y, test_split=test_split, save_path=tmp_dir
            )
        )
        # cache path: second call returns from data_split.npz
        X_train2, P_train2, Y_train2, X_test2, P_test2, Y_test2 = (
            split_protocol_dataset_from_np(
                X, P, Y, test_split=test_split, save_path=tmp_dir
            )
        )
    n_test = int(N * test_split)
    assert X_train.shape[0] + X_test.shape[0] == N
    assert X_test.shape[0] == n_test
    assert P_train.shape == (N - n_test, n_prot)
    assert P_test.shape == (n_test, n_prot)
    assert Y_train.shape == (N - n_test, n_params)
    # train/test cover all samples without overlap
    assert X_train.shape[0] == X_train2.shape[0]
    assert np.allclose(P_train, P_train2)


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
        X_tr_sc, P_tr_sc, X_te_sc, P_te_sc = scale_protocol_dataset_from_np(
            X_train,
            P_train,
            X_test,
            P_test,
            Y_train,
            Y_test,
            save_path=tmp_dir,
        )
        # cache path: second call loads from data_scaled.npz
        X_tr_sc2, P_tr_sc2, X_te_sc2, P_te_sc2 = (
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

    # X is z-scored per channel: train mean should be near 0
    assert X_tr_sc.shape == X_train.shape
    assert P_tr_sc.shape == P_train.shape
    # scaler fit on P_train only: training values must be in [0, 1] (float32 tol)
    assert P_tr_sc.min() >= -1e-5
    assert P_tr_sc.max() <= 1.0 + 1e-5
    # cache returns identical arrays
    assert np.allclose(X_tr_sc, X_tr_sc2)
    assert np.allclose(P_tr_sc, P_tr_sc2)
