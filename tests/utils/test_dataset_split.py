import os
import tempfile

import numpy as np

from batfit.utils.dataset_split import (
    split_arrays,
    split_dataset_from_np,
    split_protocol_dataset_from_np,
    split_surrogate_dataset_from_np,
)


def test_split_arrays():
    N, n_chan, T, n_params = 100, 2, 50, 4
    X = np.random.randn(N, n_chan, T).astype("float32")
    Y = np.random.randn(N, n_params).astype("float32")
    test_split = 0.2

    with tempfile.TemporaryDirectory() as tmp_dir:
        result = split_arrays(
            {"X": X, "Y": Y},
            test_split=test_split,
            save_path=tmp_dir,
            cache_filename="my_split.npz",
        )
        # cache-hit: second call returns from my_split.npz without needing arrays
        result2 = split_arrays(
            {"X": None, "Y": None},
            test_split=test_split,
            save_path=tmp_dir,
            cache_filename="my_split.npz",
        )

    n_test = int(N * test_split)
    assert result["X_train"].shape[0] + result["X_test"].shape[0] == N
    assert result["X_test"].shape[0] == n_test
    assert result["Y_train"].shape == (N - n_test, n_params)
    assert np.allclose(result["X_train"], result2["X_train"])


def test_split_dataset_from_np():
    N, n_chan, T, n_params = 100, 2, 50, 4
    X = np.random.randn(N, n_chan, T).astype("float32")
    Y = np.random.randn(N, n_params).astype("float32")
    test_split = 0.1
    with tempfile.TemporaryDirectory() as tmp_dir:
        X_train, Y_train, X_test, Y_test = split_dataset_from_np(
            X, Y, test_split=test_split, save_path=tmp_dir
        )
        assert os.path.isfile(os.path.join(tmp_dir, "data_split.npz"))
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
        # cache-hit: second call returns from data_split.npz
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


def test_split_surrogate_dataset_from_np():
    N, n_params_plus_one, n_params = 200, 5, 1
    X = np.random.randn(N, n_params_plus_one).astype("float32")
    Y = np.random.randn(N, n_params).astype("float32")
    test_split = 0.1
    with tempfile.TemporaryDirectory() as tmp_dir:
        X_train, Y_train, X_test, Y_test = split_surrogate_dataset_from_np(
            X, Y, test_split=test_split, save_path=tmp_dir
        )
        assert os.path.isfile(
            os.path.join(tmp_dir, "data_surrogate_split.npz")
        )
        # cache-hit: second call returns from data_surrogate_split.npz
        X_train2, Y_train2, X_test2, Y_test2 = split_surrogate_dataset_from_np(
            None, None, test_split=test_split, save_path=tmp_dir
        )
    assert X_train.shape[0] + X_test.shape[0] == N
    assert X_test.shape[0] == int(N * test_split)
    assert np.allclose(X_train, X_train2)
