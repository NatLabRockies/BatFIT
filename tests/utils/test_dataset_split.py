import os
import tempfile

import numpy as np

from batfit.utils.dataset_split import (
    split_arrays,
    split_dataset_from_np,
    split_protocol_dataset_from_np,
)


def test_split_arrays():
    N, n_chan, T, n_params = 100, 2, 50, 4
    X = np.random.randn(N, n_chan, T).astype("float32")
    Y = np.random.randn(N, n_params).astype("float32")
    test_split, val_split = 0.1, 0.1

    with tempfile.TemporaryDirectory() as tmp_dir:
        result = split_arrays(
            {"X": X, "Y": Y},
            test_split=test_split,
            val_split=val_split,
            save_path=tmp_dir,
            cache_filename="my_split.npz",
            random_state=0,
        )
        # cache-hit: second call returns from my_split.npz without arrays
        result2 = split_arrays(
            {"X": None, "Y": None},
            test_split=test_split,
            val_split=val_split,
            save_path=tmp_dir,
            cache_filename="my_split.npz",
        )
        # val_split=0 -> two-way fallback (no _val keys)
        two_way = split_arrays(
            {"X": X, "Y": Y},
            test_split=test_split,
            val_split=0.0,
            save_path=tmp_dir,
            cache_filename="two_way.npz",
        )

    # three-way split fractions sum to N and match 80/10/10
    n_total = (
        result["X_train"].shape[0]
        + result["X_test"].shape[0]
        + result["X_val"].shape[0]
    )
    assert n_total == N
    assert result["X_test"].shape[0] == int(N * test_split)
    assert result["X_val"].shape[0] == int(N * val_split)
    assert result["Y_val"].shape == (int(N * val_split), n_params)
    # cache-hit returns identical arrays
    assert np.allclose(result["X_train"], result2["X_train"])
    assert np.allclose(result["X_val"], result2["X_val"])
    # two-way fallback has no validation slice
    assert "X_val" not in two_way
    assert two_way["X_train"].shape[0] + two_way["X_test"].shape[0] == N


def test_split_dataset_from_np():
    N, n_chan, T, n_params = 100, 2, 50, 4
    X = np.random.randn(N, n_chan, T).astype("float32")
    Y = np.random.randn(N, n_params).astype("float32")
    test_split, val_split = 0.1, 0.1
    with tempfile.TemporaryDirectory() as tmp_dir:
        X_train, Y_train, X_test, Y_test, X_val, Y_val = split_dataset_from_np(
            X,
            Y,
            test_split=test_split,
            val_split=val_split,
            save_path=tmp_dir,
        )
        assert os.path.isfile(os.path.join(tmp_dir, "data_split.npz"))
    assert X_train.shape[0] + X_test.shape[0] + X_val.shape[0] == N
    assert X_test.shape[0] == int(N * test_split)
    assert X_val.shape[0] == int(N * val_split)
    assert X_train.shape[1:] == (n_chan, T)
    assert X_val.shape[1:] == (n_chan, T)
    assert Y_val.shape[1] == n_params


def test_split_protocol_dataset_from_np():
    N, n_chan, T, n_prot, n_params = 100, 2, 50, 3, 6
    X = np.random.randn(N, n_chan, T).astype("float32")
    P = np.random.randn(N, n_prot).astype("float32")
    Y = np.random.randn(N, n_params).astype("float32")
    test_split, val_split = 0.1, 0.1
    with tempfile.TemporaryDirectory() as tmp_dir:
        (
            X_train,
            P_train,
            Y_train,
            X_test,
            P_test,
            Y_test,
            X_val,
            P_val,
            Y_val,
        ) = split_protocol_dataset_from_np(
            X,
            P,
            Y,
            test_split=test_split,
            val_split=val_split,
            save_path=tmp_dir,
        )
        # cache-hit: second call returns from data_split.npz
        cached = split_protocol_dataset_from_np(
            X,
            P,
            Y,
            test_split=test_split,
            val_split=val_split,
            save_path=tmp_dir,
        )
    n_val = int(N * val_split)
    assert X_train.shape[0] + X_test.shape[0] + X_val.shape[0] == N
    assert X_test.shape[0] == int(N * test_split)
    assert P_val.shape == (n_val, n_prot)
    assert Y_val.shape == (n_val, n_params)
    # cache-hit returns identical val arrays (index 6 == X_val)
    assert np.allclose(X_val, cached[6])
