import os
import tempfile

import numpy as np
import torch

from batfit.utils.torch_dataset_builder import (
    make_dataset_from_np,
    make_protocol_dataset_from_np,
    make_surrogate_dataset_from_np,
)


def test_make_dataset_from_np():
    # NPE dataset is 3D: (samples, channels, time_points)
    n_samples, n_channels, n_time, n_labels = 100, 2, 50, 4
    X = np.random.randn(n_samples, n_channels, n_time).astype("float32")
    Y = np.random.randn(n_samples, n_labels).astype("float32")
    batch_size = 16

    with tempfile.TemporaryDirectory() as tmp_dir:
        loaders = make_dataset_from_np(
            batch_size=batch_size,
            np_data=X,
            np_data_label=Y,
            scale=True,
            scale_y=False,
            save_path=tmp_dir,
        )
        # val_split=0 -> no validation loader
        with tempfile.TemporaryDirectory() as tmp_dir2:
            loaders_noval = make_dataset_from_np(
                batch_size=batch_size,
                np_data=X,
                np_data_label=Y,
                val_split=0.0,
                scale=True,
                save_path=tmp_dir2,
            )

    assert set(loaders) == {"train", "test", "val"}
    assert isinstance(loaders["train"], torch.utils.data.DataLoader)
    assert isinstance(loaders["test"], torch.utils.data.DataLoader)
    assert isinstance(loaders["val"], torch.utils.data.DataLoader)
    x_batch, y_batch = next(iter(loaders["train"]))
    assert x_batch.shape == (batch_size, n_channels, n_time)
    assert y_batch.shape == (batch_size, n_labels)
    # validation batch carries the same feature dims
    xv, yv = next(iter(loaders["val"]))
    assert xv.shape[1:] == (n_channels, n_time)
    assert yv.shape[1:] == (n_labels,)
    # val_split=0 disables the validation loader
    assert loaders_noval["val"] is None


def test_make_protocol_dataset_from_np():
    n_samples, n_chan, T, n_prot, n_labels = 100, 2, 50, 3, 6
    X = np.random.randn(n_samples, n_chan, T).astype("float32")
    P = np.random.rand(n_samples, n_prot).astype("float32")
    Y = np.random.randn(n_samples, n_labels).astype("float32")
    batch_size = 16

    # scale_y=False: Y labels are unscaled
    with tempfile.TemporaryDirectory() as tmp_dir:
        loaders = make_protocol_dataset_from_np(
            batch_size=batch_size,
            np_data=X,
            np_prot_params=P,
            np_data_label=Y,
            scale=True,
            save_path=tmp_dir,
        )

    assert set(loaders) == {"train", "test", "val"}
    assert isinstance(loaders["train"], torch.utils.data.DataLoader)
    assert isinstance(loaders["val"], torch.utils.data.DataLoader)
    # each batch has three tensors: (X_signal, prot_params, Y_labels)
    x_batch, p_batch, y_batch = next(iter(loaders["train"]))
    assert x_batch.shape == (batch_size, n_chan, T)
    assert p_batch.shape == (batch_size, n_prot)
    assert y_batch.shape == (batch_size, n_labels)
    # P is MinMax-scaled to [0, 1]
    assert p_batch.min().item() >= 0.0
    assert p_batch.max().item() <= 1.0
    # validation loader yields the same three-tensor batches
    xv, pv, yv = next(iter(loaders["val"]))
    assert pv.shape[1:] == (n_prot,)
    assert yv.shape[1:] == (n_labels,)

    # scale_y=True: Y labels are z-scored; verify over all train batches
    with tempfile.TemporaryDirectory() as tmp_dir:
        loaders_sy = make_protocol_dataset_from_np(
            batch_size=16,
            np_data=X,
            np_prot_params=P,
            np_data_label=Y,
            scale=True,
            scale_y=True,
            save_path=tmp_dir,
        )
    all_y = torch.cat([yb for _, _, yb in loaders_sy["train"]], dim=0)
    assert all_y.shape[-1] == n_labels
    assert all_y.mean(dim=0).abs().max().item() < 0.2
    assert (all_y.std(dim=0) - 1.0).abs().max().item() < 0.2


def test_make_surrogate_dataset_from_np():
    # whole-curve input: (N batteries, 2 channels [time, voltage], n_points)
    N, n_deg, T = 100, 3, 10
    X = np.random.randn(N, 2, T).astype("float32")
    Y = np.random.randn(N, n_deg).astype("float32")
    batch_size = 8

    with tempfile.TemporaryDirectory() as tmp_dir:
        loaders = make_surrogate_dataset_from_np(
            batch_size=batch_size,
            np_data=X,
            np_data_label=Y,
            scale=True,
            scale_y=False,
            save_path=tmp_dir,
        )
        # split-then-slice creates both the battery split and the surrogate one
        assert os.path.isfile(os.path.join(tmp_dir, "data_split.npz"))
        assert os.path.isfile(
            os.path.join(tmp_dir, "data_surrogate_split.npz")
        )
        assert set(loaders) == {"train", "test", "val"}
        assert loaders["val"] is not None
        # surrogate rows are (time, *n_deg params) -> voltage
        x_batch, y_batch = next(iter(loaders["train"]))
        assert x_batch.shape == (batch_size, n_deg + 1)
        assert y_batch.shape == (batch_size, 1)
        xv, yv = next(iter(loaders["val"]))
        assert xv.shape[1:] == (n_deg + 1,)

    # reuse an existing battery-level NPE data_split.npz (with a val slice):
    # the surrogate must match those exact batteries, then explode into rows
    with tempfile.TemporaryDirectory() as tmp_dir:
        X_npe_train = np.random.randn(80, 2, T).astype("float32")
        X_npe_test = np.random.randn(10, 2, T).astype("float32")
        X_npe_val = np.random.randn(10, 2, T).astype("float32")
        Y_npe_train = np.random.randn(80, n_deg).astype("float32")
        Y_npe_test = np.random.randn(10, n_deg).astype("float32")
        Y_npe_val = np.random.randn(10, n_deg).astype("float32")
        np.savez(
            os.path.join(tmp_dir, "data_split.npz"),
            X_train=X_npe_train,
            Y_train=Y_npe_train,
            X_test=X_npe_test,
            Y_test=Y_npe_test,
            X_val=X_npe_val,
            Y_val=Y_npe_val,
        )
        loaders = make_surrogate_dataset_from_np(
            batch_size=4,
            np_data=X,
            np_data_label=Y,
            scale=False,
            save_path=tmp_dir,
        )
        # test rows = 10 NPE test batteries * T timesteps
        surr = np.load(os.path.join(tmp_dir, "data_surrogate_split.npz"))
        assert surr["X_test"].shape == (10 * T, n_deg + 1)
        assert surr["X_val"].shape == (10 * T, n_deg + 1)
        assert loaders["val"] is not None
