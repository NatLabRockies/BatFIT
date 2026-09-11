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
        train_loader, test_loader = make_dataset_from_np(
            batch_size=batch_size,
            np_data=X,
            np_data_label=Y,
            scale=True,
            scale_y=False,
            save_path=tmp_dir,
        )

    assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert isinstance(test_loader, torch.utils.data.DataLoader)
    x_batch, y_batch = next(iter(train_loader))
    assert x_batch.shape == (batch_size, n_channels, n_time)
    assert y_batch.shape == (batch_size, n_labels)


def test_make_protocol_dataset_from_np():
    n_samples, n_chan, T, n_prot, n_labels = 100, 2, 50, 3, 6
    X = np.random.randn(n_samples, n_chan, T).astype("float32")
    P = np.random.rand(n_samples, n_prot).astype("float32")
    Y = np.random.randn(n_samples, n_labels).astype("float32")
    batch_size = 16

    # scale_y=False: Y labels are unscaled
    with tempfile.TemporaryDirectory() as tmp_dir:
        train_loader, test_loader = make_protocol_dataset_from_np(
            batch_size=batch_size,
            np_data=X,
            np_prot_params=P,
            np_data_label=Y,
            scale=True,
            save_path=tmp_dir,
        )

    assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert isinstance(test_loader, torch.utils.data.DataLoader)
    # each batch has three tensors: (X_signal, prot_params, Y_labels)
    x_batch, p_batch, y_batch = next(iter(train_loader))
    assert x_batch.shape == (batch_size, n_chan, T)
    assert p_batch.shape == (batch_size, n_prot)
    assert y_batch.shape == (batch_size, n_labels)
    # P is MinMax-scaled to [0, 1]
    assert p_batch.min().item() >= 0.0
    assert p_batch.max().item() <= 1.0

    # scale_y=True: Y labels are z-scored; verify by concatenating all train batches
    with tempfile.TemporaryDirectory() as tmp_dir:
        train_loader_sy, _ = make_protocol_dataset_from_np(
            batch_size=16,
            np_data=X,
            np_prot_params=P,
            np_data_label=Y,
            scale=True,
            scale_y=True,
            save_path=tmp_dir,
        )
    all_y = torch.cat([yb for _, _, yb in train_loader_sy], dim=0)
    assert all_y.shape[-1] == n_labels
    # train Y should be approximately z-scored across the training set
    assert all_y.mean(dim=0).abs().max().item() < 0.2
    assert (all_y.std(dim=0) - 1.0).abs().max().item() < 0.2


def test_make_surrogate_dataset_from_np():
    N, n_features, n_params = 60, 4, 1
    X = np.random.randn(N, n_features).astype("float32")
    Y = np.random.randn(N, n_params).astype("float32")
    batch_size = 8

    with tempfile.TemporaryDirectory() as tmp_dir:
        train_loader, test_loader = make_surrogate_dataset_from_np(
            batch_size=batch_size,
            np_data=X,
            np_data_label=Y,
            scale=True,
            scale_y=False,
            save_path=tmp_dir,
        )
        assert os.path.isfile(
            os.path.join(tmp_dir, "data_surrogate_split.npz")
        )

        x_batch, y_batch = next(iter(train_loader))
        assert x_batch.shape == (batch_size, n_features)
        assert y_batch.shape == (batch_size, n_params)

    # "matching NPE split": reuse an existing NPE data_split.npz rather than
    # np_data/np_data_label (which are only there to satisfy the not-None check)
    with tempfile.TemporaryDirectory() as tmp_dir:
        n_deg, T = 3, 10
        X_npe_train = np.random.randn(40, 2, T).astype("float32")
        X_npe_test = np.random.randn(10, 2, T).astype("float32")
        Y_npe_train = np.random.randn(40, n_deg).astype("float32")
        Y_npe_test = np.random.randn(10, n_deg).astype("float32")
        np.savez(
            os.path.join(tmp_dir, "data_split.npz"),
            X_train=X_npe_train,
            Y_train=Y_npe_train,
            X_test=X_npe_test,
            Y_test=Y_npe_test,
        )
        train_loader, test_loader = make_surrogate_dataset_from_np(
            batch_size=4,
            np_data=X,
            np_data_label=Y,
            scale=False,
            save_path=tmp_dir,
        )
        assert os.path.isfile(
            os.path.join(tmp_dir, "data_surrogate_split.npz")
        )

        x_batch, y_batch = next(iter(train_loader))
        # surrogate rows are (time, *n_deg params) -> voltage
        assert x_batch.shape == (4, n_deg + 1)
        assert y_batch.shape == (4, 1)
