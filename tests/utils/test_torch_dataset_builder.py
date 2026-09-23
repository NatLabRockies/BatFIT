import os
import tempfile

import numpy as np
import torch

from batfit.utils.torch_dataset_builder import (
    make_dataset_from_np,
    make_npe_dataset_from_np,
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
    N, T = 100, 10
    rng = np.random.default_rng(0)
    time = np.tile(np.linspace(0.0, 3600.0, T), (N, 1))
    voltage = rng.uniform(3.0, 4.2, (N, T))
    X = np.stack([time, voltage], axis=1).astype("float32")
    sim_params = {
        "deg_param_names": ["i0_a", "ds_c", "x0_a"],
        "deg_i0_a_min": 0.5,
        "deg_i0_a_max": 1.5,
        "deg_ds_c_min": 0.2,
        "deg_ds_c_max": 10.0,
        "deg_x0_a_min": 0.6,
        "deg_x0_a_max": 1.0,
        "vmin": 3.0,
        "vmax": 4.2,
    }
    n_deg = 3
    low = np.array([0.5, 0.2, 0.6])
    high = np.array([1.5, 10.0, 1.0])
    Y = rng.uniform(low, high, (N, n_deg)).astype("float32")
    batch_size = 8

    with tempfile.TemporaryDirectory() as tmp_dir:
        loaders, scalers = make_surrogate_dataset_from_np(
            sim_params,
            np_data=X,
            np_data_label=Y,
            batch_size=batch_size,
            save_path=tmp_dir,
            random_state=0,
        )
        # split-then-slice creates both the battery split and the surrogate
        # one; the scaled data is not written to disk
        assert sorted(os.listdir(tmp_dir)) == [
            "data_split.npz",
            "data_surrogate_split.npz",
        ]
        assert set(loaders) == {"train", "test", "val"}
        assert set(scalers) == {"t", "Y", "V"}
        # surrogate rows are (time, *n_deg params) -> voltage
        x_batch, y_batch = next(iter(loaders["train"]))
        assert x_batch.shape == (batch_size, n_deg + 1)
        assert y_batch.shape == (batch_size, 1)
        # parameters and voltage scaled to [0, 1] from the config bounds
        x_val = torch.cat([b[0] for b in loaders["val"]])
        y_val = torch.cat([b[1] for b in loaders["val"]])
        assert torch.all((x_val[:, 1:] >= 0.0) & (x_val[:, 1:] <= 1.0))
        assert torch.all((y_val >= 0.0) & (y_val <= 1.0))
        # the cached surrogate split stays unscaled (physical)
        surr = np.load(os.path.join(tmp_dir, "data_surrogate_split.npz"))
        assert surr["Y_val"].min() >= 3.0

    # reuse an existing battery-level data_split.npz (with a val slice): the
    # surrogate must match those exact batteries, then explode into rows
    with tempfile.TemporaryDirectory() as tmp_dir:
        np.savez(
            os.path.join(tmp_dir, "data_split.npz"),
            X_train=X[:80],
            Y_train=Y[:80],
            X_test=X[80:90],
            Y_test=Y[80:90],
            X_val=X[90:],
            Y_val=Y[90:],
        )
        loaders, _ = make_surrogate_dataset_from_np(
            sim_params,
            np_data=X,
            np_data_label=Y,
            batch_size=4,
            save_path=tmp_dir,
        )
        # test rows = 10 test batteries * T timesteps
        surr = np.load(os.path.join(tmp_dir, "data_surrogate_split.npz"))
        assert surr["X_test"].shape == (10 * T, n_deg + 1)
        assert surr["X_val"].shape == (10 * T, n_deg + 1)
        assert loaders["val"] is not None


def test_make_npe_dataset_from_np():
    n_samples, n_chan, T, n_prot = 100, 2, 50, 1
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_samples, n_chan, T)).astype("float32")
    # labels drawn inside the configured bounds
    Y = np.stack(
        [rng.uniform(0.5, 1.5, n_samples), rng.uniform(10.0, 30.0, n_samples)],
        axis=1,
    ).astype("float32")
    P = rng.uniform(0.0, 10.0, (n_samples, n_prot)).astype("float32")
    sim_params = {
        "deg_param_names": ["i0_a", "ds_c"],
        "deg_i0_a_min": 0.5,
        "deg_i0_a_max": 1.5,
        "deg_ds_c_min": 10.0,
        "deg_ds_c_max": 30.0,
    }
    sim_params_prot = {
        **sim_params,
        "prot_param_names": ["amplitude"],
        "prot_amplitude_min": 0.0,
        "prot_amplitude_max": 10.0,
    }
    batch_size = 16

    with tempfile.TemporaryDirectory() as tmp_dir:
        loaders, scalers = make_npe_dataset_from_np(
            sim_params,
            np_data=X,
            np_data_label=Y,
            batch_size=batch_size,
            save_path=tmp_dir,
            random_state=0,
        )
        assert os.path.isfile(os.path.join(tmp_dir, "data_split.npz"))
        # the scaled data is not written to disk
        assert os.listdir(tmp_dir) == ["data_split.npz"]

    # plain NPE: (X, Y) batches, Y scaled to [0, 1] from the bounds
    assert set(scalers) == {"X", "Y"}
    x_batch, y_batch = next(iter(loaders["train"]))
    assert x_batch.shape == (batch_size, n_chan, T)
    assert y_batch.shape == (batch_size, 2)
    y_all = torch.cat([b[1] for b in loaders["val"]])
    assert torch.all((y_all >= 0.0) & (y_all <= 1.0))

    with tempfile.TemporaryDirectory() as tmp_dir:
        loaders_p, scalers_p = make_npe_dataset_from_np(
            sim_params_prot,
            np_data=X,
            np_data_label=Y,
            np_prot_params=P,
            batch_size=batch_size,
            val_split=0.0,
            save_path=tmp_dir,
            random_state=0,
        )

    # protocol NPE: (X, P, Y) batches, P scaled to [0, 1]; no val loader
    assert set(scalers_p) == {"X", "Y", "P"}
    x_batch, p_batch, y_batch = next(iter(loaders_p["train"]))
    assert p_batch.shape == (batch_size, n_prot)
    assert torch.all((p_batch >= 0.0) & (p_batch <= 1.0))
    assert loaders_p["val"] is None
