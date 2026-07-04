import tempfile

import numpy as np
import torch
import torch.nn as nn

from batfit.utils.torch_utils import (
    get_device_type,
    get_num_parameters,
    load_model,
    make_dataset_from_np,
    make_protocol_dataset_from_np,
    save_model,
)


def test_get_num_parameters():
    # nn.Linear(4, 8): 4*8 weights + 8 biases = 40
    assert get_num_parameters(nn.Linear(4, 8)) == 40
    # Sequential: (4*8+8) + (8*2+2) = 58
    assert (
        get_num_parameters(nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2)))
        == 58
    )


def test_get_device_type():
    assert get_device_type(enable_cuda=True, enable_mps=True) in [
        "cuda",
        "mps",
        "cpu",
    ]
    assert get_device_type(enable_cuda=False, enable_mps=False) == "cpu"


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


def test_save_load_model():
    model = nn.Linear(4, 8)
    initial_weight = model.weight.data.clone()

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_model(
            step=0,
            model=model,
            log_folder=tmp_dir,
            save_model_weights=True,
            save_model_obj=False,
            save_model_opt=False,
        )
        model.weight.data.fill_(0.0)
        model = load_model(model, state_dict_file=f"{tmp_dir}/model_0.pt")

    assert torch.allclose(model.weight.data.cpu(), initial_weight.cpu())
