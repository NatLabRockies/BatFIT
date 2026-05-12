import tempfile

import numpy as np
import torch
import torch.nn as nn

from batfit.utils.torch_utils import (
    get_device_type,
    get_num_parameters,
    load_model,
    make_dataset_from_np,
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
