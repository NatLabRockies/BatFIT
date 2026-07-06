import tempfile

import torch
import torch.nn as nn

from batfit.utils.torch_utils import (
    get_device_type,
    get_num_parameters,
    load_model,
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
