import os
import pickle
import tempfile

import torch
import torch.nn as nn

from batfit.utils.torch_utils import (
    find_best_model_file,
    get_device_type,
    get_num_parameters,
    load_frozen_model,
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


def test_find_best_model_file():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Best loss at an intermediate iteration -> closest checkpoint wins
        with open(os.path.join(tmp_dir, "test_loss.csv"), "w") as f:
            f.write("iter;loss\n100;1.0\n200;0.5\n300;0.7\n")
        for fname in [
            "model_100.pt",
            "model_200.pt",
            "model_300.pt",
            "model_final.pt",
        ]:
            open(os.path.join(tmp_dir, fname), "w").close()
        best = find_best_model_file(tmp_dir)
        assert best == os.path.join(tmp_dir, "model_200.pt")

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Best loss at the last iteration -> model_final.pt
        with open(os.path.join(tmp_dir, "test_loss.csv"), "w") as f:
            f.write("iter;loss\n100;1.0\n200;0.5\n")
        open(os.path.join(tmp_dir, "model_final.pt"), "w").close()
        best = find_best_model_file(tmp_dir)
        assert best == os.path.join(tmp_dir, "model_final.pt")

    with tempfile.TemporaryDirectory() as tmp_dir:
        # No intermediate checkpoints -> fall back to model_final.pt
        with open(os.path.join(tmp_dir, "test_loss.csv"), "w") as f:
            f.write("iter;loss\n100;0.5\n200;1.0\n")
        open(os.path.join(tmp_dir, "model_final.pt"), "w").close()
        best = find_best_model_file(tmp_dir)
        assert best == os.path.join(tmp_dir, "model_final.pt")


def test_load_frozen_model():
    model = nn.Linear(4, 8)
    with tempfile.TemporaryDirectory() as tmp_dir:
        with open(os.path.join(tmp_dir, "model.pkl"), "wb") as f:
            pickle.dump(model, f)
        torch.save(model.state_dict(), os.path.join(tmp_dir, "model_final.pt"))
        with open(os.path.join(tmp_dir, "test_loss.csv"), "w") as f:
            f.write("iter;loss\n100;0.5\n")

        loaded = load_frozen_model(tmp_dir, torch.device("cpu"))

    assert not loaded.training  # eval mode
    assert torch.allclose(loaded.weight.data.cpu(), model.weight.data.cpu())
