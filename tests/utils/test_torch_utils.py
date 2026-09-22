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
    read_restart_position,
    restart_requested,
    save_model,
    update_best_model,
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
        # model_best.pt present -> preferred, no test_loss.csv scan needed
        open(os.path.join(tmp_dir, "model_best.pt"), "w").close()
        open(os.path.join(tmp_dir, "model_final.pt"), "w").close()
        best = find_best_model_file(tmp_dir)
        assert best == os.path.join(tmp_dir, "model_best.pt")

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


def test_update_best_model():
    model = nn.Linear(4, 8)
    with tempfile.TemporaryDirectory() as tmp_dir:
        best_path = os.path.join(tmp_dir, "model_best.pt")

        # Improving loss (inf -> 0.5) writes model_best.pt and returns new best
        best = update_best_model(
            test_loss=0.5,
            best_test_loss=float("inf"),
            model=model,
            device_type="cpu",
            log_folder=tmp_dir,
        )
        assert best == 0.5
        assert os.path.isfile(best_path)

        # Non-improving loss leaves the return value and file mtime unchanged
        mtime = os.path.getmtime(best_path)
        best = update_best_model(
            test_loss=0.7,
            best_test_loss=best,
            model=model,
            device_type="cpu",
            log_folder=tmp_dir,
        )
        assert best == 0.5
        assert os.path.getmtime(best_path) == mtime

        # A tensor loss that improves is accepted and coerced to float
        best = update_best_model(
            test_loss=torch.tensor(0.1),
            best_test_loss=best,
            model=model,
            device_type="cpu",
            log_folder=tmp_dir,
        )
        assert best == float(torch.tensor(0.1))


def test_read_restart_position():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # No CSV -> start from scratch
        assert read_restart_position(tmp_dir) == (0, 0)

        # One row per epoch: 3 rows -> next epoch 3, last step 300
        with open(os.path.join(tmp_dir, "test_loss.csv"), "w") as f:
            f.write("step;loss\n100;1.0\n200;0.5\n300;0.7\n")
        assert read_restart_position(tmp_dir) == (3, 300)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Header only (no completed epochs) -> start from scratch
        with open(os.path.join(tmp_dir, "test_loss.csv"), "w") as f:
            f.write("step;loss\n")
        assert read_restart_position(tmp_dir) == (0, 0)


def test_restart_requested():
    # None -> no restart
    assert restart_requested(None) is False

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Existing file -> restart requested
        ckpt = os.path.join(tmp_dir, "model_best.pt")
        open(ckpt, "w").close()
        assert restart_requested(ckpt) is True

        # Missing file -> warn and fall back to scratch (False)
        missing = os.path.join(tmp_dir, "does_not_exist.pt")
        assert restart_requested(missing) is False


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
