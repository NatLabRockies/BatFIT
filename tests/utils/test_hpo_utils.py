import os
import tempfile

import optuna
import pytest
import torch
import yaml

from batfit.utils.hpo_utils import (
    create_or_load_study,
    make_trial_loader,
    min_test_loss,
    suggest_from_space,
    write_recipe,
)


def test_suggest_from_space():
    search_space = {
        "num_channels": {"type": "int", "low": 64, "high": 2048},
        "batch_size": {"type": "int", "low": 64, "high": 1024, "log": True},
        "lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    }

    # fixed trial: the drawn values are the fixed ones
    fixed_values = {"num_channels": 128, "batch_size": 256, "lr": 1e-3}
    values = suggest_from_space(
        optuna.trial.FixedTrial(fixed_values), search_space
    )
    assert values == fixed_values

    # real trial: values in bounds, with the declared type and log flag
    study = optuna.create_study()
    trial = study.ask()
    values = suggest_from_space(trial, search_space)
    assert isinstance(values["num_channels"], int)
    assert 64 <= values["num_channels"] <= 2048
    assert 64 <= values["batch_size"] <= 1024
    assert 1e-5 <= values["lr"] <= 1e-2
    assert trial.distributions["lr"].log
    assert not trial.distributions["num_channels"].log

    # unknown type
    with pytest.raises(ValueError):
        suggest_from_space(
            optuna.trial.FixedTrial({"x": 1}),
            {"x": {"type": "categorical", "low": 0, "high": 1}},
        )


def test_create_or_load_study():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = os.path.join(tmpdir, "hpo", "study.db")
        study = create_or_load_study(storage_path, "npe", seed=0)
        study.optimize(lambda trial: trial.suggest_float("x", 0, 1), 2)
        assert os.path.isfile(storage_path)
        assert study.direction == optuna.study.StudyDirection.MINIMIZE
        assert isinstance(study.pruner, optuna.pruners.MedianPruner)

        # reloading resumes the same study
        study_resumed = create_or_load_study(storage_path, "npe", seed=0)
        assert len(study_resumed.trials) == 2


def test_min_test_loss():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "test_loss.csv"), "w") as f:
            f.write("step;loss\n1;0.5\n2;0.2\n3;0.3\n")
        assert min_test_loss(tmpdir) == 0.2

        # a single logged epoch
        with open(os.path.join(tmpdir, "test_loss.csv"), "w") as f:
            f.write("step;loss\n1;0.7\n")
        assert min_test_loss(tmpdir) == 0.7


def test_write_recipe():
    with tempfile.TemporaryDirectory() as tmpdir:
        recipe_path = os.path.join(tmpdir, "recipe.yml")
        with open(recipe_path, "w") as f:
            yaml.safe_dump({"num_channels": 128, "lr": 0.001, "epochs": 20}, f)
        out_path = os.path.join(tmpdir, "hpo", "best_recipe.yml")
        write_recipe(recipe_path, {"num_channels": 512, "lr": 1e-4}, out_path)
        with open(out_path) as f:
            recipe = yaml.safe_load(f)
        assert recipe == {"num_channels": 512, "lr": 1e-4, "epochs": 20}


def test_make_trial_loader():
    n_samples = 10
    dataset = torch.utils.data.TensorDataset(
        torch.rand(n_samples, 2, 8), torch.rand(n_samples, 6)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=4)

    trial_loader = make_trial_loader(
        loader, batch_size=3, shuffle=True, drop_last=True
    )
    assert trial_loader.dataset is dataset
    assert trial_loader.batch_size == 3
    assert len(trial_loader) == n_samples // 3

    # batch size clamped to the dataset size
    trial_loader = make_trial_loader(
        loader, batch_size=64, shuffle=False, drop_last=True
    )
    assert trial_loader.batch_size == n_samples
    assert len(trial_loader) == 1
