"""HPO helpers."""

import os

import numpy as np
import optuna
import torch
import yaml


def suggest_from_space(
    trial: optuna.trial.BaseTrial, search_space: dict[str, dict]
) -> dict[str, int | float]:
    """Draw one hyperparameter sample from search space dict

    Parameters
    ----------
    trial: optuna.trial.BaseTrial
        Optuna trial that draws the values
    search_space: dict[str, dict]
        Dictionary of hyperparameters with bounds, type,
        and a bool that says if it is log spaced
        Log default to False

    Returns
    -------
    dict[str, int | float]
        Hyperparameter sample
    """
    values = {}
    for name, spec in search_space.items():
        log = spec.get("log", False)
        if spec["type"] == "int":
            values[name] = trial.suggest_int(
                name, int(spec["low"]), int(spec["high"]), log=log
            )
        elif spec["type"] == "float":
            values[name] = trial.suggest_float(
                name, float(spec["low"]), float(spec["high"]), log=log
            )
        else:
            raise ValueError(
                f"Unknown type {spec['type']!r} for {name!r}, "
                "expected 'int' or 'float'"
            )
    return values


def create_or_load_study(
    storage_path: str,
    study_name: str,
    seed: int,
    n_startup_trials: int = 15,
    n_warmup_steps: int = 10,
) -> optuna.study.Study:
    """Create a minimisation study or load it if it exists.
    Uses a TPE sampler (Bayesian opt)
    Uses median pruner.

    Parameters
    ----------
    storage_path: str
        Path of the Optuna db
    study_name: str
        Name of the study
    seed: int
        Seed of the TPE sampler
    n_startup_trials: int
        Number of finished trials before we starts pruning
    n_warmup_steps: int
        Number of epochs of a trial before it can be pruned

    Returns
    -------
    optuna.study.Study
        The study
    """
    storage_folder = os.path.dirname(os.path.abspath(storage_path))
    os.makedirs(storage_folder, exist_ok=True)
    storage = f"sqlite:///{os.path.abspath(storage_path)}"
    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=n_startup_trials, n_warmup_steps=n_warmup_steps
        ),
        load_if_exists=True,
    )


def min_test_loss(log_folder: str) -> float:
    """Read the lowest test loss logged in ``log_folder/test_loss.csv``.

    Parameters
    ----------
    log_folder: str
        Training log folder

    Returns
    -------
    float
        Lowest logged test loss
    """
    vals = np.loadtxt(
        os.path.join(log_folder, "test_loss.csv"), delimiter=";", skiprows=1
    )
    vals = np.atleast_2d(vals)
    return float(np.amin(vals[:, 1]))


def write_recipe(recipe_path: str, overrides: dict, out_path: str) -> None:
    """Write a copy of a YAML recipe
    Also replace some entries, useful to not carry many recipes

    Parameters
    ----------
    recipe_path: str
        Path of the source recipe
    overrides: dict
        Recipe entries to add or replace
    out_path: str
        Path of the written recipe
    """
    with open(recipe_path, "r") as f:
        recipe = yaml.safe_load(f)
    recipe.update(overrides)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(recipe, f, sort_keys=False)


def make_trial_loader(
    loader: torch.utils.data.DataLoader,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
) -> torch.utils.data.DataLoader:
    """Rebuild a DataLoader over the same (already scaled) dataset.

    Parameters
    ----------
    loader: torch.utils.data.DataLoader
        Loader whose dataset is reused
    batch_size: int
        Batch size of the new loader
    shuffle: bool
        Shuffle the samples at each epoch
    drop_last: bool
        Drop the last incomplete batch

    Returns
    -------
    torch.utils.data.DataLoader
        Loader with the new batch size
    """
    batch_size = min(batch_size, len(loader.dataset))
    return torch.utils.data.DataLoader(
        loader.dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
