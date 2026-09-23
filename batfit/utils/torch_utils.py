import os
import pickle
from pathlib import Path

import numpy as np
import torch

from batfit import logger
from batfit.utils.torch_dataset_builder import (
    make_dataset_from_np,
    make_npe_dataset_from_np,
    make_protocol_dataset_from_np,
    make_surrogate_dataset_from_np,
)

__all__ = [
    "get_num_parameters",
    "get_device_type",
    "make_dataset_from_np",
    "make_npe_dataset_from_np",
    "make_protocol_dataset_from_np",
    "make_surrogate_dataset_from_np",
    "prepare_log",
    "log_training",
    "save_model",
    "update_best_model",
    "read_restart_position",
    "restart_requested",
    "load_model",
    "find_best_model_file",
    "load_frozen_model",
]


def get_num_parameters(model: torch.nn.Module):
    """Return the number of trainable parameters in a model of type nn.Module.

    Parameters
    ----------
    model: torch.nn.Module
        Model containing trainable parameters

    Returns
    -------
    int
        Number of trainable parameters in model
    """
    num_parameters = 0
    for parameter in model.parameters():
        num_parameters += torch.numel(parameter)
    return num_parameters


def get_device_type(enable_cuda: bool = True, enable_mps: bool = True) -> str:
    """Return the best available torch device type.

    Prefers CUDA, then MPS, then CPU, subject to the enable flags.

    Parameters
    ----------
    enable_cuda: bool
        Allow selecting a CUDA device when available
    enable_mps: bool
        Allow selecting an MPS device when available

    Returns
    -------
    str
        One of ``"cuda"``, ``"mps"`` or ``"cpu"``
    """
    # Move model on GPU if available. Otherwise MPS if possible. Otherwise CPU
    if torch.cuda.is_available() and enable_cuda:
        device_type = "cuda"
    elif torch.backends.mps.is_available() and enable_mps:
        device_type = "mps"
    else:
        device_type = "cpu"
    return device_type


def prepare_log(log_folder: str, append: bool = False) -> None:
    """Create the log folder and initialise the loss CSV files.

    When ``append`` is False (default), any existing
    ``train_loss.csv``/``test_loss.csv`` in ``log_folder`` are removed and a
    fresh ``step;loss`` header is written to each. When ``append`` is True (used
    on a training restart), existing files are kept so new rows are appended;
    only missing files are created with a header.

    Parameters
    ----------
    log_folder: str
        Directory where the loss CSV files are written
    append: bool
        Keep existing loss CSV files and append to them instead of wiping them

    Returns
    -------
    None
    """
    log_dir = Path(log_folder)
    log_dir.mkdir(parents=True, exist_ok=True)
    train_loss_filename = os.path.join(log_folder, "train_loss.csv")
    test_loss_filename = os.path.join(log_folder, "test_loss.csv")
    for filename in (train_loss_filename, test_loss_filename):
        if not append:
            try:
                os.remove(filename)
            except:
                pass
        if not os.path.isfile(filename):
            f = open(filename, "a+")
            f.write("step;loss\n")
            f.close()
    return


def log_training(
    step: int,
    loss: torch.Tensor | float | list,
    log_folder: str,
    filename: str = "loss.csv",
) -> None:
    """Append a training/test loss record to a CSV file.

    Writes a semicolon-separated ``step;loss`` row to
    ``log_folder/filename``. When ``loss`` is a list, each element is written
    as an additional semicolon-separated column.

    Parameters
    ----------
    step: int
        Current training step
    loss: torch.Tensor or float or list
        Loss value(s) to record; tensors are converted via ``.item()``
    log_folder: str
        Directory containing the target CSV file
    filename: str
        Name of the CSV file to append to

    Returns
    -------
    None
    """
    filename = os.path.join(log_folder, filename)
    f = open(filename, "a+")
    if not isinstance(loss, list):
        try:
            f.write(f"{int(step)};{loss.item()}\n")
        except AttributeError:
            f.write(f"{int(step)};{loss}\n")
    else:
        try:
            string_val = f"{int(step)}"
            for element in loss:
                string_val += f";{element.item()}"
            string_val += "\n"
            f.write(string_val)
        except AttributeError:
            string_val = f"{int(step)}"
            for element in loss:
                string_val += f";{element}"
            string_val += "\n"
            f.write(string_val)
    f.close()
    return


def save_model(
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device_type: str | None = None,
    enable_cuda: bool = True,
    enable_mps: bool = True,
    log_folder: str | None = None,
    bypass: int | str | None = None,
    save_model_obj: bool = False,
    save_model_weights: bool = True,
    save_model_opt: bool = False,
    autoencoder: bool = False,
) -> None:
    """Save a model's weights, optimizer state, and/or pickled object.

    The model is temporarily moved to CPU for saving (then back to its
    original device). Files are written to ``log_folder`` with a ``step``
    (or ``bypass``) suffix.

    Parameters
    ----------
    step: int
        Current training step, used as the filename suffix
    model: torch.nn.Module
        Model to save
    optimizer: torch.optim.Optimizer or None
        Optimizer whose state dict is saved when ``save_model_opt`` is True
    device_type: str or None
        Target device type; inferred via :func:`get_device_type` when None
    enable_cuda: bool
        Passed to :func:`get_device_type` when inferring the device
    enable_mps: bool
        Passed to :func:`get_device_type` when inferring the device
    log_folder: str
        Directory the files are written to
    bypass: int or str or None
        When not None, replaces ``step`` in the filename suffix
    save_model_obj: bool
        Also pickle the full model object to ``model.pkl``
    save_model_weights: bool
        Save the model state dict
    save_model_opt: bool
        Save the optimizer state dict (requires ``optimizer``)
    autoencoder: bool
        Save encoder/decoder/autoencoder state dicts separately

    Returns
    -------
    None
    """
    # Get current model device
    current_device = next(model.parameters()).device

    if device_type is None:
        device_type = get_device_type(enable_cuda, enable_mps)

    if device_type == "cuda" or device_type == "mps":
        model = model.to(torch.device("cpu"))
    if bypass is None:
        suffix = f"{step}"
    else:
        suffix = f"{bypass}"

    if save_model_weights or save_model_obj or save_model_opt:
        # os.makedirs(log_folder, exist_ok=True)
        log_dir = Path(log_folder)
        log_dir.mkdir(parents=True, exist_ok=True)

    if save_model_weights:
        if autoencoder:
            torch.save(
                model.encoder.state_dict(),
                os.path.join(log_folder, f"encoder_{suffix}.pt"),
            )
            torch.save(
                model.decoder.state_dict(),
                os.path.join(log_folder, f"decoder_{suffix}.pt"),
            )
            torch.save(
                model.state_dict(),
                os.path.join(log_folder, f"ae_{suffix}.pt"),
            )
        else:
            torch.save(
                model.state_dict(),
                os.path.join(log_folder, f"model_{suffix}.pt"),
            )
    if optimizer is not None and save_model_opt:
        torch.save(
            optimizer.state_dict(),
            os.path.join(log_folder, f"optimizer_{suffix}.pt"),
        )

    if save_model_obj:
        with open(os.path.join(log_folder, "model.pkl"), "wb") as f:
            pickle.dump(model, f)

    model = model.to(current_device)


def update_best_model(
    test_loss: torch.Tensor | float,
    best_test_loss: float,
    model: torch.nn.Module,
    device_type: str | None = None,
    log_folder: str | None = None,
) -> float:
    """Save ``model`` to ``model_best.pt`` when the test loss improves.

    Compares the epoch's test loss against the best value seen so far and, when
    it is lower, writes the model weights to ``model_best.pt`` (via
    :func:`save_model` with ``bypass="best"``; no optimizer state is saved).

    Parameters
    ----------
    test_loss: torch.Tensor or float
        Test loss for the epoch just completed
    best_test_loss: float
        Best test loss observed so far
    model: torch.nn.Module
        Model to persist when the test loss improves
    device_type: str or None
        Target device type forwarded to :func:`save_model`
    log_folder: str
        Directory the ``model_best.pt`` checkpoint is written to

    Returns
    -------
    float
        The updated best test loss (``test_loss`` when it improved, otherwise
        ``best_test_loss``)
    """
    current = (
        float(test_loss.item())
        if hasattr(test_loss, "item")
        else float(test_loss)
    )
    if current < best_test_loss:
        save_model(
            step=0,
            model=model,
            device_type=device_type,
            log_folder=log_folder,
            bypass="best",
        )
        return current
    return best_test_loss


def read_restart_position(log_folder: str) -> tuple[int, int]:
    """Read the resume epoch and step from an existing ``test_loss.csv``.

    One ``test_loss.csv`` row is written per completed epoch, so the number of
    data rows gives the next epoch index and the last row's step gives the last
    completed training step.

    Parameters
    ----------
    log_folder: str
        Directory that may contain a ``test_loss.csv`` from a prior run

    Returns
    -------
    tuple of int
        ``(start_epoch, start_step)``; ``(0, 0)`` when no usable CSV exists
    """
    test_loss_filename = os.path.join(log_folder, "test_loss.csv")
    if not os.path.isfile(test_loss_filename):
        return 0, 0
    # Count data rows (excluding the header) before loading to avoid an empty
    # loadtxt warning on a header-only file.
    with open(test_loss_filename) as f:
        n_data_rows = sum(1 for _ in f) - 1
    if n_data_rows <= 0:
        return 0, 0
    vals = np.atleast_2d(
        np.loadtxt(test_loss_filename, delimiter=";", skiprows=1)
    )
    start_epoch = int(vals.shape[0])
    start_step = int(vals[-1, 0])
    return start_epoch, start_step


def restart_requested(restart_from: str | None) -> bool:
    """Return whether a usable restart checkpoint path was given.

    Returns False when ``restart_from`` is None. When a path is given but the
    file does not exist, logs a warning and returns False so training proceeds
    from scratch rather than failing.

    Parameters
    ----------
    restart_from: str or None
        Path to a checkpoint to restart training from, or None

    Returns
    -------
    bool
        True only when ``restart_from`` points to an existing file
    """
    if restart_from is None:
        return False
    if os.path.isfile(restart_from):
        return True
    logger.warning(
        f"restart_from checkpoint not found: {restart_from}; "
        "training from scratch"
    )
    return False


def load_model(
    model: torch.nn.Module,
    state_dict_file: str,
    device_type: str | None = None,
    enable_cuda: bool = True,
    enable_mps: bool = True,
) -> torch.nn.Module:
    """Load a saved state dict into ``model`` and move it to a device.

    Logs a warning and returns ``model`` unchanged when ``state_dict_file``
    does not exist.

    Parameters
    ----------
    model: torch.nn.Module
        Model instance to load the weights into
    state_dict_file: str
        Path to the saved ``.pt`` state dict
    device_type: str or None
        Target device type; inferred via :func:`get_device_type` when None
    enable_cuda: bool
        Passed to :func:`get_device_type` when inferring the device
    enable_mps: bool
        Passed to :func:`get_device_type` when inferring the device

    Returns
    -------
    torch.nn.Module
        The model with weights loaded, moved to the target device
    """
    if not os.path.exists(state_dict_file):
        logger.warning(
            f"Tried to load model {state_dict_file}, but could not find it"
        )
    else:
        logger.info(f"Loading model {state_dict_file}")

        if device_type is None:
            device_type = get_device_type(enable_cuda, enable_mps)
        device = torch.device(device_type)
        cpu_device = torch.device("cpu")

        if device_type == "cuda" or device_type == "mps":
            model = model.to(cpu_device)

        # model=torch.load(state_dict_file)
        model.load_state_dict(torch.load(state_dict_file, weights_only=True))
        model.to(device)

    return model


def find_best_model_file(model_dir: str) -> str:
    """Return the best checkpoint path for a trained model.

    Prefers ``model_best.pt`` (the checkpoint saved whenever the per-epoch test
    loss improved) when it exists. Otherwise falls back to reading
    ``test_loss.csv``, finding the iteration with minimum test loss, and
    returning the closest saved ``model_<iter>.pt`` checkpoint (or
    ``model_final.pt``).

    Parameters
    ----------
    model_dir: str
        Directory containing ``model_best.pt`` or ``test_loss.csv`` and
        checkpoints

    Returns
    -------
    str
        Path to the best checkpoint file
    """
    best_path = os.path.join(model_dir, "model_best.pt")
    if os.path.isfile(best_path):
        return best_path

    vals = np.loadtxt(
        os.path.join(model_dir, "test_loss.csv"), delimiter=";", skiprows=1
    )
    # Handle the case where vals has only 1 row
    vals = np.atleast_2d(vals)

    best_ind = int(np.argmin(vals[:, 1]))
    final_path = os.path.join(model_dir, "model_final.pt")
    if best_ind == vals.shape[0] - 1 and os.path.isfile(final_path):
        return final_path
    iterations = np.array(
        [
            int(fname[6 : fname.index(".pt")])
            for fname in os.listdir(model_dir)
            if fname.startswith("model_")
            and fname.endswith(".pt")
            and "final" not in fname
            and "best" not in fname
        ]
    )
    if len(iterations) == 0:
        return final_path
    best_iter = vals[best_ind, 0]
    ind = int(np.argmin(np.abs(iterations - best_iter)))
    return os.path.join(model_dir, f"model_{iterations[ind]}.pt")


def load_frozen_model(
    models_dir: str, device: torch.device
) -> torch.nn.Module:
    """Load the best checkpoint of a trained model in eval mode.

    Parameters
    ----------
    models_dir: str
        Directory containing ``model.pkl``, ``test_loss.csv`` and the
        ``model_*.pt`` checkpoints
    device: torch.device
        Device the model is moved to

    Returns
    -------
    torch.nn.Module
        Frozen model in eval mode
    """
    best_pt = find_best_model_file(models_dir)
    logger.info(f"Loading model from {best_pt}")
    with open(os.path.join(models_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    # Older pickled NPE models predate the dependent_outputs attribute
    if not hasattr(model, "dependent_outputs"):
        model.dependent_outputs = False
    # Reading on cpu and passing to device as needed
    model = load_model(model, best_pt, enable_cuda=False, enable_mps=False)
    model.to(device)
    model.eval()
    return model
