"""Model/training infrastructure: device selection, checkpointing, and logging.

Dataset/DataLoader construction (``make_*_dataset_from_np``) now lives in
:mod:`batfit.utils.torch_dataset_builder`; the names are re-exported here so
existing ``from batfit.utils.torch_utils import ...`` call sites keep working.
"""

import os
import pickle
from pathlib import Path

import numpy as np
import torch

from batfit import logger
from batfit.utils.torch_dataset_builder import (
    make_dataset_from_np,
    make_protocol_dataset_from_np,
    make_surrogate_dataset_from_np,
)

__all__ = [
    "get_num_parameters",
    "get_device_type",
    "make_dataset_from_np",
    "make_protocol_dataset_from_np",
    "make_surrogate_dataset_from_np",
    "prepare_log",
    "log_training",
    "save_model",
    "load_model",
    "find_best_model_file",
    "load_frozen_model",
]


def get_num_parameters(model: torch.nn.Module):
    """
    Returns the number of trainable parameters in a model of type nn.Module
    :param model: nn.Module containing trainable parameters
    :return: number of trainable parameters in model
    """
    num_parameters = 0
    for parameter in model.parameters():
        num_parameters += torch.numel(parameter)
    return num_parameters


def get_device_type(enable_cuda=True, enable_mps=True):
    # Move model on GPU if available. Otherwise MPS if possible. Otherwise CPU
    if torch.cuda.is_available() and enable_cuda:
        device_type = "cuda"
    elif torch.backends.mps.is_available() and enable_mps:
        device_type = "mps"
    else:
        device_type = "cpu"
    return device_type


def prepare_log(log_folder):
    log_dir = Path(log_folder)
    log_dir.mkdir(parents=True, exist_ok=True)
    # os.makedirs(log_folder, exist_ok=True)
    train_loss_filename = os.path.join(log_folder, "train_loss.csv")
    test_loss_filename = os.path.join(log_folder, "test_loss.csv")
    try:
        os.remove(train_loss_filename)
    except:
        pass
    try:
        os.remove(test_loss_filename)
    except:
        pass
    f = open(train_loss_filename, "a+")
    f.write("step;loss\n")
    f.close()
    f = open(test_loss_filename, "a+")
    f.write("step;loss\n")
    f.close()
    return


def log_training(step, loss, log_folder, filename="loss.csv"):
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
    step,
    model,
    optimizer=None,
    device_type=None,
    enable_cuda=True,
    enable_mps=True,
    log_folder=None,
    bypass=None,
    save_model_obj=False,
    save_model_weights=True,
    save_model_opt=True,
    autoencoder=False,
):
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
    # if device_type == "cuda":
    #    model = model.to(torch.device("cuda"))
    # elif device_type == "mps":
    #    model = model.to(torch.device("mps"))


def load_model(
    model, state_dict_file, device_type=None, enable_cuda=True, enable_mps=True
):
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
    """Return the checkpoint path with the lowest recorded test loss.

    Reads ``test_loss.csv`` in model_dir, finds the iteration with minimum
    test loss, and returns the closest saved ``model_<iter>.pt`` checkpoint
    (or ``model_final.pt`` when the best iteration is the last one or no
    intermediate checkpoints exist).

    :param model_dir: directory containing ``test_loss.csv`` and checkpoints
    :return: path to the best checkpoint file
    """
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

    :param models_dir: directory containing ``model.pkl``, ``test_loss.csv``
        and the ``model_*.pt`` checkpoints
    :param device: device the model is moved to
    :return: frozen model in eval mode
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
