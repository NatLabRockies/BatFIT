import os
import pickle
from pathlib import Path

import numpy as np
import torch

from batfit import logger
from batfit.utils.data_utils import (
    from_param_to_surrogate_data,
    scale_dataset_from_np,
    scale_surrogate_dataset_from_np,
    split_dataset_from_np,
    split_surrogate_dataset_from_np,
)


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


def make_dataset_from_np(
    batch_size: int = 16,
    shuffle: bool = True,
    np_data: np.ndarray[np.float32] | None = None,
    np_data_label: np.ndarray[np.float32] | None = None,
    test_split: float = 0.1,
    np_data_train: np.ndarray[np.float32] | None = None,
    np_data_test: np.ndarray[np.float32] | None = None,
    np_data_label_train: np.ndarray[np.float32] | None = None,
    np_data_label_test: np.ndarray[np.float32] | None = None,
    save_path: str = ".",
    scale: bool = True,
    scale_y: bool = False,
):

    if np_data_train is None:
        assert np_data is not None
        assert np_data_label is not None
        # Split data as needed
        X_train, Y_train, X_test, Y_test = split_dataset_from_np(
            np_data, np_data_label, test_split=test_split, save_path=save_path
        )

    else:
        logger.warning("Data provided is already split")
        assert np_data_train is not None
        assert np_data_test is not None
        assert np_data_label_train is not None
        assert np_data_label_test is not None
        X_train = np_data_train
        Y_train = np_data_label_train
        X_test = np_data_test
        Y_test = np_data_label_test

    if scale:
        X_train_scaled, Y_train_scaled, X_test_scaled, Y_test_scaled = (
            scale_dataset_from_np(
                X_train=X_train,
                X_test=X_test,
                Y_train=Y_train,
                Y_test=Y_test,
                save_path=save_path,
                scale_y=scale_y,
            )
        )
    else:
        X_train_scaled = X_train
        Y_train_scaled = Y_train
        X_test_scaled = X_test
        Y_test_scaled = Y_test

    # Make training dataset
    train_data_X = torch.from_numpy(X_train_scaled)  # .to(device)
    if scale_y:
        train_data_Y = torch.from_numpy(Y_train_scaled)  # .to(device)
    else:
        train_data_Y = torch.from_numpy(Y_train)  # .to(device)

    train_dataset = torch.utils.data.TensorDataset(train_data_X, train_data_Y)

    logger.info(f"Train on {train_data_X.shape[0]} samples")

    # Make test dataset
    test_data_X = torch.from_numpy(X_test_scaled)  # .to(device)
    if scale_y:
        test_data_Y = torch.from_numpy(Y_test_scaled)  # .to(device)
    else:
        test_data_Y = torch.from_numpy(Y_test)
    test_dataset = torch.utils.data.TensorDataset(test_data_X, test_data_Y)

    logger.info(f"Test on {test_data_X.shape[0]} samples")

    # Make into a DataLoader
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        # generator=torch.Generator(device=device),
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        # generator=torch.Generator(device=device),
    )

    num_batch = len(train_data_loader)
    num_batch_test = len(test_data_loader)

    return (
        train_data_loader,
        test_data_loader,
    )


def make_surrogate_dataset_from_np(
    batch_size: int = 16,
    shuffle: bool = True,
    np_data: np.ndarray[np.float32] | None = None,
    np_data_label: np.ndarray[np.float32] | None = None,
    test_split: float = 0.1,
    np_data_train: np.ndarray[np.float32] | None = None,
    np_data_test: np.ndarray[np.float32] | None = None,
    np_data_label_train: np.ndarray[np.float32] | None = None,
    np_data_label_test: np.ndarray[np.float32] | None = None,
    save_path: str = ".",
    scale: bool = True,
    scale_y: bool = False,
):

    data_split_filename = os.path.join(save_path, "data_surrogate_split.npz")
    if os.path.isfile(data_split_filename):
        logger.warning("Data surrogate already splitted, loading it only")
        tmp = np.load(data_split_filename)
        X_train = tmp["X_train"]
        Y_train = tmp["Y_train"]
        X_test = tmp["X_test"]
        Y_test = tmp["Y_test"]
    else:
        if np_data_train is None:
            assert np_data is not None
            assert np_data_label is not None
            if os.path.isfile(os.path.join(save_path, "data_split.npz")):
                logger.info(f"Matching param pred split")
                tmp = np.load(os.path.join(save_path, "data_split.npz"))
                X_train = tmp["X_train"]
                Y_train = tmp["Y_train"]
                X_test = tmp["X_test"]
                Y_test = tmp["Y_test"]
                X_train, Y_train = from_param_to_surrogate_data(
                    X_train, Y_train
                )
                X_test, Y_test = from_param_to_surrogate_data(X_test, Y_test)
                logger.info(
                    f"Saving splitted surrogate data at {data_split_filename}"
                )
                np.savez(
                    data_split_filename,
                    X_train=X_train.astype("float32"),
                    Y_train=Y_train.astype("float32"),
                    X_test=X_test.astype("float32"),
                    Y_test=Y_test.astype("float32"),
                )

            else:
                # Split data as needed
                X_train, Y_train, X_test, Y_test = (
                    split_surrogate_dataset_from_np(
                        np_data,
                        np_data_label,
                        test_split=test_split,
                        save_path=save_path,
                    )
                )

        else:
            logger.warning("Data provided is already split")
            assert np_data_train is not None
            assert np_data_test is not None
            assert np_data_label_train is not None
            assert np_data_label_test is not None
            X_train = np_data_train
            Y_train = np_data_label_train
            X_test = np_data_test
            Y_test = np_data_label_test

    if scale:
        X_train_scaled, Y_train_scaled, X_test_scaled, Y_test_scaled = (
            scale_surrogate_dataset_from_np(
                X_train=X_train,
                X_test=X_test,
                Y_train=Y_train,
                Y_test=Y_test,
                save_path=save_path,
                scale_y=scale_y,
            )
        )
    else:
        X_train_scaled = X_train
        Y_train_scaled = Y_train
        X_test_scaled = X_test
        Y_test_scaled = Y_test

    # Make training dataset
    train_data_X = torch.from_numpy(X_train_scaled)  # .to(device)
    if scale_y:
        train_data_Y = torch.from_numpy(Y_train_scaled)  # .to(device)
    else:
        train_data_Y = torch.from_numpy(Y_train)  # .to(device)

    train_dataset = torch.utils.data.TensorDataset(train_data_X, train_data_Y)

    logger.info(f"Train on {train_data_X.shape[0]} samples")

    # Make test dataset
    test_data_X = torch.from_numpy(X_test_scaled)  # .to(device)
    if scale_y:
        test_data_Y = torch.from_numpy(Y_test_scaled)  # .to(device)
    else:
        test_data_Y = torch.from_numpy(Y_test)
    test_dataset = torch.utils.data.TensorDataset(test_data_X, test_data_Y)

    logger.info(f"Test on {test_data_X.shape[0]} samples")

    # Make into a DataLoader
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        # generator=torch.Generator(device=device),
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        # generator=torch.Generator(device=device),
    )

    num_batch = len(train_data_loader)
    num_batch_test = len(test_data_loader)

    return (
        train_data_loader,
        test_data_loader,
    )


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


def get_device_type(enable_cuda=True, enable_mps=True):
    if torch.cuda.is_available() and enable_cuda:
        device_type = "cuda"
    elif torch.backends.mps.is_available() and enable_mps:
        device_type = "mps"
    else:
        device_type = "cpu"
    return device_type


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
