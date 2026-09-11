"""Build train/test ``DataLoader`` pairs from assembled numpy datasets.

Each ``make_*_dataset_from_np`` wires together the split step
(:mod:`batfit.utils.dataset_split`), the scaling step
(:mod:`batfit.utils.dataset_scaling`), and a shared DataLoader-construction
helper. The three wrappers keep separate top-level functions (rather than one
fully generic builder) because they genuinely differ in batch shape (2-tensor
vs. 3-tensor) and, for the surrogate case, in how the split is obtained.
"""

import os

import numpy as np
import torch

from batfit import logger
from batfit.utils.assembly import from_param_to_surrogate_data
from batfit.utils.dataset_scaling import (
    scale_dataset_from_np,
    scale_protocol_dataset_from_np,
    scale_surrogate_dataset_from_np,
)
from batfit.utils.dataset_split import (
    split_dataset_from_np,
    split_protocol_dataset_from_np,
    split_surrogate_dataset_from_np,
)


def _make_loader(
    *arrays: np.ndarray,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
) -> torch.utils.data.DataLoader:
    """Build a ``DataLoader`` over a ``TensorDataset`` of N aligned arrays."""
    tensors = [torch.from_numpy(a) for a in arrays]
    dataset = torch.utils.data.TensorDataset(*tensors)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )


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
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train/test DataLoaders for a plain ``(X, Y)`` NPE dataset."""
    if np_data_train is None:
        assert np_data is not None
        assert np_data_label is not None
        X_train, Y_train, X_test, Y_test = split_dataset_from_np(
            np_data, np_data_label, test_split=test_split, save_path=save_path
        )
    else:
        logger.warning("Data provided is already split")
        assert np_data_train is not None
        assert np_data_test is not None
        assert np_data_label_train is not None
        assert np_data_label_test is not None
        X_train, Y_train, X_test, Y_test = (
            np_data_train,
            np_data_label_train,
            np_data_test,
            np_data_label_test,
        )

    if scale:
        X_train, Y_train, X_test, Y_test = scale_dataset_from_np(
            X_train=X_train,
            X_test=X_test,
            Y_train=Y_train,
            Y_test=Y_test,
            save_path=save_path,
            scale_y=scale_y,
        )

    logger.info(f"Train on {X_train.shape[0]} samples")
    logger.info(f"Test on {X_test.shape[0]} samples")

    train_data_loader = _make_loader(
        X_train,
        Y_train,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
    )
    test_data_loader = _make_loader(
        X_test, Y_test, batch_size=batch_size, shuffle=shuffle, drop_last=False
    )

    return train_data_loader, test_data_loader


def make_protocol_dataset_from_np(
    batch_size: int = 16,
    shuffle: bool = True,
    np_data: np.ndarray[np.float32] | None = None,
    np_prot_params: np.ndarray[np.float32] | None = None,
    np_data_label: np.ndarray[np.float32] | None = None,
    test_split: float = 0.1,
    save_path: str = ".",
    scale: bool = True,
    scale_y: bool = False,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train/test DataLoaders for :class:`ProbProtParamCNN`.

    Each batch contains three tensors: ``(X_signal, prot_params, Y_labels)``.
    The signal ``X`` is standardized and protocol parameters ``P`` are
    MinMax-scaled to ``[0, 1]``; the fitted scalers are saved alongside the
    data split.

    :param np_data: electrochemical signal of shape ``(N, channels, time)``
    :param np_prot_params: protocol parameters of shape ``(N, n_prot)``
    :param np_data_label: degradation parameters of shape ``(N, n_deg)``
    """
    X_train, P_train, Y_train, X_test, P_test, Y_test = (
        split_protocol_dataset_from_np(
            np_data=np_data,
            np_prot_params=np_prot_params,
            np_data_label=np_data_label,
            test_split=test_split,
            save_path=save_path,
        )
    )

    if scale:
        X_train, P_train, Y_train, X_test, P_test, Y_test = (
            scale_protocol_dataset_from_np(
                X_train=X_train,
                P_train=P_train,
                X_test=X_test,
                P_test=P_test,
                Y_train=Y_train,
                Y_test=Y_test,
                save_path=save_path,
                scale_y=scale_y,
            )
        )

    logger.info(f"Train on {X_train.shape[0]} samples")
    logger.info(f"Test on {X_test.shape[0]} samples")

    train_data_loader = _make_loader(
        X_train,
        P_train,
        Y_train,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
    )
    test_data_loader = _make_loader(
        X_test,
        P_test,
        Y_test,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )

    return train_data_loader, test_data_loader


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
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train/test DataLoaders for the surrogate ``(time+params -> voltage)`` dataset.

    When a ``data_split.npz`` from a prior NPE run already exists at
    ``save_path``, that split is reused (via :func:`from_param_to_surrogate_data`)
    so the surrogate and NPE models train/test on matching batteries.
    """
    data_split_filename = os.path.join(save_path, "data_surrogate_split.npz")
    if os.path.isfile(data_split_filename):
        logger.warning("Data surrogate already splitted, loading it only")
        tmp = np.load(data_split_filename)
        X_train, Y_train, X_test, Y_test = (
            tmp["X_train"],
            tmp["Y_train"],
            tmp["X_test"],
            tmp["Y_test"],
        )
    elif np_data_train is None:
        assert np_data is not None
        assert np_data_label is not None
        npe_split_filename = os.path.join(save_path, "data_split.npz")
        if os.path.isfile(npe_split_filename):
            logger.info("Matching NPE split")
            tmp = np.load(npe_split_filename)
            X_train, Y_train = from_param_to_surrogate_data(
                tmp["X_train"], tmp["Y_train"]
            )
            X_test, Y_test = from_param_to_surrogate_data(
                tmp["X_test"], tmp["Y_test"]
            )
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
            X_train, Y_train, X_test, Y_test = split_surrogate_dataset_from_np(
                np_data,
                np_data_label,
                test_split=test_split,
                save_path=save_path,
            )
    else:
        logger.warning("Data provided is already split")
        assert np_data_train is not None
        assert np_data_test is not None
        assert np_data_label_train is not None
        assert np_data_label_test is not None
        X_train, Y_train, X_test, Y_test = (
            np_data_train,
            np_data_label_train,
            np_data_test,
            np_data_label_test,
        )

    if scale:
        X_train, Y_train, X_test, Y_test = scale_surrogate_dataset_from_np(
            X_train=X_train,
            X_test=X_test,
            Y_train=Y_train,
            Y_test=Y_test,
            save_path=save_path,
            scale_y=scale_y,
        )

    logger.info(f"Train on {X_train.shape[0]} samples")
    logger.info(f"Test on {X_test.shape[0]} samples")

    train_data_loader = _make_loader(
        X_train,
        Y_train,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
    )
    test_data_loader = _make_loader(
        X_test, Y_test, batch_size=batch_size, shuffle=shuffle, drop_last=False
    )

    return train_data_loader, test_data_loader
