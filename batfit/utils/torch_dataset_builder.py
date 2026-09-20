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
    val_split: float = 0.1,
    np_data_train: np.ndarray[np.float32] | None = None,
    np_data_test: np.ndarray[np.float32] | None = None,
    np_data_label_train: np.ndarray[np.float32] | None = None,
    np_data_label_test: np.ndarray[np.float32] | None = None,
    np_data_val: np.ndarray[np.float32] | None = None,
    np_data_label_val: np.ndarray[np.float32] | None = None,
    save_path: str = ".",
    scale: bool = True,
    scale_y: bool = False,
    random_state: int | None = None,
) -> dict[str, torch.utils.data.DataLoader | None]:
    """Create ``{"train","test","val"}`` DataLoaders for a plain ``(X, Y)`` dataset.

    The ``"val"`` entry is ``None`` when ``val_split == 0`` and no pre-split
    validation arrays are supplied.
    """
    if np_data_train is None:
        assert np_data is not None
        assert np_data_label is not None
        X_train, Y_train, X_test, Y_test, X_val, Y_val = split_dataset_from_np(
            np_data,
            np_data_label,
            test_split=test_split,
            val_split=val_split,
            save_path=save_path,
            random_state=random_state,
        )
    else:
        logger.warning("Data provided is already split")
        assert np_data_test is not None
        assert np_data_label_train is not None
        assert np_data_label_test is not None
        X_train, Y_train, X_test, Y_test = (
            np_data_train,
            np_data_label_train,
            np_data_test,
            np_data_label_test,
        )
        X_val, Y_val = np_data_val, np_data_label_val

    if scale:
        X_train, Y_train, X_test, Y_test, X_val, Y_val = scale_dataset_from_np(
            X_train=X_train,
            X_test=X_test,
            Y_train=Y_train,
            Y_test=Y_test,
            X_val=X_val,
            Y_val=Y_val,
            save_path=save_path,
            scale_y=scale_y,
        )

    logger.info(f"Train on {X_train.shape[0]} samples")
    logger.info(f"Test on {X_test.shape[0]} samples")

    loaders: dict[str, torch.utils.data.DataLoader | None] = {
        "train": _make_loader(
            X_train,
            Y_train,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
        ),
        "test": _make_loader(
            X_test,
            Y_test,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
        ),
        "val": None,
    }
    if X_val is not None:
        logger.info(f"Validate on {X_val.shape[0]} samples")
        loaders["val"] = _make_loader(
            X_val,
            Y_val,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

    return loaders


def make_protocol_dataset_from_np(
    batch_size: int = 16,
    shuffle: bool = True,
    np_data: np.ndarray[np.float32] | None = None,
    np_prot_params: np.ndarray[np.float32] | None = None,
    np_data_label: np.ndarray[np.float32] | None = None,
    test_split: float = 0.1,
    val_split: float = 0.1,
    save_path: str = ".",
    scale: bool = True,
    scale_y: bool = False,
    random_state: int | None = None,
) -> dict[str, torch.utils.data.DataLoader | None]:
    """Create ``{"train","test","val"}`` DataLoaders for protocol-conditioned NPE.

    Each batch contains three tensors: ``(X_signal, prot_params, Y_labels)``.
    The signal ``X`` is standardized and
    protocol parameters ``P`` are MinMax-scaled to ``[0, 1]``
    The fitted scalers are saved alongside the data split.

    Parameters
    ----------
    np_data: np.ndarray[np.float32] | None
        Electrochemical signal of shape ``(N, channels, time)``
    np_prot_params: np.ndarray[np.float32] | None
        Protocol parameters of shape ``(N, n_prot)``
    np_data_label: np.ndarray[np.float32] | None
        Degradation parameters of shape ``(N, n_deg)``
    """
    (
        X_train,
        P_train,
        Y_train,
        X_test,
        P_test,
        Y_test,
        X_val,
        P_val,
        Y_val,
    ) = split_protocol_dataset_from_np(
        np_data=np_data,
        np_prot_params=np_prot_params,
        np_data_label=np_data_label,
        test_split=test_split,
        val_split=val_split,
        save_path=save_path,
        random_state=random_state,
    )

    if scale:
        (
            X_train,
            P_train,
            Y_train,
            X_test,
            P_test,
            Y_test,
            X_val,
            P_val,
            Y_val,
        ) = scale_protocol_dataset_from_np(
            X_train=X_train,
            P_train=P_train,
            X_test=X_test,
            P_test=P_test,
            Y_train=Y_train,
            Y_test=Y_test,
            X_val=X_val,
            P_val=P_val,
            Y_val=Y_val,
            save_path=save_path,
            scale_y=scale_y,
        )

    logger.info(f"Train on {X_train.shape[0]} samples")
    logger.info(f"Test on {X_test.shape[0]} samples")

    loaders: dict[str, torch.utils.data.DataLoader | None] = {
        "train": _make_loader(
            X_train,
            P_train,
            Y_train,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
        ),
        "test": _make_loader(
            X_test,
            P_test,
            Y_test,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
        ),
        "val": None,
    }
    if X_val is not None:
        logger.info(f"Validate on {X_val.shape[0]} samples")
        loaders["val"] = _make_loader(
            X_val,
            P_val,
            Y_val,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

    return loaders


def make_surrogate_dataset_from_np(
    batch_size: int = 16,
    shuffle: bool = True,
    np_data: np.ndarray[np.float32] | None = None,
    np_data_label: np.ndarray[np.float32] | None = None,
    test_split: float = 0.1,
    val_split: float = 0.1,
    save_path: str = ".",
    scale: bool = True,
    scale_y: bool = False,
    random_state: int | None = None,
) -> dict[str, torch.utils.data.DataLoader | None]:
    """Create ``{"train","test","val"}`` DataLoaders for the surrogate dataset."""
    surrogate_split_filename = os.path.join(
        save_path, "data_surrogate_split.npz"
    )
    X_val = Y_val = None

    surrogate_cache_ok = False
    if os.path.isfile(surrogate_split_filename):
        tmp = np.load(surrogate_split_filename)
        if val_split > 0 and "X_val" not in tmp.files:
            logger.warning(
                "Surrogate split cache lacks validation slice, re-deriving"
            )
        else:
            logger.warning("Data surrogate already splitted, loading it only")
            X_train, Y_train, X_test, Y_test = (
                tmp["X_train"],
                tmp["Y_train"],
                tmp["X_test"],
                tmp["Y_test"],
            )
            X_val = tmp["X_val"] if "X_val" in tmp.files else None
            Y_val = tmp["Y_val"] if "Y_val" in tmp.files else None
            surrogate_cache_ok = True

    if not surrogate_cache_ok:
        assert np_data is not None
        assert np_data_label is not None
        # battery-level split (reuses data_split.npz if present, else creates it)
        X_tr, Y_tr, X_te, Y_te, X_va, Y_va = split_dataset_from_np(
            np_data,
            np_data_label,
            test_split=test_split,
            val_split=val_split,
            save_path=save_path,
            random_state=random_state,
        )
        # split-then-slice: explode each battery-level split into per-step rows
        X_train, Y_train = from_param_to_surrogate_data(X_tr, Y_tr)
        X_test, Y_test = from_param_to_surrogate_data(X_te, Y_te)
        if X_va is not None:
            X_val, Y_val = from_param_to_surrogate_data(X_va, Y_va)
        logger.info(
            f"Saving splitted surrogate data at {surrogate_split_filename}"
        )
        to_save = {
            "X_train": X_train.astype("float32"),
            "Y_train": Y_train.astype("float32"),
            "X_test": X_test.astype("float32"),
            "Y_test": Y_test.astype("float32"),
        }
        if X_val is not None:
            to_save["X_val"] = X_val.astype("float32")
            to_save["Y_val"] = Y_val.astype("float32")
        np.savez(surrogate_split_filename, **to_save)

    if scale:
        X_train, Y_train, X_test, Y_test, X_val, Y_val = (
            scale_surrogate_dataset_from_np(
                X_train=X_train,
                X_test=X_test,
                Y_train=Y_train,
                Y_test=Y_test,
                X_val=X_val,
                Y_val=Y_val,
                save_path=save_path,
                scale_y=scale_y,
            )
        )

    logger.info(f"Train on {X_train.shape[0]} samples")
    logger.info(f"Test on {X_test.shape[0]} samples")

    loaders: dict[str, torch.utils.data.DataLoader | None] = {
        "train": _make_loader(
            X_train,
            Y_train,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
        ),
        "test": _make_loader(
            X_test,
            Y_test,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
        ),
        "val": None,
    }
    if X_val is not None:
        logger.info(f"Validate on {X_val.shape[0]} samples")
        loaders["val"] = _make_loader(
            X_val,
            Y_val,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

    return loaders
