import os

import numpy as np
import torch

from batfit import logger
from batfit.utils.assembly import from_param_to_surrogate_data
from batfit.utils.dataset_scaling import (
    build_scalers,
    build_surrogate_scalers,
    scale_dataset_from_np,
    scale_protocol_dataset_from_np,
    scale_splits,
    scale_surrogate_splits,
)
from batfit.utils.dataset_split import (
    split_arrays,
    split_dataset_from_np,
    split_protocol_dataset_from_np,
)


def _make_loader(
    *arrays: np.ndarray,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
) -> torch.utils.data.DataLoader:
    """Build a ``DataLoader`` over a ``TensorDataset`` of N aligned arrays.

    ``batch_size`` is clamped to the number of samples so that a batch never
    exceeds the dataset; this keeps ``drop_last=True`` from discarding the only
    (partial) batch on small datasets.
    """
    n_samples = arrays[0].shape[0]
    batch_size = max(1, min(batch_size, n_samples))
    tensors = [torch.from_numpy(a) for a in arrays]
    dataset = torch.utils.data.TensorDataset(*tensors)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )


def make_npe_dataset_from_np(
    sim_params: dict,
    np_data: np.ndarray | None = None,
    np_data_label: np.ndarray | None = None,
    np_prot_params: np.ndarray | None = None,
    batch_size: int = 16,
    shuffle: bool = True,
    test_split: float = 0.1,
    val_split: float = 0.1,
    save_path: str = ".",
    random_state: int | None = None,
) -> tuple[
    dict[str, torch.utils.data.DataLoader | None],
    dict[str, torch.nn.Module],
]:
    """Create scaled train/test/val DataLoaders for parameter inference (NPE).

    The data is split (reusing ``data_split.npz`` when present), then X is
    z-scored per channel (fitted on train) and the degradation parameters Y
    (and protocol parameters P) are scaled to ``[0, 1]`` from the bounds of
    ``sim_params``. Scaling is done in place on the split arrays and the scaled
    data is not saved to disk. Batches are ``(X, Y)``, or ``(X, P, Y)`` when protocol
    parameters are used.

    Parameters
    ----------
    sim_params : dict
        Parsed experiment config (output of ``make_params``).
    np_data : numpy.ndarray | None
        Signal of shape ``(N, channels, time)``; may be None when the split
        cache already exists.
    np_data_label : numpy.ndarray | None
        Degradation parameters of shape ``(N, n_deg)``.
    np_prot_params : numpy.ndarray | None
        Protocol parameters of shape ``(N, n_prot)``. Protocol parameters are
        used when this is given or when ``sim_params`` declares them.
    batch_size : int
        Batch size of the DataLoaders.
    shuffle : bool
        Shuffle the train and test DataLoaders.
    test_split : float
        Fraction of the data held out as the test set.
    val_split : float
        Fraction of the data held out as the validation set.
    save_path : str
        Folder of the split cache ``data_split.npz``.
    random_state : int | None
        Seed of the split.

    Returns
    -------
    tuple
        ``(loaders, scalers)``: ``{"train", "test", "val"}`` DataLoaders
        (``"val"`` is None without a validation slice) and the scalers keyed
        ``"X"``, ``"Y"`` (and ``"P"``), to be handed to the model.
    """
    with_prot = np_prot_params is not None or "prot_param_names" in sim_params
    arrays = {"X": np_data, "Y": np_data_label}
    if with_prot:
        arrays["P"] = np_prot_params
    splits = split_arrays(
        arrays,
        test_split=test_split,
        val_split=val_split,
        save_path=save_path,
        random_state=random_state,
    )
    if with_prot:
        assert "P_train" in splits, (
            f"The split cache in {save_path} has no protocol parameters; "
            f"delete it to rebuild it with P"
        )

    scalers = build_scalers(splits["X_train"], sim_params, with_prot)
    # in place: the split arrays become the scaled arrays (no copy)
    scaled = scale_splits(splits, scalers)

    quantities = ["X", "P", "Y"] if with_prot else ["X", "Y"]
    logger.info(f"Train on {scaled['X_train'].shape[0]} samples")
    logger.info(f"Test on {scaled['X_test'].shape[0]} samples")
    loaders: dict[str, torch.utils.data.DataLoader | None] = {
        "train": _make_loader(
            *[scaled[f"{q}_train"] for q in quantities],
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
        ),
        "test": _make_loader(
            *[scaled[f"{q}_test"] for q in quantities],
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
        ),
        "val": None,
    }
    if "X_val" in scaled:
        logger.info(f"Validate on {scaled['X_val'].shape[0]} samples")
        loaders["val"] = _make_loader(
            *[scaled[f"{q}_val"] for q in quantities],
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )
    return loaders, scalers


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
    sim_params: dict,
    np_data: np.ndarray | None = None,
    np_data_label: np.ndarray | None = None,
    batch_size: int = 16,
    shuffle: bool = True,
    test_split: float = 0.1,
    val_split: float = 0.1,
    save_path: str = ".",
    random_state: int | None = None,
) -> tuple[
    dict[str, torch.utils.data.DataLoader | None],
    dict[str, torch.nn.Module],
]:
    """Create scaled train/test/val DataLoaders for the voltage surrogate.

    Each battery of the split (``data_split.npz``) is exploded into per-step
    rows ``(time, deg_params...) -> voltage``, cached unscaled in
    ``data_surrogate_split.npz``. Time is then z-scored (fitted on train), the
    degradation parameters and the voltage are scaled to ``[0, 1]`` from the
    bounds of ``sim_params``; scaling is done in place and not saved.

    Parameters
    ----------
    sim_params : dict
        Parsed experiment config (output of ``make_params``).
    np_data : numpy.ndarray | None
        Signal of shape ``(N, 2, time)`` (time, voltage); may be None when the
        split caches already exist.
    np_data_label : numpy.ndarray | None
        Degradation parameters of shape ``(N, n_deg)``.
    batch_size : int
        Batch size of the DataLoaders.
    shuffle : bool
        Shuffle the train and test DataLoaders.
    test_split : float
        Fraction of the batteries held out as the test set.
    val_split : float
        Fraction of the batteries held out as the validation set.
    save_path : str
        Folder of the split caches.
    random_state : int | None
        Seed of the split.

    Returns
    -------
    tuple
        ``(loaders, scalers)``: ``{"train", "test", "val"}`` DataLoaders
        (``"val"`` is None without a validation slice) and the scalers keyed
        ``"t"``, ``"Y"``, ``"V"``, to be handed to the model.
    """
    surrogate_split_filename = os.path.join(
        save_path, "data_surrogate_split.npz"
    )
    X_val = Y_val = None

    surrogate_cache_ok = False
    if os.path.isfile(surrogate_split_filename):
        tmp = np.load(surrogate_split_filename)
        has_val = "X_val" in tmp.files
        if val_split > 0 and not has_val:
            logger.warning(
                "Surrogate split cache lacks validation slice, re-deriving"
            )
        elif val_split == 0 and has_val:
            # a two-way request must not silently return a val loader
            raise ValueError(
                f"Surrogate split cache {surrogate_split_filename} contains "
                f"a validation slice but val_split == 0 was requested; delete "
                f"the file to rebuild it as a two-way split"
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
        # copy=False: the arrays are already float32, avoid duplicating them
        to_save = {
            "X_train": X_train.astype("float32", copy=False),
            "Y_train": Y_train.astype("float32", copy=False),
            "X_test": X_test.astype("float32", copy=False),
            "Y_test": Y_test.astype("float32", copy=False),
        }
        if X_val is not None:
            to_save["X_val"] = X_val.astype("float32", copy=False)
            to_save["Y_val"] = Y_val.astype("float32", copy=False)
        np.savez(surrogate_split_filename, **to_save)

    splits = {
        "X_train": X_train,
        "Y_train": Y_train,
        "X_test": X_test,
        "Y_test": Y_test,
        "X_val": X_val,
        "Y_val": Y_val,
    }
    scalers = build_surrogate_scalers(X_train, sim_params)
    # in place: the split arrays become the scaled arrays (no copy)
    scale_surrogate_splits(splits, scalers)

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

    return loaders, scalers
