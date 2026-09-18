"""Train/test/val splitting of assembled numpy datasets, with npz caching."""
import os

import numpy as np
from sklearn.model_selection import train_test_split

from batfit import logger


def split_arrays(
    arrays: dict[str, np.ndarray | None],
    test_split: float = 0.1,
    save: bool = True,
    save_path: str = ".",
    cache_filename: str = "data_split.npz",
) -> dict[str, np.ndarray]:
    """Train/test split any number of named arrays jointly, with npz caching.

    Arrays are split together (same shuffle)
    If that cache file already exists, it is loaded instead of re-splitting 

    Parameters
    ----------
    arrays: dict[str, np.ndarray]
        Mapping of array name to array, e.g. ``{"X": X, "Y": Y}``.

    Returns
    -------
    dict[str, np.ndarray]
        Dict with ``"{name}_train"``/``"{name}_test"`` keys for every name in
        ``arrays``, arrays cast to float32.
    """
    cache_file = os.path.join(save_path, cache_filename)
    if os.path.isfile(cache_file):
        # cache-hit: reuse the split already on disk instead of re-splitting
        logger.warning(f"Data already split, loading {cache_file} only")
        tmp = np.load(cache_file)
        return {key: tmp[key] for key in tmp.files}

    logger.info(
        f"Splitting the data with train/test split "
        f"({1 - test_split:.2f}/{test_split:.2f})"
    )

    names = list(arrays.keys())
    for name in names:
        assert arrays[name] is not None

    split_result = train_test_split(
        *[arrays[name] for name in names], test_size=test_split, shuffle=True
    )

    result: dict[str, np.ndarray] = {}
    for i, name in enumerate(names):
        result[f"{name}_train"] = split_result[2 * i].astype(
            "float32", copy=False
        )
        result[f"{name}_test"] = split_result[2 * i + 1].astype(
            "float32", copy=False
        )

    if save:
        logger.info(f"Saving data at {cache_file}")
        np.savez(cache_file, **result)

    return result


def split_dataset_from_np(
    np_data: np.ndarray[np.float32] | None = None,
    np_data_label: np.ndarray[np.float32] | None = None,
    test_split: float = 0.1,
    save: bool = True,
    save_path: str = ".",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train/test split a signal array ``X`` and label array ``Y`` jointly.

    Returns
    -------
    tuple
        ``X_train, Y_train, X_test, Y_test``.
    """
    result = split_arrays(
        {"X": np_data, "Y": np_data_label},
        test_split=test_split,
        save=save,
        save_path=save_path,
        cache_filename="data_split.npz",
    )
    return (
        result["X_train"],
        result["Y_train"],
        result["X_test"],
        result["Y_test"],
    )


def split_protocol_dataset_from_np(
    np_data: np.ndarray[np.float32],
    np_prot_params: np.ndarray[np.float32],
    np_data_label: np.ndarray[np.float32],
    test_split: float = 0.1,
    save: bool = True,
    save_path: str = ".",
) -> tuple:
    """Train/test split ``(X_signal, prot_params, Y_labels)`` jointly.

    Parameters
    ----------
    np_data: np.ndarray[np.float32]
        Electrochemical signal array of shape ``(N, channels, time)``
    np_prot_params: np.ndarray[np.float32]
        Protocol parameter array of shape ``(N, n_prot)``
    np_data_label: np.ndarray[np.float32]
        Degradation parameter array of shape ``(N, n_deg)``

    Returns
    -------
    tuple
        ``X_train, P_train, Y_train, X_test, P_test, Y_test``
    """
    result = split_arrays(
        {"X": np_data, "P": np_prot_params, "Y": np_data_label},
        test_split=test_split,
        save=save,
        save_path=save_path,
        cache_filename="data_split.npz",
    )
    return (
        result["X_train"],
        result["P_train"],
        result["Y_train"],
        result["X_test"],
        result["P_test"],
        result["Y_test"],
    )


def split_surrogate_dataset_from_np(
    np_data: np.ndarray[np.float32] | None = None,
    np_data_label: np.ndarray[np.float32] | None = None,
    test_split: float = 0.1,
    save: bool = True,
    save_path: str = ".",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train/test split a surrogate signal array ``X`` and label array ``Y`` jointly.

    Returns
    -------
    tuple
        ``X_train, Y_train, X_test, Y_test``.
    """
    # We don't use data_split.npz here because we may construct both a
    # surrogate dataset and an NPE dataset from the same raw data.
    result = split_arrays(
        {"X": np_data, "Y": np_data_label},
        test_split=test_split,
        save=save,
        save_path=save_path,
        cache_filename="data_surrogate_split.npz",
    )
    return (
        result["X_train"],
        result["Y_train"],
        result["X_test"],
        result["Y_test"],
    )
