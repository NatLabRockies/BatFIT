"""Train/test/val splitting of assembled numpy datasets, with npz caching."""

import os

import numpy as np
from sklearn.model_selection import train_test_split

from batfit import logger


def split_arrays(
    arrays: dict[str, np.ndarray | None],
    test_split: float = 0.1,
    val_split: float = 0.1,
    save: bool = True,
    save_path: str = ".",
    cache_filename: str = "data_split.npz",
    random_state: int | None = None,
) -> dict[str, np.ndarray]:
    """Train/test/val split any number of named arrays jointly, with caching.

    Arrays are split together (same shuffle). If the cache file already exists it
    is loaded instead of re-splitting, unless a validation slice is requested but
    the cached file predates it (then it is re-split).

    Parameters
    ----------
    arrays: dict[str, np.ndarray]
        Mapping of array name to array, e.g. ``{"X": X, "Y": Y}``.
    test_split: float
        Fraction of the original data held out as the test set.
    val_split: float
        Fraction of the original data held out as the validation set. When
        ``0.0`` a plain two-way train/test split is produced (no ``_val`` keys).
    random_state: int | None
        Seed forwarded to ``train_test_split`` for reproducible splits.

    Returns
    -------
    dict[str, np.ndarray]
        Dict with ``"{name}_train"``/``"{name}_test"`` keys (and ``"{name}_val"``
        when ``val_split > 0``) for every name in ``arrays``, cast to float32.
    """
    cache_file = os.path.join(save_path, cache_filename)
    if os.path.isfile(cache_file):
        tmp = np.load(cache_file)
        has_val = any(key.endswith("_val") for key in tmp.files)
        if val_split > 0 and not has_val:
            # stale pre-val cache: re-split so downstream _val reads succeed
            logger.warning(
                f"Cached split {cache_file} lacks a validation slice, "
                f"re-splitting"
            )
        elif val_split == 0 and has_val:
            # a two-way request must not silently return a val slice
            raise ValueError(
                f"Cached split {cache_file} contains a validation slice "
                f"but val_split == 0 was requested; delete the file to "
                f"rebuild it as a two-way split"
            )
        else:
            # cache-hit: reuse the split already on disk
            logger.warning(f"Data already split, loading {cache_file} only")
            return {key: tmp[key] for key in tmp.files}

    names = list(arrays.keys())
    for name in names:
        assert arrays[name] is not None

    def _f32(a: np.ndarray) -> np.ndarray:
        return a.astype("float32", copy=False)

    result: dict[str, np.ndarray] = {}

    if val_split > 0:
        logger.info(
            f"Splitting the data with train/test/val split "
            f"({1 - test_split - val_split:.2f}/{test_split:.2f}/"
            f"{val_split:.2f})"
        )
        # stage 1: peel the (test + val) holdout off the training set
        holdout = test_split + val_split
        stage1 = train_test_split(
            *[arrays[name] for name in names],
            test_size=holdout,
            shuffle=True,
            random_state=random_state,
        )
        train_parts = [stage1[2 * i] for i in range(len(names))]
        hold_parts = [stage1[2 * i + 1] for i in range(len(names))]

        # stage 2: split the holdout into test and val
        stage2 = train_test_split(
            *hold_parts,
            test_size=val_split / holdout,
            shuffle=True,
            random_state=random_state,
        )
        for i, name in enumerate(names):
            result[f"{name}_train"] = _f32(train_parts[i])
            result[f"{name}_test"] = _f32(stage2[2 * i])
            result[f"{name}_val"] = _f32(stage2[2 * i + 1])
    else:
        logger.info(
            f"Splitting the data with train/test split "
            f"({1 - test_split:.2f}/{test_split:.2f})"
        )
        split_result = train_test_split(
            *[arrays[name] for name in names],
            test_size=test_split,
            shuffle=True,
            random_state=random_state,
        )
        for i, name in enumerate(names):
            result[f"{name}_train"] = _f32(split_result[2 * i])
            result[f"{name}_test"] = _f32(split_result[2 * i + 1])

    if save:
        logger.info(f"Saving data at {cache_file}")
        np.savez(cache_file, **result)

    return result


def split_dataset_from_np(
    np_data: np.ndarray[np.float32] | None = None,
    np_data_label: np.ndarray[np.float32] | None = None,
    test_split: float = 0.1,
    val_split: float = 0.1,
    save: bool = True,
    save_path: str = ".",
    random_state: int | None = None,
) -> tuple:
    """Train/test/val split a signal array ``X`` and label array ``Y`` jointly.

    Returns
    -------
    tuple
        ``X_train, Y_train, X_test, Y_test, X_val, Y_val``. The ``_val`` entries
        are ``None`` when ``val_split == 0``.
    """
    result = split_arrays(
        {"X": np_data, "Y": np_data_label},
        test_split=test_split,
        val_split=val_split,
        save=save,
        save_path=save_path,
        cache_filename="data_split.npz",
        random_state=random_state,
    )
    return (
        result["X_train"],
        result["Y_train"],
        result["X_test"],
        result["Y_test"],
        result.get("X_val"),
        result.get("Y_val"),
    )


def split_protocol_dataset_from_np(
    np_data: np.ndarray[np.float32],
    np_prot_params: np.ndarray[np.float32],
    np_data_label: np.ndarray[np.float32],
    test_split: float = 0.1,
    val_split: float = 0.1,
    save: bool = True,
    save_path: str = ".",
    random_state: int | None = None,
) -> tuple:
    """Train/test/val split ``(X_signal, prot_params, Y_labels)`` jointly.

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
        ``X_train, P_train, Y_train, X_test, P_test, Y_test, X_val, P_val,
        Y_val``. The ``_val`` entries are ``None`` when ``val_split == 0``.
    """
    result = split_arrays(
        {"X": np_data, "P": np_prot_params, "Y": np_data_label},
        test_split=test_split,
        val_split=val_split,
        save=save,
        save_path=save_path,
        cache_filename="data_split.npz",
        random_state=random_state,
    )
    return (
        result["X_train"],
        result["P_train"],
        result["Y_train"],
        result["X_test"],
        result["P_test"],
        result["Y_test"],
        result.get("X_val"),
        result.get("P_val"),
        result.get("Y_val"),
    )
