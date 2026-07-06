"""Fit/apply/cache scalers for train-test-split numpy datasets.

The three dataset shapes used across the codebase (plain signal/label,
protocol-conditioned signal/protocol/label, surrogate signal/label) each get
their own top-level function below, since they genuinely differ in on-disk
cache-file naming and scaler-reuse behavior. Internally they all share the
same small set of private helpers so the fit/transform/persist logic for a
given scaler kind is implemented exactly once.
"""

import os
import pickle

import numpy as np
from sklearn import preprocessing

from batfit import logger
from batfit.utils.scalers import CustomScaler


def _fit_or_reuse_zscore_scaler(
    train_array: np.ndarray,
    scaler_file: str,
    stat_axis: int | tuple[int, ...],
    reuse_if_exists: bool,
) -> CustomScaler:
    """Fit a :class:`CustomScaler` on ``train_array``, or reuse a cached one.

    When ``reuse_if_exists`` and ``scaler_file`` already exists, the pickled
    scaler is loaded instead of re-fitting (cache-hit). Otherwise a new
    scaler is fit on ``train_array`` and persisted to ``scaler_file``.
    """
    if reuse_if_exists and os.path.isfile(scaler_file):
        # cache-hit: reuse the previously fitted scaler
        logger.warning(f"Reusing existing signal scaler from {scaler_file}")
        with open(scaler_file, "rb") as f:
            return pickle.load(f)

    scaler = CustomScaler.fit(train_array, axis=stat_axis)
    logger.info(f"Dumping scaler X at {scaler_file}")
    with open(scaler_file, "wb") as f:
        pickle.dump(scaler, f)
    return scaler


def _fit_and_dump(scaler, train_array: np.ndarray, scaler_file: str, label: str):
    """Fit a scikit-learn ``scaler`` on ``train_array`` and persist it."""
    scaler.fit(train_array)
    logger.info(f"Dumping scaler {label} at {scaler_file}")
    with open(scaler_file, "wb") as f:
        pickle.dump(scaler, f)
    return scaler


def _maybe_scale_y(
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    scaler_y_filename: str,
    scale_y: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit+apply a ``StandardScaler`` to Y when ``scale_y``, else pass through."""
    if not scale_y:
        return Y_train, Y_test
    scaler_Y = _fit_and_dump(
        preprocessing.StandardScaler(), Y_train, scaler_y_filename, "Y"
    )
    return (
        scaler_Y.transform(Y_train).astype("float32"),
        scaler_Y.transform(Y_test).astype("float32"),
    )


def scale_dataset_from_np(
    X_train: np.ndarray[np.float32],
    X_test: np.ndarray[np.float32],
    Y_train: np.ndarray[np.float32],
    Y_test: np.ndarray[np.float32],
    save_path: str = ".",
    save_scaled: bool = True,
    scale_y: bool = False,
):
    """Scale the signal X and, optionally, the degradation parameter labels Y.

    When ``scale_y=False`` the result is written to ``data_scaled.npz``.
    When ``scale_y=True`` (StandardScaler on Y) the result is written to the
    separate ``data_scaled_y.npz`` so that the two variants can coexist in the
    same directory without overwriting each other. ``scaler_X.pkl`` is shared:
    if it already exists (e.g. from a prior ``scale_y=False`` run) it is
    reused rather than re-fitted.

    Only the degradation parameter array (Y) is affected by ``scale_y``; protocol
    parameters live in a separate tensor and are scaled by their own pipeline.
    """
    scaler_x_filename = os.path.join(save_path, "scaler_X.pkl")
    scaler_y_filename = os.path.join(save_path, "scaler_Y.pkl")
    data_scaled_filename = os.path.join(save_path, "data_scaled.npz")
    data_scaled_y_filename = os.path.join(save_path, "data_scaled_y.npz")

    target_npz = data_scaled_y_filename if scale_y else data_scaled_filename

    if scale_y:
        cache_hit = (
            os.path.isfile(scaler_x_filename)
            and os.path.isfile(target_npz)
            and os.path.isfile(scaler_y_filename)
        )
    else:
        cache_hit = os.path.isfile(scaler_x_filename) and os.path.isfile(
            target_npz
        )

    if cache_hit:
        logger.warning("Data already scaled, loading scaler and data")
        tmp = np.load(target_npz)
        return (
            tmp["X_train"],
            tmp["Y_train"],
            tmp["X_test"],
            tmp["Y_test"],
        )

    logger.info("Scaling the data")

    scaler_X = _fit_or_reuse_zscore_scaler(
        X_train, scaler_x_filename, stat_axis=(0, 2), reuse_if_exists=True
    )
    X_train_scaled = scaler_X.transform(X_train).astype("float32")
    X_test_scaled = scaler_X.transform(X_test).astype("float32")

    Y_train_scaled, Y_test_scaled = _maybe_scale_y(
        Y_train, Y_test, scaler_y_filename, scale_y
    )

    if save_scaled:
        logger.info(f"Saving scaled data at {target_npz}")
        np.savez(
            target_npz,
            X_train=X_train_scaled,
            Y_train=Y_train_scaled,
            X_test=X_test_scaled,
            Y_test=Y_test_scaled,
        )

    return (
        X_train_scaled,
        Y_train_scaled,
        X_test_scaled,
        Y_test_scaled,
    )


def scale_protocol_dataset_from_np(
    X_train: np.ndarray[np.float32],
    P_train: np.ndarray[np.float32],
    X_test: np.ndarray[np.float32],
    P_test: np.ndarray[np.float32],
    Y_train: np.ndarray[np.float32],
    Y_test: np.ndarray[np.float32],
    save_path: str = ".",
    save_scaled: bool = True,
    scale_y: bool = False,
):
    """Scale X with :class:`CustomScaler`, P with MinMaxScaler, and
    optionally Y with StandardScaler.

    When ``scale_y=False`` (default) results are cached in ``data_scaled.npz``.
    When ``scale_y=True`` a ``StandardScaler`` is fitted on ``Y_train``,
    saved to ``scaler_Y.pkl``, and results are cached in ``data_scaled_y.npz``
    so the two variants can coexist in the same directory.

    Saves ``scaler_X.pkl``, ``scaler_P.pkl``, and the target ``.npz``
    (containing ``X_train``, ``P_train``, ``Y_train``, ``X_test``, ``P_test``,
    ``Y_test``) to ``save_path``.

    :return: ``X_train_scaled, P_train_scaled, Y_train_scaled,
        X_test_scaled, P_test_scaled, Y_test_scaled``.
        Y is unscaled when ``scale_y=False``.
    """
    scaler_x_filename = os.path.join(save_path, "scaler_X.pkl")
    scaler_p_filename = os.path.join(save_path, "scaler_P.pkl")
    scaler_y_filename = os.path.join(save_path, "scaler_Y.pkl")
    data_scaled_filename = os.path.join(save_path, "data_scaled.npz")
    data_scaled_y_filename = os.path.join(save_path, "data_scaled_y.npz")

    target_npz = data_scaled_y_filename if scale_y else data_scaled_filename

    if scale_y:
        cache_hit = (
            os.path.isfile(scaler_x_filename)
            and os.path.isfile(scaler_p_filename)
            and os.path.isfile(target_npz)
            and os.path.isfile(scaler_y_filename)
        )
    else:
        cache_hit = (
            os.path.isfile(scaler_x_filename)
            and os.path.isfile(scaler_p_filename)
            and os.path.isfile(target_npz)
        )

    if cache_hit:
        logger.warning("Protocol data already scaled, loading scaler and data")
        tmp = np.load(target_npz)
        return (
            tmp["X_train"],
            tmp["P_train"],
            tmp["Y_train"],
            tmp["X_test"],
            tmp["P_test"],
            tmp["Y_test"],
        )

    logger.info("Scaling the protocol dataset")

    scaler_X = _fit_or_reuse_zscore_scaler(
        X_train, scaler_x_filename, stat_axis=(0, 2), reuse_if_exists=False
    )
    X_train_scaled = scaler_X.transform(X_train).astype("float32")
    X_test_scaled = scaler_X.transform(X_test).astype("float32")

    scaler_P = _fit_and_dump(
        preprocessing.MinMaxScaler(), P_train, scaler_p_filename, "P"
    )
    P_train_scaled = scaler_P.transform(P_train).astype("float32")
    P_test_scaled = scaler_P.transform(P_test).astype("float32")

    Y_train_scaled, Y_test_scaled = _maybe_scale_y(
        Y_train, Y_test, scaler_y_filename, scale_y
    )

    if save_scaled:
        logger.info(f"Saving scaled protocol data at {target_npz}")
        np.savez(
            target_npz,
            X_train=X_train_scaled,
            P_train=P_train_scaled,
            Y_train=Y_train_scaled,
            X_test=X_test_scaled,
            P_test=P_test_scaled,
            Y_test=Y_test_scaled,
        )

    return (
        X_train_scaled,
        P_train_scaled,
        Y_train_scaled,
        X_test_scaled,
        P_test_scaled,
        Y_test_scaled,
    )


def scale_surrogate_dataset_from_np(
    X_train: np.ndarray[np.float32],
    X_test: np.ndarray[np.float32],
    Y_train: np.ndarray[np.float32],
    Y_test: np.ndarray[np.float32],
    save_path: str = ".",
    save_scaled: bool = True,
    scale_y: bool = False,
):
    """Scale a surrogate dataset's signal X and, optionally, labels Y.

    Unlike :func:`scale_dataset_from_np`, the cached ``.npz`` filename does
    not vary with ``scale_y`` (always ``data_surrogate_scaled.npz``), and the
    X scaler is always re-fit rather than reused across calls.
    """
    scaler_x_filename = os.path.join(save_path, "scaler_surrogate_X.pkl")
    data_scaled_filename = os.path.join(save_path, "data_surrogate_scaled.npz")
    scaler_y_filename = os.path.join(save_path, "scaler_surrogate_Y.pkl")

    cache_hit = os.path.isfile(scaler_x_filename) and os.path.isfile(
        data_scaled_filename
    )
    if scale_y:
        cache_hit = cache_hit and os.path.isfile(scaler_y_filename)

    if cache_hit:
        logger.warning("Data surrogate already scaled, loading scaler and data")
        tmp = np.load(data_scaled_filename)
        return (
            tmp["X_train"],
            tmp["Y_train"],
            tmp["X_test"],
            tmp["Y_test"],
        )

    logger.info("Scaling the data")

    scaler_X = _fit_or_reuse_zscore_scaler(
        X_train, scaler_x_filename, stat_axis=0, reuse_if_exists=False
    )
    X_train_scaled = scaler_X.transform(X_train).astype("float32")
    X_test_scaled = scaler_X.transform(X_test).astype("float32")

    Y_train_scaled, Y_test_scaled = _maybe_scale_y(
        Y_train, Y_test, scaler_y_filename, scale_y
    )

    if save_scaled:
        logger.info(f"Saving scaled surrogate data at {data_scaled_filename}")
        np.savez(
            data_scaled_filename,
            X_train=X_train_scaled,
            Y_train=Y_train_scaled,
            X_test=X_test_scaled,
            Y_test=Y_test_scaled,
        )

    return (
        X_train_scaled,
        Y_train_scaled,
        X_test_scaled,
        Y_test_scaled,
    )
