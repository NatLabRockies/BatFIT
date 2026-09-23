import numpy as np
import torch

from batfit import logger
from batfit.utils.scalers import BoundedScaler, ZScoreScaler


def build_scalers(
    X_train: np.ndarray,
    sim_params: dict,
    with_prot: bool = False,
) -> dict[str, torch.nn.Module]:
    """Build the scalers of a parameter-inference dataset.

    Parameters
    ----------
    X_train : numpy.ndarray
        Training signal of shape ``(N, channels, time)``; the per-channel
        z-score of X is fitted on it.
    sim_params : dict
        Parsed experiment config (output of ``make_params``) providing the
        degradation (and protocol) parameter bounds.
    with_prot : bool
        Also build the protocol-parameter scaler.

    Returns
    -------
    dict[str, torch.nn.Module]
        ``{"X": ZScoreScaler, "Y": BoundedScaler}`` plus ``"P"``
        (``BoundedScaler``) when ``with_prot`` is True.
    """
    scalers: dict[str, torch.nn.Module] = {
        "X": ZScoreScaler.fit(X_train, axis=(0, 2)),
        "Y": BoundedScaler.from_sim_params(sim_params, kind="deg"),
    }
    if with_prot:
        scalers["P"] = BoundedScaler.from_sim_params(sim_params, kind="prot")
    return scalers


def scale_splits(
    splits: dict[str, np.ndarray | None],
    scalers: dict[str, torch.nn.Module],
) -> dict[str, np.ndarray]:
    """Scale split arrays in place with the scaler of their quantity.

    Arrays are named ``"<quantity>_<split>"`` (e.g. ``"X_train"``, as produced
    by :func:`batfit.utils.dataset_split.split_arrays`) and scaled with
    ``scalers[<quantity>]``. Arrays whose quantity has no scaler, or that are
    ``None``, are dropped. Scaling is done in place so that no copy of the
    dataset is allocated; the input arrays are therefore overwritten.

    Parameters
    ----------
    splits : dict[str, numpy.ndarray | None]
        Unscaled float32 split arrays, overwritten with their scaled values.
    scalers : dict[str, torch.nn.Module]
        Scalers keyed by quantity (``"X"``, ``"Y"``, ``"P"``), each providing
        an in-place ``transform_``.

    Returns
    -------
    dict[str, numpy.ndarray]
        The scaled arrays (the same objects as in ``splits``).
    """
    logger.info("Scaling the data in place")
    scaled = {}
    for key, array in splits.items():
        quantity = key.split("_")[0]
        if array is None or quantity not in scalers:
            continue
        assert array.dtype == np.float32, f"{key} must be float32"
        scaled[key] = scalers[quantity].transform_(array)
    return scaled


def build_surrogate_scalers(
    X_train: np.ndarray,
    sim_params: dict,
) -> dict[str, torch.nn.Module]:
    """Build the scalers of a surrogate dataset.

    Surrogate rows are ``(time, deg_params...) -> voltage``.

    Parameters
    ----------
    X_train : numpy.ndarray
        Training inputs of shape ``(N, 1 + n_deg)``; column 0 is time, on
        which the time z-score is fitted.
    sim_params : dict
        Parsed experiment config providing the degradation-parameter bounds
        and ``vmin``/``vmax``.

    Returns
    -------
    dict[str, torch.nn.Module]
        ``{"t": ZScoreScaler, "Y": BoundedScaler, "V": BoundedScaler}`` for
        time, degradation parameters and voltage.
    """
    return {
        "t": ZScoreScaler.fit(X_train[:, :1], axis=0),
        "Y": BoundedScaler.from_sim_params(sim_params, kind="deg"),
        "V": BoundedScaler.from_sim_params(sim_params, kind="voltage"),
    }


def scale_surrogate_splits(
    splits: dict[str, np.ndarray | None],
    scalers: dict[str, torch.nn.Module],
) -> dict[str, np.ndarray]:
    """Scale surrogate split arrays in place.

    Inputs ``"X_<split>"`` hold ``(time, deg_params...)`` rows: column 0 is
    scaled with ``scalers["t"]`` and the other columns with ``scalers["Y"]``.
    Labels ``"Y_<split>"`` hold the voltage and are scaled with
    ``scalers["V"]``. Scaling is done in place so that no copy of the dataset
    is allocated; ``None`` entries are dropped.

    Parameters
    ----------
    splits : dict[str, numpy.ndarray | None]
        Unscaled float32 split arrays, overwritten with their scaled values.
    scalers : dict[str, torch.nn.Module]
        Output of :func:`build_surrogate_scalers`.

    Returns
    -------
    dict[str, numpy.ndarray]
        The scaled arrays (the same objects as in ``splits``).
    """
    logger.info("Scaling the surrogate data in place")
    scaled = {}
    for key, array in splits.items():
        if array is None:
            continue
        assert array.dtype == np.float32, f"{key} must be float32"
        if key.startswith("X_"):
            # column views: scaling them writes into the array itself
            scalers["t"].transform_(array[:, :1])
            scalers["Y"].transform_(array[:, 1:])
        else:
            scalers["V"].transform_(array)
        scaled[key] = array
    return scaled
