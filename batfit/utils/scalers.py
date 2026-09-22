import pickle

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


def _match_array(stat: np.ndarray, data):
    """Cast a numpy statistic to match ``data``'s array type.

    Parameters
    ----------
    stat : numpy.ndarray
        A ``means``/``stds`` array (or slice thereof) to broadcast against
        ``data``.
    data : numpy.ndarray or torch.Tensor
        The array the statistic will be combined with.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        ``stat`` unchanged when ``data`` is a numpy array; otherwise a
        ``torch.Tensor`` on ``data``'s device and dtype. The returned tensor
        is a non-leaf constant (``requires_grad=False``), so gradients flow
        through ``data`` only and no reference to it is retained on the
        scaler.
    """
    if isinstance(data, torch.Tensor):
        return torch.as_tensor(stat, dtype=data.dtype, device=data.device)
    return stat


class CustomScaler:
    """Per-channel z-score scaler for 3D signal arrays ``(N, channels, time)``.

    Falls back to the channel-1 statistics for  single-channel array.
    Accepts numpy arrays or torch tensors and returns the same type,
    propagating gradients when the input is a tensor.
    """

    def __init__(self, means: np.ndarray, stds: np.ndarray) -> None:
        """Store per-channel means and standard deviations"""
        self.means = means
        self.stds = stds

    @classmethod
    def fit(
        cls, data: np.ndarray, axis: int | tuple[int, ...]
    ) -> "CustomScaler":
        """Fit scaler from ``data``
        Mirrors the ``.fit()`` interface of scikit-learn scalers
        """
        means = np.mean(data, axis=axis, keepdims=True)
        stds = np.std(data, axis=axis, keepdims=True)
        return cls(means, stds)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Return ``(data - means) / stds``, broadcasting over channels."""
        assert len(data.shape) == len(self.means.shape)
        assert len(data.shape) == len(self.stds.shape)
        if self.stds.shape[1] == 2 and data.shape[1] == 1:
            means = self.means[:, 1, :]
            stds = self.stds[:, 1, :]
        else:
            means = self.means
            stds = self.stds
        means = _match_array(means, data)
        stds = _match_array(stds, data)
        transformed_data = (data - means) / stds
        assert transformed_data.shape == data.shape
        return transformed_data

    def inverse_transform(self, transformed_data: np.ndarray) -> np.ndarray:
        """Invert :meth:`transform`, returning data in physical units."""
        assert len(transformed_data.shape) == len(self.means.shape)
        assert len(transformed_data.shape) == len(self.stds.shape)
        if self.stds.shape[1] == 2 and transformed_data.shape[1] == 1:
            means = self.means[:, 1, :]
            stds = self.stds[:, 1, :]
        else:
            means = self.means
            stds = self.stds
        means = _match_array(means, transformed_data)
        stds = _match_array(stds, transformed_data)
        data = transformed_data * stds + means
        assert transformed_data.shape == data.shape
        return data


def _load_scaler(scaler_file: str):
    """Unpickle and return the scaler object stored at ``scaler_file``."""
    with open(scaler_file, "rb") as f:
        return pickle.load(f)


def _apply_scaler(
    data: np.ndarray,
    scaler_file: str | None,
    inverse: bool,
    allow_missing: bool,
) -> np.ndarray:
    """Load the scaler at ``scaler_file`` and transform or inverse-transform ``data``.
    If inverse is True, apply inverse_transform instead of transform
    """
    if allow_missing:
        if scaler_file is None:
            return data
        try:
            scaler = _load_scaler(scaler_file)
        except FileNotFoundError:
            return data
    else:
        scaler = _load_scaler(scaler_file)
    return (
        scaler.inverse_transform(data) if inverse else scaler.transform(data)
    )


def scale_input_from_scaler(
    X_data: np.ndarray[np.float32],
    scaler_X_file: str,
) -> np.ndarray:
    """Scale a raw signal array with the scaler pickled at ``scaler_X_file``."""
    assert len(X_data.shape) in [2, 3]
    return _apply_scaler(
        X_data, scaler_X_file, inverse=False, allow_missing=False
    )


def scale_output_from_scaler(
    Y_data: np.ndarray[np.float32],
    scaler_Y_file: str,
) -> np.ndarray:
    """Scale a raw degradation-parameter array with the pickled scaler."""
    assert len(Y_data.shape) == 2
    return _apply_scaler(
        Y_data, scaler_Y_file, inverse=False, allow_missing=False
    )


def scale_dataset_from_scaler(
    X_data: np.ndarray[np.float32],
    Y_data: np.ndarray[np.float32],
    scaler_X_file: str,
    scaler_Y_file: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale both ``X_data`` and ``Y_data`` with their respective pickled scalers."""
    X_scaled = scale_input_from_scaler(X_data, scaler_X_file)
    Y_scaled = scale_output_from_scaler(Y_data, scaler_Y_file)
    return X_scaled, Y_scaled


def unscale_input_from_scaler(
    X_data: np.ndarray[np.float32],
    scaler_X_file: str | None,
) -> np.ndarray:
    """Inverse-scale a signal array; passes through if no scaler is configured."""
    assert len(X_data.shape) == 3
    return _apply_scaler(
        X_data, scaler_X_file, inverse=True, allow_missing=True
    )


def unscale_output_from_scaler(
    Y_data: np.ndarray[np.float32],
    scaler_Y_file: str | None,
) -> np.ndarray:
    """Inverse-scale a parameter array; passes through if no scaler is configured."""
    assert len(Y_data.shape) == 2
    return _apply_scaler(
        Y_data, scaler_Y_file, inverse=True, allow_missing=True
    )


def unscale_dataset_from_scaler(
    X_data: np.ndarray[np.float32],
    Y_data: np.ndarray[np.float32],
    scaler_X_file: str | None,
    scaler_Y_file: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-scale both ``X_data`` and ``Y_data``."""
    X_data_unscaled = unscale_input_from_scaler(X_data, scaler_X_file)
    Y_data_unscaled = unscale_output_from_scaler(Y_data, scaler_Y_file)
    return X_data_unscaled, Y_data_unscaled


def unscale_pred_from_scaler(
    Y_data: np.ndarray[np.float32],
    scaler_Y_file: str | None = None,
) -> np.ndarray:
    """Inverse-scale a prediction array (alias of :func:`unscale_output_from_scaler`)."""
    return unscale_output_from_scaler(Y_data, scaler_Y_file)


def unscale_pred_std_from_scaler(
    Y_std: np.ndarray[np.float32],
    scaler_Y_file: str | None = None,
) -> np.ndarray:
    """Inverse-scale a predicted standard deviation array."""
    assert len(Y_std.shape) == 2
    if scaler_Y_file is None:
        return Y_std
    try:
        scaler = _load_scaler(scaler_Y_file)
    except FileNotFoundError:
        return Y_std
    if isinstance(scaler, StandardScaler):
        return (Y_std * scaler.scale_).astype(Y_std.dtype)
    raise NotImplementedError(
        "Std unscaling is only implemented for (x - mu) / sigma scalers "
        f"(StandardScaler); got {type(scaler).__name__}"
    )
