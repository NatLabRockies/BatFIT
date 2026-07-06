"""Scaler class and helpers to apply/invert a persisted (pickled) scaler.

All public functions here operate on a *path* to a pickled scaler object
(one with ``.transform``/``.inverse_transform`` methods, e.g. :class:`CustomScaler`
or a scikit-learn scaler) rather than the scaler object itself, so callers at
inference/test time don't need to keep the fitted scaler around in memory.
"""

import pickle

import numpy as np


class CustomScaler:
    """Per-channel z-score scaler for 3D signal arrays ``(N, channels, time)``.

    Falls back to the channel-1 statistics when asked to transform a
    single-channel array against means/stds fitted on a 2-channel array
    (used when a downstream model only consumes the voltage channel).
    """

    def __init__(self, means: np.ndarray, stds: np.ndarray) -> None:
        """Store the per-channel means and standard deviations used for scaling."""
        self.means = means
        self.stds = stds

    @classmethod
    def fit(
        cls, data: np.ndarray, axis: int | tuple[int, ...]
    ) -> "CustomScaler":
        """Fit a scaler from ``data``, reducing over ``axis`` with dims kept.

        Mirrors the ``.fit()`` interface of scikit-learn scalers so callers
        can treat :class:`CustomScaler` uniformly alongside them.
        """
        means = np.mean(data, axis=axis, keepdims=True)
        stds = np.std(data, axis=axis, keepdims=True)
        return cls(means, stds)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Return ``(data - means) / stds``, broadcasting over channels."""
        assert len(data.shape) == len(self.means.shape)
        assert len(data.shape) == len(self.stds.shape)
        if self.stds.shape[1] == 2 and data.shape[1] == 1:
            transformed_data = (data - self.means[:, 1, :]) / self.stds[
                :, 1, :
            ]
        else:
            transformed_data = (data - self.means) / self.stds
        assert transformed_data.shape == data.shape
        return transformed_data

    def inverse_transform(self, transformed_data: np.ndarray) -> np.ndarray:
        """Invert :meth:`transform`, returning data in physical units."""
        assert len(transformed_data.shape) == len(self.means.shape)
        assert len(transformed_data.shape) == len(self.stds.shape)
        if self.stds.shape[1] == 2 and transformed_data.shape[1] == 1:
            data = transformed_data * self.stds[:, 1, :] + self.means[:, 1, :]
        else:
            data = transformed_data * self.stds + self.means
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

    :param inverse: apply ``inverse_transform`` instead of ``transform``.
    :param allow_missing: when True, a ``None`` path or a missing file
        means "no scaling configured" and ``data`` is returned unchanged
        instead of raising.
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
