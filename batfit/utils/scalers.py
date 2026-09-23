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
        ``stat`` unchanged when ``data`` is a numpy array; 
        otherwise a ``torch.Tensor`` on ``data``'s device and dtype. 
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


def _buffer_like(
    buffer: torch.Tensor, data: np.ndarray | torch.Tensor
) -> np.ndarray | torch.Tensor:
    """Cast a scaler buffer to match ``data``'s array type, dtype and device.

    Parameters
    ----------
    buffer : torch.Tensor
        A statistic stored as a registered buffer on a scaler module.
    data : numpy.ndarray or torch.Tensor
        The array the statistic will be combined with.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        A numpy array of ``data``'s dtype when ``data`` is a numpy array;
        otherwise a tensor on ``data``'s device and dtype.
    """
    if isinstance(data, torch.Tensor):
        return buffer.to(dtype=data.dtype, device=data.device)
    return buffer.detach().cpu().numpy().astype(data.dtype)


class BoundedScaler(torch.nn.Module):
    """Affine scaler mapping physical bounds ``[low, high]`` onto ``[0, 1]``.

    The bounds are fixed (taken from the experiment configuration)
    Statistics are registered buffers, so they are saved in
    the ``state_dict``
    """

    def __init__(
        self, low: np.ndarray | list[float], high: np.ndarray | list[float]
    ) -> None:
        """Store the per-feature lower and upper physical bounds."""
        super().__init__()
        low_t = torch.as_tensor(np.asarray(low, dtype="float32"))
        high_t = torch.as_tensor(np.asarray(high, dtype="float32"))
        assert low_t.ndim == 1 and low_t.shape == high_t.shape
        assert torch.all(high_t > low_t), "Upper bounds must exceed lower ones"
        self.register_buffer("low", low_t)
        self.register_buffer("high", high_t)

    @classmethod
    def from_sim_params(cls, sim_params: dict, kind: str) -> "BoundedScaler":
        """Build the scaler from the bounds of a parsed experiment config.

        Parameters
        ----------
        sim_params : dict
            Output of :func:`batfit.preprocess.sim_setup.make_params`.
        kind : str
            ``"deg"`` (degradation parameters), ``"prot"`` (protocol
            parameters) or ``"voltage"`` (``vmin``/``vmax``).

        Returns
        -------
        BoundedScaler
            Scaler over the requested quantity, one feature per parameter
            (a single feature for ``"voltage"``).
        """
        if kind == "voltage":
            return cls([sim_params["vmin"]], [sim_params["vmax"]])
        assert kind in ("deg", "prot"), f"Unknown bounded quantity {kind}"
        names = sim_params[f"{kind}_param_names"]
        low = [sim_params[f"{kind}_{name}_min"] for name in names]
        high = [sim_params[f"{kind}_{name}_max"] for name in names]
        return cls(low, high)

    def transform(
        self, data: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        """Return ``(data - low) / (high - low)``."""
        low = _buffer_like(self.low, data)
        high = _buffer_like(self.high, data)
        return (data - low) / (high - low)

    def inverse_transform(
        self, data_scaled: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        """Invert :meth:`transform`, returning data in physical units."""
        low = _buffer_like(self.low, data_scaled)
        high = _buffer_like(self.high, data_scaled)
        return data_scaled * (high - low) + low

    def inverse_transform_std(
        self, std_scaled: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        """Map a standard deviation from scaled to physical units."""
        low = _buffer_like(self.low, std_scaled)
        high = _buffer_like(self.high, std_scaled)
        return std_scaled * (high - low)

    def clip_physical(
        self, data: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        """Clamp physical ``data`` to ``[low, high]``."""
        low = _buffer_like(self.low, data)
        high = _buffer_like(self.high, data)
        if isinstance(data, torch.Tensor):
            return torch.clamp(data, min=low, max=high)
        return np.clip(data, low, high)

    def to_dict(self) -> dict[str, list[float]]:
        """Return the bounds as plain lists, for a JSON export."""
        return {"low": self.low.tolist(), "high": self.high.tolist()}


class ZScoreScaler(torch.nn.Module):
    """Per-channel z-score scaler with statistics stored as buffers.

    Statistics are registered buffers so they are saved in the ``state_dict``
    Falls back to the channel-1 statistics for a single-channel
    """

    def __init__(self, means: np.ndarray, stds: np.ndarray) -> None:
        """Store means and standard deviations (dims kept for broadcasting)."""
        super().__init__()
        self.register_buffer(
            "means", torch.as_tensor(np.asarray(means, dtype="float32"))
        )
        self.register_buffer(
            "stds", torch.as_tensor(np.asarray(stds, dtype="float32"))
        )

    @classmethod
    def fit(
        cls, data: np.ndarray, axis: int | tuple[int, ...]
    ) -> "ZScoreScaler":
        """Fit the scaler on ``data``, reducing over ``axis`` with dims kept.

        Parameters
        ----------
        data : numpy.ndarray
            Training array, e.g. ``(N, channels, time)``.
        axis : int or tuple of int
            Axes to reduce over, e.g. ``(0, 2)`` for per-channel statistics.

        Returns
        -------
        ZScoreScaler
            The fitted scaler.
        """
        means = np.mean(data, axis=axis, keepdims=True)
        stds = np.std(data, axis=axis, keepdims=True)
        return cls(means, stds)

    def _stats_for(
        self, data: np.ndarray | torch.Tensor
    ) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
        """Return ``(means, stds)`` matched to ``data``'s type and channels."""
        assert data.ndim == self.means.ndim
        means, stds = self.means, self.stds
        if data.ndim == 3 and stds.shape[1] == 2 and data.shape[1] == 1:
            # single-channel input: use the channel-1 statistics
            means, stds = means[:, 1:2, :], stds[:, 1:2, :]
        return _buffer_like(means, data), _buffer_like(stds, data)

    def transform(
        self, data: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        """Return ``(data - means) / stds``, broadcasting over channels."""
        means, stds = self._stats_for(data)
        return (data - means) / stds

    def inverse_transform(
        self, data_scaled: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        """Invert :meth:`transform`, returning data in physical units."""
        means, stds = self._stats_for(data_scaled)
        return data_scaled * stds + means

    def to_dict(self) -> dict[str, list]:
        """Return the statistics as nested lists, for a JSON export."""
        return {"means": self.means.tolist(), "stds": self.stds.tolist()}


class MarginSigmoid(torch.nn.Module):
    """Sigmoid output head spanning ``[-margin, 1 + margin]``.

    Returns ``-margin + (1 + 2 * margin) * sigmoid(z)``. Used as the final
    activation of a network predicting a quantity scaled to ``[0, 1]``: the
    margin keeps the bounds ``0`` and ``1`` reachable with finite logits.
    """

    def __init__(self, margin: float = 0.05) -> None:
        """Store the margin as a buffer."""
        super().__init__()
        assert margin >= 0.0
        self.register_buffer("margin", torch.tensor(float(margin)))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Apply the widened sigmoid to ``z``."""
        return -self.margin + (1.0 + 2.0 * self.margin) * torch.sigmoid(z)

    def to_dict(self) -> dict[str, float]:
        """Return the margin, for a JSON export."""
        return {"margin": float(self.margin)}


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
