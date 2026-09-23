import numpy as np
import torch


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
    if isinstance(data, np.ndarray):
        return buffer.detach().cpu().numpy().astype(data.dtype)
    if isinstance(data, torch.Tensor):
        return buffer.to(dtype=data.dtype, device=data.device)
    # other tensor-like inputs (e.g. torch2jax tracing, where the buffer is
    # converted alongside the data) combine with the buffer as it is
    return buffer


def _readable_list(buffer: torch.Tensor) -> list:
    """Nested list of a float32 buffer, rounded to float32 precision.

    Used for the human-readable ``scaling.json`` export, so that e.g. ``0.1``
    is written as ``0.1`` rather than ``0.10000000149011612``.
    """
    values = buffer.detach().cpu().numpy()
    rounded = np.vectorize(lambda v: float(f"{v:.7g}"), otypes=[float])(values)
    return rounded.tolist()


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
        # ``** -1`` instead of ``/``: torch2jax (MCMC through the surrogate)
        # supports neither ``/`` nor ``torch.div`` on traced tensors
        return (data - low) * (high - low) ** -1

    def transform_(self, data: np.ndarray) -> np.ndarray:
        """Scale a float numpy array in place, without allocating a copy."""
        low = _buffer_like(self.low, data)
        high = _buffer_like(self.high, data)
        data -= low
        data /= high - low
        return data

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
        return {
            "low": _readable_list(self.low),
            "high": _readable_list(self.high),
        }


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
        # ``** -1`` instead of ``/``: see BoundedScaler.transform
        return (data - means) * stds**-1

    def transform_(self, data: np.ndarray) -> np.ndarray:
        """Scale a float numpy array in place, without allocating a copy."""
        means, stds = self._stats_for(data)
        data -= means
        data /= stds
        return data

    def inverse_transform(
        self, data_scaled: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        """Invert :meth:`transform`, returning data in physical units."""
        means, stds = self._stats_for(data_scaled)
        return data_scaled * stds + means

    def to_dict(self) -> dict[str, list]:
        """Return the statistics as nested lists, for a JSON export."""
        return {
            "means": _readable_list(self.means),
            "stds": _readable_list(self.stds),
        }


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
        # written without unary minus, which torch2jax does not support
        return (1.0 + 2.0 * self.margin) * torch.sigmoid(z) - self.margin

    def to_dict(self) -> dict[str, float]:
        """Return the margin, for a JSON export."""
        return {"margin": float(f"{float(self.margin):.7g}")}


def scaling_to_dict(model: torch.nn.Module) -> dict:
    """Collect the scalers and margin heads of a model into a plain dict.

    Every :class:`BoundedScaler`, :class:`ZScoreScaler` and
    :class:`MarginSigmoid` submodule is listed under its attribute path (e.g.
    ``"scaler_Y"``), with its type and statistics. The parameter names of the
    experiment configuration are added when the model holds them, so that the
    bounds can be read per parameter.

    Parameters
    ----------
    model : torch.nn.Module
        Model holding scaler submodules.

    Returns
    -------
    dict
        JSON-serialisable summary; ``{"scalers": {}}`` when the model holds
        no scaler.
    """
    summary: dict = {"scalers": {}}
    for name, module in model.named_modules():
        if isinstance(module, (BoundedScaler, ZScoreScaler, MarginSigmoid)):
            summary["scalers"][name] = {
                "type": type(module).__name__,
                **module.to_dict(),
            }
    sim_params = getattr(model, "sim_params", None)
    if sim_params is not None:
        summary["sim_config"] = str(getattr(model, "sim_config", None))
        for key in ("deg_param_names", "prot_param_names"):
            if key in sim_params:
                summary[key] = list(sim_params[key])
    return summary
