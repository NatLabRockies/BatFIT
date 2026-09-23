"""End-time encoding of a ``(time, voltage)`` signal."""

import numpy as np
import torch

# - "zscore": the (time, voltage) signal, z-scored per channel
# - "time_dependent_zscore": the voltage on the rescaled time grid [0, 1],
#   z-scored per time point, plus the end time t_end as a separate input
SIGNAL_SCALINGS = ("zscore", "time_dependent_zscore")


def split_end_time(
    X: np.ndarray | torch.Tensor, atol: float = 1e-3
) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
    """Split a ``(time, voltage)`` signal into its voltage and end time.

    Each signal is sampled on an equidistant grid ``linspace(0, t_end, n)``,
    so the time channel is fully described by ``t_end``.

    Parameters
    ----------
    X : numpy.ndarray or torch.Tensor
        Physical signal of shape ``(N, 2, n_points)``; channel 0 is time (s),
        channel 1 is voltage (V).
    atol : float
        Tolerance (s) on the grid starting at time 0.

    Returns
    -------
    tuple
        ``(V, T)``: the voltage of shape ``(N, 1, n_points)``, a view of
        ``X`` (in-place changes to ``V`` modify ``X``), and the end times of
        shape ``(N, 1)``.
    """
    assert X.ndim == 3 and X.shape[1] == 2, (
        f"Expected a (N, 2, n_points) (time, voltage) signal, got "
        f"{tuple(X.shape)}"
    )
    # the grid must start at 0 for t_end to describe it
    assert float(abs(X[:, 0, 0]).max()) <= atol, "Time grids must start at 0"
    V = X[:, 1:, :]
    T = X[:, 0, -1:]
    return V, T
