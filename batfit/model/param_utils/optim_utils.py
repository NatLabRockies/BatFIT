"""
NPE-based protocol optimization pipelines.
"""

import numpy as np
import scipy.optimize
import torch

from .model_utils import _ProbParamFMBase
from .noise_utils import apply_noise


def predict_mu_sigma(
    X_scaled: np.ndarray,
    npe_model: torch.nn.Module,
    noise_levels: torch.Tensor,
    a_min: torch.Tensor,
    a_max: torch.Tensor,
    n_noise: int,
    device: torch.device,
    P_scaled: np.ndarray = None,
    n_samples: int = 1000,
    n_ode_steps: int = 100,
    batch_size: int = None,
) -> tuple:
    """Run a frozen NPE with noise averaging and return physical (mu, sigma).

    Each curve is tiled n_noise times
    (mu, sigma) are averaged over noise realisations.
    Dispatches on the NPE architecture and conditioning:

    - CNN-style NPE (``ProbParamCNN`` / ``ProbProtParamCNN``): one forward
      pass gives scaled (mu, gamma), mapped to physical space by
      ``npe_model.to_physical``.
    - Flow-matching NPE (``ProbParamFM`` / ``ProbProtParamFM``): draws
      n_samples posterior samples per noisy copy, mapped to physical space by
      ``npe_model.to_physical``; their mean/std are used instead.
    - ``P_scaled=None`` selects the protocol-free call signature
      (``forward(x)`` / ``sample(x, ...)``); otherwise protocol parameters
      are passed as the second argument.

    Parameters
    ----------
    X_scaled: np.ndarray
        Signal z-scored with ``npe_model.scaler_X``, shape
        ``(n_curves, channels, time)``
    npe_model: torch.nn.Module
        Frozen NPE model; its ``scaler_X`` is used to apply noise in physical
        space
    noise_levels: torch.Tensor
        Per-channel noise levels from make_noise_levels
    a_min: torch.Tensor
        Per-channel lower clip bound from make_noise_levels
    a_max: torch.Tensor
        Per-channel upper clip bound from make_noise_levels
    n_noise: int
        Number of noise realisations averaged per curve
    device: torch.device
        Compute device
    P_scaled: np.ndarray, optional
        Protocol params scaled with ``npe_model.scaler_P``, shape
        ``(n_curves, n_prot)``; None for an NPE trained without protocol
        conditioning
    n_samples: int
        FM only — posterior samples drawn per noisy copy
    n_ode_steps: int
        FM only — ODE integration steps for model.sample()
    batch_size: int, optional
        Curves processed per forward pass (None = all at once)

    Returns
    -------
    tuple
        ``(mu, sigma)`` in physical space, each shape ``(n_curves, n_deg)``
    """
    n_curves = X_scaled.shape[0]
    n_deg = npe_model.n_param_pred
    is_fm = isinstance(npe_model, _ProbParamFMBase)
    if batch_size is None:
        batch_size = n_curves

    mu_list = []
    sigma_list = []
    for start in range(0, n_curves, batch_size):
        end = min(start + batch_size, n_curves)
        B = end - start
        x_t = torch.from_numpy(X_scaled[start:end])  # (B, C, T)
        # Tile to (B * n_noise, ...) so one pass covers all realisations
        x_tiled = (
            x_t.unsqueeze(1)
            .expand(-1, n_noise, -1, -1)
            .reshape(B * n_noise, x_t.shape[1], x_t.shape[2])
        )
        x_noisy = apply_noise(
            x_tiled, npe_model.scaler_X, noise_levels, a_min, a_max
        )
        args = [x_noisy.to(device)]
        if P_scaled is not None:
            p_t = torch.from_numpy(P_scaled[start:end])  # (B, n_prot)
            p_tiled = (
                p_t.unsqueeze(1)
                .expand(-1, n_noise, -1)
                .reshape(B * n_noise, p_t.shape[1])
            )
            args.append(p_tiled.to(device))

        with torch.no_grad():
            if is_fm:
                # args is [x] or [x, p], matching the model's sample
                samples_flow = npe_model.sample(
                    *args, n_samples=n_samples, n_steps=n_ode_steps
                )  # (B*n_noise, n_samples, n_deg), flow space
                samples_phys = npe_model.to_physical(samples_flow)
                samples_phys = samples_phys.cpu().numpy()
                mu_np = samples_phys.mean(axis=1)
                sigma_np = samples_phys.std(axis=1)
            else:
                # args is [x] or [x, p], matching the model's forward
                mu_scaled, sigma_scaled = npe_model(*args)
                mu_s, sigma_s = npe_model.to_physical(mu_scaled, sigma_scaled)
                mu_np = mu_s.cpu().numpy()
                sigma_np = sigma_s.cpu().numpy()

        # Average over noise realisations
        mu_list.append(mu_np.reshape(B, n_noise, n_deg).mean(axis=1))
        sigma_list.append(sigma_np.reshape(B, n_noise, n_deg).mean(axis=1))

    mu = np.vstack(mu_list).astype("float32")
    sigma = np.vstack(sigma_list).astype("float32")
    return mu, sigma


def evaluate_sigma(
    P_scaled: np.ndarray,
    mu_scaled: np.ndarray,
    var_model: torch.nn.Module,
    device: torch.device,
) -> np.ndarray:
    """Return physical sigma for all parameters at one (P_scaled, mu_scaled).

    Parameters
    ----------
    P_scaled: np.ndarray
        Protocol params scaled with ``var_model.scaler_P``, shape
        ``(n_prot,)``
    mu_scaled: np.ndarray
        Degradation param mean scaled with ``var_model.scaler_Y``, shape
        ``(n_deg,)``
    var_model: torch.nn.Module
        Trained VariancePredFCNN
    device: torch.device
        Compute device

    Returns
    -------
    np.ndarray
        Physical sigma, shape ``(n_deg,)``
    """
    p_t = torch.from_numpy(P_scaled.reshape(1, -1)).to(device)
    mu_t = torch.from_numpy(mu_scaled.reshape(1, -1)).to(device)
    with torch.no_grad():
        sigma_scaled = var_model(p_t, mu_t)
        sigma_phys = var_model.to_physical(sigma_scaled)
    return sigma_phys.cpu().numpy().flatten()


def optimize_protocol(
    mu_scaled: np.ndarray,
    var_model: torch.nn.Module,
    param_idx: int,
    bounds: list,
    n_restarts: int,
    device: torch.device,
) -> tuple:
    """Find P_scaled that minimises the physical sigma of one parameter.

    Runs L-BFGS-B (bounded quasi-Newton) with exact gradients
    restarted from n_restarts random initial points sampled
    uniformly within bounds.

    Single-parameter objective is isolated below; a joint criterion such
    as D-optimality (e.g. minimising the sum of log-sigmas over all
    degradation parameters) can later be added by swapping that objective.

    Parameters
    ----------
    mu_scaled: np.ndarray
        Fixed degradation param mean scaled with ``var_model.scaler_Y``,
        shape ``(1, n_deg)``
    var_model: torch.nn.Module
        Trained VariancePredFCNN
    param_idx: int
        Index of the degradation parameter whose sigma is minimised
    bounds: list
        List of ``(low, high)`` tuples in scaled protocol space, one per
        protocol parameter; ``(0, 1)`` is the full protocol range
    n_restarts: int
        Number of L-BFGS-B restarts
    device: torch.device
        Compute device

    Returns
    -------
    tuple
        ``(P_scaled_opt, sigma_physical_opt)`` for the target parameter
    """
    mu_t = torch.from_numpy(mu_scaled).to(device)  # (1, n_deg)
    n_prot = len(bounds)

    def objective_and_grad(p_np: np.ndarray) -> tuple:
        p_t = torch.tensor(
            p_np.reshape(1, n_prot),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        sigma_scaled = var_model(p_t, mu_t)
        sigma_phys = var_model.to_physical(sigma_scaled)
        obj = sigma_phys[0, param_idx]
        obj.backward()
        grad = p_t.grad.detach().cpu().numpy().flatten()
        return obj.item(), grad

    lows = np.array([b[0] for b in bounds])
    highs = np.array([b[1] for b in bounds])
    best = None
    for _ in range(n_restarts):
        p0 = np.random.uniform(lows, highs)
        result = scipy.optimize.minimize(
            objective_and_grad,
            x0=p0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-8},
        )
        if best is None or result.fun < best.fun:
            best = result

    return best.x.astype("float32"), float(best.fun)
