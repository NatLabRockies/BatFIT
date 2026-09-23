"""
NPE-based protocol optimization pipelines.
"""

import numpy as np
import scipy.optimize
import torch
from sklearn.preprocessing import StandardScaler

from .model_utils import _ProbParamFMBase
from .noise_utils import apply_noise


def predict_mu_sigma(
    X_scaled: np.ndarray,
    npe_model: torch.nn.Module,
    scaler_x,
    noise_levels: torch.Tensor,
    a_min: torch.Tensor,
    a_max: torch.Tensor,
    n_noise: int,
    device: torch.device,
    P_scaled: np.ndarray = None,
    scaler_Y=None,
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
    - Flow-matching NPE (``ProbParamFM`` / ``ProbProtParamFM``): draws n_samples posterior
      samples per noisy copy (z-scored) whose mean/std after
      ``scaler_Y.inverse_transform`` are used instead.
    - ``P_scaled=None`` selects the protocol-free call signature
      (``forward(x)`` / ``sample(x, ...)``); otherwise protocol parameters
      are passed as the second argument.

    Parameters
    ----------
    X_scaled: np.ndarray
        Z-scored signal, shape ``(n_curves, channels, time)``
    npe_model: torch.nn.Module
        Frozen NPE model
    scaler_x: CustomScaler
        The NPE's CustomScaler (needed by apply_noise)
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
        MinMax-scaled protocol params, shape ``(n_curves, n_prot)``; None
        for an NPE trained without protocol conditioning
    scaler_Y: object, optional
        FM only — inverse-transforms posterior samples from z-scored to
        physical space; required for a flow-matching NPE
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
    if is_fm:
        assert scaler_Y is not None, "scaler_Y is required for an FM NPE"
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
        x_noisy = apply_noise(x_tiled, scaler_x, noise_levels, a_min, a_max)
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
                samples_z = npe_model.sample(
                    *args, n_samples=n_samples, n_steps=n_ode_steps
                )  # (B*n_noise, n_samples, n_deg), z-scored
                samples_phys = scaler_Y.inverse_transform(
                    samples_z.cpu().numpy().reshape(-1, n_deg)
                ).reshape(B * n_noise, n_samples, n_deg)
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


def sigma_physical(
    sigma_out: torch.Tensor,
    var_model: torch.nn.Module,
    scaler_sigma,
    device: torch.device,
) -> torch.Tensor:
    """Convert the variance estimator's raw output to physical sigma.

    - StandardScaler (``log_sigma: true``): output is z-scored log sigma;
      sigma = exp(out * scale_ + mean_)
    - MinMaxScaler (``scale_sigma: true``): output is MinMax-scaled sigma;
      sigma = out / scale_ + data_min_
    - None: Sigmoid output rescaled via inv_transform_gamma (amp_par)

    Parameters
    ----------
    sigma_out: torch.Tensor
        Raw VariancePredFCNN output
    var_model: torch.nn.Module
        The variance estimator (provides inv_transform_gamma)
    scaler_sigma: object
        The sigma scaler saved by gen_var_dataset.py
        (``scaler_logsigma.pkl`` or ``scaler_sigma.pkl``), or None
    device: torch.device
        Compute device

    Returns
    -------
    torch.Tensor
        Sigma in physical space, same shape as sigma_out
    """
    if isinstance(scaler_sigma, StandardScaler):
        # Reverse z-scored log sigma: sigma = exp(z * scale + mean)
        scale = torch.tensor(
            scaler_sigma.scale_, dtype=torch.float32, device=device
        )
        mean = torch.tensor(
            scaler_sigma.mean_, dtype=torch.float32, device=device
        )
        return torch.exp(sigma_out * scale + mean)
    if scaler_sigma is not None:
        # Reverse the MinMax transform: x_physical = x_scaled / scale + min
        scale = torch.tensor(
            scaler_sigma.scale_, dtype=torch.float32, device=device
        )
        min_val = torch.tensor(
            scaler_sigma.data_min_, dtype=torch.float32, device=device
        )
        return sigma_out / scale + min_val
    return var_model.inv_transform_gamma(
        sigma_out, var_model.amp_par.to(device)
    )


def evaluate_sigma(
    P_scaled: np.ndarray,
    mu_scaled: np.ndarray,
    var_model: torch.nn.Module,
    scaler_sigma,
    device: torch.device,
) -> np.ndarray:
    """Return physical sigma for all parameters at one (P_scaled, mu_scaled).

    Parameters
    ----------
    P_scaled: np.ndarray
        Protocol params in the variance estimator's MinMax space,
        shape ``(n_prot,)``
    mu_scaled: np.ndarray
        MinMax-scaled degradation param mean, shape ``(n_deg,)``
    var_model: torch.nn.Module
        Trained VariancePredFCNN
    scaler_sigma: object
        Sigma scaler or None (see sigma_physical)
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
        sigma_out = var_model(p_t, mu_t)
        sigma_phys = sigma_physical(sigma_out, var_model, scaler_sigma, device)
    return sigma_phys.cpu().numpy().flatten()


def optimize_protocol(
    mu_scaled: np.ndarray,
    var_model: torch.nn.Module,
    param_idx: int,
    bounds: list,
    n_restarts: int,
    scaler_sigma,
    device: torch.device,
) -> tuple:
    """Find P_scaled that minimises sigma_physical[param_idx] for a fixed mu.

    Runs L-BFGS-B (bounded quasi-Newton) with exact gradients
    restarted from n_restarts random initial points sampled
    uniformly within bounds.

    Single-parameter objective is isolated below; a joint criterion such
    as D-optimality (e.g. minimising the sum of log-sigmas over all
    degradation parameters) can later be added by swapping that objective.

    Parameters
    ----------
    mu_scaled: np.ndarray
        Fixed MinMax-scaled degradation param mean, shape ``(1, n_deg)``
    var_model: torch.nn.Module
        Trained VariancePredFCNN
    param_idx: int
        Index of the degradation parameter whose sigma is minimised
    bounds: list
        List of ``(low, high)`` tuples in scaled protocol space, one per
        protocol parameter
    n_restarts: int
        Number of L-BFGS-B restarts
    scaler_sigma: object
        Sigma scaler or None (see sigma_physical)
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
        sigma_out = var_model(p_t, mu_t)
        sigma_phys = sigma_physical(sigma_out, var_model, scaler_sigma, device)
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
