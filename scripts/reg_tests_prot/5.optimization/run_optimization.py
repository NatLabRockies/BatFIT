"""
Optimize protocol parameters to minimize NPE variance for a chosen degradation
parameter, using the trained variance estimator as a differentiable objective.

For each selected test voltage curve:
  1. Run the frozen NPE (with noise averaging) to get mu (degradation param mean)
  2. Fix mu_scaled; treat protocol params P_scaled as the optimization variable
  3. Run L-BFGS-B with PyTorch autograd gradients to minimize
     sigma_pred[param_idx](P_scaled, mu_scaled)
  4. Report variance reduction and save results

Usage:
    python run_optimization.py training_recipes/recipe_optimization.yml
"""

import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import pickle
import sys

import numpy as np
import scipy.optimize
import torch

from batfit import logger
from batfit.basicutilityc import ReadInput as ri
from batfit.model.param_utils.noise_utils import apply_noise, make_noise_levels
from batfit.model.paramNN import ProbProtParamFM
from batfit.preprocess.sim_setup import make_params
from batfit.utils.data_utils import scale_input_from_scaler
from batfit.utils.torch_utils import get_device_type, load_frozen_model

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_npe(inp, device):
    """Load the best NPE checkpoint."""
    logger.info("Loading NPE")
    return load_frozen_model(inp.npe_models_dir, device)


def _load_var_pred(inp, device):
    """Load the best variance predictor checkpoint."""
    logger.info("Loading variance predictor")
    return load_frozen_model(inp.var_pred_models_dir, device)


# ---------------------------------------------------------------------------
# NPE inference
# ---------------------------------------------------------------------------


def predict_mu_batch(
    X_scaled: np.ndarray,
    P_npe_scaled: np.ndarray,
    npe_model,
    scaler_x,
    noise_levels: torch.Tensor,
    a_min: torch.Tensor,
    a_max: torch.Tensor,
    n_noise_npe: int,
    device: torch.device,
    scaler_Y=None,
    n_samples: int = 1000,
    n_ode_steps: int = 100,
) -> np.ndarray:
    """Run NPE on a batch of curves and return averaged mu in physical space.

    Tiles each curve n_noise_npe times, applies independent noise to each copy,
    and averages mu over noise realisations for a stable estimate. Works with
    either NPE architecture:

    - ProbProtParamCNN: one forward pass gives mu directly.
    - ProbProtParamFM: no closed-form mu; model.sample() draws n_samples
      posterior samples per noisy copy (in z-scored space) and their mean
      (after scaler_Y.inverse_transform) is used as that copy's mu.

    :param X_scaled: z-scored signal, shape (n_curves, channels, time)
    :param P_npe_scaled: MinMax-scaled protocol params, shape (n_curves, n_prot)
    :param scaler_Y: FM only — inverse-transforms samples from z-scored to
        physical space; required when npe_model is a ProbProtParamFM.
    :param n_samples: FM only — posterior samples drawn per noisy copy.
    :param n_ode_steps: FM only — ODE integration steps for model.sample().
    :return: physical mu, shape (n_curves, n_deg)
    """
    n_curves = X_scaled.shape[0]
    n_deg = npe_model.n_param_pred
    x_t = torch.from_numpy(X_scaled)  # (n_curves, C, T)
    p_t = torch.from_numpy(P_npe_scaled)  # (n_curves, n_prot)

    # Tile: (n_curves * n_noise_npe, …)
    x_tiled = (
        x_t.unsqueeze(1)
        .expand(-1, n_noise_npe, -1, -1)
        .reshape(n_curves * n_noise_npe, x_t.shape[1], x_t.shape[2])
    )
    p_tiled = (
        p_t.unsqueeze(1)
        .expand(-1, n_noise_npe, -1)
        .reshape(n_curves * n_noise_npe, p_t.shape[1])
    )
    x_noisy = apply_noise(x_tiled, scaler_x, noise_levels, a_min, a_max)

    with torch.no_grad():
        if isinstance(npe_model, ProbProtParamFM):
            samples_z = npe_model.sample(
                x_noisy.to(device),
                p_tiled.to(device),
                n_samples=n_samples,
                n_steps=n_ode_steps,
            )  # (n_curves*n_noise_npe, n_samples, n_deg), z-scored
            samples_phys = scaler_Y.inverse_transform(
                samples_z.cpu().numpy().reshape(-1, n_deg)
            ).reshape(n_curves * n_noise_npe, n_samples, n_deg)
            mu_np = samples_phys.mean(axis=1)  # (n_curves*n_noise_npe, n_deg)
        else:
            mu_s, _ = npe_model(x_noisy.to(device), p_tiled.to(device))
            if npe_model.constrain_output:
                mu_s = npe_model.inv_transform_mu(
                    mu_s,
                    npe_model.min_par.to(device),
                    npe_model.amp_par.to(device),
                )
            mu_np = mu_s.cpu().numpy()
    mu_np = mu_np.reshape(n_curves, n_noise_npe, n_deg)
    return mu_np.mean(axis=1).astype("float32")  # (n_curves, n_deg)


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------


def _sigma_physical(
    sigma_out: torch.Tensor,
    var_model,
    scale_sigma: bool,
    scaler_sigma,
    device: torch.device,
) -> torch.Tensor:
    """Convert var-pred Sigmoid output to physical sigma."""
    if scale_sigma:
        # sigma_out is in [0,1] scaled space; inverse-transform to physical
        # We do this differentiably by reversing the MinMax transform:
        # sigma_physical_j = sigma_scaled_j * (max_j - min_j) + min_j
        scale = torch.tensor(
            scaler_sigma.scale_, dtype=torch.float32, device=device
        )  # (n_deg,) — 1/(max-min)
        min_val = torch.tensor(
            scaler_sigma.data_min_, dtype=torch.float32, device=device
        )
        # inverse: x_physical = x_scaled / scale + min_val
        return sigma_out / scale + min_val
    else:
        return var_model.inv_transform_gamma(
            sigma_out, var_model.amp_par.to(device)
        )


def optimize_single_curve(
    mu_scaled: np.ndarray,
    var_model,
    param_idx: int,
    n_prot: int,
    n_restarts: int,
    scale_sigma: bool,
    scaler_sigma,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    """Find P_scaled that minimises sigma_physical[param_idx] for a fixed mu.

    Uses L-BFGS-B (bounded quasi-Newton) with exact PyTorch gradients.

    :param mu_scaled: fixed scaled degradation param mean, shape (1, n_deg)
    :return: (P_scaled_opt, sigma_physical_opt) for the target parameter
    """
    mu_t = torch.from_numpy(mu_scaled).to(device)  # (1, n_deg)

    def objective_and_grad(p_np: np.ndarray) -> tuple[float, np.ndarray]:
        p_t = torch.tensor(
            p_np.reshape(1, n_prot),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        sigma_out = var_model(p_t, mu_t)
        sigma_phys = _sigma_physical(
            sigma_out, var_model, scale_sigma, scaler_sigma, device
        )
        obj = sigma_phys[0, param_idx]
        obj.backward()
        grad = p_t.grad.detach().cpu().numpy().flatten()
        return obj.item(), grad

    bounds = [(0.0, 1.0)] * n_prot
    best = None

    for _ in range(n_restarts):
        p0 = np.random.uniform(0.0, 1.0, n_prot)
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


def evaluate_sigma(
    P_scaled: np.ndarray,
    mu_scaled: np.ndarray,
    var_model,
    scale_sigma: bool,
    scaler_sigma,
    device: torch.device,
) -> np.ndarray:
    """Return physical sigma for all parameters at a given (P_scaled, mu_scaled)."""
    p_t = torch.from_numpy(P_scaled.reshape(1, -1)).to(device)
    mu_t = torch.from_numpy(mu_scaled.reshape(1, -1)).to(device)
    with torch.no_grad():
        sigma_out = var_model(p_t, mu_t)
        sigma_phys = _sigma_physical(
            sigma_out, var_model, scale_sigma, scaler_sigma, device
        )
    return sigma_phys.cpu().numpy().flatten()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_optimization(inp) -> None:
    """Run protocol optimization for n_curves test voltage curves.

    :param inp: recipe object from recipe_optimization.yml
    """
    np.random.seed(inp.random_seed)
    os.makedirs(inp.save_path, exist_ok=True)

    device = torch.device(get_device_type())

    # --- Load models ---
    npe_model = _load_npe(inp, device)
    var_model = _load_var_pred(inp, device)

    # --- Detect sigma scaling ---
    scaler_sigma_path = os.path.join(
        inp.var_pred_save_path, "scaler_sigma.pkl"
    )
    scale_sigma = os.path.isfile(scaler_sigma_path)
    scaler_sigma = None
    if scale_sigma:
        with open(scaler_sigma_path, "rb") as f:
            scaler_sigma = pickle.load(f)
        logger.info(
            "scaler_sigma.pkl found — optimising in scaled sigma space"
        )

    # --- Load scalers ---
    with open(inp.scaler_path, "rb") as f:
        scaler_x = pickle.load(f)
    with open(inp.scaler_P_path, "rb") as f:
        scaler_p_npe = pickle.load(f)
    with open(
        os.path.join(inp.var_pred_save_path, "scaler_P_varpred.pkl"), "rb"
    ) as f:
        scaler_p_vp = pickle.load(f)
    with open(
        os.path.join(inp.var_pred_save_path, "scaler_mu.pkl"), "rb"
    ) as f:
        scaler_mu = pickle.load(f)

    # scaler_Y only applies to a ProbProtParamFM NPE (trained with
    # scale_y=True); it's fit on the NPE's own training data, not var-pred data.
    scaler_y = None
    if isinstance(npe_model, ProbProtParamFM):
        with open(os.path.join(inp.data_path, "scaler_Y.pkl"), "rb") as f:
            scaler_y = pickle.load(f)

    # --- Load test data ---
    split_file = os.path.join(inp.data_path, "data_split.npz")
    assert os.path.isfile(
        split_file
    ), f"data_split.npz not found at {split_file}"
    split_data = np.load(split_file)
    X_test = split_data["X_test"]  # (N_test, channels, time) — physical
    P_test = split_data["P_test"]  # (N_test, n_prot) — physical
    Y_test = split_data["Y_test"]  # (N_test, n_deg)  — physical (ground truth)

    n_test = X_test.shape[0]
    n_curves = min(inp.n_curves, n_test)
    indices = np.random.choice(n_test, size=n_curves, replace=False)
    logger.info(f"Selected {n_curves} test curves: {indices}")

    X_sel = X_test[indices]  # (n_curves, C, T)
    P_sel = P_test[indices]  # (n_curves, n_prot)

    # --- Resolve parameter index ---
    sim_params = make_params(inp.sim_config)
    param_names = sim_params["deg_param_names"]
    assert (
        inp.param_to_minimize in param_names
    ), f"param_to_minimize='{inp.param_to_minimize}' not in {param_names}"
    param_idx = param_names.index(inp.param_to_minimize)
    logger.info(
        f"Minimising sigma of '{inp.param_to_minimize}' (index {param_idx})"
    )

    # --- Noise settings for NPE ---
    noise_levels, a_min, a_max = make_noise_levels(
        target_mode=inp.target_mode,
        noise_levels=[
            0,
            0.001444 * 2 * inp.noise_factor,
            0.001786 * 2,
            2.01 * 2,
        ],
        cyc_mode=inp.cyc_mode,
    )

    # --- Batch NPE inference over all selected curves ---
    X_scaled = scaler_x.transform(X_sel).astype("float32")
    P_npe_scaled = scaler_p_npe.transform(P_sel).astype("float32")

    logger.info(
        f"Running NPE on {n_curves} curves (n_noise_npe={inp.n_noise_npe}) …"
    )
    mu_physical = predict_mu_batch(
        X_scaled=X_scaled,
        P_npe_scaled=P_npe_scaled,
        npe_model=npe_model,
        scaler_x=scaler_x,
        noise_levels=noise_levels,
        a_min=a_min,
        a_max=a_max,
        n_noise_npe=inp.n_noise_npe,
        device=device,
        scaler_Y=scaler_y,
        n_samples=getattr(inp, "n_samples", 1000),
        n_ode_steps=getattr(inp, "n_ode_steps", 100),
    )  # (n_curves, n_deg)
    mu_scaled = scaler_mu.transform(mu_physical).astype("float32")

    # --- Optimize protocol for each curve ---
    n_prot = P_sel.shape[1]
    P_opt_scaled = np.zeros((n_curves, n_prot), dtype="float32")
    sigma_init_all = np.zeros((n_curves, len(param_names)), dtype="float32")
    sigma_opt_all = np.zeros((n_curves, len(param_names)), dtype="float32")

    # Scale the actual test P for the initial-sigma evaluation
    P_sel_vp_scaled = scaler_p_vp.transform(P_sel).astype("float32")

    for i in range(n_curves):
        mu_i = mu_scaled[i : i + 1]  # (1, n_deg)

        # Sigma at the actual protocol used to generate this curve
        sigma_init_all[i] = evaluate_sigma(
            P_sel_vp_scaled[i],
            mu_i,
            var_model,
            scale_sigma,
            scaler_sigma,
            device,
        )

        # Optimise
        p_opt_i, _ = optimize_single_curve(
            mu_scaled=mu_i,
            var_model=var_model,
            param_idx=param_idx,
            n_prot=n_prot,
            n_restarts=inp.n_restarts,
            scale_sigma=scale_sigma,
            scaler_sigma=scaler_sigma,
            device=device,
        )
        P_opt_scaled[i] = p_opt_i

        sigma_opt_all[i] = evaluate_sigma(
            p_opt_i, mu_i, var_model, scale_sigma, scaler_sigma, device
        )

        reduction = (
            (sigma_init_all[i, param_idx] - sigma_opt_all[i, param_idx])
            / sigma_init_all[i, param_idx]
            * 100
        )
        logger.info(
            f"Curve {i:3d} | sigma_init={sigma_init_all[i, param_idx]:.4e} "
            f"sigma_opt={sigma_opt_all[i, param_idx]:.4e} "
            f"reduction={reduction:.1f}%"
        )

    # Unscale optimised P to physical units
    P_opt_physical = scaler_p_vp.inverse_transform(P_opt_scaled).astype(
        "float32"
    )

    # --- Save results ---
    results_file = os.path.join(inp.save_path, "optimization_results.npz")
    np.savez(
        results_file,
        indices=indices,
        mu_physical=mu_physical,
        P_test=P_sel,
        P_opt=P_opt_physical,
        sigma_init=sigma_init_all,
        sigma_opt=sigma_opt_all,
        param_names=np.array(param_names),
        param_to_minimize=inp.param_to_minimize,
    )
    logger.info(f"Results saved to {results_file}")

    # Summary
    mean_reduction = (
        (sigma_init_all[:, param_idx] - sigma_opt_all[:, param_idx])
        / sigma_init_all[:, param_idx]
        * 100
    ).mean()
    logger.info(
        f"Mean sigma reduction for '{inp.param_to_minimize}': {mean_reduction:.1f}%"
    )
    logger.info(f"Optimised P (physical) — mean over curves:")
    prot_names = sim_params["prot_param_names"]
    for j, name in enumerate(prot_names):
        logger.info(
            f"  {name}: {P_opt_physical[:, j].mean():.4f} "
            f"± {P_opt_physical[:, j].std():.4f}"
        )


if __name__ == "__main__":
    inp = ri.basic_input(sys.argv[1])
    run_optimization(inp)
