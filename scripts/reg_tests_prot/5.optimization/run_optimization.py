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

import sys

import numpy as np
import torch

from batfit import logger
from batfit.basicutilityc import ReadInput as ri
from batfit.model.param_utils.noise_utils import make_noise_levels
from batfit.model.param_utils.optim_utils import (
    evaluate_sigma,
    optimize_protocol,
    predict_mu_sigma,
)
from batfit.model.paramNN import ProbProtParamFM
from batfit.preprocess.sim_setup import make_params
from batfit.utils.data_utils import load_pickle
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
    scaler_sigma = None
    if os.path.isfile(scaler_sigma_path):
        scaler_sigma = load_pickle(scaler_sigma_path)
        logger.info(
            "scaler_sigma.pkl found — optimising in scaled sigma space"
        )

    # --- Load scalers ---
    scaler_x = load_pickle(inp.scaler_path)
    scaler_p_npe = load_pickle(inp.scaler_P_path)
    scaler_p_vp = load_pickle(
        os.path.join(inp.var_pred_save_path, "scaler_P_varpred.pkl")
    )
    scaler_mu = load_pickle(
        os.path.join(inp.var_pred_save_path, "scaler_mu.pkl")
    )

    # scaler_Y only applies to a ProbProtParamFM NPE (trained with
    # scale_y=True); it's fit on the NPE's own training data, not var-pred data.
    scaler_y = None
    if isinstance(npe_model, ProbProtParamFM):
        scaler_y = load_pickle(os.path.join(inp.data_path, "scaler_Y.pkl"))

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
    mu_physical, _ = predict_mu_sigma(
        X_scaled,
        npe_model,
        scaler_x,
        noise_levels,
        a_min,
        a_max,
        n_noise=inp.n_noise_npe,
        device=device,
        P_scaled=P_npe_scaled,
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

    bounds = [(0.0, 1.0)] * n_prot
    for i in range(n_curves):
        mu_i = mu_scaled[i : i + 1]  # (1, n_deg)

        # Sigma at the actual protocol used to generate this curve
        sigma_init_all[i] = evaluate_sigma(
            P_sel_vp_scaled[i], mu_i, var_model, scaler_sigma, device
        )

        # Optimise
        p_opt_i, _ = optimize_protocol(
            mu_scaled=mu_i,
            var_model=var_model,
            param_idx=param_idx,
            bounds=bounds,
            n_restarts=inp.n_restarts,
            scaler_sigma=scaler_sigma,
            device=device,
        )
        P_opt_scaled[i] = p_opt_i

        sigma_opt_all[i] = evaluate_sigma(
            p_opt_i, mu_i, var_model, scaler_sigma, device
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
