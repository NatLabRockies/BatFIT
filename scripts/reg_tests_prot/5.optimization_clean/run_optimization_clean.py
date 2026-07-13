"""
Evaluate the benefit of adding a chirp to a plain 1C charge.

For each selected nochirp (chargecc) test curve:
  1. Run an NPE trained WITHOUT chirp to get (mu, sigma_nochirp) — the
     state-of-health estimate and its uncertainty from the plain charge.
  2. Fix mu and recommend a chirp: minimise the chirp-side variance
     estimator sigma over the protocol parameters (L-BFGS-B, autograd
     gradients), looping the target over every degradation parameter.
  3. Also minimise with the amplitude clamped to 0 (physically a plain 1C
     charge) as an internal-consistency baseline of the variance estimator.

Results are saved to a single npz; plots are made by
plot_optimization_clean.py.

Usage:
    python run_optimization_clean.py training_recipes/recipe_clean.yml
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
from batfit.model.paramNN import ProbParamFM
from batfit.preprocess.sim_setup import make_params
from batfit.utils.data_utils import load_pickle
from batfit.utils.torch_utils import get_device_type, load_frozen_model


def run_optimization_clean(inp) -> None:
    """Quantify chirp-induced variance reduction over nochirp observations.

    :param inp: recipe object from recipe_clean.yml
    """
    np.random.seed(inp.random_seed)
    os.makedirs(inp.save_path, exist_ok=True)
    device = torch.device(get_device_type())

    # --- Models and scalers ---
    npe = load_frozen_model(inp.nochirp_npe_models_dir, device)
    var_model = load_frozen_model(inp.var_pred_models_dir, device)
    scaler_x = load_pickle(os.path.join(inp.nochirp_data_path, "scaler_X.pkl"))
    scaler_mu = load_pickle(
        os.path.join(inp.var_pred_save_path, "scaler_mu.pkl")
    )
    scaler_p_vp = load_pickle(
        os.path.join(inp.var_pred_save_path, "scaler_P_varpred.pkl")
    )
    scaler_sigma_path = os.path.join(
        inp.var_pred_save_path, "scaler_sigma.pkl"
    )
    scaler_sigma = (
        load_pickle(scaler_sigma_path)
        if os.path.isfile(scaler_sigma_path)
        else None
    )
    # scaler_Y only applies to a flow-matching nochirp NPE (scale_y=True)
    scaler_y = None
    if isinstance(npe, ProbParamFM):
        scaler_y = load_pickle(
            os.path.join(inp.nochirp_data_path, "scaler_Y.pkl")
        )

    # --- Nochirp observations (test split, ground truth kept for plots) ---
    A = np.load(os.path.join(inp.nochirp_data_path, "data_split.npz"))
    X_test, Y_test = A["X_test"], A["Y_test"]
    n_curves = min(inp.n_curves, X_test.shape[0])
    indices = np.random.choice(X_test.shape[0], size=n_curves, replace=False)
    X_sel = X_test[indices]
    Y_sel = Y_test[indices]
    logger.info(f"Selected {n_curves} nochirp test curves")

    # --- Parameter names and protocol bounds (from the chirp config) ---
    sim_params = make_params(inp.sim_config)
    param_names = sim_params["deg_param_names"]
    prot_names = sim_params["prot_param_names"]
    n_deg = len(param_names)
    n_prot = len(prot_names)
    amp_idx = prot_names.index("amplitude")

    # --- NPE inference: (mu, sigma) from the plain charge ---
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
    X_scaled = scaler_x.transform(X_sel).astype("float32")
    mu_physical, sigma_nochirp = predict_mu_sigma(
        X_scaled,
        npe,
        scaler_x,
        noise_levels,
        a_min,
        a_max,
        n_noise=inp.n_noise_npe,
        device=device,
        scaler_Y=scaler_y,
        n_samples=getattr(inp, "n_samples", 1000),
        n_ode_steps=getattr(inp, "n_ode_steps", 100),
    )
    # scaler_mu was fitted on chirp-NPE mus; nochirp mus may fall slightly
    # outside [0, 1]
    mu_scaled = scaler_mu.transform(mu_physical).astype("float32")

    # --- Optimize the chirp for each target parameter and curve ---
    bounds_full = [(0.0, 1.0)] * n_prot
    # Amplitude clamped to the scaled lower bound = smallest amplitude seen
    # in the variance-estimator training data (~0, i.e. no chirp)
    bounds_amp0 = list(bounds_full)
    bounds_amp0[amp_idx] = (0.0, 0.0)

    P_opt = np.zeros((n_deg, n_curves, n_prot), dtype="float32")
    sigma_opt = np.zeros((n_deg, n_curves, n_deg), dtype="float32")
    sigma_amp0 = np.zeros((n_deg, n_curves, n_deg), dtype="float32")

    for k, name in enumerate(param_names):
        logger.info(f"Optimizing chirp for '{name}' ({k + 1}/{n_deg})")
        for i in range(n_curves):
            mu_i = mu_scaled[i : i + 1]  # (1, n_deg)
            p_opt, _ = optimize_protocol(
                mu_i,
                var_model,
                k,
                bounds_full,
                inp.n_restarts,
                scaler_sigma,
                device,
            )
            P_opt[k, i] = p_opt
            sigma_opt[k, i] = evaluate_sigma(
                p_opt, mu_i.flatten(), var_model, scaler_sigma, device
            )
            p_amp0, _ = optimize_protocol(
                mu_i,
                var_model,
                k,
                bounds_amp0,
                inp.n_restarts,
                scaler_sigma,
                device,
            )
            sigma_amp0[k, i] = evaluate_sigma(
                p_amp0, mu_i.flatten(), var_model, scaler_sigma, device
            )
        red_npe = (
            (sigma_nochirp[:, k] - sigma_opt[k, :, k])
            / sigma_nochirp[:, k]
            * 100
        )
        red_amp0 = (
            (sigma_amp0[k, :, k] - sigma_opt[k, :, k])
            / sigma_amp0[k, :, k]
            * 100
        )
        logger.info(
            f"  mean reduction: {red_npe.mean():.1f}% (vs nochirp NPE), "
            f"{red_amp0.mean():.1f}% (vs amplitude-0 estimator)"
        )

    # Unscale optimised protocols to physical units
    P_opt_physical = scaler_p_vp.inverse_transform(
        P_opt.reshape(-1, n_prot)
    ).reshape(P_opt.shape)

    results_file = os.path.join(
        inp.save_path, "optimization_clean_results.npz"
    )
    np.savez(
        results_file,
        indices=indices,
        Y_true=Y_sel,
        mu_physical=mu_physical,
        sigma_nochirp=sigma_nochirp,
        P_opt=P_opt_physical.astype("float32"),
        sigma_opt=sigma_opt,
        sigma_amp0=sigma_amp0,
        param_names=np.array(param_names),
        prot_names=np.array(prot_names),
    )
    logger.info(f"Results saved to {results_file}")


if __name__ == "__main__":
    inp = ri.basic_input(sys.argv[1])
    run_optimization_clean(inp)
