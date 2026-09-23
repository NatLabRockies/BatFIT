"""
Evaluate the benefit of adding a chirp

For each validation curve:
  1. Run an NPE trained WITHOUT chirp to get (mu, sigma_nochirp)
  2. Run the CHIRP NPE on the same signal (requires to interpolate
     to a finer time grid) with protocol input (time_start, amplitude=0,
     length), averaged over n_amp0_draws random (these extra inputs
     should not matter under amplitude=0)
     This gives sigma_amp0
  3. Fix mu and recommend a chirp: minimise the variance
     estimator wrt the protocol parameters.
     Loop over each degradation parameter.
     The optimisation is run twice: from the NPE estimate mu (deployable case)
     and from the ground-truth parameters (P_opt_true, sigma_opt_true) to
     isolate the effect of mu inaccuracy.

Results are saved to a single npz; plots are made separately
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
from batfit.preprocess.sim_setup import make_params
from batfit.utils.data_utils import load_pickle
from batfit.utils.torch_utils import get_device_type, load_frozen_model


def interp_signal(X: np.ndarray, n_points_new: int) -> np.ndarray:
    """Interpolate signals onto a new number of equidistant points.

    Both data pipelines sample each curve on n_points equidistant points
    spanning that curve's duration, so re-gridding is a 1D interpolation
    on the normalised point index, channel by channel.

    :param X: signals of shape (n_curves, channels, n_points_old)
    :param n_points_new: number of points of the target grid
    :return: signals of shape (n_curves, channels, n_points_new)
    """
    n_curves, n_channels, n_points_old = X.shape
    if n_points_new == n_points_old:
        return X
    grid_old = np.linspace(0.0, 1.0, n_points_old)
    grid_new = np.linspace(0.0, 1.0, n_points_new)
    X_new = np.empty((n_curves, n_channels, n_points_new), dtype=X.dtype)
    for i in range(n_curves):
        for ch in range(n_channels):
            X_new[i, ch] = np.interp(grid_new, grid_old, X[i, ch])
    return X_new


def run_optimization_clean(inp) -> None:
    """Quantify chirp-induced variance reduction over nochirp observations.

    :param inp: recipe object from recipe_clean.yml
    """
    np.random.seed(inp.random_seed)
    os.makedirs(inp.save_path, exist_ok=True)
    device = torch.device(get_device_type())

    # --- Models and scalers ---
    npe = load_frozen_model(inp.nochirp_npe_models_dir, device)
    chirp_npe = load_frozen_model(inp.chirp_npe_models_dir, device)
    var_model = load_frozen_model(inp.var_pred_models_dir, device)
    # the NPEs carry their own signal/protocol/parameter scalers
    scaler_mu = load_pickle(
        os.path.join(inp.var_pred_save_path, "scaler_mu.pkl")
    )
    scaler_p_vp = load_pickle(
        os.path.join(inp.var_pred_save_path, "scaler_P_varpred.pkl")
    )
    # Sigma target scaler: scaler_logsigma.pkl (log_sigma mode, StandardScaler
    # on log sigma) takes precedence over scaler_sigma.pkl (scale_sigma mode,
    # MinMax); sigma_physical dispatches on the scaler type. None = amp_par.
    scaler_logsigma_path = os.path.join(
        inp.var_pred_save_path, "scaler_logsigma.pkl"
    )
    scaler_sigma_path = os.path.join(
        inp.var_pred_save_path, "scaler_sigma.pkl"
    )
    assert not (
        os.path.isfile(scaler_logsigma_path)
        and os.path.isfile(scaler_sigma_path)
    ), (
        f"Both scaler_logsigma.pkl and scaler_sigma.pkl found in "
        f"{inp.var_pred_save_path}: target parameterisation is ambiguous. "
        "Regenerate the dataset in a fresh var_pred_save_path."
    )
    if os.path.isfile(scaler_logsigma_path):
        scaler_sigma = load_pickle(scaler_logsigma_path)
        logger.info("Using log-sigma parameterisation (scaler_logsigma.pkl)")
    elif os.path.isfile(scaler_sigma_path):
        scaler_sigma = load_pickle(scaler_sigma_path)
    else:
        scaler_sigma = None

    # --- Nochirp observations (val split, ground truth kept for plots) ---
    A = np.load(os.path.join(inp.nochirp_data_path, "data_split.npz"))
    assert "X_val" in A.files, (
        f"{inp.nochirp_data_path}/data_split.npz has no validation slice; "
        "regenerate the nochirp NPE split with val_split > 0"
    )
    X_val, Y_val = A["X_val"], A["Y_val"]
    n_curves = min(inp.n_curves, X_val.shape[0])
    indices = np.random.choice(X_val.shape[0], size=n_curves, replace=False)
    X_sel = X_val[indices]
    Y_sel = Y_val[indices]
    logger.info(f"Selected {n_curves} nochirp val curves")

    # --- Parameter names and protocol bounds (from the chirp config) ---
    sim_params = make_params(inp.sim_config)
    param_names = sim_params["deg_param_names"]
    prot_names = sim_params["prot_param_names"]
    n_deg = len(param_names)
    n_prot = len(prot_names)

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
        vmin=npe.sim_params["vmin"],
        vmax=npe.sim_params["vmax"],
    )
    X_scaled = npe.scaler_X.transform(X_sel).astype("float32")
    mu_physical, sigma_nochirp = predict_mu_sigma(
        X_scaled,
        npe,
        noise_levels,
        a_min,
        a_max,
        n_noise=inp.n_noise_npe,
        device=device,
        n_samples=getattr(inp, "n_samples", 1000),
        n_ode_steps=getattr(inp, "n_ode_steps", 100),
        batch_size=getattr(inp, "gen_batch_size", 256),
    )
    # scaler_mu was fitted on chirp-NPE mus; nochirp mus may fall slightly
    # outside [0, 1]
    mu_scaled = scaler_mu.transform(mu_physical).astype("float32")
    # Ground-truth parameters in the same space: optimizing with these
    # instead of the NPE estimate isolates the effect of mu inaccuracy on
    # the recommended chirp
    mu_true_scaled = scaler_mu.transform(Y_sel).astype("float32")

    # --- Chirp NPE on the same signals at amplitude 0 ---
    # A chargecc charge is physically identical to a chirp charge with
    # amplitude 0, whatever time_start and length: the chirp NPE is fed the
    # same observations (re-gridded and re-scaled for its pipeline) with
    # n_amp0_draws random (time_start, length) draws, and sigma_amp0 is the
    # average over the draws. The per-draw sigmas quantify the sensitivity
    # of the chirp NPE to the physically irrelevant protocol input.
    n_amp0_draws = getattr(inp, "n_amp0_draws", 20)
    ts_idx = prot_names.index("time_start")
    len_idx = prot_names.index("length")
    P_amp0_draws = np.zeros((n_amp0_draws, n_prot), dtype="float32")
    P_amp0_draws[:, ts_idx] = np.random.uniform(
        sim_params["prot_time_start_min"],
        sim_params["prot_time_start_max"],
        n_amp0_draws,
    )
    P_amp0_draws[:, len_idx] = np.random.uniform(
        sim_params["prot_length_min"],
        sim_params["prot_length_max"],
        n_amp0_draws,
    )
    P_amp0_scaled = chirp_npe.scaler_P.transform(P_amp0_draws).astype(
        "float32"
    )

    noise_levels_chirp, a_min_chirp, a_max_chirp = make_noise_levels(
        target_mode=inp.target_mode,
        noise_levels=[
            0,
            0.001444 * 2 * inp.noise_factor,
            0.001786 * 2,
            2.01 * 2,
        ],
        cyc_mode="chirp",
        vmin=chirp_npe.sim_params["vmin"],
        vmax=chirp_npe.sim_params["vmax"],
    )
    # The chirp NPE's input grid size is recorded in the recipe saved next
    # to its checkpoint at training time
    chirp_npe_recipe = ri.basic_input(
        os.path.join(inp.chirp_npe_models_dir, "recipe.yml")
    )
    X_chirp = interp_signal(X_sel, int(chirp_npe_recipe.n_points))
    X_chirp_scaled = chirp_npe.scaler_X.transform(X_chirp).astype("float32")
    sigma_amp0_draws = np.zeros(
        (n_amp0_draws, n_curves, n_deg), dtype="float32"
    )
    for j in range(n_amp0_draws):
        logger.info(
            f"Chirp NPE at amp=0, draw {j + 1}/{n_amp0_draws}: "
            f"time_start={P_amp0_draws[j, ts_idx]:.0f}s, "
            f"length={P_amp0_draws[j, len_idx]:.0f}s"
        )
        P_tiled = np.tile(P_amp0_scaled[j : j + 1], (n_curves, 1))
        _, sigma_amp0_draws[j] = predict_mu_sigma(
            X_chirp_scaled,
            chirp_npe,
            noise_levels_chirp,
            a_min_chirp,
            a_max_chirp,
            n_noise=inp.n_noise_npe,
            device=device,
            P_scaled=P_tiled,
            n_samples=getattr(inp, "n_samples", 1000),
            n_ode_steps=getattr(inp, "n_ode_steps", 100),
            batch_size=getattr(inp, "gen_batch_size", 256),
        )
    sigma_amp0 = sigma_amp0_draws.mean(axis=0)  # (n_curves, n_deg)

    # --- Optimize the chirp for each target parameter and curve ---
    bounds_full = [(0.0, 1.0)] * n_prot

    P_opt = np.zeros((n_deg, n_curves, n_prot), dtype="float32")
    # sigma_opt[k, i, :] = estimator sigma of ALL deg params at the protocol
    # optimized FOR param k on curve i; only the diagonal [k, :, k] is used
    # by the reduction plots, off-diagonals record cross-parameter effects
    sigma_opt = np.zeros((n_deg, n_curves, n_deg), dtype="float32")
    # Same, with the optimization run from the TRUE parameters instead of mu
    P_opt_true = np.zeros((n_deg, n_curves, n_prot), dtype="float32")
    sigma_opt_true = np.zeros((n_deg, n_curves, n_deg), dtype="float32")

    for k, name in enumerate(param_names):
        logger.info(f"Optimizing chirp for '{name}' ({k + 1}/{n_deg})")
        from prettyPlot.progressBar import print_progress_bar

        print_progress_bar(
            0,
            n_curves,
            prefix=f"Curves 0 / {n_curves} ",
            suffix="Complete",
            length=50,
        )
        for i in range(n_curves):
            # Two passes: from the NPE estimate mu (deployable case) and
            # from the true parameters (mu-accuracy check)
            for mu_arr, P_dst, sigma_dst in (
                (mu_scaled, P_opt, sigma_opt),
                (mu_true_scaled, P_opt_true, sigma_opt_true),
            ):
                mu_i = mu_arr[i : i + 1]  # (1, n_deg)
                p_opt, _ = optimize_protocol(
                    mu_i,
                    var_model,
                    k,
                    bounds_full,
                    inp.n_restarts,
                    scaler_sigma,
                    device,
                )
                P_dst[k, i] = p_opt
                sigma_dst[k, i] = evaluate_sigma(
                    p_opt, mu_i.flatten(), var_model, scaler_sigma, device
                )
            print_progress_bar(
                i + 1,
                n_curves,
                prefix=f"Curves {i+1} / {n_curves} ",
                suffix="Complete",
                length=50,
            )
        red_npe = (
            (sigma_nochirp[:, k] - sigma_opt[k, :, k])
            / sigma_nochirp[:, k]
            * 100
        )
        # Per-draw reduction so the spread over draws is visible in the log
        red_amp0_draws = (
            (sigma_amp0_draws[:, :, k] - sigma_opt[k, :, k])
            / sigma_amp0_draws[:, :, k]
            * 100
        ).mean(axis=1)
        red_npe_true = (
            (sigma_nochirp[:, k] - sigma_opt_true[k, :, k])
            / sigma_nochirp[:, k]
            * 100
        )
        logger.info(
            f"  mean reduction: {red_npe.mean():.1f}% (vs nochirp NPE), "
            f"{red_amp0_draws.mean():.1f}% +/- {red_amp0_draws.std():.1f}% "
            f"(vs amplitude-0 chirp NPE, spread over draws), "
            f"{red_npe_true.mean():.1f}% (vs nochirp NPE, true params)"
        )

    # Unscale optimised protocols to physical units
    P_opt_physical = scaler_p_vp.inverse_transform(
        P_opt.reshape(-1, n_prot)
    ).reshape(P_opt.shape)
    P_opt_true_physical = scaler_p_vp.inverse_transform(
        P_opt_true.reshape(-1, n_prot)
    ).reshape(P_opt_true.shape)

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
        P_opt_true=P_opt_true_physical.astype("float32"),
        sigma_opt_true=sigma_opt_true,
        sigma_amp0=sigma_amp0,
        sigma_amp0_draws=sigma_amp0_draws,
        P_amp0_draws=P_amp0_draws,
        param_names=np.array(param_names),
        prot_names=np.array(prot_names),
    )
    logger.info(f"Results saved to {results_file}")


if __name__ == "__main__":
    inp = ri.basic_input(sys.argv[1])
    run_optimization_clean(inp)
