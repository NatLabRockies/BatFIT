"""
Bayesian Optimisation of protocol parameters to minimise NPE sigma for a
chosen degradation parameter.

For each selected test voltage curve:
  1. Extract the battery's true degradation parameters from Y_test.
  2. For each BO step: propose a protocol via Expected Improvement (EI),
     simulate a NEW voltage curve with BatMODS-lite using the proposed
     protocol and the known degradation state, then evaluate the NPE on
     that fresh curve to obtain sigma.
  3. After n_bo_steps total evaluations (the first n_init_points are a
     random design of experiments, then EI takes over), report the best
     protocol found and the variance reduction.

The Gaussian Process surrogate and EI acquisition are handled by
scikit-optimize (skopt). Protocol parameters are optimised in their
physical units — skopt reads bounds directly from the sim_config YAML.

Usage:
    python run_bo.py training_recipes/recipe_bo.yml
"""

import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import pickle
import sys

import numpy as np
import torch
from skopt import Optimizer
from skopt.space import Real

from batfit import logger
from batfit.basicutilityc import ReadInput as ri
from batfit.model.param_utils.noise_utils import apply_noise, make_noise_levels
from batfit.model.param_utils.train_utils import create_model_from_log
from batfit.model.paramNN import ProbProtParamFM
from batfit.preprocess.sim_setup import make_params
from batfit.preprocess.sol_gen import single_run
from batfit.preprocess.utils import (
    from_degparamlist_to_degparamdict,
    from_protparamlist_to_protparamdict,
)
from batfit.utils.torch_utils import find_best_model_file, get_device_type

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_npe(inp, device: torch.device):
    """Load the best NPE checkpoint and move it to device."""
    model_pkl = os.path.join(inp.npe_models_dir, "model.pkl")
    best_pt = find_best_model_file(inp.npe_models_dir)
    logger.info(f"Loading NPE from {best_pt}")
    model = create_model_from_log(
        model_obj_file=model_pkl, model_state_dict_file=best_pt
    )
    model.to(device)
    model.eval()
    return model


def _sol_to_signal(
    rootsol, n_points: int, target_mode: str
) -> np.ndarray | None:
    """Extract a (channels, n_points) signal array from a BatMODS-lite solution.

    Mirrors the ``diff_cap=False`` branch of ``from_sol_dict_to_xy`` used
    for chirp mode: channel 0 is the uniform time grid, channel 1 is the
    interpolated voltage (when target_mode contains "phi").

    :param rootsol: solution object returned by ``single_run``.
    :param n_points: number of time points for interpolation.
    :param target_mode: signal channels to extract (e.g. "phi").
    :return: float32 array of shape (channels, n_points), or None on failure.
    """
    try:
        t = rootsol.vars["time_s"]
        phis_c = rootsol.vars["voltage_V"]
        assert np.amax(phis_c) - np.amin(phis_c) > 0.1
        t_grid = np.linspace(t.min(), t.max(), n_points)
        x = t_grid.reshape(1, -1)
        if "phi" in target_mode.lower():
            phi_grid = np.interp(t_grid, t, phis_c)
            x = np.vstack((x, phi_grid.reshape(1, -1)))
        return x.astype("float32")
    except (AssertionError, AttributeError, TypeError, KeyError):
        return None


# ---------------------------------------------------------------------------
# NPE oracle
# ---------------------------------------------------------------------------


def predict_sigma_npe(
    p_npe_scaled: np.ndarray,
    x_npe_scaled: np.ndarray,
    npe_model,
    scaler_x,
    noise_levels: torch.Tensor,
    a_min: torch.Tensor,
    a_max: torch.Tensor,
    n_noise: int,
    device: torch.device,
    scaler_Y=None,
    n_samples: int = 1000,
    n_ode_steps: int = 100,
) -> np.ndarray:
    """Evaluate the NPE sigma on a single voltage curve.

    Tiles the signal n_noise times, applies independent noise to each copy,
    and averages sigma over realisations for a stable estimate. Works with
    either NPE architecture:

    - ProbProtParamCNN: one forward pass gives sigma directly per noisy copy.
    - ProbProtParamFM: no closed-form sigma; model.sample() draws n_samples
      posterior samples per noisy copy (in z-scored space) and their std
      (after scaler_Y.inverse_transform) is used as that copy's sigma.

    :param p_npe_scaled: MinMax-scaled protocol params, shape (n_prot,).
    :param x_npe_scaled: z-scored signal, shape (channels, n_points).
    :param scaler_Y: FM only — inverse-transforms samples from z-scored to
        physical space; required when npe_model is a ProbProtParamFM.
    :param n_samples: FM only — posterior samples drawn per noisy copy.
    :param n_ode_steps: FM only — ODE integration steps for model.sample().
    :return: physical sigma for all degradation parameters, shape (n_deg,).
    """
    n_deg = npe_model.n_param_pred
    x_t = torch.from_numpy(x_npe_scaled).unsqueeze(0)  # (1, C, T)
    p_t = torch.from_numpy(p_npe_scaled.astype("float32")).unsqueeze(
        0
    )  # (1, n_prot)

    x_tiled = (
        x_t.unsqueeze(1)
        .expand(-1, n_noise, -1, -1)
        .reshape(n_noise, x_t.shape[1], x_t.shape[2])
    )
    p_tiled = (
        p_t.unsqueeze(1).expand(-1, n_noise, -1).reshape(n_noise, p_t.shape[1])
    )
    x_noisy = apply_noise(x_tiled, scaler_x, noise_levels, a_min, a_max)

    with torch.no_grad():
        if isinstance(npe_model, ProbProtParamFM):
            samples_z = npe_model.sample(
                x_noisy.to(device),
                p_tiled.to(device),
                n_samples=n_samples,
                n_steps=n_ode_steps,
            )  # (n_noise, n_samples, n_deg), z-scored
            samples_phys = scaler_Y.inverse_transform(
                samples_z.cpu().numpy().reshape(-1, n_deg)
            ).reshape(n_noise, n_samples, n_deg)
            sigma_np = samples_phys.std(axis=1)  # (n_noise, n_deg)
        else:
            mu_s, sigma_s = npe_model(x_noisy.to(device), p_tiled.to(device))
            if npe_model.constrain_output:
                mu_s, sigma_s = npe_model.inv_transform_output(
                    mu_s,
                    sigma_s,
                    npe_model.min_par.to(device),
                    npe_model.amp_par.to(device),
                )
            sigma_np = sigma_s.cpu().numpy()
    return sigma_np.mean(axis=0).astype("float32")  # (n_deg,)


# ---------------------------------------------------------------------------
# Simulation oracle
# ---------------------------------------------------------------------------


def evaluate_sigma_at_P(
    p_physical: list[float],
    deg_param_sample: dict,
    sim_params: dict,
    npe_model,
    scaler_x,
    scaler_p_npe,
    noise_levels: torch.Tensor,
    a_min: torch.Tensor,
    a_max: torch.Tensor,
    n_points: int,
    target_mode: str,
    n_noise: int,
    param_idx: int,
    device: torch.device,
    sigma_fail: float,
    scaler_Y=None,
    n_samples: int = 1000,
    n_ode_steps: int = 100,
) -> float:
    """Simulate a new voltage curve at the proposed protocol, then return NPE sigma.

    This is the black-box objective for the BO loop. Each call:
      1. Runs a BatMODS-lite chirp simulation with the proposed protocol and
         the battery's known degradation state.
      2. Interpolates the resulting voltage onto a uniform time grid.
      3. Evaluates the NPE (with noise averaging) on the new signal.

    :param p_physical: proposed protocol params in physical units, length n_prot.
    :param deg_param_sample: degradation parameter dict for this battery.
    :param sigma_fail: value returned when the simulation fails.
    :return: physical sigma for ``param_idx``, or ``sigma_fail`` on failure.
    """
    prot_param_sample = from_protparamlist_to_protparamdict(
        p_physical, sim_params
    )
    _, _, rootsol = single_run(
        deg_param_sample=deg_param_sample,
        sim_params=sim_params,
        prot_param_sample=prot_param_sample,
    )

    if rootsol is None:
        logger.warning(
            f"Simulation failed for P={p_physical} — returning sigma_fail"
        )
        return sigma_fail

    x_phys = _sol_to_signal(rootsol, n_points, target_mode)
    if x_phys is None:
        logger.warning("Signal extraction failed — returning sigma_fail")
        return sigma_fail

    x_scaled = scaler_x.transform(x_phys[np.newaxis])[0].astype("float32")
    p_npe_scaled = scaler_p_npe.transform(
        np.array(p_physical, dtype="float32").reshape(1, -1)
    )[0]

    sigma_phys = predict_sigma_npe(
        p_npe_scaled,
        x_scaled,
        npe_model,
        scaler_x,
        noise_levels,
        a_min,
        a_max,
        n_noise,
        device,
        scaler_Y=scaler_Y,
        n_samples=n_samples,
        n_ode_steps=n_ode_steps,
    )
    return float(sigma_phys[param_idx])


# ---------------------------------------------------------------------------
# BO loop for a single curve
# ---------------------------------------------------------------------------


def run_bo_single_curve(
    deg_param_sample: dict,
    p_test_physical: np.ndarray,
    sim_params: dict,
    npe_model,
    scaler_x,
    scaler_p_npe,
    noise_levels: torch.Tensor,
    a_min: torch.Tensor,
    a_max: torch.Tensor,
    param_idx: int,
    n_points: int,
    target_mode: str,
    n_init_points: int,
    n_bo_steps: int,
    n_noise: int,
    sigma_fail: float,
    random_state: int,
    device: torch.device,
    scaler_Y=None,
    n_samples: int = 1000,
    n_ode_steps: int = 100,
) -> dict:
    """Run Bayesian Optimisation for a single test battery.

    The first ``n_init_points`` evaluations are random (DoE); all subsequent
    evaluations use the EI acquisition function from scikit-optimize. Physical
    bounds for each protocol parameter are read from ``sim_params``.

    :param deg_param_sample: degradation state of this battery (ground truth).
    :param p_test_physical: protocol used in the original measurement, shape (n_prot,).
        Used only to evaluate the baseline sigma.
    :return: dict with keys:

        - ``sigma_init``      — sigma at the original protocol, (n_deg,)
        - ``P_history``       — all evaluated protocols in physical units, (n_bo_steps, n_prot)
        - ``sigma_history``   — sigma[param_idx] at each eval, (n_bo_steps,)
        - ``P_best``          — best protocol found, physical, (n_prot,)
        - ``sigma_best``      — lowest sigma[param_idx] achieved, scalar
    """
    prot_names = sim_params["prot_param_names"]
    search_space = [
        Real(
            sim_params[f"prot_{name}_min"],
            sim_params[f"prot_{name}_max"],
            name=name,
        )
        for name in prot_names
    ]

    optimizer = Optimizer(
        dimensions=search_space,
        base_estimator="GP",
        acq_func="EI",
        n_initial_points=n_init_points,
        acq_optimizer="lbfgs",
        random_state=random_state,
    )

    # Baseline: simulate at the original test protocol
    sigma_init = np.full(npe_model.n_param_pred, np.nan, dtype="float32")
    x_init_phys = _sol_to_signal(
        single_run(
            deg_param_sample=deg_param_sample,
            sim_params=sim_params,
            prot_param_sample=from_protparamlist_to_protparamdict(
                p_test_physical.tolist(), sim_params
            ),
        )[2],
        n_points,
        target_mode,
    )
    if x_init_phys is not None:
        x_init_scaled = scaler_x.transform(x_init_phys[np.newaxis])[0].astype(
            "float32"
        )
        p_npe_scaled_init = scaler_p_npe.transform(
            p_test_physical.reshape(1, -1).astype("float32")
        )[0]
        sigma_init = predict_sigma_npe(
            p_npe_scaled_init,
            x_init_scaled,
            npe_model,
            scaler_x,
            noise_levels,
            a_min,
            a_max,
            n_noise,
            device,
            scaler_Y=scaler_Y,
            n_samples=n_samples,
            n_ode_steps=n_ode_steps,
        )

    # BO loop
    p_history = np.zeros((n_bo_steps, len(prot_names)), dtype="float32")
    sigma_history = np.zeros(n_bo_steps, dtype="float32")

    common = dict(
        deg_param_sample=deg_param_sample,
        sim_params=sim_params,
        npe_model=npe_model,
        scaler_x=scaler_x,
        scaler_p_npe=scaler_p_npe,
        noise_levels=noise_levels,
        a_min=a_min,
        a_max=a_max,
        n_points=n_points,
        target_mode=target_mode,
        n_noise=n_noise,
        param_idx=param_idx,
        device=device,
        sigma_fail=sigma_fail,
        scaler_Y=scaler_Y,
        n_samples=n_samples,
        n_ode_steps=n_ode_steps,
    )

    for step in range(n_bo_steps):
        p_next = optimizer.ask()  # physical units
        sigma_next = evaluate_sigma_at_P(p_next, **common)
        optimizer.tell(p_next, sigma_next)

        p_history[step] = np.array(p_next, dtype="float32")
        sigma_history[step] = sigma_next

        best_so_far = float(np.min(sigma_history[: step + 1]))
        phase = "init" if step < n_init_points else "EI"
        logger.info(
            f"    [{phase}] step {step + 1:3d}/{n_bo_steps} | "
            f"sigma={sigma_next:.4e} | best={best_so_far:.4e}"
        )

    best_idx = int(np.argmin(sigma_history))
    return {
        "sigma_init": sigma_init,
        "P_history": p_history,
        "sigma_history": sigma_history,
        "P_best": p_history[best_idx],
        "sigma_best": float(sigma_history[best_idx]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_bo(inp) -> None:
    """Run Bayesian Optimisation on n_curves test batteries.

    :param inp: recipe object from recipe_bo.yml.
    """
    np.random.seed(inp.random_seed)
    os.makedirs(inp.save_path, exist_ok=True)

    device = torch.device(get_device_type())
    npe_model = _load_npe(inp, device)

    with open(inp.scaler_path, "rb") as f:
        scaler_x = pickle.load(f)
    with open(inp.scaler_P_path, "rb") as f:
        scaler_p_npe = pickle.load(f)

    # scaler_Y only applies to a ProbProtParamFM NPE (trained with
    # scale_y=True); it's fit on the NPE's own training data.
    scaler_y = None
    if isinstance(npe_model, ProbProtParamFM):
        with open(os.path.join(inp.data_path, "scaler_Y.pkl"), "rb") as f:
            scaler_y = pickle.load(f)
    n_samples = getattr(inp, "n_samples", 1000)
    n_ode_steps = getattr(inp, "n_ode_steps", 100)

    split_file = os.path.join(inp.data_path, "data_split.npz")
    assert os.path.isfile(
        split_file
    ), f"data_split.npz not found at {split_file}"
    split_data = np.load(split_file)
    P_test = split_data["P_test"]  # (N_test, n_prot) — physical
    Y_test = split_data["Y_test"]  # (N_test, n_deg)  — physical (ground truth)

    n_test = P_test.shape[0]
    n_curves = min(inp.n_curves, n_test)
    indices = np.random.choice(n_test, size=n_curves, replace=False)
    logger.info(f"Selected {n_curves} test curves: {indices}")

    sim_params = make_params(inp.sim_config)
    param_names = sim_params["deg_param_names"]
    prot_names = sim_params["prot_param_names"]
    assert (
        inp.param_to_minimize in param_names
    ), f"param_to_minimize='{inp.param_to_minimize}' not in {param_names}"
    param_idx = param_names.index(inp.param_to_minimize)
    logger.info(
        f"Minimising sigma of '{inp.param_to_minimize}' (index {param_idx})"
    )

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

    n_prot = P_test.shape[1]
    sigma_init_all = np.full(
        (n_curves, len(param_names)), np.nan, dtype="float32"
    )
    sigma_best_all = np.zeros(n_curves, dtype="float32")
    P_opt_all = np.zeros((n_curves, n_prot), dtype="float32")
    sigma_history_all = np.zeros((n_curves, inp.n_bo_steps), dtype="float32")
    P_history_all = np.zeros(
        (n_curves, inp.n_bo_steps, n_prot), dtype="float32"
    )

    for curve_i, idx in enumerate(indices):
        p_phys = P_test[idx]  # (n_prot,)
        y_phys = Y_test[idx]  # (n_deg,)

        deg_param_sample = from_degparamlist_to_degparamdict(
            y_phys.tolist(), sim_params
        )

        logger.info(
            f"Curve {curve_i:3d} (test idx {idx}) | "
            f"deg params = {dict(zip(param_names, [f'{v:.3f}' for v in y_phys]))}"
        )
        logger.info(
            f"  Running BO: {inp.n_init_points} random init + "
            f"{inp.n_bo_steps - inp.n_init_points} EI steps "
            f"({inp.n_bo_steps} total)"
        )

        res = run_bo_single_curve(
            deg_param_sample=deg_param_sample,
            p_test_physical=p_phys,
            sim_params=sim_params,
            npe_model=npe_model,
            scaler_x=scaler_x,
            scaler_p_npe=scaler_p_npe,
            noise_levels=noise_levels,
            a_min=a_min,
            a_max=a_max,
            param_idx=param_idx,
            n_points=inp.n_points,
            target_mode=inp.target_mode,
            n_init_points=inp.n_init_points,
            n_bo_steps=inp.n_bo_steps,
            n_noise=inp.n_noise,
            sigma_fail=inp.sigma_fail,
            random_state=inp.random_seed + curve_i,
            device=device,
            scaler_Y=scaler_y,
            n_samples=n_samples,
            n_ode_steps=n_ode_steps,
        )

        sigma_init_all[curve_i] = res["sigma_init"]
        sigma_best_all[curve_i] = res["sigma_best"]
        P_opt_all[curve_i] = res["P_best"]
        sigma_history_all[curve_i] = res["sigma_history"]
        P_history_all[curve_i] = res["P_history"]

        sigma_init_target = float(res["sigma_init"][param_idx])
        if np.isfinite(sigma_init_target) and sigma_init_target > 0:
            reduction = (
                (sigma_init_target - res["sigma_best"])
                / sigma_init_target
                * 100
            )
            logger.info(
                f"  sigma_init={sigma_init_target:.4e}  "
                f"sigma_best={res['sigma_best']:.4e}  "
                f"reduction={reduction:.1f}%"
            )
        else:
            logger.info(
                f"  sigma_init=N/A (init sim failed)  "
                f"sigma_best={res['sigma_best']:.4e}"
            )

    # Running-minimum convergence trace
    convergence_all = np.minimum.accumulate(sigma_history_all, axis=1)

    out_file = os.path.join(inp.save_path, "bo_results.npz")
    np.savez(
        out_file,
        indices=indices,
        sigma_init=sigma_init_all,
        sigma_best=sigma_best_all,
        P_test=P_test[indices],
        P_opt=P_opt_all,
        Y_test=Y_test[indices],
        sigma_history=sigma_history_all,
        P_history=P_history_all,
        convergence=convergence_all,
        param_names=np.array(param_names),
        prot_param_names=np.array(prot_names),
        param_to_minimize=inp.param_to_minimize,
    )
    logger.info(f"Results saved to {out_file}")

    # Summary (only over curves where init simulation succeeded)
    valid = np.isfinite(sigma_init_all[:, param_idx]) & (
        sigma_init_all[:, param_idx] > 0
    )
    if valid.any():
        mean_reduction = (
            (sigma_init_all[valid, param_idx] - sigma_best_all[valid])
            / sigma_init_all[valid, param_idx]
            * 100
        ).mean()
        logger.info(
            f"Mean sigma reduction for '{inp.param_to_minimize}' "
            f"({valid.sum()} curves): {mean_reduction:.1f}%"
        )

    logger.info("Optimised P (physical) — mean over curves:")
    for j, name in enumerate(prot_names):
        logger.info(
            f"  {name}: {P_opt_all[:, j].mean():.4f} "
            f"± {P_opt_all[:, j].std():.4f}"
        )


if __name__ == "__main__":
    inp = ri.basic_input(sys.argv[1])
    run_bo(inp)
