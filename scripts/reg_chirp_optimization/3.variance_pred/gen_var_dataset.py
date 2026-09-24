"""
Generate the variance predictor training and test datasets from a frozen NPE.
Works with Gaussian NPE and FM

For every data point (X_i, P_i, Y_i) in the NPE data split:
  - Apply inp.n_noise independent noise realisations to X_i
  - Obtain (mu_k, sigma_k) for each noisy copy k (forward pass for CNN,
    sample mean/std for FM)
  - Average across realisations: mu_avg = mean(mu_k), sigma_avg = mean(sigma_k)
  - Store the feature as (P_i, mu_avg) or (P_i, Y_i) depending on inp.use_true_y
"""

import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys

import numpy as np
import torch

from batfit import logger
from batfit.basicutilityc import ReadInput as ri
from batfit.model.param_utils.noise_utils import make_noise_levels
from batfit.model.param_utils.optim_utils import predict_mu_sigma
from batfit.model.param_utils.train_utils import create_model_from_log
from batfit.model.paramNN import ProbProtParamFM
from batfit.utils.torch_utils import find_best_model_file, get_device_type


def _load_npe(inp):
    """Load the best NPE checkpoint and move to the compute device."""
    model_pkl = os.path.join(inp.npe_models_dir, "model.pkl")
    best_pt = find_best_model_file(inp.npe_models_dir)
    logger.info(f"Loading NPE from {best_pt}")
    # the NPE carries its own signal/protocol/parameter scalers
    model = create_model_from_log(
        model_obj_file=model_pkl,
        model_state_dict_file=best_pt,
    )

    device = torch.device(get_device_type())
    model.to(device)
    model.eval()
    return model, device


def _process_split(
    X_np: np.ndarray,
    P_np: np.ndarray,
    Y_np: np.ndarray,
    model,
    noise_levels: torch.Tensor,
    a_min: torch.Tensor,
    a_max: torch.Tensor,
    n_noise: int,
    use_true_y: bool,
    gen_batch_size: int,
    device: torch.device,
    n_samples: int = 1000,
    n_ode_steps: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the frozen NPE over one data split with noise augmentation.

    Uses :func:`predict_mu_sigma` on the physical data: each curve gets
    ``n_noise`` noise realisations, (mu, sigma) come from a forward pass
    (Gaussian NPE) or the mean/std of posterior samples (FM NPE) and are
    averaged over the realisations. With ``use_true_y`` the stored mu is
    the ground truth instead.
    """
    mu_np, sigma_np = predict_mu_sigma(
        X_np,
        model,
        noise_levels,
        a_min,
        a_max,
        n_noise=n_noise,
        device=device,
        P=P_np.astype("float32"),
        n_samples=n_samples,
        n_ode_steps=n_ode_steps,
        batch_size=gen_batch_size,
    )
    mu_out = Y_np.astype("float32") if use_true_y else mu_np
    return P_np.astype("float32"), mu_out, sigma_np


def gen_var_dataset(inp) -> None:
    """Build and save the variance predictor dataset."""
    os.makedirs(inp.var_pred_save_path, exist_ok=True)

    # Load data split (unscaled physical values)
    split_file = os.path.join(inp.data_path, "data_split.npz")
    assert os.path.isfile(
        split_file
    ), f"data_split.npz not found at {split_file}"
    A = np.load(split_file)
    X_train, P_train, Y_train = A["X_train"], A["P_train"], A["Y_train"]
    X_test, P_test, Y_test = A["X_test"], A["P_test"], A["Y_test"]
    assert "X_val" in A.files, (
        f"{split_file} has no validation slice; regenerate the NPE split with "
        "val_split > 0"
    )
    X_val, P_val, Y_val = A["X_val"], A["P_val"], A["Y_val"]
    logger.info(
        f"Loaded split: train={X_train.shape[0]}, test={X_test.shape[0]}, "
        f"val={X_val.shape[0]}"
    )

    model, device = _load_npe(inp)

    noise_levels, a_min, a_max = make_noise_levels(
        target_mode=inp.target_mode,
        noise_levels=[
            0,
            0.001444 * 2 * inp.noise_factor,
            0.001786 * 2,
            2.01 * 2,
        ],
        cyc_mode=inp.cyc_mode,
        vmin=model.sim_params["vmin"],
        vmax=model.sim_params["vmax"],
    )

    # n_samples/n_ode_steps only apply to a ProbProtParamFM NPE; CNN recipes
    # don't set them, so fall back to reasonable defaults.
    n_samples = getattr(inp, "n_samples", 1000)
    n_ode_steps = getattr(inp, "n_ode_steps", 100)
    logger.info(
        f"Generating variance dataset: n_noise={inp.n_noise}, "
        f"use_true_y={inp.use_true_y}"
        + (
            f", n_samples={n_samples}, n_ode_steps={n_ode_steps}"
            if isinstance(model, ProbProtParamFM)
            else ""
        )
    )

    shared = dict(
        model=model,
        noise_levels=noise_levels,
        a_min=a_min,
        a_max=a_max,
        n_noise=inp.n_noise,
        n_samples=n_samples,
        n_ode_steps=n_ode_steps,
        use_true_y=inp.use_true_y,
        gen_batch_size=inp.gen_batch_size,
        device=device,
    )

    logger.info("Processing train split …")
    P_tr, mu_tr, sigma_tr = _process_split(X_train, P_train, Y_train, **shared)
    logger.info("Processing test split …")
    P_te, mu_te, sigma_te = _process_split(X_test, P_test, Y_test, **shared)
    logger.info("Processing val split …")
    P_va, mu_va, sigma_va = _process_split(X_val, P_val, Y_val, **shared)

    # Physical values: the variance estimator holds its own scalers (bounds of
    # the config for P and mu, log-sigma z-score fitted by train_var_pred.py)
    assert (
        sigma_tr.min() > 0 and sigma_te.min() > 0 and sigma_va.min() > 0
    ), "sigma must be strictly positive to train on log sigma"
    dataset_file = os.path.join(inp.var_pred_save_path, "var_pred_dataset.npz")
    np.savez(
        dataset_file,
        P_train=P_tr,
        Mu_train=mu_tr,
        Sigma_train=sigma_tr,
        P_test=P_te,
        Mu_test=mu_te,
        Sigma_test=sigma_te,
        P_val=P_va,
        Mu_val=mu_va,
        Sigma_val=sigma_va,
    )
    logger.info(f"Variance predictor dataset saved to {dataset_file}")
    logger.info(
        f"  train: P={P_tr.shape}, mu={mu_tr.shape}, sigma={sigma_tr.shape}"
    )
    logger.info(
        f"  test:  P={P_te.shape}, mu={mu_te.shape}, sigma={sigma_te.shape}"
    )
    logger.info(
        f"  val:   P={P_va.shape}, mu={mu_va.shape}, sigma={sigma_va.shape}"
    )


if __name__ == "__main__":
    inp = ri.basic_input(sys.argv[1])
    gen_var_dataset(inp)
