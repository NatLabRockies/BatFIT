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
from batfit.model.param_utils.noise_utils import apply_noise, make_noise_levels
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

    For each batch of size B:
      - Tiles X and P to (B * n_noise, …) so one forward pass covers
        all noise realisations simultaneously.
      - For ProbProtParamCNN: one forward pass gives (mu_k, sigma_k) per
        noisy copy directly.
      - For ProbProtParamFM: draws n_samples posterior samples per noisy
        copy via model.sample(...), then takes their mean/std (after
        model.to_physical) as that copy's (mu_k, sigma_k).
      - Averages (mu_k, sigma_k) over the n_noise dimension.
    """
    N = X_np.shape[0]
    n_deg = Y_np.shape[1]

    mu_list: list[np.ndarray] = []
    sigma_list: list[np.ndarray] = []

    for start in range(0, N, gen_batch_size):
        end = min(start + gen_batch_size, N)
        X_batch = X_np[start:end]  # (B, channels, time)
        P_batch = P_np[start:end]  # (B, n_prot)
        Y_batch = Y_np[start:end]  # (B, n_deg)
        B = X_batch.shape[0]

        # Scale X with the NPE's z-score scaler
        X_scaled = model.scaler_X.transform(X_batch)  # (B, channels, time)
        X_tensor = torch.from_numpy(X_scaled)  # float32

        # Scale P to [0, 1] with the NPE's protocol scaler
        P_scaled = model.scaler_P.transform(P_batch).astype("float32")
        P_tensor = torch.from_numpy(P_scaled)

        # Tile to (B * n_noise, …) for vectorised noise application
        X_tiled = (
            X_tensor.unsqueeze(1)
            .expand(-1, n_noise, -1, -1)
            .reshape(B * n_noise, X_tensor.shape[1], X_tensor.shape[2])
        )  # (B*n_noise, channels, time)
        P_tiled = (
            P_tensor.unsqueeze(1)
            .expand(-1, n_noise, -1)
            .reshape(B * n_noise, P_tensor.shape[1])
        )  # (B*n_noise, n_prot)

        # Each of the B*n_noise copies gets independent noise
        X_noisy = apply_noise(
            X_tiled, model.scaler_X, noise_levels, a_min, a_max
        )

        with torch.no_grad():
            if isinstance(model, ProbProtParamFM):
                samples_flow = model.sample(
                    X_noisy.to(device),
                    P_tiled.to(device),
                    n_samples=n_samples,
                    n_steps=n_ode_steps,
                )  # (B*n_noise, n_samples, n_deg), flow space
                samples_phys = model.to_physical(samples_flow)
                samples_phys = samples_phys.cpu().numpy()
                mu_np = samples_phys.mean(axis=1)  # (B*n_noise, n_deg)
                sigma_np = samples_phys.std(axis=1)  # (B*n_noise, n_deg)
            else:
                mu_scaled, sigma_scaled = model(
                    X_noisy.to(device), P_tiled.to(device)
                )
                mu_s, sigma_s = model.to_physical(mu_scaled, sigma_scaled)
                mu_np = mu_s.cpu().numpy()  # (B*n_noise, n_deg)
                sigma_np = sigma_s.cpu().numpy()  # (B*n_noise, n_deg)

        # Average over noise realisations
        mu_np = mu_np.reshape(B, n_noise, n_deg).mean(axis=1)  # (B, n_deg)
        sigma_np = sigma_np.reshape(B, n_noise, n_deg).mean(axis=1)

        if use_true_y:
            mu_list.append(Y_batch.astype("float32"))
        else:
            mu_list.append(mu_np.astype("float32"))
        sigma_list.append(sigma_np.astype("float32"))

    mu_out = np.vstack(mu_list)  # (N, n_deg)
    sigma_out = np.vstack(sigma_list)  # (N, n_deg)
    return P_np.astype("float32"), mu_out, sigma_out


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
