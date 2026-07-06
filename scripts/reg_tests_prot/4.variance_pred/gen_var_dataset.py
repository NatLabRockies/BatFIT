"""
Generate the variance predictor training and test datasets from a frozen NPE.

Works with either NPE architecture:
  - ProbProtParamCNN: one forward pass gives (mu, gamma) directly.
  - ProbProtParamFM: no closed-form (mu, gamma); model.sample() draws
    posterior samples (in z-scored space) and their mean/std (after
    scaler_Y.inverse_transform) is used as (mu, sigma) instead.

For every data point (X_i, P_i, Y_i) in the NPE data split:
  - Apply inp.n_noise independent noise realisations to X_i
  - Obtain (mu_k, sigma_k) for each noisy copy k (forward pass for CNN,
    sample mean/std for FM)
  - Average across realisations: mu_avg = mean(mu_k), sigma_avg = mean(sigma_k)
  - Store the feature as (P_i, mu_avg) or (P_i, Y_i) depending on inp.use_true_y

Both splits (train and test) are processed so the variance predictor can be
evaluated on held-out data. Only the train split is used to fit scalers.

Outputs written to inp.var_pred_save_path:
  var_pred_dataset.npz  — raw physical (P, Mu, Sigma) for train and test
  scaler_P_varpred.pkl  — MinMaxScaler fitted on P_train
  scaler_Mu.pkl         — MinMaxScaler fitted on Mu_train
"""

import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import pickle
import sys

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from batfit import logger
from batfit.basicutilityc import ReadInput as ri
from batfit.model.param_utils.noise_utils import apply_noise, make_noise_levels
from batfit.model.param_utils.train_utils import create_model_from_log
from batfit.model.paramNN import ProbProtParamFM
from batfit.utils.data_utils import scale_input_from_scaler
from batfit.utils.torch_utils import get_device_type


def _find_best_model_file(model_dir: str) -> str:
    """Return the checkpoint path with the lowest recorded test loss."""
    vals = np.loadtxt(
        os.path.join(model_dir, "test_loss.csv"), delimiter=";", skiprows=1
    )
    best_ind = int(np.argmin(vals[:, 1]))
    final_path = os.path.join(model_dir, "model_final.pt")
    if best_ind == vals.shape[0] - 1 and os.path.isfile(final_path):
        return final_path
    iterations = np.array(
        [
            int(fname[6 : fname.index(".pt")])
            for fname in os.listdir(model_dir)
            if fname.startswith("model_")
            and fname.endswith(".pt")
            and "final" not in fname
        ]
    )
    if len(iterations) == 0:
        return final_path
    best_iter = vals[best_ind, 0]
    ind = int(np.argmin(np.abs(iterations - best_iter)))
    return os.path.join(model_dir, f"model_{iterations[ind]}.pt")


def _load_npe(inp):
    """Load the best NPE checkpoint and move to the compute device.

    :return: (model, scaler_X, scaler_Y, device). scaler_Y is None for
        ProbProtParamCNN (which uses constrain_output/inv_transform_output
        instead); for ProbProtParamFM it's loaded from inp.data_path since
        the FM NPE is trained with scale_y=True.
    """
    model_pkl = os.path.join(inp.npe_models_dir, "model.pkl")
    best_pt = _find_best_model_file(inp.npe_models_dir)
    logger.info(f"Loading NPE from {best_pt}")
    model = create_model_from_log(
        model_obj_file=model_pkl,
        model_state_dict_file=best_pt,
    )
    with open(inp.scaler_path, "rb") as f:
        scaler_X = pickle.load(f)

    scaler_Y = None
    if isinstance(model, ProbProtParamFM):
        with open(os.path.join(inp.data_path, "scaler_Y.pkl"), "rb") as f:
            scaler_Y = pickle.load(f)

    device = torch.device(get_device_type())
    model.to(device)
    model.eval()
    return model, scaler_X, scaler_Y, device


def _process_split(
    X_np: np.ndarray,
    P_np: np.ndarray,
    Y_np: np.ndarray,
    model,
    scaler_X,
    scaler_Y,
    scaler_P,
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
      - Tiles X and P to (B * n_noise, …) so one forward pass (CNN) or ODE
        integration (FM) covers all noise realisations simultaneously.
      - For ProbProtParamCNN: one forward pass gives (mu_k, sigma_k) per
        noisy copy directly.
      - For ProbProtParamFM: draws n_samples posterior samples per noisy
        copy via model.sample(...), then takes their mean/std (after
        scaler_Y.inverse_transform) as that copy's (mu_k, sigma_k).
      - Averages (mu_k, sigma_k) over the n_noise dimension.

    :param n_samples: FM only — posterior samples drawn per noisy copy.
    :param n_ode_steps: FM only — ODE integration steps for model.sample().
    :return: (P_raw, Mu_raw, Sigma_raw) all in physical (unscaled) space
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
        X_scaled = scaler_X.transform(X_batch)  # (B, channels, time)
        X_tensor = torch.from_numpy(X_scaled)  # float32

        # Scale P with the NPE's MinMax scaler
        P_scaled = scaler_P.transform(P_batch).astype("float32")
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
        X_noisy = apply_noise(X_tiled, scaler_X, noise_levels, a_min, a_max)

        with torch.no_grad():
            if isinstance(model, ProbProtParamFM):
                samples_z = model.sample(
                    X_noisy.to(device),
                    P_tiled.to(device),
                    n_samples=n_samples,
                    n_steps=n_ode_steps,
                )  # (B*n_noise, n_samples, n_deg), z-scored
                samples_phys = scaler_Y.inverse_transform(
                    samples_z.cpu().numpy().reshape(-1, n_deg)
                ).reshape(B * n_noise, n_samples, n_deg)
                mu_np = samples_phys.mean(axis=1)  # (B*n_noise, n_deg)
                sigma_np = samples_phys.std(axis=1)  # (B*n_noise, n_deg)
            else:
                mu_s, sigma_s = model(X_noisy.to(device), P_tiled.to(device))
                if model.constrain_output:
                    mu_s, sigma_s = model.inv_transform_output(
                        mu_s,
                        sigma_s,
                        model.min_par.to(device),
                        model.amp_par.to(device),
                    )
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
    """Build and save the variance predictor dataset.

    Reads inp.data_path/data_split.npz, runs the frozen NPE over both splits,
    fits MinMax scalers on the train set, and writes the scaled dataset plus
    scalers to inp.var_pred_save_path.

    :param inp: recipe object with attributes defined in recipe_rel.yml
    """
    os.makedirs(inp.var_pred_save_path, exist_ok=True)

    # Load data split (unscaled physical values)
    split_file = os.path.join(inp.data_path, "data_split.npz")
    assert os.path.isfile(
        split_file
    ), f"data_split.npz not found at {split_file}"
    A = np.load(split_file)
    X_train, P_train, Y_train = A["X_train"], A["P_train"], A["Y_train"]
    X_test, P_test, Y_test = A["X_test"], A["P_test"], A["Y_test"]
    logger.info(
        f"Loaded split: train={X_train.shape[0]}, test={X_test.shape[0]}"
    )

    # Load the NPE's P scaler (used inside _process_split to scale P before the NPE)
    with open(inp.scaler_P_path, "rb") as f:
        scaler_P_npe = pickle.load(f)

    model, scaler_X, scaler_Y, device = _load_npe(inp)

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
        scaler_X=scaler_X,
        scaler_Y=scaler_Y,
        scaler_P=scaler_P_npe,
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

    # Fit scalers on train only
    scaler_P_vp = MinMaxScaler()
    scaler_P_vp.fit(P_tr)
    scaler_mu = MinMaxScaler()
    scaler_mu.fit(mu_tr)

    save_path = inp.var_pred_save_path
    with open(os.path.join(save_path, "scaler_P_varpred.pkl"), "wb") as f:
        pickle.dump(scaler_P_vp, f)
    with open(os.path.join(save_path, "scaler_mu.pkl"), "wb") as f:
        pickle.dump(scaler_mu, f)

    # Optional per-parameter sigma scaling: maps each sigma_j to [0, 1] using
    # MinMaxScaler fitted on the train set. Recommended when degradation parameters
    # have very different variance magnitudes. When enabled, scaler_sigma.pkl is
    # saved and train_var_pred.py will use the scaled sigma as the direct training
    # target (bypassing amp_par scaling).
    if inp.scale_sigma:
        scaler_sigma = MinMaxScaler()
        scaler_sigma.fit(sigma_tr)
        with open(os.path.join(save_path, "scaler_sigma.pkl"), "wb") as f:
            pickle.dump(scaler_sigma, f)
        sigma_tr = scaler_sigma.transform(sigma_tr).astype("float32")
        sigma_te = scaler_sigma.transform(sigma_te).astype("float32")
        logger.info("sigma MinMax-scaled per parameter (scale_sigma=true)")

    logger.info(f"Scalers saved to {save_path}")

    P_tr_sc = scaler_P_vp.transform(P_tr).astype("float32")
    mu_tr_sc = scaler_mu.transform(mu_tr).astype("float32")
    P_te_sc = scaler_P_vp.transform(P_te).astype("float32")
    mu_te_sc = scaler_mu.transform(mu_te).astype("float32")

    dataset_file = os.path.join(save_path, "var_pred_dataset.npz")
    np.savez(
        dataset_file,
        P_train=P_tr_sc,
        Mu_train=mu_tr_sc,
        Sigma_train=sigma_tr,
        P_test=P_te_sc,
        Mu_test=mu_te_sc,
        Sigma_test=sigma_te,
    )
    logger.info(f"Variance predictor dataset saved to {dataset_file}")
    logger.info(
        f"  train: P={P_tr_sc.shape}, mu={mu_tr_sc.shape}, sigma={sigma_tr.shape}"
    )
    logger.info(
        f"  test:  P={P_te_sc.shape}, mu={mu_te_sc.shape}, sigma={sigma_te.shape}"
    )


if __name__ == "__main__":
    inp = ri.basic_input(sys.argv[1])
    gen_var_dataset(inp)
