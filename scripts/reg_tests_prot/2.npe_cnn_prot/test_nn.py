import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import pickle
import sys

import numpy as np
import torch
from scipy.stats import norm
from train_nn import define_model

from batfit import logger
from batfit.basicutilityc import ReadInput as ri
from batfit.model.param_utils.noise_utils import apply_noise, make_noise_levels
from batfit.utils.data_utils import scale_input_from_scaler
from batfit.utils.torch_utils import find_best_model_file, get_device_type


def norm_coverage(n):
    """Percentage of data within ±n standard deviations for a standard normal."""
    return 100.0 * (norm.cdf(n) - norm.cdf(-n))


def from_mom_to_samples(mu, sigma, n=500):
    assert mu.shape == sigma.shape
    return np.random.normal(loc=mu, scale=sigma, size=(n, len(sigma)))


def test_perf(inp):
    data_path = inp.data_path
    split_file = os.path.join(data_path, "data_split.npz")
    if not os.path.isfile(split_file):
        logger.warning(f"Split file not found at {split_file}, skipping test")
        return

    A = np.load(split_file)
    X_scaled = scale_input_from_scaler(
        A["X_test"], os.path.join(data_path, "scaler_X.pkl")
    )
    with open(inp.scaler_P_path, "rb") as f:
        scaler_P = pickle.load(f)
    P_scaled = scaler_P.transform(A["P_test"]).astype("float32")
    Y_test = A["Y_test"]

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

    model, scaler_X = define_model(inp)
    best_model_file = find_best_model_file(inp.models_dir)
    logger.info(f"Loading {best_model_file}")
    model.load_state_dict(torch.load(best_model_file, weights_only=True))

    device = torch.device(get_device_type())
    model.to(device)
    model.eval()

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(X_scaled),
            torch.from_numpy(P_scaled),
            torch.from_numpy(Y_test),
        ),
        batch_size=min(X_scaled.shape[0], 256),
        shuffle=False,
    )

    mu_preds, sigma_preds, truth_all, noisy_voltage_all = [], [], [], []

    with torch.no_grad():
        for batch in test_loader:
            batch_in = apply_noise(
                batch_in=batch[0],
                scaler_X=scaler_X,
                noise_levels=noise_levels,
                a_min=a_min,
                a_max=a_max,
            )
            mu, sigma = model(batch_in.to(device), batch[1].to(device))
            if model.constrain_output:
                mu = model.inv_transform_mu(
                    mu.cpu(), model.min_par.numpy(), model.amp_par.numpy()
                )
                sigma = model.inv_transform_gamma(
                    sigma.cpu(), model.amp_par.numpy()
                )
            mu_preds.append(mu.cpu().numpy())
            sigma_preds.append(sigma.cpu().numpy())
            truth_all.append(batch[2].numpy())
            noisy_voltage_all.append(
                scaler_X.inverse_transform(batch_in.cpu()).numpy()
            )

    mu_preds = np.vstack(mu_preds)
    sigma_preds = np.vstack(sigma_preds)
    truth = np.vstack(truth_all)
    err = np.abs(mu_preds - truth)

    mean_err = np.mean(err, axis=0)
    rmse = np.sqrt(np.mean(err**2, axis=0))
    mean_std = np.mean(sigma_preds, axis=0)
    amp = np.amax(Y_test, axis=0) - np.amin(Y_test, axis=0)
    perf_metric = np.sum(mean_err / amp) / Y_test.shape[1]

    n_test = truth.shape[0]
    n_params = truth.shape[1]
    non_cov = {1: np.zeros(n_test), 2: np.zeros(n_test), 3: np.zeros(n_test)}
    for i in range(n_test):
        for j in range(n_params):
            for k in (1, 2, 3):
                if (
                    truth[i, j] > mu_preds[i, j] + k * sigma_preds[i, j]
                    or truth[i, j] < mu_preds[i, j] - k * sigma_preds[i, j]
                ):
                    non_cov[k][i] += 1

    cov = {k: 1 - np.sum(non_cov[k]) / (n_test * n_params) for k in (1, 2, 3)}
    true_cov = {k: 0.01 * norm_coverage(k) for k in (1, 2, 3)}
    cov_discrepancy = sum(abs(cov[k] - true_cov[k]) for k in (1, 2, 3))

    post_file = os.path.join(inp.models_dir, "post_test.txt")
    with open(post_file, "w") as f:
        f.write(f"MAE: {mean_err}\n")
        f.write(f"RMSE: {rmse}\n")
        f.write(f"STD: {mean_std}\n")
        f.write(f"PERF: {perf_metric}\n")
        for k in (1, 2, 3):
            f.write(f"COV_{k}: {cov[k]:.4f} (target {true_cov[k]:.4f})\n")
        f.write(f"COV_DISCREPANCY: {cov_discrepancy:.4f}\n")

    np.savez(
        os.path.join(inp.models_dir, "post_test.npz"),
        err=err,
        std=sigma_preds,
    )
    logger.info(f"Results written to {post_file}")


if __name__ == "__main__":
    inp = ri.basic_input(sys.argv[1])
    test_perf(inp)
