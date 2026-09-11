import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import pickle
import sys

import numpy as np
import torch
from scipy.stats import norm

from batfit import logger
from batfit.basicutilityc import ReadInput as ri
from batfit.model.param_utils.noise_utils import apply_noise, make_noise_levels
from batfit.model.param_utils.train_utils import create_model_from_log
from batfit.utils.data_utils import scale_input_from_scaler
from batfit.utils.torch_utils import find_best_model_file, get_device_type


def norm_coverage(n):
    """Percentage of data within ±n standard deviations for a standard normal."""
    return 100.0 * (norm.cdf(n) - norm.cdf(-n))


def empirical_coverage(
    samples: np.ndarray, truth: np.ndarray, k: int
) -> float:
    """Fraction of (test, param) pairs where truth falls within the central
    norm_coverage(k)% percentile interval of the posterior samples.

    Unlike the mu ± k*sigma check, this doesn't assume a Gaussian posterior —
    it uses the empirical sample distribution directly.

    :param samples: (n_test, n_samples, n_params) physical-space posterior samples
    :param truth: (n_test, n_params) physical-space ground truth
    :param k: interval width expressed as a "k-sigma equivalent", i.e. the
        interval covers norm_coverage(k)% of the samples
    :return: fraction of (test, param) pairs covered, in [0, 1]
    """
    alpha = 100.0 - norm_coverage(k)
    lower = np.percentile(samples, alpha / 2.0, axis=1)
    upper = np.percentile(samples, 100.0 - alpha / 2.0, axis=1)
    covered = (truth >= lower) & (truth <= upper)
    return covered.mean()


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

    # Training uses scale_y=True, so model.sample() returns posterior samples
    # in z-scored parameter space; scaler_Y brings them back to physical units.
    with open(os.path.join(data_path, "scaler_Y.pkl"), "rb") as f:
        scaler_Y = pickle.load(f)

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

    with open(inp.scaler_path, "rb") as f:
        scaler_X = pickle.load(f)

    # model.pkl was pickled (by train_fm_model) right after train_nn.py called
    # set_prior_data(), so it already has the correct architecture and a
    # correctly populated Y_prior buffer; load_state_dict then only needs to
    # overwrite it with the trained weights -- no manual buffer reconstruction.

    best_model_file = find_best_model_file(inp.models_dir)
    model = create_model_from_log(
        os.path.join(inp.models_dir, "model.pkl"), best_model_file
    )

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

    samples_z_all, truth_all = [], []

    with torch.no_grad():
        for batch in test_loader:
            batch_in = apply_noise(
                batch_in=batch[0],
                scaler_X=scaler_X,
                noise_levels=noise_levels,
                a_min=a_min,
                a_max=a_max,
            )
            samps = model.sample(
                batch_in.to(device),
                batch[1].to(device),
                n_samples=inp.n_samples,
                n_steps=inp.n_ode_steps,
            )
            samples_z_all.append(samps.cpu().numpy())
            truth_all.append(batch[2].numpy())

    samples_z = np.vstack(
        samples_z_all
    )  # (n_test, n_samples, n_params), z-scored
    truth = np.vstack(truth_all)  # physical

    # Inverse-transform every posterior sample (not mu/sigma separately) so
    # sigma never gets the training-set mean incorrectly added to it.
    n_test, n_samples, n_params = samples_z.shape
    samples_physical = (
        scaler_Y.inverse_transform(samples_z.reshape(-1, n_params))
        .reshape(n_test, n_samples, n_params)
        .astype("float32")
    )
    mu_preds = samples_physical.mean(axis=1)
    sigma_preds = samples_physical.std(axis=1)

    err = np.abs(mu_preds - truth)
    mean_err = np.mean(err, axis=0)
    rmse = np.sqrt(np.mean(err**2, axis=0))
    mean_std = np.mean(sigma_preds, axis=0)
    amp = np.amax(Y_test, axis=0) - np.amin(Y_test, axis=0)
    perf_metric = np.sum(mean_err / amp) / Y_test.shape[1]

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

    cov_pct = {
        k: empirical_coverage(samples_physical, truth, k) for k in (1, 2, 3)
    }
    cov_pct_discrepancy = sum(abs(cov_pct[k] - true_cov[k]) for k in (1, 2, 3))

    post_file = os.path.join(inp.models_dir, "post_test.txt")
    with open(post_file, "w") as f:
        f.write(f"MAE: {mean_err}\n")
        f.write(f"RMSE: {rmse}\n")
        f.write(f"STD: {mean_std}\n")
        f.write(f"PERF: {perf_metric}\n")
        for k in (1, 2, 3):
            f.write(f"COV_{k}: {cov[k]:.4f} (target {true_cov[k]:.4f})\n")
        f.write(f"COV_DISCREPANCY: {cov_discrepancy:.4f}\n")
        for k in (1, 2, 3):
            f.write(
                f"COV_PCT_{k}: {cov_pct[k]:.4f} (target {true_cov[k]:.4f})\n"
            )
        f.write(f"COV_PCT_DISCREPANCY: {cov_pct_discrepancy:.4f}\n")

    np.savez(
        os.path.join(inp.models_dir, "post_test.npz"),
        err=err,
        std=sigma_preds,
    )
    logger.info(f"Results written to {post_file}")


if __name__ == "__main__":
    inp = ri.basic_input(sys.argv[1])
    test_perf(inp)
