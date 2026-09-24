"""Test the nochirp FM NPE on validation split"""

import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import sys

import numpy as np
import torch
from scipy.stats import norm

from batfit import logger
from batfit.basicutilityc import ReadInput as ri
from batfit.model.param_utils.noise_utils import (
    apply_noise_unscaled,
    make_noise_levels,
)
from batfit.model.param_utils.train_utils import create_model_from_log
from batfit.utils.torch_utils import find_best_model_file, get_device_type


def norm_coverage(n):
    """Percentage of data within ±n standard deviations for a standard normal."""
    return 100.0 * (norm.cdf(n) - norm.cdf(-n))


def empirical_coverage(
    samples: np.ndarray, truth: np.ndarray, k: int
) -> float:
    """Fraction of (val, param) pairs whose truth falls in the central interval.

    Parameters
    ----------
    samples : np.ndarray
        Physical-space posterior samples of shape
        ``(n_val, n_samples, n_params)``.
    truth : np.ndarray
        Physical-space ground truth of shape ``(n_val, n_params)``.
    k : int
        Interval width expressed as a "k-sigma equivalent", i.e. the interval
        covers ``norm_coverage(k)%`` of the samples.

    Returns
    -------
    float
        Fraction of (val, param) pairs covered, in ``[0, 1]``.
    """
    alpha = 100.0 - norm_coverage(k)
    lower = np.percentile(samples, alpha / 2.0, axis=1)
    upper = np.percentile(samples, 100.0 - alpha / 2.0, axis=1)
    covered = (truth >= lower) & (truth <= upper)
    return covered.mean()


def test_perf(inp):
    """Report MAE/RMSE/STD/coverage on the validation slice of data_split.npz."""
    data_path = inp.data_path
    split_file = os.path.join(data_path, "data_split.npz")
    if not os.path.isfile(split_file):
        logger.warning(f"Split file not found at {split_file}, skipping test")
        return

    A = np.load(split_file)
    Y_val = A["Y_val"]

    # model.pkl was pickled (by train_fm_model) right after train_nn.py called
    # set_prior_data(), so it already has the correct architecture and a
    # correctly populated Y_prior buffer; load_state_dict then only needs to
    # overwrite it with the trained weights -- no manual buffer reconstruction.
    best_model_file = find_best_model_file(inp.models_dir)
    model = create_model_from_log(
        os.path.join(inp.models_dir, "model.pkl"), best_model_file
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
        vmin=model.sim_params["vmin"],
        vmax=model.sim_params["vmax"],
    )

    device = torch.device(get_device_type())
    model.to(device)
    model.eval()

    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(A["X_val"]), torch.from_numpy(Y_val)
        ),
        batch_size=min(Y_val.shape[0], 256),
        shuffle=False,
    )

    samples_phys_all, truth_all = [], []

    with torch.no_grad():
        for batch in val_loader:
            # noise in physical space, then the physical API
            batch_in = apply_noise_unscaled(
                batch_in=batch[0],
                noise_levels=noise_levels,
                a_min=a_min,
                a_max=a_max,
            )
            # physical samples, clamped to the bounds
            samples_phys = model.sample_physical(
                batch_in.to(device),
                n_samples=inp.n_samples,
                n_steps=inp.n_ode_steps,
            )
            samples_phys_all.append(samples_phys.cpu().numpy())
            truth_all.append(batch[1].numpy())

    # (n_val, n_samples, n_params), physical
    samples_physical = np.vstack(samples_phys_all)
    truth = np.vstack(truth_all)  # physical

    n_val, n_samples, n_params = samples_physical.shape
    mu_preds = samples_physical.mean(axis=1)
    sigma_preds = samples_physical.std(axis=1)

    err = np.abs(mu_preds - truth)
    mean_err = np.mean(err, axis=0)
    rmse = np.sqrt(np.mean(err**2, axis=0))
    mean_std = np.mean(sigma_preds, axis=0)
    amp = np.amax(Y_val, axis=0) - np.amin(Y_val, axis=0)
    perf_metric = np.sum(mean_err / amp) / Y_val.shape[1]

    non_cov = {1: np.zeros(n_val), 2: np.zeros(n_val), 3: np.zeros(n_val)}
    for i in range(n_val):
        for j in range(n_params):
            for k in (1, 2, 3):
                if (
                    truth[i, j] > mu_preds[i, j] + k * sigma_preds[i, j]
                    or truth[i, j] < mu_preds[i, j] - k * sigma_preds[i, j]
                ):
                    non_cov[k][i] += 1

    cov = {k: 1 - np.sum(non_cov[k]) / (n_val * n_params) for k in (1, 2, 3)}
    true_cov = {k: 0.01 * norm_coverage(k) for k in (1, 2, 3)}
    cov_discrepancy = sum(abs(cov[k] - true_cov[k]) for k in (1, 2, 3))

    cov_pct = {
        k: empirical_coverage(samples_physical, truth, k) for k in (1, 2, 3)
    }
    cov_pct_discrepancy = sum(abs(cov_pct[k] - true_cov[k]) for k in (1, 2, 3))

    post_file = os.path.join(inp.models_dir, "post_val.txt")
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
        os.path.join(inp.models_dir, "post_val.npz"),
        err=err,
        std=sigma_preds,
    )
    logger.info(f"Results written to {post_file}")


if __name__ == "__main__":
    inp = ri.basic_input(sys.argv[1])
    test_perf(inp)
