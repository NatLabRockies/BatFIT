"""Evaluate a trained ProbParamFM (no protocol conditioning) on the val split."""

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
from batfit.model.surrogate_utils.losses import mae_loss as mae_loss_surr
from batfit.model.surrogateNN import SurrogateFCNN
from batfit.utils.torch_utils import (
    find_best_model_file,
    get_device_type,
    get_num_parameters,
)


class ForwardModel(torch.nn.Module):
    """Wraps a trained SurrogateFCNN so it can be called as v(t; degradation_params)."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.n_param_pred = model.n_param_pred

    def forward(
        self, degradation_parameters: np.ndarray, t_tens: torch.Tensor
    ) -> torch.Tensor:
        degradation_parameters = torch.tensor(degradation_parameters).view(
            1, -1
        )
        # same physical parameters at every time step of the grid
        degradation_parameters = degradation_parameters.expand(
            t_tens.shape[0], -1
        )
        voltage = self.model.predict_physical(t_tens, degradation_parameters)
        return voltage[:, 0]


def define_surrogate_model(inp) -> "SurrogateFCNN":
    """Instantiate the frozen surrogate; its scalers are filled by
    load_state_dict.

    :param inp: parsed surrogate recipe (from inp.surrogate_model_recipe,
        not the FM recipe)
    """
    model = SurrogateFCNN(
        fc_list=inp.fc_units,
        sim_config=inp.sim_config,
        loss_fn=mae_loss_surr,
        cyc_mode=inp.cyc_mode,
        voltage_margin=getattr(inp, "voltage_margin", 0.5),
    )
    logger.info(f"Surrogate trainable parameters: {get_num_parameters(model)}")
    return model


def load_surrogate_model(inp):
    """Load the frozen, trained surrogate model (it carries its scalers)."""
    model = define_surrogate_model(inp)
    best_model_file = find_best_model_file(inp.models_dir)
    logger.info(f"Loading {best_model_file}")
    model.load_state_dict(torch.load(best_model_file, weights_only=True))
    model.eval()
    return model


def norm_coverage(n: float) -> float:
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


def test_perf(inp, mode: str = "val") -> None:
    """Evaluate the trained FM model and write a report.

    Metrics are reported on the held-out validation slice of ``data_split.npz``
    in the training dataset (``inp.data_path``).
    """
    data_path = inp.data_path
    split_file = os.path.join(data_path, "data_split.npz")
    if not os.path.isfile(split_file):
        logger.warning(f"Split file not found at {split_file}, skipping val")
        return
    A = np.load(split_file)
    Y_test = A["Y_val"]

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

    # physical (time, voltage) signals: the model carries its scalers and
    # scales them itself
    X_val = A["X_val"]
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(X_val), torch.from_numpy(Y_test)
        ),
        batch_size=min(X_val.shape[0], 256),
        shuffle=False,
    )

    samples_phys_all, truth_all, noisy_voltage_all = [], [], []

    with torch.no_grad():
        for batch in test_loader:
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
            noisy_voltage_all.append(batch_in.numpy())

    # (n_test, n_samples, n_params), physical
    samples_physical = np.vstack(samples_phys_all)
    truth = np.vstack(truth_all)  # physical
    noisy_voltage = np.vstack(noisy_voltage_all)  # (n_test, 2, n_points)

    n_test, n_samples, n_params = samples_physical.shape
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

    # Voltage-fit check: round-trip real posterior samples (not a Gaussian
    # resample from mu/sigma, since FM already gives us the true samples)
    # through the frozen surrogate and compare against the noisy test voltage.
    n_samp_per_obs = 10
    logger.info(
        f"Using {n_samp_per_obs} posterior samples per observation for the "
        "voltage-fit check (should match the number of MCMC samples)"
    )
    samples_pred_params = samples_physical[:, :n_samp_per_obs, :].copy()
    truth_params = truth[:, np.newaxis, :]
    truth_phi = noisy_voltage[:, 1:2, :]
    truth_time = noisy_voltage[:, 0:1, :]
    post_file = f"post_{mode.lower()}"
    np.savez(
        os.path.join(inp.models_dir, f"{post_file}_samples.npz"),
        pred_params=samples_pred_params,
        truth_params=truth_params,
        truth_phi=truth_phi,
        truth_time=truth_time,
    )

    surr_inp = ri.basic_input(inp.surrogate_model_recipe)
    # Rebase the surrogate recipe's relative models_dir/data_path onto its own
    # step dir so they resolve from our CWD.
    surr_base = os.path.dirname(os.path.dirname(inp.surrogate_model_recipe))
    surr_inp.models_dir = os.path.join(surr_base, surr_inp.models_dir)
    surr_inp.data_path = os.path.join(surr_base, surr_inp.data_path)
    surrogate = load_surrogate_model(surr_inp)
    forward_model = ForwardModel(surrogate)
    voltage_error = np.zeros(samples_pred_params.shape[:2])
    logger.info("Computing voltage error")
    # Samples are already clamped to the prior bounds of the experiment config
    # by model.to_physical, so the surrogate never sees out-of-distribution
    # degradation parameters.

    for i in range(samples_pred_params.shape[0]):
        for j in range(samples_pred_params.shape[1]):
            pred_voltage = forward_model(
                samples_pred_params[i, j].astype("float32"),
                torch.reshape(
                    torch.tensor(truth_time[i].astype("float32")), (-1, 1)
                ),
            )
            true_voltage = truth_phi[i][0, :]
            voltage_error[i, j] = np.mean(
                abs(pred_voltage.detach().numpy() - true_voltage)
            )
    voltage_error_fit = np.mean(voltage_error, axis=1)

    post_file_txt = os.path.join(inp.models_dir, f"{post_file}.txt")
    with open(post_file_txt, "w") as f:
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
        f.write(
            f"Voltage fit median: {1000*np.median(voltage_error_fit)} mV\n"
        )
        f.write(f"Voltage fit mean: {1000*np.mean(voltage_error_fit)} mV\n")

    np.savez(
        os.path.join(inp.models_dir, f"{post_file}.npz"),
        err=err,
        std=sigma_preds,
    )
    logger.info(f"Results written to {post_file_txt}")


if __name__ == "__main__":
    inp = ri.basic_input(sys.argv[1])
    test_perf(inp)
