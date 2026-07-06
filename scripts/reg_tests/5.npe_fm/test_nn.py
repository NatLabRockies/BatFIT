"""Evaluate a trained ProbParamFM (no protocol conditioning) on held-out data.

Evaluates both the held-out "test" split (inp.data_path) and the separate
synthetic "val" dataset (inp.data_val_path), matching 4.npe_cnn/test_nn.py.
Also includes the surrogate voltage-fit round-trip check, adapted to use the
FM's real posterior samples directly (instead of resampling a Gaussian from
mu/sigma, as the CNN version does).
"""

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
from batfit.model.surrogate_utils.losses import mae_loss as mae_loss_surr
from batfit.model.surrogateNN import SurrogateFCNN
from batfit.utils.data_utils import scale_input_from_scaler
from batfit.utils.torch_utils import get_device_type, get_num_parameters


class ForwardModel(torch.nn.Module):
    """Wraps a trained SurrogateFCNN so it can be called as v(t; degradation_params)."""

    def __init__(self, model: torch.nn.Module, scaler):
        super().__init__()
        self.model = model
        self.n_param_pred = model.n_param_pred
        self.means = scaler.means
        self.stds = 1.0 / scaler.stds

    def forward(
        self, degradation_parameters: np.ndarray, t_tens: torch.Tensor
    ) -> torch.Tensor:
        means = torch.tensor(self.means)
        stds = torch.tensor(self.stds)
        degradation_parameters = torch.tensor(degradation_parameters).view(
            1, -1
        )
        degradation_parameters = degradation_parameters.expand(
            t_tens.shape[0], -1
        )
        x_input = torch.cat((t_tens, degradation_parameters), dim=1)
        x_input = (x_input - means) * stds
        output = self.model(x_input)
        if self.model.constrain_output:
            output = self.model.inv_transform_output(
                output, float(self.model.min_v), float(self.model.amp_v)
            )
        return output[:, 0]


def define_surrogate_model(inp) -> tuple["SurrogateFCNN", object]:
    """Instantiate the frozen surrogate and load its signal scaler.

    :param inp: parsed surrogate recipe (from inp.surrogate_model_recipe,
        not the FM recipe)
    """
    model = SurrogateFCNN(
        fc_list=inp.fc_units,
        loss_fn=mae_loss_surr,
        n_param_pred=inp.n_param_pred,
        sim_config=inp.sim_config,
        cyc_mode=inp.cyc_mode,
        constrain_output=inp.constrain_output,
    )
    logger.info(f"Surrogate trainable parameters: {get_num_parameters(model)}")

    with open(
        os.path.join(inp.data_path, "scaler_surrogate_X.pkl"), "rb"
    ) as f:
        scaler_X = pickle.load(f)

    return model, scaler_X


def load_surrogate_model(inp):
    """Load the frozen, trained surrogate model and its scaler."""
    model, scaler = define_surrogate_model(inp)
    best_model_file = find_best_model_file(inp.models_dir)
    logger.info(f"Loading {best_model_file}")
    model.load_state_dict(torch.load(best_model_file, weights_only=True))
    model.eval()
    return model, scaler


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


def get_model_it(model_dir: str) -> np.ndarray:
    """Return the checkpoint step numbers found in model_dir (excluding "final")."""
    iterations = []
    for fname in os.listdir(model_dir):
        if (
            fname.startswith("model_")
            and fname.endswith(".pt")
            and "final" not in fname
        ):
            iterations.append(int(fname[6 : fname.index(".pt")]))
    return np.array(iterations)


def read_test_loss(model_dir: str):
    """Return the step (or "final") with the lowest recorded test loss."""
    vals = np.loadtxt(
        os.path.join(model_dir, "test_loss.csv"), delimiter=";", skiprows=1
    )
    best_ind = np.argmin(vals[:, 1])
    if best_ind == vals.shape[0] - 1 and os.path.isfile(
        os.path.join(model_dir, "model_final.pt")
    ):
        return "final"
    return vals[best_ind, 0]


def find_best_model_file(model_dir: str) -> str:
    """Return the checkpoint path with the lowest test loss."""
    best_iter = read_test_loss(model_dir)
    if best_iter == "final":
        return os.path.join(model_dir, "model_final.pt")
    iterations = get_model_it(model_dir)
    if len(iterations) == 0:
        return os.path.join(model_dir, "model_final.pt")
    ind = np.argmin(abs(iterations - best_iter))
    return os.path.join(model_dir, f"model_{iterations[ind]}.pt")


def test_perf(inp, mode: str = "test") -> None:
    """Evaluate the trained FM model and write a report.

    :param mode: "test" evaluates the held-out test split of the training
        dataset (inp.data_path); "val" evaluates the separate synthetic
        validation dataset (inp.data_val_path), matching 4.npe_cnn/test_nn.py.
    """
    if mode.lower() == "test":
        data_path = inp.data_path
        split_file = os.path.join(data_path, "data_split.npz")
        if not os.path.isfile(split_file):
            logger.warning(
                f"Split file not found at {split_file}, skipping test"
            )
            return
        A = np.load(split_file)
        X_scaled = scale_input_from_scaler(
            A["X_test"], os.path.join(data_path, "scaler_X.pkl")
        )
        Y_test = A["Y_test"]
    elif mode.lower() == "val":
        data_path = inp.data_val_path
        assembled_file = os.path.join(data_path, "assembled_data.npz")
        if not os.path.isfile(assembled_file):
            logger.warning(
                f"Assembled data not found at {assembled_file}, skipping val"
            )
            return
        A = np.load(assembled_file)
        X_scaled = scale_input_from_scaler(
            A["X_data"], os.path.join(data_path, "scaler_X.pkl")
        )
        Y_test = A["Y_data"]
    else:
        raise NotImplementedError(mode)

    # Training uses scale_y=True, so model.sample() returns posterior samples
    # in z-scored parameter space; scaler_Y brings them back to physical
    # units. It's only ever fit on the training set (inp.data_path), so it's
    # always loaded from there regardless of mode.
    with open(os.path.join(inp.data_path, "scaler_Y.pkl"), "rb") as f:
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

    with open(os.path.join(data_path, "scaler_X.pkl"), "rb") as f:
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
            torch.from_numpy(X_scaled), torch.from_numpy(Y_test)
        ),
        batch_size=min(X_scaled.shape[0], 256),
        shuffle=False,
    )

    samples_z_all, truth_all, noisy_voltage_all = [], [], []

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
                n_samples=inp.n_samples,
                n_steps=inp.n_ode_steps,
            )
            samples_z_all.append(samps.cpu().numpy())
            truth_all.append(batch[1].numpy())
            noisy_voltage_all.append(
                scaler_X.inverse_transform(batch_in.cpu()).numpy()
            )

    samples_z = np.vstack(
        samples_z_all
    )  # (n_test, n_samples, n_params), z-scored
    truth = np.vstack(truth_all)  # physical
    noisy_voltage = np.vstack(noisy_voltage_all)  # (n_test, 2, n_points)

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

    surrogate, surrogate_scaler = load_surrogate_model(
        ri.basic_input(inp.surrogate_model_recipe)
    )
    forward_model = ForwardModel(surrogate, surrogate_scaler)
    voltage_error = np.zeros(samples_pred_params.shape[:2])
    logger.info("Computing voltage error")
    # Clip samples to the physical bounds used when generating the discharge
    # dataset (spm_discharge.yaml), to avoid feeding the surrogate
    # out-of-distribution degradation parameters.
    samples_pred_params[:, :, 0] = np.clip(
        samples_pred_params[:, :, 0], 0.1, 4.0
    )
    samples_pred_params[:, :, 1] = np.clip(
        samples_pred_params[:, :, 1], 0.2, 10.0
    )
    samples_pred_params[:, :, 2] = np.clip(
        samples_pred_params[:, :, 2], 0.6, 1.077
    )
    samples_pred_params[:, :, 3] = np.clip(
        samples_pred_params[:, :, 3], 0.88, 1.6
    )
    samples_pred_params[:, :, 4] = np.clip(
        samples_pred_params[:, :, 4], 0.1, 1.6
    )
    samples_pred_params[:, :, 5] = np.clip(
        samples_pred_params[:, :, 5], 0.7, 1.0
    )

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
    test_perf(inp, mode="test")
    test_perf(inp, mode="val")
