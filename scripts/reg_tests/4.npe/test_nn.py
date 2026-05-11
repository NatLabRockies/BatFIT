import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable MPS fallback
import pickle

import numpy as np
import torch
from prettyPlot.plotting import *
from scipy.stats import norm
from train_nn import define_model, define_surrogate_model

from batfit import BATFIT_DIR, BATFIT_EXP, logger
from batfit.basicutilityc import ReadInput as ri
from batfit.model.paramNN import *
from batfit.model.param_utils.noise_utils import *
from batfit.utils.data_utils import *
from batfit.utils.torch_utils import *


class ForwardModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, scaler):
        super(ForwardModel, self).__init__()
        self.model = model
        self.n_param_pred = model.n_param_pred
        self.means = scaler.means
        self.stds = 1.0 / scaler.stds

    def forward(self, degradation_parameters: list, t_tens: torch.Tensor):
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


def norm_coverage(n):
    """
    Compute the percentage of data within ±n standard deviations
    for a standard normal distribution.
    """
    prob = norm.cdf(n) - norm.cdf(-n)
    return prob * 100  # convert to percentage


def from_mom_to_samples(mu, sigma, n=500):
    assert mu.shape == sigma.shape
    samples = np.random.normal(
        loc=mu,
        scale=sigma,
        size=(n, len(sigma)),
    )
    return samples


def get_model_it(model_dirs):
    filenames = os.listdir(model_dirs)
    iterations = []
    for filename in filenames:
        if (
            filename.startswith("model_")
            and filename.endswith(".pt")
            and "final" not in filename
        ):
            ind_end = filename.index(".pt")
            ind_start = 6
            iterations.append(int(filename[ind_start:ind_end]))
    return np.array(iterations)


def read_test_loss(model_dirs):
    filename = os.path.join(model_dirs, "test_loss.csv")
    vals = np.loadtxt(filename, delimiter=";", skiprows=1)
    best_ind = np.argmin(vals[:, 1])
    if best_ind == vals.shape[0] - 1 and os.path.isfile(
        os.path.join(model_dirs, "model_final.pt")
    ):
        return "final"
    else:
        return vals[best_ind, 0]


def find_best_model_file(model_dirs: str) -> str:
    """
    Get model checkpoint that correspond to the best test loss
    """
    best_iter = read_test_loss(model_dirs)
    if best_iter == "final":
        return os.path.join(model_dirs, "model_final.pt")
    else:
        iterations = get_model_it(model_dirs)
        if len(iterations) == 0:
            return os.path.join(model_dirs, f"model_final.pt")
        else:
            ind = np.argmin(abs(iterations - best_iter))
            return os.path.join(model_dirs, f"model_{iterations[ind]}.pt")


def load_model(inp):
    model, scaler = define_model(inp)
    best_model_file = find_best_model_file(inp.models_dir)
    logger.info(f"Loading {best_model_file}")
    model.load_state_dict(torch.load(best_model_file, weights_only=True))
    model.eval()
    return model, scaler


def load_surrogate_model(inp):
    model, scaler = define_surrogate_model(inp)
    best_model_file = find_best_model_file(inp.models_dir)
    logger.info(f"Loading {best_model_file}")
    model.load_state_dict(torch.load(best_model_file, weights_only=True))
    model.eval()
    return model, scaler


def load_synthetic_data(inp):
    t = {}
    phi = {}
    truth = {}
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
    data_path = inp.data_path_discharge
    tmp = np.load(os.path.join(data_path, "assembled_data.npz"))
    batch_in_unscaled = apply_noise_unscaled(
        torch.tensor(tmp["X_data"]),
        noise_levels=noise_levels,
        a_min=a_min,
        a_max=a_max,
    )
    t["discharge"] = batch_in_unscaled[:, 0, :]
    phi["discharge"] = batch_in_unscaled[:, 1, :]
    truth["discharge"] = tmp["Y_data"][:, :]

    return (
        t,
        phi,
        truth,
    )


def test_perf(inp, mode="test"):
    if mode.lower() == "test":
        data_path = inp.data_path
        if not os.path.isfile(os.path.join(data_path, "data_split.npz")):
            return
        # Make dataset
        A = np.load(os.path.join(data_path, "data_split.npz"))
        X_scaled = scale_input_from_scaler(
            A["X_test"],
            os.path.join(data_path, "scaler_X.pkl"),
        )
        Y_test = A["Y_test"]
    if mode.lower() == "val":
        data_path = inp.data_val_path
        if not os.path.isfile(os.path.join(data_path, "assembled_data.npz")):
            return
        # Make dataset
        A = np.load(os.path.join(data_path, "assembled_data.npz"))
        X_scaled = scale_input_from_scaler(
            A["X_data"],
            os.path.join(data_path, "scaler_X.pkl"),
        )
        Y_test = A["Y_data"]

    input_data = torch.Tensor(X_scaled)
    output_data = torch.Tensor(Y_test)
    shape_in = input_data[0].shape
    test_data_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(input_data, output_data),
        batch_size=min(X_scaled.shape[0], 256),
        shuffle=False,
    )

    # Make model
    model, scaler = define_model(inp)
    best_model_file = find_best_model_file(inp.models_dir)
    logger.info(f"Loading {best_model_file}")
    model.load_state_dict(torch.load(best_model_file, weights_only=True))

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

    device = torch.device(get_device_type())
    model.to(device)
    model.eval()

    logger.info("Forward pass")

    # Forward pass
    with torch.no_grad():
        for ibatch, batch in enumerate(test_data_loader):
            batch_in = apply_noise(
                batch_in=batch[0],
                scaler_X=scaler,
                noise_levels=noise_levels,
                a_min=a_min,
                a_max=a_max,
            )
            tmpmu_preds, tmpsigma_preds = model(batch_in.to(device))
            if model.constrain_output:
                tmpmu_preds = model.inv_transform_mu(
                    tmpmu_preds.cpu(),
                    model.min_par.numpy(),
                    model.amp_par.numpy(),
                )
                tmpsigma_preds = model.inv_transform_gamma(
                    tmpsigma_preds.cpu(), model.amp_par.numpy()
                )

            tmpsigma_preds = tmpsigma_preds.numpy()
            tmpmu_preds = tmpmu_preds.numpy()

            tmptruth = batch[1].cpu().numpy()

            tmperr = abs(tmpmu_preds - tmptruth)
            tmpmu_preds = tmpmu_preds
            tmpsigma_preds = tmpsigma_preds
            tmp_noisy_voltage = scaler.inverse_transform(batch_in).numpy()

            if ibatch == 0:
                mu_preds = tmpmu_preds
                err = tmperr
                truth = tmptruth
                sigma_preds = tmpsigma_preds
                noisy_voltage = tmp_noisy_voltage
            else:
                mu_preds = np.vstack((mu_preds, tmpmu_preds))
                err = np.vstack((err, tmperr))
                sigma_preds = np.vstack((sigma_preds, tmpsigma_preds))
                truth = np.vstack((truth, tmptruth))
                noisy_voltage = np.vstack((noisy_voltage, tmp_noisy_voltage))

    logger.info("Computing error metrics")

    mean_err = np.mean(err, axis=0)
    rmse = np.sqrt(np.mean(err**2, axis=0))
    mean_std = np.mean(sigma_preds, axis=0)
    amp = np.amax(Y_test, axis=0) - np.amin(Y_test, axis=0)
    perf_metric = np.sum(mean_err / amp) / Y_test.shape[1]
    coverage = np.mean(abs(mean_std - rmse))

    logger.info("coverage")
    n_test_samples = truth.shape[0]
    n_params = truth.shape[1]
    non_coverage_1 = np.zeros(n_test_samples)
    non_coverage_2 = np.zeros(n_test_samples)
    non_coverage_3 = np.zeros(n_test_samples)
    for i in range(n_test_samples):
        for j in range(n_params):
            if (
                truth[i, j] > mu_preds[i, j] + sigma_preds[i, j]
                or truth[i, j] < mu_preds[i, j] - sigma_preds[i, j]
            ):
                non_coverage_1[i] += 1
            if (
                truth[i, j] > mu_preds[i, j] + 2 * sigma_preds[i, j]
                or truth[i, j] < mu_preds[i, j] - 2 * sigma_preds[i, j]
            ):
                non_coverage_2[i] += 1
            if (
                truth[i, j] > mu_preds[i, j] + 3 * sigma_preds[i, j]
                or truth[i, j] < mu_preds[i, j] - 3 * sigma_preds[i, j]
            ):
                non_coverage_3[i] += 1

    coverage_1 = 1 - np.sum(non_coverage_1) / (len(non_coverage_1) * n_params)
    coverage_2 = 1 - np.sum(non_coverage_2) / (len(non_coverage_2) * n_params)
    coverage_3 = 1 - np.sum(non_coverage_3) / (len(non_coverage_3) * n_params)
    true_coverage_1 = 0.01 * norm_coverage(1.0)
    true_coverage_2 = 0.01 * norm_coverage(2.0)
    true_coverage_3 = 0.01 * norm_coverage(3.0)
    coverage_discrepancy = (
        abs(coverage_1 - true_coverage_1)
        + abs(coverage_2 - true_coverage_2)
        + abs(coverage_3 - true_coverage_3)
    )

    post_file = f"post_{mode}"

    # Sample samples for 100 realizations
    samples_pred_params = []
    truth_params = []
    truth_phi = []
    truth_time = []
    n_samp_per_obs = 10
    logger.info(
        f"Sampling {n_samp_per_obs} (This should match the number of MCMC samples)"
    )
    for i in range(mu_preds.shape[0]):
        tmpsamples = from_mom_to_samples(
            mu_preds[i, :], sigma_preds[i, :], n=n_samp_per_obs
        )
        samples_pred_params.append(tmpsamples)
        truth_params.append(np.reshape(truth[i, :], (1, -1)))
        truth_phi.append(np.reshape(noisy_voltage[i, 1, :], (1, -1)))
        truth_time.append(np.reshape(noisy_voltage[i, 0, :], (1, -1)))
    samples_pred_params = np.array(samples_pred_params)
    truth_params = np.array(truth_params)
    truth_phi = np.array(truth_phi)
    truth_time = np.array(truth_time)
    np.savez(
        os.path.join(inp.models_dir, f"{post_file}_samples.npz"),
        pred_params=samples_pred_params,
        truth_params=truth_params,
        truth_phi=truth_phi,
        truth_time=truth_time,
    )

    # Compute voltage ierror
    surrogate, surrogate_scaler = load_surrogate_model(
        ri.basic_input(inp.surrogate_model_recipe)
    )
    forward_model = ForwardModel(surrogate, surrogate_scaler)
    voltage_error = np.zeros(samples_pred_params.shape[:2])
    logger.info("Voltage error")
    # Clip samples to avoid issues
    samples_pred_params_clip = samples_pred_params
    samples_pred_params_clip[:, :, 0] = np.clip(
        samples_pred_params[:, :, 0], 0.1, 4.0
    )
    samples_pred_params_clip[:, :, 1] = np.clip(
        samples_pred_params[:, :, 1], 0.2, 10.0
    )
    samples_pred_params_clip[:, :, 2] = np.clip(
        samples_pred_params[:, :, 2], 0.6, 1.077
    )
    samples_pred_params_clip[:, :, 3] = np.clip(
        samples_pred_params[:, :, 3], 0.88, 1.6
    )
    samples_pred_params_clip[:, :, 4] = np.clip(
        samples_pred_params[:, :, 4], 0.1, 1.6
    )
    samples_pred_params_clip[:, :, 5] = np.clip(
        samples_pred_params[:, :, 5], 0.7, 1.0
    )

    for i in range(samples_pred_params.shape[0]):
        logger.info(i)
        for j in range(samples_pred_params.shape[1]):
            pred_voltage = forward_model(
                samples_pred_params_clip[i, j].astype("float32"),
                torch.reshape(
                    torch.tensor(truth_time[i].astype("float32")), (-1, 1)
                ),
            )
            true_voltage = truth_phi[i][0, :]
            voltage_error[i, j] = np.mean(
                abs(pred_voltage.detach().numpy() - true_voltage)
            )
    voltage_error_fit = np.mean(voltage_error, axis=1)

    with open(os.path.join(inp.models_dir, f"{post_file}.txt"), "w+") as f:
        f.write(f"MAE: {mean_err}\n")
        f.write(f"RMSE: {rmse}\n")
        f.write(f"STD: {mean_std}\n")
        f.write(f"PERF: {perf_metric}\n")
        f.write(f"COV: {coverage}\n")
        f.write(f"COV_ERROR: {coverage_discrepancy}\n")
        f.write(f"COV_1: {coverage_1} (instead of {true_coverage_1})\n")
        f.write(f"COV_2: {coverage_2} (instead of {true_coverage_2})\n")
        f.write(f"COV_3: {coverage_3} (instead of {true_coverage_3})\n")
        f.write(
            f"Voltage fit median: {1000*np.median(voltage_error_fit)} mV\n"
        )
        f.write(f"Voltage fit mean: {1000*np.mean(voltage_error_fit)} mV\n")

    np.savez(
        os.path.join(inp.models_dir, f"{post_file}.npz"),
        err=err,
        std=sigma_preds,
    )

    figure_folder = os.path.join(inp.models_dir, "Figures")
    os.makedirs(figure_folder, exist_ok=True)


if __name__ == "__main__":
    import shutil
    import sys

    # from postproc import plot_forw,plot_dist,plot_samples, plot_repeated_samples
    # from prettyPlot.plotting import *
    inp = ri.basic_input(sys.argv[1])
    test_perf(inp, mode="test")
    test_perf(inp, mode="val")
