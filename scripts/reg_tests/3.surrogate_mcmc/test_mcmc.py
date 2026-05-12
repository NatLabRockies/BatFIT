import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable MPS fallback
import pickle

import numpy as np
import torch
from jax._src import config
from prettyPlot.plotting import *
from torch2jax import j2t, t2j

from batfit import BATFIT_DIR, BATFIT_EXP, logger
from batfit.basicutilityc import ReadInput as ri
#from batfit.model.paramNN import *
from batfit.model.param_utils.noise_utils import make_noise_levels, apply_noise_unscaled
from batfit.preprocess.sim_setup import make_params
from batfit.utils.data_utils import *
from batfit.utils.torch_utils import *

config.update("jax_platforms", "cpu")
import sys

import corner
import jax
import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from jax import config
from numpyro.infer import MCMC, NUTS, SA
from numpyro.infer.initialization import *
from scipy.stats import norm
from utils import define_model, find_best_model_file

from batfit.basicutilityc import ReadInput as ri


def norm_coverage(n):
    """
    Compute the percentage of data within ±n standard deviations
    for a standard normal distribution.
    """
    prob = norm.cdf(n) - norm.cdf(-n)
    return prob * 100  # convert to percentage


def load_model(inp):
    model, scaler = define_model(inp)
    best_model_file = find_best_model_file(inp.models_dir)
    logger.info(f"Loading {best_model_file}")
    model.load_state_dict(torch.load(best_model_file, weights_only=True))
    model.eval()
    return model, scaler


def load_surrogates(inp):
    models = {}
    inp_discharge = ri.basic_input(inp.model_discharge_recipe)
    tmp_d = load_model(inp_discharge)
    models["discharge"] = {
        "torch_model": tmp_d[0],
        "scaler": tmp_d[1],
        "sim_params": make_params(inp_discharge.sim_config),
    }
    return models


class TorchScaler(torch.nn.Module):
    def __init__(self, scaler):
        super(TorchScaler, self).__init__()
        self.means = torch.tensor(scaler.means)
        self.stds = torch.tensor(scaler.stds)

    def transform(self, data):
        transformed_data = (data - self.means) / self.stds
        return transformed_data

    def inverse_transform(self, transformed_data):
        data = transformed_data * self.stds + self.means
        return data


# device = torch.device(get_device_type())


class ForwardModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, t_tens: torch.Tensor, scaler):
        super(ForwardModel, self).__init__()
        self.model = model
        self.n_param_pred = model.n_param_pred
        self.means = scaler.means
        self.stds = 1.0 / scaler.stds
        self.t_tens_shape = t_tens.shape
        self.t_tens = t_tens

    def forward(self, degradation_parameters: list):
        t_tens = torch.tensor(self.t_tens)
        means = torch.tensor(self.means)
        stds = torch.tensor(self.stds)
        degradation_parameters = torch.tensor(degradation_parameters).view(
            1, -1
        )
        degradation_parameters = degradation_parameters.expand(
            self.t_tens_shape[0], -1
        )
        x_input = torch.cat((t_tens, degradation_parameters), dim=1)
        x_input = (x_input - means) * stds

        # x_input = x_input.to(device)
        # self.model = self.model.to(device)
        output = self.model(x_input)
        # self.model = self.model.to("cpu")
        # x_input = x_input.to("cpu")
        # output = output.to("cpu")
        if self.model.constrain_output:
            output = self.model.inv_transform_output(
                output, float(self.model.min_v), float(self.model.amp_v)
            )
        return output[:, 0]


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


def compute_fit_error(inp, samples, models):
    (
        total_data_t,
        total_data_phi,
        total_truth,
    ) = load_synthetic_data(inp)
    voltage_err = {}
    for key in models:
        voltage_err[key] = np.zeros(samples.shape[:2])

    for i_sample_test in range(samples.shape[0]):
        # for i_sample_test in range(2):
        logger.info(i_sample_test)
        for key in cycle_types:
            data_t[key] = total_data_t[key][i_sample_test, :]
            data_phis_c[key] = total_data_phi[key][i_sample_test, :]
            deg_param_truth[key] = total_truth[key][i_sample_test, :]

        sim_params_dict = {}
        for key in models:
            sim_params_dict[key] = models[key]["sim_params"]

        t = {}
        t_tens = {}

        for key in models:
            t[key] = np.reshape(data_t[key], (data_t[key].shape[0], 1))
            t_tens[key] = torch.Tensor(t[key])

        forward_dict = {}
        for key in models:
            forw_dis = ForwardModel(
                models["discharge"]["torch_model"],
                t_tens["discharge"],
                models["discharge"]["scaler"],
            )
            forward_dict["discharge"] = forw_dis

        size_inpt = {}
        jax_func_dict = {}
        jax_params_dict = {}
        for key in models:
            size_inpt[key] = models[key]["torch_model"].n_param_pred
            p = np.random.normal(size=(size_inpt[key],)).astype(np.float32)
            jax_params_dict[key] = {
                k: t2j(v) for k, v in forward_dict[key].named_parameters()
            }
            jax_func_dict[key] = lambda p: t2j(forward_dict[key])(
                p, state_dict=jax_params_dict[key]
            )

        for key in models:
            # fig=plt.figure()
            true_v = np.array(data_phis_c[key])
            # plt.plot(true_v, color="k")
            for j_sample_test in range(samples.shape[1]):
                pred_v = np.array(
                    jax_func_dict[key](
                        jnp.array(samples[i_sample_test, j_sample_test, :-1])
                    )
                )
                voltage_err[key][i_sample_test, j_sample_test] = np.mean(
                    abs(true_v - pred_v)
                )
                # plt.plot(pred_v, color="b")
            # plt.show()

    return voltage_err


if __name__ == "__main__":
    import time

    import batfit.utils.parallel as parallel_env

    # Read input
    inp = ri.basic_input(sys.argv[1])
    min_sigma = inp.min_sigma
    max_sigma = inp.max_sigma
    calibrate_sigma = inp.calibrate_sigma
    mcmc_method = inp.mcmc_method
    num_warmup = inp.num_warmup
    num_samples = inp.num_samples
    step_size = inp.step_size
    num_chains = inp.num_chains
    # numpyro.set_host_device_count(num_chains)
    cyc_mode = inp.cyc_mode
    target_mode = inp.target_mode
    if target_mode != "phi":
        raise NotImplementedError(
            "No support for differential capacity for now"
        )

    # Load surrogates
    models = load_surrogates(inp)
    (
        total_data_t,
        total_data_phi,
        total_truth,
    ) = load_synthetic_data(inp)

    cycle_types = list(models.keys())

    # get_data
    data_t = {}
    data_phis_c = {}
    deg_param_truth = {}
    mode = "normal"
    truths = {}
    samples = np.load(os.path.join(inp.models_dir, "samples.npz"))["samples"]
    for key in cycle_types:
        truths[key] = np.load(os.path.join(inp.models_dir, "samples.npz"))[
            f"truths_{key}"
        ]
    truth = truths[cyc_mode][:]
    A = total_truth[cyc_mode][:]
    n_test_samples = samples.shape[0]

    # Compute Param errors
    mu_preds = np.mean(samples[:, :, :-1], axis=1)
    err = abs(mu_preds - truth)
    sigma_preds = np.std(samples[:, :, :-1], axis=1)

    # Compute Voltage errors
    voltage_err = compute_fit_error(inp, samples, models)[cyc_mode]

    # Compute MAE
    mean_err = np.mean(err, axis=0)
    rmse = np.sqrt(np.mean(err**2, axis=0))
    mean_std = np.mean(sigma_preds, axis=0)
    amp = np.amax(A, axis=0) - np.amin(A, axis=0)
    perf_metric = np.sum(mean_err / amp) / A.shape[1]
    coverage = np.mean(abs(mean_std - rmse))
    n_test_samples = truth.shape[0]
    n_params = truth.shape[1]

    ptile_p1sig = np.zeros((n_test_samples, n_params))
    ptile_m1sig = np.zeros((n_test_samples, n_params))
    ptile_p2sig = np.zeros((n_test_samples, n_params))
    ptile_m2sig = np.zeros((n_test_samples, n_params))
    ptile_p3sig = np.zeros((n_test_samples, n_params))
    ptile_m3sig = np.zeros((n_test_samples, n_params))

    for i in range(n_test_samples):
        for j in range(n_params):
            ptile_p1sig[i, j] = np.percentile(
                samples[i, :, j], 50 + 34.13447460685429
            )
            ptile_m1sig[i, j] = np.percentile(
                samples[i, :, j], 50 - 34.13447460685429
            )
            ptile_p2sig[i, j] = np.percentile(
                samples[i, :, j], 50 + 47.72498680518208
            )
            ptile_m2sig[i, j] = np.percentile(
                samples[i, :, j], 50 - 47.72498680518208
            )
            ptile_p3sig[i, j] = np.percentile(
                samples[i, :, j], 50 + 49.86501019683699
            )
            ptile_m3sig[i, j] = np.percentile(
                samples[i, :, j], 50 - 49.86501019683699
            )

    non_coverage_1 = np.zeros(n_test_samples)
    non_coverage_2 = np.zeros(n_test_samples)
    non_coverage_3 = np.zeros(n_test_samples)
    for i in range(n_test_samples):
        for j in range(n_params):
            if (
                truth[i, j] > ptile_p1sig[i, j]
                or truth[i, j] < ptile_m1sig[i, j]
            ):
                non_coverage_1[i] += 1
            if (
                truth[i, j] > ptile_p2sig[i, j]
                or truth[i, j] < ptile_m2sig[i, j]
            ):
                non_coverage_2[i] += 1
            if (
                truth[i, j] > ptile_p3sig[i, j]
                or truth[i, j] < ptile_m3sig[i, j]
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

    voltage_error_fit = np.mean(voltage_err, axis=1)

    post_file = "post"
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
