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
from batfit.model.paramNN import *
from batfit.preprocess.sim_setup import make_params
from batfit.utils.data_utils import *
from batfit.utils.data_utils import scale_input_from_scaler
from batfit.utils.torch_utils import *
from batfit.utils.torch_utils import get_device_type

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
from utils import define_model, find_best_model_file

from batfit.basicutilityc import ReadInput as ri
from batfit.calibration.cal_utils import (
    mcmc_iter,
    mcmc_iter_synth,
)
from batfit.calibration.data_utils import (
    collect_observation_files,
    get_neighbor_cycles,
    load_observation_data,
    obs_filename,
)
from batfit.model.paramNN import apply_noise, apply_noise_unscaled


def make_val_data(inp):
    data_root_folder = inp.data_path_discharge
    n_points = inp.n_points
    n_param_pred = inp.n_param_pred
    cyc_mode = inp.cyc_mode

    # This is optional, and is useful to look at the voltage curve prediction
    X_npe_data, Y_npe_data = assemble_all_data(
        data_root_folder,
        n_points=n_points,
        combined_pickle_file=os.path.join(data_root_folder, "sols.pkl"),
        target_mode="phi",
        save_data=True,
        cyc_mode=cyc_mode,
        save_path=data_root_folder,
    )

    return


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
        output = self.model(x_input)
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


if __name__ == "__main__":
    import time

    import batfit.utils.parallel as parallel_env

    inp = ri.basic_input(sys.argv[1])
    if parallel_env.irank == parallel_env.iroot:
        make_val_data(inp)
    parallel_env.barrier()
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

    min_test = np.inf
    for key in cycle_types:
        if total_data_t[key].shape[0] < min_test:
            min_test = total_data_t[key].shape[0]
    # only do 4 mcmc for regression test
    min_test = 4

    n_test_, start_test_ = parallel_env.partitionData(min_test)

    for i_sample_test in range(start_test_, start_test_ + n_test_):
        time_s = time.time()
        for key in cycle_types:
            data_t[key] = total_data_t[key][i_sample_test, :]
            data_phis_c[key] = total_data_phi[key][i_sample_test, :]
            deg_param_truth[key] = total_truth[key][i_sample_test, :]

        sim_params_dict = {}
        for key in cycle_types:
            sim_params_dict[key] = models[key]["sim_params"]

        t = {}
        t_tens = {}

        for key in cycle_types:
            t[key] = np.reshape(data_t[key], (data_t[key].shape[0], 1))
            t_tens[key] = torch.Tensor(t[key])

        forward_dict = {}
        for key in cycle_types:
            if key == "discharge":
                forw_dis = ForwardModel(
                    models["discharge"]["torch_model"],
                    t_tens["discharge"],
                    models["discharge"]["scaler"],
                )
                forward_dict["discharge"] = forw_dis

        size_inpt = {}
        jax_func_dict = {}
        jax_params_dict = {}
        for key in cycle_types:
            size_inpt[key] = models[key]["torch_model"].n_param_pred
            p = np.random.normal(size=(size_inpt[key],)).astype(np.float32)
            jax_params_dict[key] = {
                k: t2j(v) for k, v in forward_dict[key].named_parameters()
            }
            jax_func_dict[key] = lambda p: t2j(forward_dict[key])(
                p, state_dict=jax_params_dict[key]
            )

        _, results = mcmc_iter_synth(
            mcmc_method=mcmc_method,
            cyc_mode=inp.cyc_mode,
            cal_sigma=True,
            num_chains=num_chains,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            sim_params_dict=sim_params_dict,
            target_list=["phis_c"],
            data_phis_c=data_phis_c,
            jax_func_dict=jax_func_dict,
            jax_params_dict=jax_params_dict,
            forward_dict=forward_dict,
            num_warmup=num_warmup,
            num_samples=num_samples,
            parallel_env=parallel_env,
        )
        if i_sample_test == start_test_:
            mcmc_samples_ = results["samples"][np.newaxis, :, :]
        else:
            mcmc_samples_ = np.vstack(
                (mcmc_samples_, results["samples"][np.newaxis, :, :])
            )
        time_e = time.time()

        parallel_env.printAll(
            f"Elapsed time ({i_sample_test-start_test_+1}/{n_test_}) = {time_e-time_s:.2f}s"
        )

    NGlob = int(parallel_env.allsumScalar(mcmc_samples_.shape[0]))
    mcmc_samples = parallel_env.gather3DArray(
        mcmc_samples_,
        parallel_env.iroot,
        mcmc_samples_.shape[0],
        NGlob,
        mcmc_samples_.shape[1],
        mcmc_samples_.shape[2],
    )

    if parallel_env.irank == parallel_env.iroot:
        os.makedirs(inp.models_dir, exist_ok=True)
        np.savez(
            os.path.join(inp.models_dir, "samples.npz"),
            samples=mcmc_samples,
            truths_discharge=total_truth["discharge"][:min_test],
            obs_t_discharge=total_data_t["discharge"][:min_test],
            obs_phi_discharge=total_data_phi["discharge"][:min_test],
        )
