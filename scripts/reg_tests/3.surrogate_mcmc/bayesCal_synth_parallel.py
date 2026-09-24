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
from batfit.model.param_utils.noise_utils import (
    apply_noise,
    apply_noise_unscaled,
    make_noise_levels,
)


def load_model(inp):
    model = define_model(inp)
    best_model_file = find_best_model_file(inp.models_dir)
    logger.info(f"Loading {best_model_file}")
    model.load_state_dict(torch.load(best_model_file, weights_only=True))
    model.eval()
    return model


def load_surrogates(inp):
    models = {}
    inp_discharge = ri.basic_input(inp.model_discharge_recipe)
    # Rebase the surrogate recipe's relative models_dir/data_path onto its own
    # step dir so they resolve from our CWD.
    surr_base = os.path.dirname(os.path.dirname(inp.model_discharge_recipe))
    inp_discharge.models_dir = os.path.join(
        surr_base, inp_discharge.models_dir
    )
    inp_discharge.data_path = os.path.join(surr_base, inp_discharge.data_path)
    surrogate = load_model(inp_discharge)
    models["discharge"] = {
        "torch_model": surrogate,
        "sim_params": surrogate.sim_params,
    }
    return models


class ForwardModel(torch.nn.Module):
    """Wrap the surrogate as ``voltage(deg_params)`` over a fixed time grid."""

    def __init__(self, model: torch.nn.Module, t_tens: torch.Tensor):
        super(ForwardModel, self).__init__()
        self.model = model
        self.n_param_pred = model.n_param_pred
        # torch2jax intercepts every torch call while tracing, including
        # .shape on plain tensors: keep the time grid as numpy and its size
        # as an int, and rebuild the tensor inside forward
        self.t_np = t_tens.numpy()
        self.n_times = t_tens.shape[0]

    def forward(self, degradation_parameters: list):
        t_tens = torch.tensor(self.t_np)
        degradation_parameters = torch.tensor(degradation_parameters).view(
            1, -1
        )
        # same physical parameters at every time step of the grid
        degradation_parameters = degradation_parameters.expand(
            self.n_times, -1
        )
        voltage = self.model.predict_physical(t_tens, degradation_parameters)
        return voltage[:, 0]


def load_synthetic_data(inp, sim_params):
    """Noisy validation observations; noise clipped to the config's vmin/vmax."""
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
        vmin=sim_params["vmin"],
        vmax=sim_params["vmax"],
    )
    data_path = inp.data_path_discharge
    # Use the held-out validation slice as the synthetic observations to invert.
    tmp = np.load(os.path.join(data_path, "data_split.npz"))
    batch_in_unscaled = apply_noise_unscaled(
        torch.tensor(tmp["X_val"]),
        noise_levels=noise_levels,
        a_min=a_min,
        a_max=a_max,
    )
    t["discharge"] = batch_in_unscaled[:, 0, :]
    phi["discharge"] = batch_in_unscaled[:, 1, :]
    truth["discharge"] = tmp["Y_val"][:, :]

    return (
        t,
        phi,
        truth,
    )


if __name__ == "__main__":
    import time

    import batfit.utils.parallel as parallel_env

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

    models = load_surrogates(inp)
    (
        total_data_t,
        total_data_phi,
        total_truth,
    ) = load_synthetic_data(inp, models["discharge"]["sim_params"])
    cycle_types = list(models.keys())

    # get_data
    data_t = {}
    data_phis_c = {}
    deg_param_truth = {}

    min_test = np.inf
    for key in cycle_types:
        if total_data_t[key].shape[0] < min_test:
            min_test = total_data_t[key].shape[0]
    # MCMC is far more expensive than NPE, so cap how many val curves we invert
    # (never exceeding the available val count).
    min_test = min(int(min_test), inp.n_val_mcmc)

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
                )
                forward_dict["discharge"] = forw_dis

        size_inpt = {}
        jax_func_dict = {}
        jax_params_dict = {}
        for key in cycle_types:
            size_inpt[key] = models[key]["torch_model"].n_param_pred
            p = np.random.normal(size=(size_inpt[key],)).astype(np.float32)
            # full state dict: torch2jax needs the weights AND the buffers
            # (the surrogate's scalers)
            jax_params_dict[key] = {
                k: t2j(v) for k, v in forward_dict[key].state_dict().items()
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
