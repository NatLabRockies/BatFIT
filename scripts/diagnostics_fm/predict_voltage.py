import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable MPS fallback
import concurrent.futures
import pickle
import sys

import numpy as np
import torch
from prettyPlot.plotting import *

from batfit import BATFIT_DIR, BATFIT_EXP, logger
from batfit.basicutilityc import ReadInput as ri
from batfit.preprocess.sim_setup import make_params
from batfit.preprocess.sol_gen import single_run


class ForwardModel:
    def __init__(self, sim_params):
        self.sim_params = sim_params
        self.sim_params[f"deg_eps_s_c_max"] = 1.0
        self.sim_params[f"deg_eps_s_a_max"] = 1.0

    def __call__(self, degradation_parameters: list):
        deg_param_sample = {}
        for ideg_param, deg_param_name in enumerate(
            self.sim_params["deg_param_names"]
        ):
            deg_param_sample[deg_param_name] = float(
                max(
                    min(
                        degradation_parameters[ideg_param],
                        self.sim_params[f"deg_{deg_param_name}_max"],
                    ),
                    self.sim_params[f"deg_{deg_param_name}_min"],
                )
            )
        _, _, rootsol = single_run(deg_param_sample, self.sim_params)
        t_sol = None
        phi_sol = None
        if rootsol is not None:
            t_sol = rootsol.vars["time_s"]
            phi_sol = rootsol.vars["voltage_V"]
        return t_sol, phi_sol


def draw_deg_parameter_samples(
    deg_parameters: dict, n_samples: int = 10
) -> np.ndarray:
    """Draw a random subset of the stored FM posterior samples.

    The FM NPE provides true posterior draws (saved under "samples" by
    predict_param.py), so no Gaussian resampling from mu/sigma is needed.

    :param deg_parameters: per-cell dict with a "samples" key of shape
        (n_stored_samples, n_param_pred)
    :param n_samples: number of samples to draw without replacement
    :return: samples of shape (n_samples, n_param_pred)
    """
    samples = deg_parameters["samples"]
    if n_samples > samples.shape[0]:
        raise ValueError(
            f"Requested {n_samples} samples but only {samples.shape[0]} "
            "posterior samples were stored by predict_param.py"
        )
    indices = np.random.choice(samples.shape[0], size=n_samples, replace=False)
    return samples[indices, :]


def run_simulation(protocol_name, cell, sample_params, forward_model):
    t_tmp, phi_tmp = forward_model(sample_params)
    return protocol_name, cell, t_tmp, phi_tmp


def do_voltage_pred(rpt_id: int, data_type: str, dim: int = 12, n_samples=10):

    assert data_type.lower() in ["posthppc", "diffcap", "diffcap-posthppc"]

    logger.info(f"Doing RPT {rpt_id} for {data_type.lower()}")
    if data_type.lower() == "diffcap-posthppc":
        model_recipe = f"recipes/diffcap/{dim}dim/recipe.yml"
    else:
        model_recipe = f"recipes/{data_type.lower()}/{dim}dim/recipe.yml"
    inp = ri.basic_input(model_recipe)
    sim_params = make_params(inp.sim_config)
    forward_model = ForwardModel(sim_params)

    # Folder management
    voltage_output_folder = os.path.join("output_voltage")
    os.makedirs(voltage_output_folder, exist_ok=True)
    voltage_output_folder = os.path.join(
        voltage_output_folder, data_type.lower()
    )
    os.makedirs(voltage_output_folder, exist_ok=True)
    voltage_output_folder = os.path.join(voltage_output_folder, f"{dim}dim")
    os.makedirs(voltage_output_folder, exist_ok=True)
    param_output_folder = os.path.join(
        "output", data_type.lower(), f"{dim}dim"
    )
    if rpt_id == -1:
        folder_leaf = "BOL"
    else:
        folder_leaf = f"RPT_{rpt_id}"
    voltage_output_folder_leaf = os.path.join(
        voltage_output_folder, folder_leaf
    )
    param_output_folder_leaf = os.path.join(param_output_folder, folder_leaf)
    os.makedirs(voltage_output_folder_leaf, exist_ok=True)

    logger.info(f"\tLoading predicted parameters")
    with open(
        os.path.join(param_output_folder_leaf, "deg_parameters.pkl"), "rb"
    ) as f:
        deg_parameters = pickle.load(f)

    # Load target data
    target_folder = f"/projects/mlbatt/LHX_2/extract_hdvolts_data/data_target/{folder_leaf}"

    tasks = []
    target_data_store = {}
    # 2. FLATTEN THE LOOPS: Gather all targets and parameters into a single task list
    for protocol in deg_parameters:
        # for protocol in ["LH-1"]:
        logger.info(f"\tPreparing tasks for {protocol}")
        target_file = os.path.join(
            target_folder, f"{protocol}_{data_type.lower()}.pkl"
        )
        with open(target_file, "rb") as f:
            target_data = pickle.load(f)

        for cell_id in target_data:
            # for cell_id in [13]: # or loop over target_data
            target_data_cell = target_data[cell_id]
            t_target = target_data_cell["t"]
            phi_target = target_data_cell["phis_c"]

            # Store the target arrays so we can attach them to the final dict later
            target_data_store[(protocol, cell_id)] = {
                "t_target": t_target,
                "phi_target": phi_target,
            }

            deg_samples = draw_deg_parameter_samples(
                deg_parameters[protocol][cell_id], n_samples=n_samples
            )

            # Append every individual sample as a completely independent task
            for isample in range(n_samples):
                tasks.append((protocol, cell_id, deg_samples[isample, :]))

    n_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    logger.info(f"\tRunning {len(tasks)} total simulations on {n_cores} cores")
    results = []

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_cores
    ) as executor:
        futures = [
            executor.submit(run_simulation, p, c, params, forward_model)
            for p, c, params in tasks
        ]

        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    phi_pred = {}
    for protocol, cell_id, t_tmp, phi_tmp in results:
        if protocol not in phi_pred:
            phi_pred[protocol] = {}
        if cell_id not in phi_pred[protocol]:
            phi_pred[protocol][cell_id] = {
                "phi": [],
                "t": [],
                "t_target": target_data_store[(protocol, cell_id)]["t_target"],
                "phi_target": target_data_store[(protocol, cell_id)][
                    "phi_target"
                ],
                "error": [],
            }

        # Append the simulation outputs to the correct lists
        phi_pred[protocol][cell_id]["phi"].append(phi_tmp)
        phi_pred[protocol][cell_id]["t"].append(t_tmp)

        if phi_tmp is not None and t_tmp is not None:
            t_target = phi_pred[protocol][cell_id]["t_target"]
            phi_target = phi_pred[protocol][cell_id]["phi_target"]
            t_interp = np.linspace(
                max(t_target.min(), t_tmp.min()),
                min(t_target.max(), t_tmp.max()),
                2048,
            )
            phi_target_interp = np.interp(t_interp, t_target, phi_target)
            phi_sim_interp = np.interp(t_interp, t_tmp, phi_tmp)
            phi_pred[protocol][cell_id]["error"].append(
                1000 * np.mean(abs(phi_sim_interp - phi_target_interp))
            )
        else:
            phi_pred[protocol][cell_id]["error"].append(np.nan)

    # Save the final structured dictionary
    # with open(os.path.join(voltage_output_folder_leaf, "phi_pred.pkl"), "wb") as f:
    with open(
        os.path.join(voltage_output_folder_leaf, "phi_pred.pkl"), "wb"
    ) as f:
        pickle.dump(phi_pred, f)


if __name__ == "__main__":

    n_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    # for data_type in ["diffcap", "posthppc"]:
    for data_type in ["posthppc"]:
        # for dim in [12, 17, 19]:
        for dim in [12]:
            for rpt_id in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                do_voltage_pred(rpt_id, data_type, n_samples=10, dim=dim)
