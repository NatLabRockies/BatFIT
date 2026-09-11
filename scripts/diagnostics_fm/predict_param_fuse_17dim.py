import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable MPS fallback
import pickle
import sys

import numpy as np
import torch
from prettyPlot.plotting import *
from utils import define_model, find_best_model_file, single_forward_pass

from batfit import BATFIT_DIR, BATFIT_EXP, logger
from batfit.basicutilityc import ReadInput as ri
from batfit.model.paramNN import *
from batfit.utils.data_utils import *
from batfit.utils.data_utils import scale_input_from_scaler
from batfit.utils.torch_utils import *
from batfit.utils.torch_utils import get_device_type


def classify_samples(candidate_samples):
    sys.path.append("/projects/mlbatt/LHX_2/fuse_post_17dim_ext")
    from classifier import JointBoundaryClassifier
    from fuse_utils import load_config

    config = "/projects/mlbatt/LHX_2/BatFIT/batfit/default_exps/p2d_diffcap_17dim_ext.yaml"
    mins, maxs, param_names = load_config(config)
    model = JointBoundaryClassifier(input_dim=17, hidden_dim=256, num_layers=2)
    model = load_model(
        model,
        "/scratch/mhassana/LHX_2/train_ext/fuse_post_17dim_ext/models/model_final.pt",
    )
    device = model.device
    model.eval()

    if isinstance(candidate_samples, np.ndarray):
        candidate_samples = torch.Tensor(candidate_samples).to(device)

    accepted_mask = model.filter_samples(candidate_samples, threshold=0.5)
    final_valid_samples = candidate_samples[accepted_mask.squeeze()].to("cpu")
    return final_valid_samples.numpy()


def get_mu_sigma_combined(
    deg_parameters_posthppc, deg_parameters_diffcap, n_samples=100000
):
    deg_parameters = {}
    for protocol in deg_parameters_posthppc:
        deg_parameters[protocol] = {}
        for cell_id in deg_parameters_posthppc[protocol]:
            deg_parameters[protocol][cell_id] = {}
            diffcap = deg_parameters_diffcap[protocol][cell_id]
            posthppc = deg_parameters_posthppc[protocol][cell_id]
            assert len(diffcap["sigma"]) == len(diffcap["mu"])
            assert len(posthppc["sigma"]) == len(posthppc["mu"])
            assert len(posthppc["sigma"]) == len(diffcap["sigma"])
            n_params = len(diffcap["sigma"])

            sigma = np.zeros(n_params)
            mu = np.zeros(n_params)
            for i in range(n_params):
                sigma[i] = (
                    1.0 / posthppc["sigma"][i] ** 2
                    + 1.0 / diffcap["sigma"][i] ** 2
                )
                sigma[i] = np.sqrt(1.0 / sigma[i])
                mu[i] = sigma[i] ** 2 * (
                    posthppc["mu"][i] / posthppc["sigma"][i] ** 2
                    + diffcap["mu"][i] / diffcap["sigma"][i] ** 2
                )

            samples = np.random.normal(
                loc=mu, scale=sigma, size=(n_samples, len(mu))
            )
            final_valid_samples = classify_samples(samples)

            mu = np.mean(final_valid_samples, axis=0)
            sigma = np.std(final_valid_samples, axis=0)

            deg_parameters[protocol][cell_id] = {"mu": mu, "sigma": sigma}

    return deg_parameters


def get_combined_posterior(rpt_id, output_folder="output", n_samples=100000):
    output_posthppc = os.path.join(output_folder, "posthppc/17dim")
    output_diffcap = os.path.join(output_folder, "diffcap/17dim")
    if rpt_id == -1:
        folder_leaf = "BOL"
    else:
        folder_leaf = f"RPT_{rpt_id}"

    with open(
        os.path.join(output_posthppc, folder_leaf, "deg_parameters.pkl"), "rb"
    ) as f:
        deg_parameters_posthppc = pickle.load(f)
    with open(
        os.path.join(output_diffcap, folder_leaf, "deg_parameters.pkl"), "rb"
    ) as f:
        deg_parameters_diffcap = pickle.load(f)

    deg_parameters_combined = get_mu_sigma_combined(
        deg_parameters_posthppc, deg_parameters_diffcap
    )

    return deg_parameters_combined


if __name__ == "__main__":
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    for data_type in ["diffcap-posthppc"]:
        output_folder2 = os.path.join(
            output_folder, data_type.lower(), "17dim"
        )
        os.makedirs(output_folder2, exist_ok=True)
        for rpt_id in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            if rpt_id == -1:
                folder_leaf = "BOL"
            else:
                folder_leaf = f"RPT_{rpt_id}"
            output_folder3 = os.path.join(output_folder2, folder_leaf)
            os.makedirs(output_folder3, exist_ok=True)
            deg_param = get_combined_posterior(
                rpt_id, output_folder=output_folder, n_samples=100000
            )
            with open(
                os.path.join(output_folder3, "deg_parameters.pkl"), "wb"
            ) as f:
                pickle.dump(deg_param, f)
