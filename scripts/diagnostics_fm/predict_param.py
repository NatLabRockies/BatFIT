import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable MPS fallback
import pickle
import sys

import numpy as np
import torch
from prettyPlot.plotting import *
from utils import load_fm_model, single_forward_pass

from batfit import BATFIT_DIR, BATFIT_EXP, logger
from batfit.basicutilityc import ReadInput as ri
from batfit.model.paramNN import *
from batfit.preprocess.sim_setup import make_params
from batfit.utils.data_utils import *
from batfit.utils.data_utils import scale_input_from_scaler
from batfit.utils.torch_utils import *
from batfit.utils.torch_utils import get_device_type


def do_diagnostics(rpt_id: int, data_type: str, dim: int):

    assert data_type.lower() in ["posthppc", "diffcap"]

    logger.info(f"Doing RPT {rpt_id} for {data_type.lower()}")

    # Load model
    model_recipe = f"recipes/{data_type.lower()}/{dim}dim/recipe.yml"
    inp = ri.basic_input(model_recipe)
    assert inp.n_param_pred == dim
    scaler_file = inp.scaler_path
    # scaler_Y.pkl is fitted at FM training time (scale_y=True) and saved
    # next to the training data; it maps z-scored posterior samples back to
    # physical parameter space
    scaler_Y_file = os.path.join(inp.data_path, "scaler_Y.pkl")
    model = load_fm_model(inp)
    sim_params = make_params(inp.sim_config)
    device = torch.device(get_device_type())
    model.to(device)

    # Load target data
    if rpt_id == -1:
        folder_leaf = "BOL"
    else:
        folder_leaf = f"RPT_{rpt_id}"
    target_folder = f"/projects/mlbatt/LHX_2/extract_hdvolts_data/data_target/{folder_leaf}"
    sys.path.append("/projects/mlbatt/LHX_2/extract_hdvolts_data/")
    from file_management import cells_protocols_pairs

    pairs = cells_protocols_pairs()

    # Forward pass
    deg_parameters = {}
    for protocol in pairs:
        deg_parameters[protocol] = {}
        logger.info(f"\tDoing {protocol}")
        target_file = os.path.join(
            target_folder, f"{protocol}_{data_type.lower()}.pkl"
        )
        with open(target_file, "rb") as f:
            target_data = pickle.load(f)
        for cell_id in target_data:
            target_data_cell = target_data[cell_id]
            samples, mu, sigma = single_forward_pass(
                target_data_cell["t"],
                target_data_cell["phis_c"],
                scaler_file,
                scaler_Y_file,
                model,
                n_samples=inp.n_samples,
                n_ode_steps=inp.n_ode_steps,
            )
            # keep the mu/sigma keys so plot_aging.py and plot_rpt.py work
            # unchanged; samples are the raw FM posterior draws used by
            # predict_voltage.py
            deg_parameters[protocol][cell_id] = {
                "mu": np.squeeze(mu),
                "sigma": np.squeeze(sigma),
                "samples": samples,
            }

    return deg_parameters


if __name__ == "__main__":
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    for data_type in ["posthppc"]:
        # for data_type in ["diffcap"]:
        output_folder2 = os.path.join(output_folder, data_type.lower())
        os.makedirs(output_folder2, exist_ok=True)
        # for dim in [12, 17, 19]:
        for dim in [12]:
            output_folder3 = os.path.join(output_folder2, f"{dim}dim")
            os.makedirs(output_folder3, exist_ok=True)
            for rpt_id in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                if rpt_id == -1:
                    folder_leaf = "BOL"
                else:
                    folder_leaf = f"RPT_{rpt_id}"
                output_folder4 = os.path.join(output_folder3, folder_leaf)
                os.makedirs(output_folder4, exist_ok=True)
                deg_param = do_diagnostics(rpt_id, data_type, dim=dim)
                with open(
                    os.path.join(output_folder4, "deg_parameters.pkl"), "wb"
                ) as f:
                    pickle.dump(deg_param, f)
