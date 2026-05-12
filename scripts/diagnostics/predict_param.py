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


def do_diagnostics(rpt_id: int, data_type: str):

    assert data_type.lower() in ["hppc", "diffcap"]

    logger.info(f"Doing RPT {rpt_id} for {data_type.lower()}")

    # Load model
    model_recipe = f"recipes/{data_type.lower()}/recipe.yml"
    inp = ri.basic_input(model_recipe)
    scaler_file = inp.scaler_path
    model, scaler_X = define_model(inp)
    best_model_file = find_best_model_file(inp.models_dir)
    logger.info(f"Loading {best_model_file}")
    model.load_state_dict(torch.load(best_model_file, weights_only=True))
    sim_params = make_params(inp.sim_config)
    device = torch.device(get_device_type())
    model.to(device)
    model.eval()

    # Load target data
    if rpt_id == -1:
        folder_leaf = "BOL"
    else:
        folder_leaf = f"RPT_{rpt_id}"
    target_folder = f"/Users/mhassana/Desktop/GitHub/BatFIT_mar26/scripts/extract_hdvolts_data/data_target/{folder_leaf}"
    sys.path.append(
        "/Users/mhassana/Desktop/GitHub/BatFIT_mar26/scripts/extract_hdvolts_data/"
    )
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
            mu, sigma = single_forward_pass(
                target_data_cell["t"],
                target_data_cell["phis_c"],
                scaler_file,
                model,
            )
            deg_parameters[protocol][cell_id] = {
                "mu": np.squeeze(mu),
                "sigma": np.squeeze(sigma),
            }

    return deg_parameters


if __name__ == "__main__":
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    for data_type in ["hppc", "diffcap"]:
        output_folder2 = os.path.join(output_folder, data_type.lower())
        os.makedirs(output_folder2, exist_ok=True)
        for rpt_id in [-1, 1, 2, 3, 4, 5, 6, 7, 8]:
            if rpt_id == -1:
                folder_leaf = "BOL"
            else:
                folder_leaf = f"RPT_{rpt_id}"
            output_folder3 = os.path.join(output_folder2, folder_leaf)
            os.makedirs(output_folder3, exist_ok=True)
            deg_param = do_diagnostics(rpt_id, data_type)
            with open(
                os.path.join(output_folder3, "deg_parameters.pkl"), "wb"
            ) as f:
                pickle.dump(deg_param, f)
