import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable MPS fallback
import pickle

import numpy as np
import torch
from prettyPlot.plotting import *

from batfit import BATFIT_DIR, BATFIT_EXP, logger
from batfit.basicutilityc import ReadInput as ri
from batfit.model.paramNN import *
from batfit.model.surrogateNN import SurrogateFCNN, mae_loss
from batfit.utils.data_utils import *
from batfit.utils.torch_utils import *


def make_data_loaders(inp):
    data_root_folder = inp.data_path
    n_points = inp.n_points
    target_mode = inp.target_mode
    cyc_mode = inp.cyc_mode
    n_param_pred = inp.n_param_pred
    enforce_licons = inp.enforce_licons

    X_data, Y_data = assemble_all_data(
        data_root_folder,
        n_points=n_points,
        combined_pickle_file=os.path.join(data_root_folder, "sols.pkl"),
        target_mode=target_mode,
        save_data=True,
        cyc_mode=cyc_mode,
        save_path=data_root_folder,
    )
    tmp = np.load(os.path.join(data_root_folder, "assembled_data.npz"))

    if enforce_licons and target_mode == "discharge-chargecc":
        logger.info("Removing last input")
        X_data = tmp["X_data"]
        Y_data = tmp["Y_data"][:, :-1]
    else:
        X_data = tmp["X_data"]
        Y_data = tmp["Y_data"]

    BATCH_SIZE = min(inp.batch_size, int(Y_data.shape[0] * 0.9))
    train_data_loader, test_data_loader = make_dataset_from_np(
        batch_size=BATCH_SIZE,
        np_data=X_data,
        np_data_label=Y_data,
        scale=True,
        scale_y=False,
        save_path=data_root_folder,
    )

    return train_data_loader, test_data_loader


def define_surrogate_model(inp):
    data_root_folder = inp.data_path
    n_points = inp.n_points
    n_param_pred = inp.n_param_pred
    cyc_mode = inp.cyc_mode

    model = SurrogateFCNN(
        fc_list=inp.fc_units,
        loss_fn=mae_loss,
        n_param_pred=n_param_pred,
        sim_config=inp.sim_config,
        cyc_mode=cyc_mode,
        constrain_output=inp.constrain_output,
    )
    num_parameters = get_num_parameters(model)
    print(f"No. Trainable Parameters: {num_parameters}")

    with open(
        os.path.join(inp.data_path, "scaler_surrogate_X.pkl"), "rb"
    ) as f:
        scaler_X = pickle.load(f)

    return model, scaler_X


def define_model(inp):
    data_root_folder = inp.data_path
    n_points = inp.n_points
    target_mode = inp.target_mode
    cyc_mode = inp.cyc_mode
    n_param_pred = inp.n_param_pred
    enforce_licons = inp.enforce_licons
    if target_mode != "encoded":
        input_shape = (2, inp.n_points)

    model = ProbParamCNN(
        input_shape=input_shape,
        chan_list=[inp.num_channels] * inp.num_convs,
        fc_list=[inp.num_fc_units] * inp.num_fc_hidden,
        fc_mu_list=[inp.num_fc_gamma_mu_units] * inp.num_fc_gamma_mu_hidden,
        fc_gamma_list=[inp.num_fc_gamma_mu_units] * inp.num_fc_gamma_mu_hidden,
        loss_fn=independent_normal_loss,
        cyc_mode=cyc_mode,
        n_param_pred=n_param_pred,
        constrain_output=True,
        dependent_outputs=False,
        enforce_licons=enforce_licons,
        sim_config=inp.sim_config,
    )
    num_parameters = get_num_parameters(model)
    print(f"No. Trainable Parameters: {num_parameters}")

    with open(os.path.join(inp.data_path, "scaler_X.pkl"), "rb") as f:
        scaler_X = pickle.load(f)

    return model, scaler_X


def do_training(inp, model, train_data_loader, test_data_loader, scaler_X):
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

    model, loss_hist = train_model(
        model,
        train_data_loader=train_data_loader,
        test_data_loader=test_data_loader,
        learning_rate=inp.lr,
        num_epochs=inp.epochs,
        scaler_X=scaler_X,
        noise_levels=noise_levels,
        a_min=a_min,
        a_max=a_max,
        enable_cuda=True,
        enable_mps=True,
        log_folder=inp.models_dir,
    )


if __name__ == "__main__":
    import shutil
    import sys

    inp = ri.basic_input(sys.argv[1])
    train_data_loader, test_data_loader = make_data_loaders(inp)
    model, scaler_X = define_model(inp)
    do_training(inp, model, train_data_loader, test_data_loader, scaler_X)
    shutil.copy(sys.argv[1], os.path.join(inp.models_dir, "recipe.yml"))
