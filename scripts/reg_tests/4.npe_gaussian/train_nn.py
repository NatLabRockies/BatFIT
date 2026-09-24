import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable MPS fallback

import numpy as np
import torch
from prettyPlot.plotting import *

from batfit import BATFIT_DIR, BATFIT_EXP, logger
from batfit.basicutilityc import ReadInput as ri
from batfit.model.param_utils.losses import (
    independent_normal_loss as independent_normal_loss_param,
)
from batfit.model.param_utils.noise_utils import make_signal_noise_levels
from batfit.model.param_utils.train_utils import (
    train_model as train_model_param,
)
from batfit.model.paramNN import ProbParamCNN
from batfit.model.surrogate_utils.losses import mae_loss as mae_loss_surr
from batfit.model.surrogateNN import SurrogateFCNN
from batfit.preprocess.sim_setup import make_params
from batfit.utils.data_utils import *
from batfit.utils.torch_utils import *


def make_data_loaders(inp):
    data_root_folder = inp.data_path
    n_points = inp.n_points
    target_mode = inp.target_mode
    cyc_mode = inp.cyc_mode

    X_data, Y_data = assemble_all_data(
        data_root_folder,
        n_points=n_points,
        combined_pickle_file="sols.pkl",
        target_mode=target_mode,
        save_data=True,
        cyc_mode=cyc_mode,
        save_path=data_root_folder,
    )
    tmp = np.load(os.path.join(data_root_folder, "assembled_data.npz"))

    X_data = tmp["X_data"]
    Y_data = tmp["Y_data"]

    BATCH_SIZE = min(inp.batch_size, int(Y_data.shape[0] * 0.8))
    loaders, scalers = make_npe_dataset_from_np(
        make_params(inp.sim_config),
        np_data=X_data,
        np_data_label=Y_data,
        batch_size=BATCH_SIZE,
        save_path=data_root_folder,
        signal_scaling=getattr(inp, "signal_scaling", "zscore"),
    )

    return loaders, scalers


def define_surrogate_model(inp):
    """Build the frozen surrogate from its recipe; its scalers are filled by
    load_state_dict."""
    model = SurrogateFCNN(
        fc_list=inp.fc_units,
        sim_config=inp.sim_config,
        loss_fn=mae_loss_surr,
        cyc_mode=inp.cyc_mode,
        voltage_margin=getattr(inp, "voltage_margin", 0.5),
    )
    num_parameters = get_num_parameters(model)
    print(f"No. Trainable Parameters: {num_parameters}")

    return model


def define_model(inp, scaler_X=None, scaler_T=None):
    """Instantiate a ProbParamCNN; scaler_X=None (and scaler_T=None) leave
    placeholders that load_state_dict fills from the checkpoint."""
    data_root_folder = inp.data_path
    n_points = inp.n_points
    target_mode = inp.target_mode
    cyc_mode = inp.cyc_mode
    if target_mode != "encoded":
        input_shape = (2, inp.n_points)

    model = ProbParamCNN(
        input_shape=input_shape,
        chan_list=[inp.num_channels] * inp.num_convs,
        fc_list=[inp.num_fc_units] * inp.num_fc_hidden,
        fc_mu_list=[inp.num_fc_gamma_mu_units] * inp.num_fc_gamma_mu_hidden,
        fc_gamma_list=[inp.num_fc_gamma_mu_units] * inp.num_fc_gamma_mu_hidden,
        loss_fn=independent_normal_loss_param,
        sim_config=inp.sim_config,
        cyc_mode=cyc_mode,
        scaler_X=scaler_X,
        param_margin=getattr(inp, "param_margin", 0.05),
        signal_scaling=getattr(inp, "signal_scaling", "zscore"),
        scaler_T=scaler_T,
    )
    num_parameters = get_num_parameters(model)
    print(f"No. Trainable Parameters: {num_parameters}")

    return model


def do_training(inp, model, train_data_loader, test_data_loader):
    noise_levels, a_min, a_max = make_signal_noise_levels(
        model,
        target_mode=inp.target_mode,
        noise_levels=[
            0,
            0.001444 * 2 * inp.noise_factor,
            0.001786 * 2,
            2.01 * 2,
        ],
        cyc_mode=inp.cyc_mode,
    )

    model, loss_hist = train_model_param(
        model,
        train_data_loader=train_data_loader,
        test_data_loader=test_data_loader,
        learning_rate=inp.lr,
        num_epochs=inp.epochs,
        noise_levels=noise_levels,
        a_min=a_min,
        a_max=a_max,
        enable_cuda=True,
        enable_mps=True,
        log_folder=inp.models_dir,
        restart_from=getattr(inp, "restart_from", None) or None,
    )


if __name__ == "__main__":
    import shutil
    import sys

    inp = ri.basic_input(sys.argv[1])
    loaders, scalers = make_data_loaders(inp)
    model = define_model(inp, scaler_X=scalers["X"], scaler_T=scalers.get("T"))
    do_training(inp, model, loaders["train"], loaders["test"])
    shutil.copy(sys.argv[1], os.path.join(inp.models_dir, "recipe.yml"))
