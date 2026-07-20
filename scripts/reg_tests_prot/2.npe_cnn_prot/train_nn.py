import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import shutil
import sys

import numpy as np
import torch
from prettyPlot.plotting import *

from batfit import BATFIT_DIR, BATFIT_EXP, logger
from batfit.basicutilityc import ReadInput as ri
from batfit.model.param_utils.losses import (
    independent_normal_loss as independent_normal_loss_param,
)
from batfit.model.param_utils.noise_utils import make_noise_levels
from batfit.model.param_utils.train_utils import (
    train_model as train_model_param,
)
from batfit.model.paramNN import ProbProtParamCNN
from batfit.utils.data_utils import assemble_all_data
from batfit.utils.torch_utils import (
    get_num_parameters,
    make_protocol_dataset_from_np,
)


def make_data_loaders(inp):
    """Assemble chirp data and build 3-tensor DataLoaders (X, P, Y)."""
    data_root_folder = inp.data_path
    n_points = inp.n_points

    X_data, P_data, Y_data = assemble_all_data(
        data_root_folder,
        n_points=n_points,
        combined_pickle_file=os.path.join(data_root_folder, "sols.pkl"),
        target_mode=inp.target_mode,
        save_data=True,
        cyc_mode=inp.cyc_mode,
        save_path=data_root_folder,
        return_prot_params=True,
    )

    BATCH_SIZE = min(inp.batch_size, int(Y_data.shape[0] * 0.9))
    train_data_loader, test_data_loader = make_protocol_dataset_from_np(
        batch_size=BATCH_SIZE,
        np_data=X_data,
        np_prot_params=P_data,
        np_data_label=Y_data,
        scale=True,
        scale_y=False,
        save_path=data_root_folder,
    )
    return train_data_loader, test_data_loader


def define_model(inp):
    """Instantiate a ProbProtParamCNN from recipe parameters."""
    import pickle

    input_shape = (2, inp.n_points)
    model = ProbProtParamCNN(
        input_shape=input_shape,
        chan_list=[inp.num_channels] * inp.num_convs,
        fc_list=[inp.num_fc_units] * inp.num_fc_hidden,
        fc_prot_list=[inp.num_fc_prot_units] * inp.num_fc_prot_hidden,
        fc_mu_list=[inp.num_fc_gamma_mu_units] * inp.num_fc_gamma_mu_hidden,
        fc_gamma_list=[inp.num_fc_gamma_mu_units] * inp.num_fc_gamma_mu_hidden,
        loss_fn=independent_normal_loss_param,
        n_prot_params=inp.n_prot_params,
        cyc_mode=inp.cyc_mode,
        n_param_pred=inp.n_param_pred,
        constrain_output=True,
        dependent_outputs=False,
        sim_config=inp.sim_config,
    )
    num_parameters = get_num_parameters(model)
    logger.info(f"No. Trainable Parameters: {num_parameters}")

    with open(inp.scaler_path, "rb") as f:
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

    train_model_param(
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
    inp = ri.basic_input(sys.argv[1])
    train_data_loader, test_data_loader = make_data_loaders(inp)
    model, scaler_X = define_model(inp)
    do_training(inp, model, train_data_loader, test_data_loader, scaler_X)
    shutil.copy(sys.argv[1], os.path.join(inp.models_dir, "recipe.yml"))
