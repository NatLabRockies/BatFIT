import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable MPS fallback
import pickle
from pathlib import Path

import numpy as np
import torch
from prettyPlot.plotting import *
from torchsummary import summary

from batfit import BATFIT_DIR, BATFIT_EXP, BATFIT_REG, logger
from batfit.basicutilityc import ReadInput as ri
from batfit.model.surrogate_utils.losses import mae_loss as mae_loss_surr
from batfit.model.surrogate_utils.train_utils import (
    train_model as train_model_surr,
)
from batfit.model.surrogateNN import SurrogateFCNN
from batfit.utils.data_utils import *
from batfit.utils.torch_utils import *


def make_data_loaders(inp):
    data_root_folder = inp.data_path
    n_points = inp.n_points
    cyc_mode = inp.cyc_mode

    # Whole-curve data; the surrogate builder splits batteries then explodes.
    # This just loads a numpy file if preproc was called before.
    X_data, Y_data = assemble_all_data(
        data_root_folder,
        n_points=n_points,
        combined_pickle_file="sols.pkl",
        target_mode="phi",
        save_data=True,
        cyc_mode=cyc_mode,
        save_path=data_root_folder,
    )
    loaders = make_surrogate_dataset_from_np(
        batch_size=inp.batch_size,
        np_data=X_data,
        np_data_label=Y_data,
        scale=True,
        scale_y=False,
        save_path=data_root_folder,
    )

    return loaders


def define_model(inp):
    data_root_folder = inp.data_path
    n_points = inp.n_points
    n_param_pred = inp.n_param_pred
    cyc_mode = inp.cyc_mode

    model = SurrogateFCNN(
        fc_list=inp.fc_units,
        loss_fn=mae_loss_surr,
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


def do_training(inp, model, train_data_loader, test_data_loader):

    # summary(model.cpu(), (model.n_param_pred + 1,))
    model, loss_hist = train_model_surr(
        model,
        train_data_loader=train_data_loader,
        test_data_loader=test_data_loader,
        learning_rate=inp.lr,
        num_epochs=inp.epochs,
        enable_cuda=True,
        enable_mps=True,
        log_folder=inp.models_dir,
        restart_from=getattr(inp, "restart_from", None) or None,
    )


if __name__ == "__main__":
    import shutil
    import sys

    inp = ri.basic_input(sys.argv[1])
    loaders = make_data_loaders(inp)
    model, scaler_X = define_model(inp)
    do_training(inp, model, loaders["train"], loaders["test"])
    shutil.copy(sys.argv[1], os.path.join(inp.models_dir, "recipe.yml"))
