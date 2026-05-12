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
    data_root_folder_val = inp.data_val_path
    n_points = inp.n_points
    n_param_pred = inp.n_param_pred
    cyc_mode = inp.cyc_mode

    # This will just load a numpy file if preproc was called before
    X_data, Y_data = assemble_surrogate_data(
        data_root_folder,
        n_points=n_points,
        n_param_pred=n_param_pred,
        combined_pickle_file=os.path.join(data_root_folder, "sols.pkl"),
        cyc_mode=cyc_mode,
        save_data=True,
        save_path=data_root_folder,
    )
    tmp = np.load(
        os.path.join(data_root_folder, "assembled_surrogate_data.npz")
    )
    BATCH_SIZE = min(inp.batch_size, int(Y_data.shape[0] * 0.9))
    train_data_loader, test_data_loader = make_surrogate_dataset_from_np(
        batch_size=BATCH_SIZE,
        np_data=X_data,
        np_data_label=Y_data,
        scale=True,
        scale_y=False,
        save_path=data_root_folder,
    )

    return train_data_loader, test_data_loader


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
    )


if __name__ == "__main__":
    import shutil
    import sys

    inp = ri.basic_input(sys.argv[1])
    train_data_loader, test_data_loader = make_data_loaders(inp)
    model, scaler_X = define_model(inp)
    do_training(inp, model, train_data_loader, test_data_loader)
    shutil.copy(sys.argv[1], os.path.join(inp.models_dir, "recipe.yml"))
