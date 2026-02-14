import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable MPS fallback
import pickle

import numpy as np
import torch
from prettyPlot.plotting import *
from torchsummary import summary

from batfit import BATFIT_DIR, BATFIT_EXP, BATFIT_REG, logger
from batfit.basicutilityc import ReadInput as ri
from batfit.model.surrogateNN import *
from batfit.utils.data_utils import *
from batfit.utils.torch_utils import *


def pre_inp(inp):
    if not os.path.isabs(inp.data_path):
        inp.data_path = os.path.join(BATFIT_REG, inp.data_path)
        logger.warning(
            f"Data path not absolute and replaced with {inp.data_path}"
        )
    if not os.path.isabs(inp.scaler_path):
        inp.scaler_path = os.path.join(BATFIT_REG, inp.data_path)
        logger.warning(
            f"Scaler path not absolute and replaced with {inp.scaler_path}"
        )
    if not os.path.isabs(inp.sim_config):
        inp.sim_config = os.path.join(BATFIT_EXP, inp.sim_config)
        logger.warning(
            f"Sim config path not absolute and replaced with {inp.sim_config}"
        )
    if not os.path.isabs(inp.models_dir):
        inp.models_dir = os.path.join(
            BATFIT_REG, "2.train_surrogate", inp.models_dir
        )
        logger.warning(
            f"Models dir path not absolute and replaced with {inp.models_dir}"
        )
    return inp


def make_data_loaders(inp):
    data_root_folder = inp.data_path
    n_points = inp.n_points
    n_param_pred = inp.n_param_pred
    cyc_mode = inp.cyc_mode

    # This is optional, and is useful to look at the voltage curve prediction
    X_npe_data, Y_npe_data = assemble_all_data(
        data_root_folder,
        n_points=n_points,
        combined_pickle_file=os.path.join(data_root_folder, "sols.pkl"),
        target_mode="phi",
        save_data=True,
        cyc_mode=cyc_mode,
        save_path=data_root_folder,
    )
    tmp = np.load(os.path.join(data_root_folder, "assembled_data.npz"))
    BATCH_SIZE = min(inp.batch_size, int(Y_npe_data.shape[0] * 0.9))
    _, _ = make_dataset_from_np(
        batch_size=BATCH_SIZE,
        np_data=X_npe_data,
        np_data_label=Y_npe_data,
        scale=True,
        scale_y=False,
        save_path=data_root_folder,
    )

    # This is non optional
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


def do_training(inp, model, train_data_loader, test_data_loader):

    # summary(model.cpu(), (model.n_param_pred + 1,))
    model, loss_hist = train_model(
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
    inp = pre_inp(inp)
    train_data_loader, test_data_loader = make_data_loaders(inp)
    model, scaler_X = define_model(inp)
    do_training(inp, model, train_data_loader, test_data_loader)
    shutil.copy(sys.argv[1], os.path.join(inp.models_dir, "recipe.yml"))
