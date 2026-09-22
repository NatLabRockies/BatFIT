"""Train the nochirp Gaussian NPE (ProbParamCNN) on the train split."""

import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import pickle
import shutil
import sys

import numpy as np

from batfit import logger
from batfit.basicutilityc import ReadInput as ri
from batfit.model.param_utils.losses import (
    independent_normal_loss as independent_normal_loss_param,
)
from batfit.model.param_utils.noise_utils import make_noise_levels
from batfit.model.param_utils.train_utils import (
    train_model as train_model_param,
)
from batfit.model.paramNN import ProbParamCNN
from batfit.utils.data_utils import assemble_all_data
from batfit.utils.torch_utils import (
    get_num_parameters,
    make_dataset_from_np,
)


def make_data_loaders(inp):
    """Assemble nochirp data and build the train/test/val (X, Y) loaders."""
    data_root_folder = inp.data_path

    assemble_all_data(
        data_root_folder,
        n_points=inp.n_points,
        combined_pickle_file="sols.pkl",
        target_mode=inp.target_mode,
        save_data=True,
        cyc_mode=inp.cyc_mode,
        save_path=data_root_folder,
    )
    tmp = np.load(os.path.join(data_root_folder, "assembled_data.npz"))
    X_data = tmp["X_data"]
    Y_data = tmp["Y_data"]

    BATCH_SIZE = min(inp.batch_size, int(Y_data.shape[0] * 0.8))
    loaders = make_dataset_from_np(
        batch_size=BATCH_SIZE,
        np_data=X_data,
        np_data_label=Y_data,
        test_split=0.1,
        val_split=0.1,
        scale=True,
        scale_y=False,
        save_path=data_root_folder,
    )
    return loaders


def define_model(inp):
    """Instantiate a ProbParamCNN from recipe parameters."""
    input_shape = (2, inp.n_points)
    model = ProbParamCNN(
        input_shape=input_shape,
        chan_list=[inp.num_channels] * inp.num_convs,
        fc_list=[inp.num_fc_units] * inp.num_fc_hidden,
        fc_mu_list=[inp.num_fc_gamma_mu_units] * inp.num_fc_gamma_mu_hidden,
        fc_gamma_list=[inp.num_fc_gamma_mu_units] * inp.num_fc_gamma_mu_hidden,
        loss_fn=independent_normal_loss_param,
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
    """Fit the model, selecting the best checkpoint on the test split."""
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
        restart_from=getattr(inp, "restart_from", None) or None,
    )


if __name__ == "__main__":
    inp = ri.basic_input(sys.argv[1])
    loaders = make_data_loaders(inp)
    model, scaler_X = define_model(inp)
    do_training(inp, model, loaders["train"], loaders["test"], scaler_X)
    shutil.copy(sys.argv[1], os.path.join(inp.models_dir, "recipe.yml"))
