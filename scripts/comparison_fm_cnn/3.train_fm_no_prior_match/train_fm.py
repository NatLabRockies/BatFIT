"""Train ProbParamFM (conditional flow matching) on SPM discharge data."""

import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import pickle
import shutil
import sys

import numpy as np

from batfit import BATFIT_EXP, logger
from batfit.basicutilityc import ReadInput as ri
from batfit.model.param_utils.noise_utils import make_noise_levels
from batfit.model.param_utils.train_fm_utils import train_fm_model
from batfit.model.paramNN import ProbParamFM
from batfit.utils.data_utils import assemble_all_data
from batfit.utils.torch_utils import get_num_parameters, make_dataset_from_np


def make_data_loaders(
    inp,
) -> tuple["torch.utils.data.DataLoader", "torch.utils.data.DataLoader"]:
    """Assemble signals and build train/test DataLoaders.

    :param inp: parsed recipe (DotMap)
    :return: (train_data_loader, test_data_loader)
    """
    assemble_all_data(
        inp.data_path,
        n_points=inp.n_points,
        combined_pickle_file=os.path.join(inp.data_path, "sols.pkl"),
        target_mode=inp.target_mode,
        save_data=True,
        cyc_mode=inp.cyc_mode,
        save_path=inp.data_path,
    )
    tmp = np.load(os.path.join(inp.data_path, "assembled_data.npz"))
    X_data = tmp["X_data"]
    Y_data = tmp["Y_data"]

    batch_size = min(inp.batch_size, int(Y_data.shape[0] * 0.9))
    return make_dataset_from_np(
        batch_size=batch_size,
        np_data=X_data,
        np_data_label=Y_data,
        scale=True,
        scale_y=True,
        save_path=inp.data_path,
    )


def define_model(inp) -> tuple["ProbParamFM", object]:
    """Instantiate ProbParamFM and load the signal scaler.

    :param inp: parsed recipe
    :return: (model, scaler_X)
    """
    sim_config = os.path.join(BATFIT_EXP, inp.sim_config)
    model = ProbParamFM(
        input_shape=(2, inp.n_points),
        chan_list=[inp.num_channels] * inp.num_convs,
        fc_list=[inp.num_fc_units] * inp.num_fc_hidden,
        vf_hidden_list=[inp.num_vf_units] * inp.num_vf_hidden,
        cyc_mode=inp.cyc_mode,
        n_param_pred=inp.n_param_pred,
        sim_config=sim_config,
        use_prior_matching=inp.use_prior_matching,
    )
    logger.info(f"Trainable parameters: {get_num_parameters(model)}")
    with open(os.path.join(inp.data_path, "scaler_X.pkl"), "rb") as f:
        scaler_X = pickle.load(f)
    return model, scaler_X


def do_training(inp, model, train_data_loader, test_data_loader, scaler_X):
    """Run the FM training loop.

    :param inp: parsed recipe
    :param model: ProbParamFM
    :param train_data_loader: training DataLoader
    :param test_data_loader: test DataLoader
    :param scaler_X: signal normalisation scaler
    """
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
    train_fm_model(
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
    train_dl, test_dl = make_data_loaders(inp)
    model, scaler_X = define_model(inp)
    do_training(inp, model, train_dl, test_dl, scaler_X)
    shutil.copy(sys.argv[1], os.path.join(inp.models_dir, "recipe.yml"))
