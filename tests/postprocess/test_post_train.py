import os
import tempfile

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

from batfit.model.param_utils.losses import independent_normal_loss
from batfit.model.paramNN import ProbParamCNN
from batfit.postprocess.post_train import check_on_test, plot_loss


def test_plot_loss():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Write a dummy train loss
        loss_file = os.path.join(tmp_dir, "train_loss.csv")
        with open(loss_file, "w") as f:
            f.write("step;loss\n")
            for i in range(10):
                f.write(f"{i};{1.0 / (i + 1):.6f}\n")

        figure_folder = os.path.join(tmp_dir, "figures")
        plot_loss(loss_file, figure_folder=figure_folder, fig_name="loss.png")

        assert os.path.exists(os.path.join(figure_folder, "loss.png"))


def test_check_on_test():
    n_channels, n_points, n_param_pred = 2, 64, 3
    model = ProbParamCNN(
        input_shape=(n_channels, n_points),
        chan_list=[8],
        fc_list=[16],
        fc_mu_list=[8],
        fc_gamma_list=[8],
        loss_fn=independent_normal_loss,
        cyc_mode="discharge",
        n_param_pred=n_param_pred,
        constrain_output=False,
    )

    X_test = np.random.randn(2, n_channels, n_points).astype("float32")
    Y_test = np.random.randn(2, n_param_pred).astype("float32")

    with tempfile.TemporaryDirectory() as tmp_dir:
        data_file = os.path.join(tmp_dir, "data_split.npz")
        np.savez(data_file, X_test=X_test, Y_test=Y_test)

        figure_folder = os.path.join(tmp_dir, "figures")
        check_on_test(
            model=model,
            scaled_data_file=data_file,
            scaler_X_file=None,
            scaler_Y_file=None,
            scale_y=False,
            figure_folder=figure_folder,
            fig_name="test_check.png",
        )

        assert os.path.exists(os.path.join(figure_folder, "test_check.png"))
