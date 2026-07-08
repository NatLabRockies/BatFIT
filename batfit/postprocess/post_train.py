import os
import pickle
from pathlib import Path

import numpy as np
import torch
from prettyPlot.plotting import *

from batfit.model.paramNN import ProbParamCNN, ProbParamFCNN
from batfit.utils.data_utils import (
    scale_dataset_from_scaler,
    scale_input_from_scaler,
    scale_output_from_scaler,
    unscale_dataset_from_scaler,
    unscale_input_from_scaler,
    unscale_output_from_scaler,
    unscale_pred_from_scaler,
    unscale_pred_std_from_scaler,
)
from batfit.utils.torch_utils import (
    get_num_parameters,
    load_model,
    make_dataset_from_np,
)


def plot_loss(loss_hist_file, figure_folder="Figures", fig_name="loss.png"):
    loss_data = np.genfromtxt(loss_hist_file, delimiter=";", skip_header=1)
    fig = plt.figure()
    plt.plot(loss_data[:, 0], loss_data[:, 1], color="k")
    pretty_labels("# Step", "Loss", fontsize=16, fontname="Times")
    # os.makedirs(figure_folder, exist_ok=True)
    log_dir = Path(figure_folder)
    log_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(figure_folder, fig_name))
    plt.close()


def check_on_test(
    model,
    scaled_data_file,
    scaler_X_file,
    scaler_Y_file,
    scale_y,
    figure_folder="Figures",
    fig_name="test_check.png",
):
    model.eval()
    model.to("cpu")

    X_scaled = np.load(scaled_data_file)["X_test"]
    Y_scaled = np.load(scaled_data_file)["Y_test"]

    num_test = min(2, X_scaled.shape[0])

    if isinstance(model, ProbParamCNN):
        pred_scaled, gamma_scaled = model(
            torch.from_numpy(X_scaled[:num_test])
        )
        pred_scaled = pred_scaled.detach().numpy()
        gamma_scaled = gamma_scaled.detach().numpy()
        gamma_unscaled = unscale_pred_std_from_scaler(
            gamma_scaled, scaler_Y_file
        )
        inp_unscaled, pred_unscaled = unscale_dataset_from_scaler(
            X_scaled[:num_test], pred_scaled, scaler_X_file, scaler_Y_file
        )
        truth_scaled = Y_scaled[:num_test]
        truth_unscaled = unscale_pred_from_scaler(truth_scaled, scaler_Y_file)
        probabilistic = True

    else:
        raise NotImplementedError

    if not scale_y:
        pred_unscaled = pred_scaled
        if isinstance(model, ProbParamCNN):
            gamma_unscaled = gamma_scaled

    fig, axs = plt.subplots(1, num_test, figsize=(8 * num_test, 4))
    # plt.subplots returns a bare Axes (not an array) when num_test == 1
    axs = np.atleast_1d(axs)
    for i_test in range(num_test):
        axs[i_test].plot(
            inp_unscaled[i_test, 0, :], inp_unscaled[i_test, 1, :]
        )
        list_pred = [
            f"{pred_unscaled[i_test,i]:.2f}"
            for i in range(pred_unscaled.shape[1])
        ]
        list_truth = [
            f"{truth_unscaled[i_test,i]:.2f}"
            for i in range(pred_unscaled.shape[1])
        ]
        if probabilistic:
            list_unc = [
                f"{gamma_unscaled[i_test,i]:.2f}"
                for i in range(pred_unscaled.shape[1])
            ]
            title = (
                f"Pred = {list_pred}\nUnc = {list_unc}\nTrue = {list_truth}"
            )
        else:
            title = f"Pred = {list_pred}\nTrue = {list_truth}"
        pretty_labels(
            "",
            "",
            16,
            ax=axs[i_test],
            title=title,
            grid=False,
        )

    log_dir = Path(figure_folder)
    log_dir.mkdir(parents=True, exist_ok=True)
    # os.makedirs(figure_folder, exist_ok=True)
    plt.savefig(os.path.join(figure_folder, fig_name))
    plt.close()
