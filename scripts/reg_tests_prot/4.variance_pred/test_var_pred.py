"""
Postprocessing: parity plots between variance predictor output and NPE sigma
on the test set produced by gen_var_dataset.py.

One subplot per degradation parameter. The x-axis shows the NPE-averaged sigma
stored in var_pred_dataset.npz (ground truth for the variance predictor), and
the y-axis shows the variance predictor's sigma prediction. Two figures are
saved: parity_plot.png (linear axes) and parity_plot_log.png (log-log axes,
i.e. parity on log sigma — where small-sigma accuracy is visible).

Per-parameter errors in physical sigma space (RMSE and median relative error)
are written to test_error_sigma.txt in models_dir.

All three sigma target parameterisations are supported (see
train_var_pred.detect_sigma_mode): amp_par, scale_sigma, log_sigma.

Usage:
    python test_var_pred.py training_recipes/recipe_var_pred.yml
"""

import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import pickle
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from train_var_pred import detect_sigma_mode

from batfit import logger
from batfit.basicutilityc import ReadInput as ri
from batfit.model.param_utils.train_utils import create_model_from_log
from batfit.preprocess.sim_setup import make_params
from batfit.utils.torch_utils import find_best_model_file, get_device_type


def _parity_figure(
    sigma_true: np.ndarray,
    sigma_pred: np.ndarray,
    param_names: list[str],
    log_axes: bool,
    out_file: str,
) -> None:
    """Save a per-parameter parity plot of predicted vs NPE sigma.

    :param sigma_true: NPE sigma (ground truth), shape (N, n_deg)
    :param sigma_pred: variance predictor sigma, shape (N, n_deg)
    :param param_names: degradation parameter names
    :param log_axes: if True, use log-log axes (parity on log sigma)
    :param out_file: path of the saved figure
    """
    n_deg = len(param_names)
    ncols = 3
    nrows = int(np.ceil(n_deg / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, name in enumerate(param_names):
        ax = axes[i]
        ax.scatter(
            sigma_true[:, i],
            sigma_pred[:, i],
            s=8,
            alpha=0.4,
            rasterized=True,
        )
        # Diagonal reference line
        lims = [
            min(sigma_true[:, i].min(), sigma_pred[:, i].min()),
            max(sigma_true[:, i].max(), sigma_pred[:, i].max()),
        ]
        ax.plot(lims, lims, "k--", linewidth=1.0, label="perfect")
        if log_axes:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_xlabel("NPE sigma (ground truth)")
        ax.set_ylabel("Predicted sigma")
        ax.set_title(name)
        ax.legend(fontsize=8)

    # Hide unused subplots
    for j in range(n_deg, len(axes)):
        axes[j].set_visible(False)

    scale_lab = "log axes" if log_axes else "linear axes"
    fig.suptitle(
        f"Variance predictor parity plot ({scale_lab}) — test set",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    logger.info(f"Parity plot saved to {out_file}")


def parity_plot(inp) -> None:
    """Run inference on the test set, save parity plots and a sigma error table.

    :param inp: recipe object from recipe_var_pred.yml
    """
    dataset_file = os.path.join(inp.var_pred_save_path, "var_pred_dataset.npz")
    assert os.path.isfile(dataset_file), (
        f"var_pred_dataset.npz not found at {dataset_file}; "
        "run gen_var_dataset.py first"
    )
    A = np.load(dataset_file)
    assert (
        "P_test" in A and "Mu_test" in A and "Sigma_test" in A
    ), "Test keys missing from var_pred_dataset.npz"

    # Load model
    model_pkl = os.path.join(inp.models_dir, "model.pkl")
    best_pt = find_best_model_file(inp.models_dir)
    logger.info(f"Loading variance predictor from {best_pt}")
    model = create_model_from_log(
        model_obj_file=model_pkl,
        model_state_dict_file=best_pt,
    )
    device = torch.device(get_device_type())
    model.to(device)
    model.eval()
    amp_par = model.amp_par.to(device)

    # Detect how sigma targets were parameterised at dataset generation
    sigma_mode = detect_sigma_mode(inp.var_pred_save_path)
    logger.info(f"sigma_mode={sigma_mode}")

    p_test = torch.from_numpy(A["P_test"])
    mu_test = torch.from_numpy(A["Mu_test"])

    with torch.no_grad():
        sigma_out = model(p_test.to(device), mu_test.to(device)).cpu().numpy()

    # Convert both predicted and stored sigma to physical space
    if sigma_mode == "log_sigma":
        with open(
            os.path.join(inp.var_pred_save_path, "scaler_logsigma.pkl"), "rb"
        ) as f:
            scaler_logsigma = pickle.load(f)
        sigma_pred = np.exp(scaler_logsigma.inverse_transform(sigma_out))
        sigma_true = np.exp(
            scaler_logsigma.inverse_transform(A["Sigma_test"])
        )
    elif sigma_mode == "scale_sigma":
        with open(
            os.path.join(inp.var_pred_save_path, "scaler_sigma.pkl"), "rb"
        ) as f:
            scaler_sigma = pickle.load(f)
        sigma_pred = scaler_sigma.inverse_transform(sigma_out)
        sigma_true = scaler_sigma.inverse_transform(A["Sigma_test"])
    else:
        sigma_pred = model.inv_transform_gamma(
            torch.from_numpy(sigma_out), model.amp_par
        ).numpy()
        sigma_true = A["Sigma_test"]

    # Parameter names from sim config
    sim_params = make_params(inp.sim_config)
    param_names = sim_params["deg_param_names"]
    n_deg = len(param_names)

    # Parity plots: linear axes and log-log axes (parity on log sigma)
    _parity_figure(
        sigma_true,
        sigma_pred,
        param_names,
        log_axes=False,
        out_file=os.path.join(inp.models_dir, "parity_plot.png"),
    )
    _parity_figure(
        sigma_true,
        sigma_pred,
        param_names,
        log_axes=True,
        out_file=os.path.join(inp.models_dir, "parity_plot_log.png"),
    )

    # Per-parameter error in physical sigma space, logged and saved to file
    err = sigma_pred - sigma_true
    rmse = np.sqrt(np.mean(err**2, axis=0))
    med_rel = np.median(np.abs(err) / sigma_true, axis=0) * 100
    err_file = os.path.join(inp.models_dir, "test_error_sigma.txt")
    with open(err_file, "w") as f:
        f.write(
            f"{'param':16s} {'rmse_sigma':>14s} {'median_rel_err_%':>18s}\n"
        )
        for name, r, m in zip(param_names, rmse, med_rel):
            f.write(f"{name:16s} {r:14.6e} {m:18.2f}\n")
            logger.info(
                f"  RMSE sigma [{name}]: {r:.4e} | median rel err: {m:.2f}%"
            )
    logger.info(f"Sigma error table saved to {err_file}")


if __name__ == "__main__":
    inp = ri.basic_input(sys.argv[1])
    parity_plot(inp)
