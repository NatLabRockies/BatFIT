import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

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
    """Save a per-parameter parity plot of predicted vs NPE sigma."""
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
        f"Variance predictor parity plot ({scale_lab}) — val set",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    logger.info(f"Parity plot saved to {out_file}")


def parity_plot(inp) -> None:
    """Run inference on the val set, save parity plots and a sigma error table."""
    dataset_file = os.path.join(inp.var_pred_save_path, "var_pred_dataset.npz")
    assert os.path.isfile(dataset_file), (
        f"var_pred_dataset.npz not found at {dataset_file}; "
        "run gen_var_dataset.py first"
    )
    A = np.load(dataset_file)
    assert (
        "P_val" in A and "Mu_val" in A and "Sigma_val" in A
    ), "Val keys missing from var_pred_dataset.npz"

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

    # physical inputs -> physical sigma (the model holds its scalers)
    p_val = torch.from_numpy(A["P_val"]).to(device)
    mu_val = torch.from_numpy(A["Mu_val"]).to(device)
    with torch.no_grad():
        sigma_pred = model.predict_physical(p_val, mu_val).cpu().numpy()
    sigma_true = A["Sigma_val"]

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
    err_file = os.path.join(inp.models_dir, "val_error_sigma.txt")
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
