"""
Postprocessing: parity plot between variance predictor output and NPE sigma
on the test set produced by gen_var_dataset.py.

One subplot per degradation parameter. The x-axis shows the NPE-averaged sigma
stored in var_pred_dataset.npz (ground truth for the variance predictor), and
the y-axis shows the variance predictor's sigma prediction.

Usage:
    python test_var_pred.py training_recipes/recipe_var_pred.yml
"""

import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from batfit import logger
from batfit.basicutilityc import ReadInput as ri
from batfit.model.param_utils.train_utils import create_model_from_log
from batfit.preprocess.sim_setup import make_params
from batfit.utils.torch_utils import get_device_type


def _find_best_model_file(model_dir: str) -> str:
    """Return the checkpoint path with the lowest recorded test loss."""
    vals = np.loadtxt(
        os.path.join(model_dir, "test_loss.csv"), delimiter=";", skiprows=1
    )
    best_ind = int(np.argmin(vals[:, 1]))
    final_path = os.path.join(model_dir, "model_final.pt")
    if best_ind == vals.shape[0] - 1 and os.path.isfile(final_path):
        return final_path
    iterations = np.array(
        [
            int(fname[6 : fname.index(".pt")])
            for fname in os.listdir(model_dir)
            if fname.startswith("model_")
            and fname.endswith(".pt")
            and "final" not in fname
        ]
    )
    if len(iterations) == 0:
        return final_path
    best_iter = vals[best_ind, 0]
    ind = int(np.argmin(np.abs(iterations - best_iter)))
    return os.path.join(model_dir, f"model_{iterations[ind]}.pt")


def parity_plot(inp) -> None:
    """Run inference on the test set and save a sigma parity plot.

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
    best_pt = _find_best_model_file(inp.models_dir)
    logger.info(f"Loading variance predictor from {best_pt}")
    model = create_model_from_log(
        model_obj_file=model_pkl,
        model_state_dict_file=best_pt,
    )
    device = torch.device(get_device_type())
    model.to(device)
    model.eval()
    amp_par = model.amp_par.to(device)

    # Detect whether sigma was MinMax-scaled during dataset generation
    scaler_sigma_path = os.path.join(
        inp.var_pred_save_path, "scaler_sigma.pkl"
    )
    scale_sigma = os.path.isfile(scaler_sigma_path)
    if scale_sigma:
        import pickle

        with open(scaler_sigma_path, "rb") as f:
            scaler_sigma = pickle.load(f)
        logger.info(
            "scaler_sigma.pkl found — will inverse-transform to physical sigma"
        )

    p_test = torch.from_numpy(A["P_test"])
    mu_test = torch.from_numpy(A["Mu_test"])

    with torch.no_grad():
        sigma_out = model(p_test.to(device), mu_test.to(device)).cpu().numpy()

    # Convert both predicted and stored sigma to physical space for the parity plot
    if scale_sigma:
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

    # Parity plot: one subplot per degradation parameter
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
        ax.set_xlabel("NPE sigma (ground truth)")
        ax.set_ylabel("Predicted sigma")
        ax.set_title(name)
        ax.legend(fontsize=8)

    # Hide unused subplots
    for j in range(n_deg, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Variance predictor parity plot — test set", fontsize=13)
    fig.tight_layout()

    out_file = os.path.join(inp.models_dir, "parity_plot.png")
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    logger.info(f"Parity plot saved to {out_file}")

    # Log per-parameter RMSE for quick diagnostics
    rmse = np.sqrt(np.mean((sigma_pred - sigma_true) ** 2, axis=0))
    for name, r in zip(param_names, rmse):
        logger.info(f"  RMSE sigma [{name}]: {r:.4e}")


if __name__ == "__main__":
    inp = ri.basic_input(sys.argv[1])
    parity_plot(inp)
