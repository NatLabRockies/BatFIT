"""Global sensitivity of the NPE sigma to (mu, protocol)."""

import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from SALib.analyze import delta

from batfit import logger
from batfit.basicutilityc import ReadInput as ri
from batfit.preprocess.sim_setup import make_params


def compute_sensitivity(inp) -> None:
    """Delta sensitivity of each parameter's sigma to (mu, protocol) inputs.
    Parameters
    ----------
    inp : object
        Parsed recipe (recipe_var_pred_*.yml): needs ``var_pred_save_path``,
        ``sim_config`` and ``models_dir``.
    """
    dataset_file = os.path.join(inp.var_pred_save_path, "var_pred_dataset.npz")
    assert os.path.isfile(dataset_file), (
        f"var_pred_dataset.npz not found at {dataset_file}; "
        "run gen_var_dataset.py first"
    )
    A = np.load(dataset_file)
    P = A["P_train"]
    Mu = A["Mu_train"]
    Sigma = A["Sigma_train"]

    sim_params = make_params(inp.sim_config)
    param_names = sim_params["deg_param_names"]
    prot_names = sim_params["prot_param_names"]
    n_deg = len(param_names)

    X = np.hstack((Mu, P))
    names = [f"mu[{n}]" for n in param_names] + [f"P[{n}]" for n in prot_names]
    num_vars = len(names)
    bounds = [[X[:, i].min(), X[:, i].max()] for i in range(num_vars)]
    problem = {"num_vars": num_vars, "names": names, "bounds": bounds}

    ncols = 3
    nrows = int(np.ceil(n_deg / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()
    for i in range(n_deg):
        Si = delta.analyze(problem, X, Sigma[:, i], num_resamples=10)
        # SALib renamed the Borgonovo delta keys across versions: older
        # releases return "delta"/"delta_conf", newer ones return the
        # bias-corrected "delta_balanced"/"delta_balanced_conf".
        if "delta" in Si:
            delta_idx, delta_conf = Si["delta"], Si["delta_conf"]
        else:
            delta_idx = Si["delta_balanced"]
            delta_conf = Si["delta_balanced_conf"]
        ax = axes[i]
        ax.bar(
            names,
            delta_idx,
            yerr=delta_conf,
            capsize=4,
            color="#4C72B0",
            edgecolor="black",
            alpha=0.85,
        )
        ax.set_title(f"sigma {param_names[i]}")
        ax.set_ylabel("delta index")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
    for j in range(n_deg, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Variance-estimator input sensitivity (delta index)")
    fig.tight_layout()
    out_file = os.path.join(inp.models_dir, "sensitivity.png")
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {out_file}")


if __name__ == "__main__":
    inp = ri.basic_input(sys.argv[1])
    compute_sensitivity(inp)
