"""
Plot conditional-averaged results of run_optimization_clean.py.

For every SOH (degradation) parameter targeted by the chirp optimization,
two figures are produced, each with one subplot per conditioning SOH
parameter (ground-truth values, equal-width bins):

  - var_reduction_<param>.png — conditionally averaged variance reduction
    in percent, against both baselines (nochirp NPE and amplitude-0
    variance estimator).
  - opt_chirp_<param>.png — conditionally averaged optimal chirp
    parameters, MinMax-scaled to [0, 1] with their physical range in the
    legend so the three magnitudes share one axis.

Usage:
    python plot_optimization_clean.py training_recipes/recipe_clean.yml
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from batfit import logger
from batfit.basicutilityc import ReadInput as ri
from batfit.preprocess.sim_setup import make_params


def conditional_average(
    x: np.ndarray, y: np.ndarray, nbins: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a 1D conditional average of y with respect to x
    The conditional average is distributed to neighbors of the binned array when needed

    Adapted from BioReactorDesign (bird/utilities/mathtools.py), with the
    distribution weights measured in x (sample position within its bin)
    rather than y.

    Parameters
    ----------
    x: np.ndarray
        1D array with respect to which conditional averaged is performed
    y : np.ndarray
        1D array conditioned
    nbins: int
        Number of bins through x

    Returns
    ----------
    x_cond: np.ndarray
        The binned array of values conditioned againsts
    y_cond: np.ndarray
        The conditional averages at each bin
    """
    # Check the shape of input arrays
    try:
        assert len(x.shape) <= 2
        assert len(y.shape) <= 2
        if len(x.shape) == 2:
            assert x.shape[1] == 1
        if len(y.shape) == 2:
            assert y.shape[1] == 1
    except AssertionError:
        error_msg = "conditional average of tensors is ambiguous"
        error_msg += f"\nx shape =  {x.shape}"
        error_msg += f"\ny shape =  {y.shape}"
        logger.error(error_msg)
        raise AssertionError(error_msg)
    if len(x.shape) == 2:
        x = x[:, 0]
    if len(y.shape) == 2:
        y = y[:, 0]
    try:
        assert len(x) == len(y)
    except AssertionError:
        error_msg = "conditional average x and y have different dimension"
        error_msg += f"\ndim x =  {len(x)}"
        error_msg += f"\ndim y =  {len(y)}"
        logger.error(error_msg)
        raise AssertionError(error_msg)

    # Bin conditional space
    mag = np.amax(x) - np.amin(x)
    x_bin = np.linspace(
        np.amin(x) - mag / (2 * nbins), np.amax(x) + mag / (2 * nbins), nbins
    )
    weight = np.zeros(nbins)
    weightVal = np.zeros(nbins)
    asum = np.zeros(nbins)
    bsum = np.zeros(nbins)
    avalsum = np.zeros(nbins)
    bvalsum = np.zeros(nbins)
    inds = np.digitize(x, x_bin)

    a = abs(x - x_bin[inds - 1])
    b = abs(x - x_bin[inds])
    c = a + b
    a = a / c
    b = b / c

    # Conditional average at each bin
    for i in range(nbins):
        asum[i] = np.sum(a[np.argwhere(inds == i)])
        bsum[i] = np.sum(b[np.argwhere(inds == i + 1)])
        avalsum[i] = np.sum(
            a[np.argwhere(inds == i)] * y[np.argwhere(inds == i)]
        )
        bvalsum[i] = np.sum(
            b[np.argwhere(inds == i + 1)] * y[np.argwhere(inds == i + 1)]
        )
    weight = asum + bsum
    weightVal = avalsum + bvalsum

    # Assemble output
    x_cond = x_bin
    y_cond = weightVal / (weight)

    return x_cond, y_cond


def _make_grid(n_panels: int) -> tuple:
    """Return (fig, flat axes) for a 3-column subplot grid."""
    ncols = 3
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()
    for j in range(n_panels, len(axes)):
        axes[j].set_visible(False)
    return fig, axes


def plot_optimization_clean(inp) -> None:
    """Make the conditional-average plots from the saved npz results.

    :param inp: recipe object from recipe_clean.yml
    """
    results_file = os.path.join(
        inp.save_path, "optimization_clean_results.npz"
    )
    d = np.load(results_file)
    Y_true = d["Y_true"]  # (n_curves, n_deg)
    sigma_nochirp = d["sigma_nochirp"]  # (n_curves, n_deg)
    sigma_opt = d["sigma_opt"]  # (n_deg, n_curves, n_deg)
    sigma_amp0 = d["sigma_amp0"]  # (n_deg, n_curves, n_deg)
    P_opt = d["P_opt"]  # (n_deg, n_curves, n_prot), physical
    param_names = [str(n) for n in d["param_names"]]
    prot_names = [str(n) for n in d["prot_names"]]
    n_deg = len(param_names)
    n_bins = int(getattr(inp, "n_bins", 10))

    # Physical protocol bounds, used to normalise the chirp params to [0, 1]
    sim_params = make_params(inp.sim_config)
    prot_min = np.array(
        [sim_params[f"prot_{name}_min"] for name in prot_names]
    )
    prot_max = np.array(
        [sim_params[f"prot_{name}_max"] for name in prot_names]
    )

    figure_folder = os.path.join(inp.save_path, "figures")
    os.makedirs(figure_folder, exist_ok=True)

    for k, target in enumerate(param_names):
        # Per-curve variance reduction (%) for the target parameter
        red_npe = (
            (sigma_nochirp[:, k] - sigma_opt[k, :, k])
            / sigma_nochirp[:, k]
            * 100
        )
        red_amp0 = (
            (sigma_amp0[k, :, k] - sigma_opt[k, :, k])
            / sigma_amp0[k, :, k]
            * 100
        )

        # --- Figure 1: variance reduction vs each SOH parameter ---
        fig, axes = _make_grid(n_deg)
        for j, cond in enumerate(param_names):
            ax = axes[j]
            for red, label in [
                (red_npe, "vs nochirp NPE"),
                (red_amp0, "vs amplitude-0 estimator"),
            ]:
                centers, means = conditional_average(Y_true[:, j], red, n_bins)
                ax.plot(centers, means, "o-", label=label)
            ax.axhline(0.0, color="k", linewidth=0.8, linestyle=":")
            ax.set_xlabel(cond)
            ax.set_ylabel("sigma reduction [%]")
            ax.legend(fontsize=8)
        fig.suptitle(
            f"Chirp-induced variance reduction for '{target}'", fontsize=13
        )
        fig.tight_layout()
        out_file = os.path.join(figure_folder, f"var_reduction_{target}.png")
        fig.savefig(out_file, dpi=150)
        plt.close(fig)
        logger.info(f"Saved {out_file}")

        # --- Figure 2: optimal chirp parameters vs each SOH parameter ---
        P_scaled = (P_opt[k] - prot_min) / (prot_max - prot_min)
        fig, axes = _make_grid(n_deg)
        for j, cond in enumerate(param_names):
            ax = axes[j]
            for p, pname in enumerate(prot_names):
                centers, means = conditional_average(
                    Y_true[:, j], P_scaled[:, p], n_bins
                )
                ax.plot(
                    centers,
                    means,
                    "o-",
                    label=f"{pname} [{prot_min[p]:g}, {prot_max[p]:g}]",
                )
            ax.set_xlabel(cond)
            ax.set_ylabel("optimal chirp param (scaled)")
            ax.set_ylim(-0.05, 1.05)
            ax.legend(fontsize=8)
        fig.suptitle(f"Optimal chirp when targeting '{target}'", fontsize=13)
        fig.tight_layout()
        out_file = os.path.join(figure_folder, f"opt_chirp_{target}.png")
        fig.savefig(out_file, dpi=150)
        plt.close(fig)
        logger.info(f"Saved {out_file}")


if __name__ == "__main__":
    inp = ri.basic_input(sys.argv[1])
    plot_optimization_clean(inp)
