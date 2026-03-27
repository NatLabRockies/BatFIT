import os

import corner
import numpy as np
import yaml
from prettyPlot.plotting import *

from batfit import BATFIT_EXP, logger


def generate_corner_plot(
    bad_par_txt: str,
    sim_config_yaml: str,
    plot_params: list[int] | list[str] | None = None,
    fontsize: int = 12,
    figname: str | None = None,
):
    """
    Reads parameter values and a YAML configuration to generate a corner plot.

    Args:
        bad_par_txt (str): Path to the text file with parameter values.
        sim_config_yaml (str): Path to the YAML file with parameter configurations.
        plot_params (list, optional): List of parameter indices or names to display. Plots all if None.
        fontsize (int, optional): Adjustable font size for the plot.
    """

    # 1. Configure Matplotlib Font Settings
    # Use Times/Times New Roman and set everything to bold
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times", "Times New Roman"],
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "font.size": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
        }
    )

    # 2. Load the Data
    # Assuming space-delimited text data
    samples = np.loadtxt(bad_par_txt)

    # 3. Load and Parse the YAML File
    with open(sim_config_yaml, "r") as file:
        config = yaml.safe_load(file)

    # Parse parameter names (handle the comma-separated string)
    raw_names = config.get("degradation parameter names", "")
    if isinstance(raw_names, str):
        param_names = [name.strip() for name in raw_names.split(",")]
    else:
        param_names = raw_names

    min_params = config.get("min degradation parameter", {})
    max_params = config.get("max degradation parameter", {})

    # Extract domains (ranges) matching the order of the parameters
    param_ranges = []
    for name in param_names:
        p_min = float(min_params.get(name, 0.0))
        p_max = float(max_params.get(name, 1.0))
        param_ranges.append((p_min, p_max))

    # 4. Filter by Indices or Names (If Provided)
    if plot_params is not None:
        indices = []
        for p in plot_params:
            if isinstance(p, int):
                indices.append(p)
            elif isinstance(p, str):
                if p in param_names:
                    indices.append(param_names.index(p))
                else:
                    raise ValueError(
                        f"Parameter '{p}' not found in the YAML configuration."
                    )
            else:
                raise TypeError(
                    "Elements of plot_params must be either integers or strings."
                )

        # Ensure indices are valid
        samples = samples[:, indices]
        param_names = [param_names[i] for i in indices]
        param_ranges = [param_ranges[i] for i in indices]

    # 5. Generate the Corner Plot
    # `range=param_ranges` sets the bounds of the 1D and 2D histograms to the YAML edges
    fig = corner.corner(
        samples,
        labels=param_names,
        range=param_ranges,
        show_titles=False,
        plot_datapoints=True,
    )

    # 6. Explicitly enforce edge values on the axes
    # To guarantee the "edge of the domain" is explicitly shown as ticks
    ndim = len(param_names)
    axes = np.array(fig.axes).reshape((ndim, ndim))

    for i in range(ndim):
        for j in range(i + 1):
            ax = axes[i, j]

            # X-axis ticks for the bottom row (and the diagonals)
            if i == ndim - 1 or i == j:
                ax.set_xticks([param_ranges[j][0], param_ranges[j][1]])
                ax.set_xticklabels(
                    [f"{param_ranges[j][0]}", f"{param_ranges[j][1]}"]
                )

            # Y-axis ticks for the first column (off-diagonal only)
            if j == 0 and i > 0:
                ax.set_yticks([param_ranges[i][0], param_ranges[i][1]])
                ax.set_yticklabels(
                    [f"{param_ranges[i][0]}", f"{param_ranges[i][1]}"]
                )

    if figname is None:
        plt.show()
    else:
        plt.savefig(f"{figname}.png")
        plt.savefig(f"{figname}.pdf")
    return fig


if __name__ == "__main__":
    fig = generate_corner_plot(
        "bad_par_diffcap.txt",
        os.path.join(BATFIT_EXP, "p2d_diffcap.yaml"),
        plot_params=[
            "i0_a",
            "x0_c",
            "eps_s_c",
            "eps_s_a",
            "eps_el_c",
            "eps_el_a",
            "l_a",
            "l_c",
        ],
        fontsize=14,
        figname="diffcap_fail",
    )
    fig = generate_corner_plot(
        "bad_par_hppc.txt",
        os.path.join(BATFIT_EXP, "p2d_hppc.yaml"),
        plot_params=[
            "i0_a",
            "x0_c",
            "eps_s_c",
            "eps_s_a",
            "eps_el_c",
            "eps_el_a",
            "l_a",
            "l_c",
        ],
        fontsize=14,
        figname="hppc_fail",
    )
