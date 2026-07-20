import os
import sys

import corner
import yaml
from prettyPlot.plotting import *
from utils import figure_org


def make_folder_leaf(rpt_id):
    if rpt_id == -1:
        folder_leaf = "BOL"
    else:
        folder_leaf = f"RPT_{rpt_id}"
    return folder_leaf


def plot_corner_prior(data_path, sim_config):

    with open(sim_config, "r") as f:
        config = yaml.safe_load(f)
    raw_names = config.get("degradation parameter names", "")
    labels = [name.strip() for name in raw_names.split(",")]
    min_bounds = config.get("min degradation parameter", {})
    max_bounds = config.get("max degradation parameter", {})

    plot_ranges = []
    for label in labels:
        if label in min_bounds and label in max_bounds:
            plot_ranges.append((min_bounds[label], max_bounds[label]))
        else:
            raise ValueError(
                f"Bounds for parameter '{label}' are missing in the YAML file."
            )

    npz_path = os.path.join(data_path, "data_split.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Could not find {npz_path}")

    data = np.load(npz_path)

    y_train = data["Y_train"]
    y_test = data["Y_test"]
    all_data = np.vstack((y_train, y_test))

    if all_data.shape[1] != len(labels):
        raise ValueError(
            f"Data dimension ({all_data.shape[1]}) does not match the number of parameters ({len(labels)})."
        )

    fig = corner.corner(
        all_data,
        labels=labels,
        range=plot_ranges,
        # quantiles=[0.16, 0.5, 0.84], # Optional: shows 1-sigma distribution lines
        show_titles=True,  # Optional: shows median and uncertainty above 1D histograms
        title_kwargs={"fontsize": 10},
        label_kwargs={"fontsize": 12},
    )

    figure_folder = "Figures"
    os.makedirs(figure_folder, exist_ok=True)
    figure_folder = os.path.join(figure_folder, "prior")
    os.makedirs(figure_folder, exist_ok=True)
    plt.savefig(os.path.join(figure_folder, f"prior.png"))
    plt.close()


if __name__ == "__main__":
    data_path = "/scratch/mhassana/LHX_2/data_p2d_diffcap_12dim_ext_foc2_2M/"
    sim_config = "/projects/mlbatt/LHX_2/BatFIT/batfit/default_exps/p2d_diffcap_12dim_ext_foc.yaml"
    plot_corner_prior(data_path=data_path, sim_config=sim_config)
