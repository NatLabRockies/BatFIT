import os
import pickle
import sys

from prettyPlot.plotting import *
from utils import figure_org

from batfit.basicutilityc import ReadInput as ri
from batfit.model.paramNN import *
from batfit.preprocess.sim_setup import make_params


def make_folder_leaf(rpt_id):
    if rpt_id == -1:
        folder_leaf = "BOL"
    else:
        folder_leaf = f"RPT_{rpt_id}"
    return folder_leaf


def plot_aging(data_type, mode="lhx", dim=12, one_cell=False):

    assert mode in ["lhx", "rhx", "lhrh"]
    if mode.lower() == "lhx":
        protocols = ["LH-4", "LH-3", "LH-2", "LH-1"]
    elif mode.lower() == "rhx":
        protocols = ["RH-4", "RH-3", "RH-2", "RH-1"]
    elif mode.lower() == "lhrh":
        protocols = ["LH-3", "RH-3"]

    figure_folder = "Figures"
    os.makedirs(figure_folder, exist_ok=True)
    assert data_type.lower() in ["posthppc", "diffcap", "diffcap-posthppc"]
    figure_folder = os.path.join(figure_folder, data_type.lower())
    os.makedirs(figure_folder, exist_ok=True)
    figure_folder = os.path.join(figure_folder, f"{dim}dim")
    os.makedirs(figure_folder, exist_ok=True)

    if data_type.lower() == "diffcap-posthppc":
        model_recipe = f"recipes/diffcap/{dim}dim/recipe.yml"
    else:
        model_recipe = f"recipes/{data_type.lower()}/{dim}dim/recipe.yml"
    inp = ri.basic_input(model_recipe)
    sim_params = make_params(inp.sim_config)

    sys.path.append("/projects/mlbatt/LHX_2/extract_hdvolts_data/")
    from file_management import cells_protocols_pairs

    pair = cells_protocols_pairs()

    symbol = ["s", "^", "o", "v"]
    color = ["b", "r", "k", "lightcoral", "c", "m", "lawngreen", "brown"]

    diag_folder = os.path.join("output", data_type.lower(), f"{dim}dim")
    deg_param_list = []
    for rpt_id in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        with open(
            os.path.join(
                diag_folder, make_folder_leaf(rpt_id), "deg_parameters.pkl"
            ),
            "rb",
        ) as f:
            deg_param_list.append(pickle.load(f))

    n_params = inp.n_param_pred
    assert n_params == dim
    fig_org = figure_org(n_params)
    rows = fig_org["rows"]
    cols = fig_org["cols"]
    fig, axes = plt.subplots(rows, cols, figsize=(rows * 2 + 3, cols * 2))
    axes = axes.flatten()
    for ideg, deg_param_name in enumerate(sim_params["deg_param_names"]):
        ax = axes[ideg]
        for iprot, protocol in enumerate(protocols):
            cell_list = pair[protocol]
            if one_cell:
                cell_list = [cell_list[0]]
            for icell, cell_id in enumerate(cell_list):
                rpt_x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                mu_deg = [
                    deg_param[protocol][cell_id]["mu"][ideg]
                    for deg_param in deg_param_list
                ]
                std_deg = [
                    deg_param[protocol][cell_id]["sigma"][ideg]
                    for deg_param in deg_param_list
                ]
                ax.errorbar(
                    rpt_x,
                    mu_deg,
                    yerr=std_deg,
                    fmt=symbol[icell],
                    color=color[iprot],
                    alpha=0.3,
                    linewidth=2,
                    capsize=6,
                    label=f"{protocol}_{cell_id}",
                )
        ax.plot(
            np.linspace(0, len(rpt_x) + 1, 10),
            np.ones(10) * sim_params[f"deg_{deg_param_name}_min"],
            color="r",
            linewidth=3,
        )
        ax.plot(
            np.linspace(0, len(rpt_x) + 1, 10),
            np.ones(10) * sim_params[f"deg_{deg_param_name}_max"],
            color="r",
            linewidth=3,
        )
        pretty_labels(
            "",
            deg_param_name,
            fontsize=10,
            fontname="Times",
            grid=False,
            ax=ax,
        )
    figure_rpt_folder = os.path.join(figure_folder, "aging")
    os.makedirs(figure_rpt_folder, exist_ok=True)
    for i in range(len(sim_params["deg_param_names"]), rows * cols):
        ax = axes[i]
        ax.axis("off")
    if not one_cell:
        ax = axes[fig_org["ax_id"]]

    handles, labels = axes[0].get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="center",
        edgecolor="none",
        prop={"family": "Times New Roman", "size": 12, "weight": "bold"},
    )
    if one_cell:
        plt.savefig(
            os.path.join(figure_rpt_folder, f"combined_{mode.lower()}_one.png")
        )
    else:
        plt.savefig(
            os.path.join(figure_rpt_folder, f"combined_{mode.lower()}.png")
        )

    plt.close()


if __name__ == "__main__":
    # for data_type in ["posthppc", "diffcap", "diffcap-posthppc"]:
    # for data_type in ["diffcap", "posthppc", "diffcap-posthppc"]:
    for data_type in ["posthppc"]:
        # for dim in [12, 17, 19]:
        for dim in [12]:
            for one_cell in [True, False]:
                plot_aging(data_type, "lhx", dim=dim, one_cell=one_cell)
                plot_aging(data_type, "rhx", dim=dim, one_cell=one_cell)
                plot_aging(data_type, "lhrh", dim=dim, one_cell=one_cell)
