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


def plot_rpt(data_type, rpt_id, dim=12, one_cell=False):

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

    symbol = ["s", "^", "o"]
    color = ["b", "r", "k", "lightcoral", "c", "m", "lawngreen", "brown"]

    diag_folder = os.path.join(
        "output", data_type.lower(), f"{dim}dim", make_folder_leaf(rpt_id)
    )
    with open(os.path.join(diag_folder, "deg_parameters.pkl"), "rb") as f:
        deg_param = pickle.load(f)

    for ideg, deg_param_name in enumerate(sim_params["deg_param_names"]):
        fig = plt.figure()
        for iprot, protocol in enumerate(list(pair.keys())):
            for icell, cell_id in enumerate(pair[protocol]):
                plt.errorbar(
                    iprot,
                    deg_param[protocol][cell_id]["mu"][ideg],
                    yerr=deg_param[protocol][cell_id]["sigma"][ideg],
                    fmt=symbol[icell],
                    color=color[iprot],
                    linewidth=2,
                    capsize=6,
                )
        plt.plot(
            np.linspace(0, len(protocol) + 1, 10),
            np.ones(10) * sim_params[f"deg_{deg_param_name}_min"],
            color="r",
            linewidth=3,
        )
        plt.plot(
            np.linspace(0, len(protocol) + 1, 10),
            np.ones(10) * sim_params[f"deg_{deg_param_name}_max"],
            color="r",
            linewidth=3,
        )
        ax = plt.gca()
        ax.set_xticks(list(range(len(pair.keys()))))
        ax.set_xticklabels([protocol for protocol in list(pair.keys())])
        pretty_labels(
            "", deg_param_name, fontsize=16, fontname="Times", grid=False
        )

        figure_rpt_folder = os.path.join(
            figure_folder, make_folder_leaf(rpt_id)
        )
        os.makedirs(figure_rpt_folder, exist_ok=True)
        plt.savefig(os.path.join(figure_rpt_folder, f"{deg_param_name}.png"))
        plt.close()

    n_params = inp.n_param_pred
    fig_org = figure_org(n_params)
    rows = fig_org["rows"]
    cols = fig_org["cols"]
    fig, axes = plt.subplots(rows, cols, figsize=(rows * 4 + 3, cols * 2))
    axes = axes.flatten()
    for ideg, deg_param_name in enumerate(sim_params["deg_param_names"]):
        ax = axes[ideg]
        for iprot, protocol in enumerate(list(pair.keys())):
            cell_list = pair[protocol]
            if one_cell:
                cell_list = [cell_list[0]]
            for icell, cell_id in enumerate(cell_list):
                ax.errorbar(
                    iprot,
                    deg_param[protocol][cell_id]["mu"][ideg],
                    yerr=deg_param[protocol][cell_id]["sigma"][ideg],
                    fmt=symbol[icell],
                    color=color[iprot],
                    linewidth=2,
                    capsize=6,
                )
        ax.plot(
            np.linspace(0, len(protocol) + 1, 10),
            np.ones(10) * sim_params[f"deg_{deg_param_name}_min"],
            color="r",
            linewidth=3,
        )
        ax.plot(
            np.linspace(0, len(protocol) + 1, 10),
            np.ones(10) * sim_params[f"deg_{deg_param_name}_max"],
            color="r",
            linewidth=3,
        )
        ax.set_xticks(list(range(len(pair.keys()))))
        ax.set_xticklabels([protocol for protocol in list(pair.keys())])
        pretty_labels(
            "",
            deg_param_name,
            fontsize=10,
            fontname="Times",
            grid=False,
            ax=ax,
        )
    figure_rpt_folder = os.path.join(figure_folder, make_folder_leaf(rpt_id))
    os.makedirs(figure_rpt_folder, exist_ok=True)
    for i in range(len(sim_params["deg_param_names"]), rows * cols):
        ax = axes[i]
        ax.axis("off")
    if not one_cell:
        ax = axes[fig_org["ax_id"]]
    handles, labels = axes[0].get_legend_handles_labels()
    if one_cell:
        plt.savefig(os.path.join(figure_rpt_folder, f"combined_one.png"))
    else:
        plt.savefig(os.path.join(figure_rpt_folder, f"combined.png"))

    plt.close()


if __name__ == "__main__":
    # for data_type in ["posthppc", "diffcap", "diffcap-posthppc"]:
    for data_type in ["posthppc"]:
        # for dim in [12, 17, 19]:
        for dim in [12]:
            # for rpt_id in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            for rpt_id in [0, 10]:
                for one_cell in [True, False]:
                    plot_rpt(data_type, rpt_id, dim=dim, one_cell=one_cell)
