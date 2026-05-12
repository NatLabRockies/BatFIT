import os
import sys

from prettyPlot.plotting import *

from batfit.basicutilityc import ReadInput as ri
from batfit.model.paramNN import *


def make_folder_leaf(rpt_id):
    if rpt_id == -1:
        folder_leaf = "BOL"
    else:
        folder_leaf = f"RPT_{rpt_id}"
    return folder_leaf


def plot_aging(data_type, mode="lhx"):

    assert mode in ["lhx", "rhx", "lhrh"]
    if mode.lower() == "lhx":
        protocols = ["LH-3", "LH-1"]
    elif mode.lower() == "rhx":
        protocols = ["RH-3", "RH-2", "RH-1"]
    elif mode.lower() == "lhrh":
        protocols = [
            "LH-3",
            "RH-3",
        ]

    figure_folder = "Figures"
    os.makedirs(figure_folder, exist_ok=True)
    assert data_type.lower() in ["hppc", "diffcap", "diffcap-hppc"]
    figure_folder = os.path.join(figure_folder, data_type.lower())
    os.makedirs(figure_folder, exist_ok=True)

    model_recipe = f"recipes/hppc/recipe.yml"
    inp = ri.basic_input(model_recipe)
    sim_params = make_params(inp.sim_config)

    sys.path.append(
        "/Users/mhassana/Desktop/GitHub/BatFIT_mar26/scripts/extract_hdvolts_data/"
    )
    from file_management import cells_protocols_pairs

    pair = cells_protocols_pairs()

    symbol = ["s", "^", "o"]
    color = ["b", "r", "k", "lightcoral", "c", "m", "lawngreen"]

    diag_folder = os.path.join("output", data_type.lower())
    deg_param_list = []
    for rpt_id in [-1, 1, 2, 3, 4, 5, 6, 7, 8]:
        with open(
            os.path.join(
                diag_folder, make_folder_leaf(rpt_id), "deg_parameters.pkl"
            ),
            "rb",
        ) as f:
            deg_param_list.append(pickle.load(f))

    rows = 4
    cols = 8
    fig, axes = plt.subplots(rows, cols, figsize=(20, 8))
    axes = axes.flatten()
    for ideg, deg_param_name in enumerate(sim_params["deg_param_names"]):
        ax = axes[ideg]
        for iprot, protocol in enumerate(protocols):
            for icell, cell_id in enumerate(pair[protocol]):
                rpt_x = [0, 1, 2, 3, 4, 5, 6, 7, 8]
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
    # fig.delaxes(axes[31])
    ax = axes[31]
    ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="center",
        edgecolor="none",
        prop={"family": "Times New Roman", "size": 12, "weight": "bold"},
    )
    plt.savefig(
        os.path.join(figure_rpt_folder, f"combined_{mode.lower()}.png")
    )
    plt.close()


if __name__ == "__main__":
    for data_type in ["hppc", "diffcap", "diffcap-hppc"]:
        plot_aging(data_type, "lhx")
        plot_aging(data_type, "rhx")
        plot_aging(data_type, "lhrh")
