from prettyPlot.plotting import *
from batfit.basicutilityc import ReadInput as ri
from batfit.model.paramNN import *
import os
import sys

def make_folder_leaf(rpt_id):
    if rpt_id == -1:
        folder_leaf = "BOL"
    else:
        folder_leaf = f"RPT_{rpt_id}"
    return folder_leaf

def plot_rpt(data_type, rpt_id):

    figure_folder = "Figures"
    os.makedirs(figure_folder, exist_ok=True)
    assert data_type.lower() in ["hppc", "diffcap", "diffcap-hppc"]
    figure_folder = os.path.join(figure_folder, data_type.lower())
    os.makedirs(figure_folder, exist_ok=True)

    model_recipe = f"recipes/hppc/recipe.yml"
    inp = ri.basic_input(model_recipe)
    sim_params = make_params(inp.sim_config)
   
    sys.path.append("/Users/mhassana/Desktop/GitHub/BatFIT_mar26/scripts/extract_hdvolts_data/")
    from file_management import cells_protocols_pairs
    pair = cells_protocols_pairs()

    symbol = ["s", "^", "o"]
    color = ["b", "r", "k", "lightcoral", "c", "m", "lawngreen"]
    
    diag_folder = os.path.join("output", data_type.lower(), make_folder_leaf(rpt_id))
    with open(os.path.join(diag_folder, "deg_parameters.pkl"), "rb") as f:
        deg_param = pickle.load(f)

    for ideg, deg_param_name in enumerate(sim_params["deg_param_names"]):
        fig = plt.figure()
        for iprot, protocol in enumerate(list(pair.keys())):
            for icell, cell_id in enumerate(pair[protocol]):
                plt.errorbar(iprot, deg_param[protocol][cell_id]["mu"][ideg], yerr=deg_param[protocol][cell_id]["sigma"][ideg], fmt=symbol[icell], color=color[iprot], linewidth=2, capsize=6)
        plt.plot(np.linspace(0, len(protocol)+1, 10), np.ones(10)*sim_params[f"deg_{deg_param_name}_min"], color="r", linewidth=3)
        plt.plot(np.linspace(0, len(protocol)+1, 10), np.ones(10)*sim_params[f"deg_{deg_param_name}_max"], color="r", linewidth=3)
        ax = plt.gca()
        ax.set_xticks(list(range(len(pair.keys()))))
        ax.set_xticklabels([protocol for protocol in list(pair.keys())])
        pretty_labels("", deg_param_name, fontsize=16, fontname="Times", grid=False)

        figure_rpt_folder = os.path.join(figure_folder, make_folder_leaf(rpt_id))
        os.makedirs(figure_rpt_folder, exist_ok=True)
        plt.savefig(os.path.join(figure_rpt_folder, f"{deg_param_name}.png"))
        plt.close()

    rows = 4
    cols = 8
    fig, axes = plt.subplots(rows, cols, figsize=(23, 6))
    axes = axes.flatten()
    for ideg, deg_param_name in enumerate(sim_params["deg_param_names"]):
        ax = axes[ideg]
        for iprot, protocol in enumerate(list(pair.keys())):
            for icell, cell_id in enumerate(pair[protocol]):
                ax.errorbar(iprot, deg_param[protocol][cell_id]["mu"][ideg], yerr=deg_param[protocol][cell_id]["sigma"][ideg], fmt=symbol[icell], color=color[iprot], linewidth=2, capsize=6)
        ax.plot(np.linspace(0, len(protocol)+1, 10), np.ones(10)*sim_params[f"deg_{deg_param_name}_min"], color="r", linewidth=3)
        ax.plot(np.linspace(0, len(protocol)+1, 10), np.ones(10)*sim_params[f"deg_{deg_param_name}_max"], color="r", linewidth=3)
        ax.set_xticks(list(range(len(pair.keys()))))
        ax.set_xticklabels([protocol for protocol in list(pair.keys())])
        pretty_labels("", deg_param_name, fontsize=10, fontname="Times", grid=False, ax=ax)
    figure_rpt_folder = os.path.join(figure_folder, make_folder_leaf(rpt_id))
    os.makedirs(figure_rpt_folder, exist_ok=True)
    fig.delaxes(axes[31])
    plt.savefig(os.path.join(figure_rpt_folder, f"combined.png"))
    plt.close()


if __name__ == "__main__":
    for data_type in ["hppc", "diffcap", "diffcap-hppc"]:
        for rpt_id in [-1, 1, 2, 3, 4, 5, 6, 7, 8]:
            plot_rpt(data_type, rpt_id)
