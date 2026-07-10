import pickle

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

def plot_voltage(data_type, rpt_id, dim):

    figure_folder = "Figures"
    os.makedirs(figure_folder, exist_ok=True)
    assert data_type.lower() in ["posthppc", "diffcap", "diffcap-posthppc"]
    figure_folder = os.path.join(figure_folder, data_type.lower())
    os.makedirs(figure_folder, exist_ok=True)
    figure_folder = os.path.join(figure_folder, f"{dim}dim")
    os.makedirs(figure_folder, exist_ok=True)
    figure_folder = os.path.join(figure_folder, "forward")
    os.makedirs(figure_folder, exist_ok=True)
    folder_leaf = make_folder_leaf(rpt_id)
    figure_folder = os.path.join(figure_folder, folder_leaf)
    os.makedirs(figure_folder, exist_ok=True)

    # load data
    phi_pred_file= f"output_voltage/{data_type.lower()}/{dim}dim/{folder_leaf}/phi_pred.pkl"
    with open(phi_pred_file, "rb") as f:
        phi_pred = pickle.load(f)

    for protocol in phi_pred:
        for cell_id in phi_pred[protocol]:
            fig = plt.figure(figsize=(12, 6))
            plt.plot(phi_pred[protocol][cell_id]["t_target"], phi_pred[protocol][cell_id]["phi_target"], color="r", linewidth=3)
            for t,p  in zip(phi_pred[protocol][cell_id]["t"], phi_pred[protocol][cell_id]["phi"]):
                if t is not None and p is not None:
                    plt.plot(t,p, linewidth=1, color='k')
            mean_mae = np.nanmean(np.array(phi_pred[protocol][cell_id]["error"]))
            std_mae = np.nanstd(np.array(phi_pred[protocol][cell_id]["error"]))
            min_mae = np.nanmin(np.array(phi_pred[protocol][cell_id]["error"]))
            max_mae = np.nanmax(np.array(phi_pred[protocol][cell_id]["error"]))


            pretty_labels("t [s]", r"$\phi$ [V]", title=f"MAE = {mean_mae:.2g} +/- {std_mae:.2g} mV;  Min-max error = {min_mae:.2g}-{max_mae:.2g} mV",fontsize=16, fontname="Times", grid=False)
            plt.savefig(os.path.join(figure_folder, f"{protocol}_{cell_id}.png"))
            plt.close()


if __name__ == "__main__":
    #for data_type in ["posthppc", "diffcap", "diffcap-posthppc"]:
    #for data_type in ["diffcap", "posthppc"]: #"diffcap-posthppc"]:
    for data_type in ["posthppc"]: #"diffcap-posthppc"]:
        #for dim in [12, 17, 19]:
        for dim in [12]:
            for rpt_id in range(0,11):
                plot_voltage(data_type, rpt_id, dim=dim)
