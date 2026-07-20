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


def plot_voltage(data_type, rpt_id):

    figure_folder = "Figures"
    os.makedirs(figure_folder, exist_ok=True)
    assert data_type.lower() in ["posthppc", "diffcap", "diffcap-posthppc"]
    figure_folder = os.path.join(figure_folder, data_type.lower())
    os.makedirs(figure_folder, exist_ok=True)
    figure_folder = os.path.join(figure_folder, "forward")
    os.makedirs(figure_folder, exist_ok=True)
    folder_leaf = make_folder_leaf(rpt_id)
    figure_folder = os.path.join(figure_folder, folder_leaf)
    os.makedirs(figure_folder, exist_ok=True)

    # load data
    phi_pred_file = (
        f"output_voltage/{data_type.lower()}/{folder_leaf}/phi_pred.pkl"
    )
    with open(phi_pred_file, "rb") as f:
        phi_pred = pickle.load(f)

    for dim in [10, 15, 19, 23]:
        A = np.load(
            f"/scratch/mhassana/LHX_2/data_p2d_diffcap_{dim}dim_4M/data_split.npz"
        )["X_train"]
        for protocol in phi_pred:
            for cell_id in phi_pred[protocol]:
                fig = plt.figure()
                for i in range(1000):
                    plt.plot(A[i, 0, :], A[i, 1, :], linewidth=1, color="grey")
                plt.plot(
                    phi_pred[protocol][cell_id]["t_target"],
                    phi_pred[protocol][cell_id]["phi_target"],
                    color="r",
                    linewidth=3,
                )
                for t, p in zip(
                    phi_pred[protocol][cell_id]["t"],
                    phi_pred[protocol][cell_id]["phi"],
                ):
                    if t is not None and p is not None:
                        plt.plot(t, p, linewidth=1, color="k")
                pretty_labels(
                    "t [s]",
                    r"$\phi$ [V]",
                    fontsize=16,
                    fontname="Times",
                    grid=False,
                )
                plt.savefig(
                    os.path.join(
                        figure_folder,
                        f"{protocol}_{cell_id}_back_{dim}dim.png",
                    )
                )
                plt.close()


if __name__ == "__main__":
    # for data_type in ["posthppc", "diffcap", "diffcap-posthppc"]:
    for data_type in ["diffcap"]:
        for rpt_id in [0]:
            plot_voltage(data_type, rpt_id)
