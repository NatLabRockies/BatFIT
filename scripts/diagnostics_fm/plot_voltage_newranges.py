import os
import sys

from prettyPlot.plotting import *

from batfit.basicutilityc import ReadInput as ri
from batfit.model.paramNN import *


def plot_voltage(data_type):

    figure_folder = "Figures_back"
    os.makedirs(figure_folder, exist_ok=True)
    assert data_type.lower() in ["posthppc", "diffcap", "diffcap-posthppc"]

    # Load target data
    t_target = []
    phi_target = []

    for rpt_id in range(0, 11):
        target_folder = f"/projects/mlbatt/LHX_2/extract_hdvolts_data/data_target/RPT_{rpt_id}"
        for protocol in [
            "LH-1",
            "LH-2",
            "LH-3",
            "LH-4",
            "RH-1",
            "RH-2",
            "RH-3",
            "RH-4",
        ]:
            target_file = os.path.join(
                target_folder, f"{protocol}_{data_type.lower()}.pkl"
            )
            with open(target_file, "rb") as f:
                target_data = pickle.load(f)

            for cell_id in target_data:
                target_data_cell = target_data[cell_id]
                t_target.append(target_data_cell["t"])
                phi_target.append(target_data_cell["phis_c"])

    # for dim in [12, 31]:
    for dim in [12]:
        if data_type.lower() == "diffcap":
            A = np.load(
                f"/scratch/mhassana/LHX_2/data_p2d_{data_type}_{dim}dim_ext_2M/data_split.npz"
            )["X_train"]
        elif data_type.lower() == "posthppc":
            A = np.load(
                f"/scratch/mhassana/LHX_2/data_p2d_{data_type}_{dim}dim_ext_2M/data_split.npz"
            )["X_train"]
        fig = plt.figure()
        print(A.shape[0])
        for i in range(min(10000, A.shape[0])):
            plt.plot(A[i, 0, :], A[i, 1, :], linewidth=1, color="grey")
        for t, p in zip(t_target, phi_target):
            plt.plot(t, p, linewidth=1, color="k")
        plt.savefig(
            os.path.join(figure_folder, f"{data_type}_back_{dim}dim_ext.png")
        )
        plt.close()


if __name__ == "__main__":
    # for data_type in ["posthppc", "diffcap", "diffcap-posthppc"]:
    plot_voltage("diffcap")
    plot_voltage("posthppc")
