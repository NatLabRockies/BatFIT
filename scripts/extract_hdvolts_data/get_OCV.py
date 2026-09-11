import os

import bmlite as bm
import pandas as pd
from prettyPlot.plotting import *

from batfit import logger

# logger.setLevel("DEBUG")
logger.setLevel("INFO")


def get_raw_ca_OCV(root_folder="/Users/mhassana/Desktop/GitHub/HDVOLTS_data"):
    half_cell_data_folder = os.path.join(root_folder, "Half cell")
    raw = pd.read_csv(
        os.path.join(half_cell_data_folder, "NMC C-50 raw.csv"), skiprows=0
    )
    # breakpoint()
    return raw


def get_raw_an_OCV(root_folder="/Users/mhassana/Desktop/GitHub/HDVOLTS_data"):
    half_cell_data_folder = os.path.join(root_folder, "Half cell")
    raw = pd.read_csv(
        os.path.join(half_cell_data_folder, "Gr-Si C-50 raw.csv"), skiprows=0
    )
    # breakpoint()
    return raw


def get_an_OCV(root_folder="/Users/mhassana/Desktop/GitHub/HDVOLTS_data"):
    half_cell_data_folder = os.path.join(root_folder, "Half cell")
    raw = pd.read_csv(
        os.path.join(half_cell_data_folder, "Gr-Si OCV C-50.csv"), skiprows=0
    )
    SOC = raw["SOC_avg (%)"] * 0.01
    OCV = raw["Voltage_avg (V)"]
    return SOC, OCV


def get_ca_OCV(root_folder="/Users/mhassana/Desktop/GitHub/HDVOLTS_data"):
    half_cell_data_folder = os.path.join(root_folder, "Half cell")
    raw = pd.read_csv(
        os.path.join(half_cell_data_folder, "NMC OCV C-50.csv"), skiprows=0
    )
    SOC = raw["SOC_avg (%)"] * 0.01
    OCV = raw["Voltage_avg (V)"]
    return SOC, OCV


def get_xmin_nmc811():
    array = pd.read_csv(
        "nmc811_garrick.png.traj.csv", skiprows=0, delimiter=" "
    ).to_numpy()
    array[:, 0] = array[:, 0] / 10

    # Remove edges
    array = array[20:, :]

    # Find the value of intercallation fraction at y=4.2V
    sort_indices = np.argsort(array[:, 1])
    y_vals_sorted = array[sort_indices, 1]
    x_vals_sorted = array[sort_indices, 0]
    interpolated_x = np.interp(4.2, y_vals_sorted, x_vals_sorted)

    return interpolated_x


def soc2x(soc, x_min, x_max):
    return -soc * (x_max - x_min) + (x_max)


if __name__ == "__main__":
    nmc811_xmin = get_xmin_nmc811()
    SOC_an, OCV_an = get_an_OCV()
    SOC_ca, OCV_ca = get_ca_OCV()
    X_an = soc2x(SOC_an.to_numpy(), x_min=0.0, x_max=1.0)
    X_ca = soc2x(SOC_ca.to_numpy(), x_min=nmc811_xmin, x_max=1.0)

    bm_an = bm.materials.GraphiteSiOx(0.5, 0.5, 29.583)
    x_bm_an = np.linspace(0, 1, 1000)
    ocv_bm_an = bm_an.get_Eeq(x_bm_an)
    bm_ca = bm.materials.NMC811(0.5, 0.5, 51.765)
    x_bm_ca = np.linspace(0, 1, 1000)
    ocv_bm_ca = bm_ca.get_Eeq(x_bm_ca)

    from scipy.interpolate import CubicSpline, PchipInterpolator

    # Extrapolate
    sort_idx_an = np.argsort(X_an)
    sort_idx_ca = np.argsort(X_ca)
    X_an_extrap = X_an[sort_idx_an]
    X_ca_extrap = X_ca[sort_idx_ca]
    OCV_an_extrap = OCV_an.to_numpy()[sort_idx_an]
    OCV_ca_extrap = OCV_ca.to_numpy()[sort_idx_ca]

    # left Linear extrap to 6 for cathode
    OCV_ca_extrap = np.delete(OCV_ca_extrap, [0])
    X_ca_extrap = np.delete(X_ca_extrap, [0])
    X_add = np.linspace(0, X_ca_extrap[0], 21)[:20]
    OCV_add = 6 + (X_add / X_ca_extrap[0]) * (OCV_ca_extrap[0] - 6)
    X_ca_extrap = np.insert(X_ca_extrap, 0, X_add)
    OCV_ca_extrap = np.insert(OCV_ca_extrap, 0, OCV_add)

    # Right Linear extrap to 2.5 for cathode
    OCV_ca_extrap = np.delete(OCV_ca_extrap, [-1])
    X_ca_extrap = np.delete(X_ca_extrap, [-1])
    X_add = np.linspace(X_ca_extrap[-1], 1, 11)[1:]
    OCV_add = OCV_ca_extrap[-1] + (X_add - X_ca_extrap[-1]) / (
        1.0 - X_ca_extrap[-1]
    ) * (2.5 - OCV_ca_extrap[-1])
    OCV_ca_extrap = np.delete(OCV_ca_extrap, [-1, -2, -3, -4])
    X_ca_extrap = np.delete(X_ca_extrap, [-1, -2, -3, -4])
    X_ca_extrap = np.append(X_ca_extrap, X_add)
    OCV_ca_extrap = np.append(OCV_ca_extrap, OCV_add)

    # Right Linear extrap to 0 for anode
    OCV_an_extrap = np.delete(OCV_an_extrap, [-1])
    X_an_extrap = np.delete(X_an_extrap, [-1])
    X_add = np.linspace(X_an_extrap[-1], 1, 11)[1:]
    OCV_add = OCV_an_extrap[-1] + (X_add - X_an_extrap[-1]) / (
        1.0 - X_an_extrap[-1]
    ) * (0.0 - OCV_an_extrap[-1])
    OCV_an_extrap = np.delete(OCV_an_extrap, [-1])
    X_an_extrap = np.delete(X_an_extrap, [-1])
    X_an_extrap = np.append(X_an_extrap, X_add)
    OCV_an_extrap = np.append(OCV_an_extrap, OCV_add)

    data_ocv_an = {"x": X_an_extrap, "V": OCV_an_extrap}
    data_ocv_ca = {"x": X_ca_extrap, "V": OCV_ca_extrap}

    df_an = pd.DataFrame(data_ocv_an)
    df_an.to_csv("graphite_siox_ocv.csv", index=False)

    df_ca = pd.DataFrame(data_ocv_ca)
    df_ca.to_csv("nmc811_ocv.csv", index=False)

    cs_an = PchipInterpolator(X_an_extrap, OCV_an_extrap)
    cs_ca = PchipInterpolator(X_ca_extrap, OCV_ca_extrap)
    # cs_an = CubicSpline(X_an_extrap, OCV_an_extrap)
    # cs_ca = CubicSpline(X_ca_extrap, OCV_ca_extrap)

    x_dense = np.linspace(0, 1, 10000)
    y_dense_an = cs_an(x_dense)
    y_dense_ca = cs_ca(x_dense)

    fig = plt.figure()
    plt.plot(x_bm_an, ocv_bm_an, color="k", linewidth=1, label="paper")
    plt.plot(X_an, OCV_an, color="b", linewidth=3, label="bumjun")
    plt.plot(x_dense, y_dense_an, "--", color="r", linewidth=3, label="Spline")
    # plt.plot(X_an_extrap, OCV_an_extrap, "o", color="k", linewidth=3)
    ax = plt.gca()
    ax.set_ylim([0, 1.5])
    pretty_labels(
        "X / SOC", "OCV Gr Si [V]", fontsize=16, fontname="Times", grid=False
    )
    pretty_legend(fontsize=16, fontname="Times")
    fig = plt.figure()
    plt.plot(x_bm_ca, ocv_bm_ca, color="k", linewidth=1, label="paper")
    plt.plot(X_ca, OCV_ca, color="b", linewidth=3, label="bumjun")
    plt.plot(x_dense, y_dense_ca, "--", color="r", linewidth=3, label="Spline")
    # plt.plot(X_ca_extrap, OCV_ca_extrap, "o", color="k", linewidth=3)
    pretty_labels(
        "X / SOC", "OCV NMC 811 [V]", fontsize=16, fontname="Times", grid=False
    )
    pretty_legend(fontsize=16, fontname="Times")
    plt.show()

    # raw_ca = get_raw_ca_OCV()
    # fig = plt.figure()
    # plt.plot(raw_ca["Capacity_ch (mAh)"], raw_ca["Voltage_ch (V)"], color="r", linewidth=3)
    # plt.plot(raw_ca["Capacity_disch (mAh)"], raw_ca["Voltage_disch (V)"], color="b", linewidth=3)
    # raw_an = get_raw_an_OCV()
    # fig = plt.figure()
    # plt.plot(raw_an["Capacity_ch (mAh)"], raw_an["Voltage_ch (V)"], color="r", linewidth=3)
    # plt.plot(raw_an["Capacity_disch (mAh)"], raw_an["Voltage_disch (V)"], color="b", linewidth=3)
    ##plt.show()
