import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
import scipy
from prettyPlot.plotting import *

from batfit import logger
from batfit.utils.text_utils import shuffle_substrings


# ReadFile
def readFile(filename, min_t, max_t):
    df = pd.read_csv(filename).to_numpy()
    data_t_raw = df[:, 2]
    data_phis_c_raw = df[:, 5]
    data_c_raw = df[:, 4]
    data_state_raw = df[:, -1]
    ind_min = np.argmin(abs(data_t_raw * 60 - min_t))
    if max_t is not None:
        ind_max = np.argmin(abs(data_t_raw * 60 - max_t))
    else:
        ind_max = len(data_t_raw) - 1
    data_t = data_t_raw[ind_min : ind_max + 1]
    data_phis_c = data_phis_c_raw[ind_min : ind_max + 1]
    data_c = data_c_raw[ind_min : ind_max + 1] / 0.0189
    data_c[0] = data_c[1]
    data_state = data_state_raw[ind_min : ind_max + 1]
    # if np.amax(np.diff(data_phis_c))>0:
    #   data_dict = None
    # else:
    #   data_dict = {"t": data_t*60, "phis_c": data_phis_c, "c" : data_c}
    data_dict = {
        "t": data_t * 60,
        "phis_c": data_phis_c,
        "c": data_c,
        "state": data_state,
    }
    return data_dict


def grad_hand(x, y, winsize=35, poly=1, t_tmp=None):
    if len(x) > 1500:
        winsize = max(len(x) // 60, 25)
    else:
        winsize = max(len(x) // 40, 7)
    xold = x
    yold = y
    x = scipy.signal.savgol_filter(x, winsize, poly)
    y = scipy.signal.savgol_filter(y, winsize, poly)
    assert len(x) == len(y)
    assert len(x) == len(t_tmp)
    n = len(x)
    grads = np.zeros(n)
    for i in range(1, n - 1):
        if abs(x[i + 1] - x[i]) < 1e-8:
            grads[i] = np.nan
            t_tmp[i] = np.nan
        else:
            try:
                grads[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
            except ZeroDivisionError:
                grads[i] = np.nan
                t_tmp[i] = np.nan
    grads[0] = np.nan
    grads[1] = np.nan
    t_tmp[0] = np.nan
    t_tmp[1] = np.nan
    grads[-1] = np.nan
    grads[-2] = np.nan
    t_tmp[-1] = np.nan
    t_tmp[-2] = np.nan

    ind = np.argwhere(~np.isnan(grads))[:, 0]
    x = x[ind]
    y = y[ind]
    grads = grads[ind]
    t_tmp = t_tmp[ind]
    ind = np.argwhere(abs(grads) < np.inf)[:, 0]
    x = x[ind]
    y = y[ind]
    grads = grads[ind]
    t_tmp = t_tmp[ind]

    # plt.plot(x,grads)
    # plt.draw()
    # plt.pause(0.1)
    # plt.close()
    # plt.show(block=False)

    # if np.amax(abs(grads)>1e6):

    return x, y, grads, t_tmp


def compute_exo_data(data_dict):
    C_rate_cand = np.array(
        [
            0.5026548517235131,
            4.021070903968223,
            6.031749128431443,
            9.044948010744898,
        ]
    )
    mean_c = np.mean(data_dict["c"][:10])
    ind_c = np.argmin(abs(mean_c - C_rate_cand))
    act_c = C_rate_cand[ind_c]

    if abs(act_c - mean_c) > 0.1:
        logger.warning(f"measured c = {mean_c}, used c = {act_c}")
        breakpoint()

    minT = np.amin(data_dict["t"])
    maxT = np.amax(data_dict["t"])
    tmin = minT + (maxT - minT) / 250
    tmax = maxT - (maxT - minT) / 250
    indmin = np.argmin(abs(data_dict["t"] - tmin)) + 1
    indmax = np.argmin(abs(data_dict["t"] - tmax)) - 1
    Q = act_c * 0.0189 * 1000 * data_dict["t"][indmin : indmax + 1] / 3600
    told = data_dict["t"][indmin : indmax + 1].copy()

    dV_dQ_x, dV_dQ_y, dV_dQ, t_tmp = grad_hand(
        Q, data_dict["phis_c"][indmin : indmax + 1], t_tmp=told
    )
    tVQ = t_tmp
    # dQ_dV_x, dQ_dV_y, dQ_dV, t_tmp = grad_hand(data_dict['phis_c'][indmin:], Q, t_tmp=told)
    tQV = tVQ
    # breakpoint()
    dQ_dV = 1.0 / dV_dQ
    dQ_dV_x = dV_dQ_y
    dQ_dV_y = dV_dQ_x

    assert np.linalg.norm(tQV - tVQ) < 1e-12

    assert len(t_tmp) == len(dV_dQ_x)
    data_dict["Q"] = act_c * 0.0189 * 1000 * tVQ / 3600
    data_dict["t_QV"] = tVQ
    data_dict["dV_dQ"] = dV_dQ
    data_dict["dQ_dV"] = dQ_dV
    data_dict["dV_dQ_x"] = dV_dQ_x
    data_dict["dQ_dV_x"] = dQ_dV_x

    return data_dict


def read_cycling_data(
    dataFolder,
    cellFolder,
    min_t,
    max_t,
    mean_cell_c_dis,
    mean_cell_c_ch4,
    mean_cell_c_ch6,
    mean_cell_c_ch9,
    mean_cell_time_dis,
    mean_cell_time_ch4,
    mean_cell_time_ch6,
    mean_cell_time_ch9,
    mean_cell_end_phi_dis,
    mean_cell_end_phi_ch4,
    mean_cell_end_phi_ch6,
    mean_cell_end_phi_ch9,
    cycles_extract,
    figureFolder,
    dataFolderSave,
    cyc_mode="discharge",
    n_points_per_curve=128,
):

    # setup what dataset is being read
    discharging = False
    charging = False
    cc = False
    cccv = False
    if cyc_mode.lower() == "discharge":
        discharging = True
    elif cyc_mode.lower() == "chargecc":
        charging = True
        cc = True
    elif cyc_mode.lower() == "chargecccv":
        charging = True
        cccv = True
    else:
        logger.error(f"Mode {cyc_mode} not recognized")
        sys.exit()
    if discharging:
        print("Discharging")
    if charging and cc:
        print("Charging CC")
    if charging and cccv:
        print("Charging CC-CV")

    if discharging:
        fullFold = os.path.join(dataFolder, "Discharge", cellFolder)
    elif charging:
        fullFold = os.path.join(dataFolder, "Charge", cellFolder)
    files = os.listdir(fullFold)
    if discharging:
        rootFile = f"P492_{cellFolder.lower()}_discharge_cycle"
    elif charging:
        rootFile = f"P492_{cellFolder.lower()}_charge_cycle"

    cycles = {}
    for file in files:
        if file.startswith(rootFile):
            ind = file.index("_cycle")
            cycle_id = int(file[ind + 6 : -4]) - 1
            if discharging:
                filename = os.path.join(
                    dataFolder, "Discharge", cellFolder, file
                )
            else:
                filename = os.path.join(dataFolder, "Charge", cellFolder, file)

            data_dict = readFile(filename, min_t, max_t)
            err_ind = np.argwhere(data_dict["state"] == "S")
            if len(err_ind) > 0:
                # print(filename)
                # print(err_ind)
                delete = [err_ind[i][0] for i in range(len(err_ind))]
                data_dict["t"] = np.delete(data_dict["t"], delete)
                data_dict["phis_c"] = np.delete(data_dict["phis_c"], delete)
                data_dict["c"] = np.delete(data_dict["c"], delete)

            if charging:
                indCCCV = np.argwhere(
                    data_dict["c"] < data_dict["c"][0] * 0.01
                )
                if not len(indCCCV) == 1:
                    breakpoint()

            if cc:
                low = None
                high = indCCCV[0][0]
                data_dict["t"] = data_dict["t"][low:high]
                data_dict["phis_c"] = data_dict["phis_c"][low:high]
                data_dict["c"] = data_dict["c"][low:high]
            elif cccv:
                delete = [indCCCV[0][0], indCCCV[0][0] + 1]
                # fig=plt.figure()
                # plt.plot(data_dict["t"], data_dict["phis_c"])
                # plt.show()
                data_dict["t"] = np.delete(data_dict["t"], delete)
                data_dict["phis_c"] = np.delete(data_dict["phis_c"], delete)
                data_dict["c"] = np.delete(data_dict["c"], delete)
                # fig=plt.figure()
                # plt.plot(data_dict["t"], data_dict["phis_c"])
                # plt.show()
            else:
                low = None
                high = None

            if data_dict is not None:
                data_dict["t"] = data_dict["t"].astype("float32")
                data_dict["phis_c"] = data_dict["phis_c"].astype("float32")
                data_dict["c"] = data_dict["c"].astype("float32")
                data_dict = compute_exo_data(data_dict)
                # breakpoint()
                # reduce datasize
                # if len(data_dict["t"])<100:
                #    data_dict_final = data_dict
                # else:
                #    data_dict_final = {}
                #    data_dict_final["t"] = np.linspace(data_dict["t"][0], data_dict["t"][-1], 100).astype('float32')
                #    data_dict_final["phis_c"] = np.interp(data_dict_final["t"], data_dict["t"], data_dict["phis_c"]).astype('float32')
                #    data_dict_final["c"] = np.interp(data_dict_final["t"], data_dict["t"], data_dict["c"]).astype('float32')
                # cycles[cycle_id] = data_dict_final
                cycles[cycle_id] = data_dict

    # compute mean C
    mean_c = 0
    mean_time = 0
    mean_end_phi = 0
    min_c = None
    max_c = None
    min_time = None
    max_time = None

    for cyc in cycles:
        ind_arr = np.argwhere(cycles[cyc]["c"] > 0)
        c_arr = cycles[cyc]["c"][ind_arr]
        t_arr = cycles[cyc]["t"][ind_arr]
        phi_arr = cycles[cyc]["phis_c"][ind_arr]

        mean_c += np.mean(c_arr)
        mean_time += t_arr[-1]
        mean_end_phi += phi_arr[-1]

        ind_min = np.argmin(c_arr)
        min_c_tmp = c_arr[ind_min]
        ind_max = np.argmax(c_arr)
        max_c_tmp = c_arr[ind_max]
        if min_c_tmp < 0.4:
            msg = f"ERROR: {cellFolder} for cycle {cyc}"
            msg += f"\nmin c {min_c_tmp} index {ind_min}"
            msg += f"\n{c_arr}"
            logger.error(msg)
            sys.exit()
        if max_c_tmp > 15:
            msg = f"ERROR: {cellFolder} for cycle {cyc}"
            msg += f"\nmin c {max_c_tmp} index {ind_max}"
            msg += f"\n{c_arr}"
            logger.error(msg)
            sys.exit()

        if max_c is None or max_c_tmp > max_c:
            max_c = max_c_tmp
        if min_c is None or min_c_tmp < min_c:
            min_c = min_c_tmp
        if max_time is None or t_arr[-1] > max_time:
            max_time = t_arr[-1]
        if min_time is None or t_arr[-1] < min_time:
            min_time = t_arr[-1]

    mean_c /= len(cycles)
    mean_time /= len(cycles)
    mean_end_phi /= len(cycles)
    print(f"\tmean c for {cellFolder} = {mean_c} ({min_c} - {max_c})")
    print(
        f"\tmean time for {cellFolder} = {mean_time} ({min_time} - {max_time})"
    )
    print(f"\tmean end_phi for {cellFolder} = {mean_end_phi}")
    if discharging:
        mean_cell_c_dis.append(mean_c)
        mean_cell_time_dis.append(mean_time)
        mean_cell_end_phi_dis.append(mean_end_phi)
    elif charging and cc and np.abs(mean_c - 4) < 1:
        mean_cell_c_ch4.append(mean_c)
        mean_cell_time_ch4.append(mean_time)
        mean_cell_end_phi_ch4.append(mean_end_phi)
    elif charging and cc and np.abs(mean_c - 6) < 1:
        mean_cell_c_ch6.append(mean_c)
        mean_cell_time_ch6.append(mean_time)
        mean_cell_end_phi_ch6.append(mean_end_phi)
    elif charging and cc and np.abs(mean_c - 9) < 1:
        mean_cell_c_ch9.append(mean_c)
        mean_cell_time_ch9.append(mean_time)
        mean_cell_end_phi_ch9.append(mean_end_phi)

    targets = ["phis_c", "dV_dQ", "dQ_dV"]

    # Sort cycles
    cyc_id = list(cycles.keys())
    cyc_id.sort()
    sorted_cycles = {i: cycles[i] for i in cyc_id}
    cycles = sorted_cycles
    for cycle_extract in cycles_extract:
        neighbors = get_neighbor_cycles(
            cycle_extract, n_window_min=1, n_window_max=1
        )
        available = True
        for cyc in neighbors:
            if cyc not in cycles:
                available = False
        if not available:
            break

        for target in targets:
            data_ext_x = None
            data_ext_x_exo = None
            data_ext_y = None

            fig = plt.figure()

            ## Balance cycle data
            # ndat_min = 0
            # for cyc in neighbors:
            #    ndat = len(cycles[cyc]["t"])
            #    # print(f"\tndat = {ndat}, cyc={cyc}")
            #    if ndat > 3000:
            #        ndat = ndat // 10
            #    if ndat_min == 0:
            #        ndat_min = ndat
            #    else:
            #        ndat_min = min(ndat, ndat_min)
            # print(f"ndat_min = {ndat_min}, cyc={cyc}")

            for cyc in neighbors:
                t_interp = np.linspace(
                    cycles[cyc]["t"][0],
                    cycles[cyc]["t"][-1],
                    n_points_per_curve,
                )
                if target == "phis_c":
                    phis_c_interp = np.interp(
                        t_interp, cycles[cyc]["t"], cycles[cyc]["phis_c"]
                    ).astype("float32")
                elif target == "dQ_dV":
                    t_exo_interp = np.linspace(
                        np.amin(cycles[cyc]["t_QV"]),
                        np.amax(cycles[cyc]["t_QV"]),
                        n_points_per_curve,
                    ).astype("float32")
                    dQ_dV_x_interp = np.linspace(
                        np.amin(cycles[cyc]["dQ_dV_x"]),
                        np.amax(cycles[cyc]["dQ_dV_x"]),
                        n_points_per_curve,
                    ).astype("float32")
                    # ind = np.argsort(cycles[cyc]["dQ_dV_x"])
                    # breakpoint()
                    dQ_dV_y_interp = np.interp(
                        t_exo_interp, cycles[cyc]["t_QV"], cycles[cyc]["dQ_dV"]
                    ).astype("float32")
                elif target == "dV_dQ":
                    t_exo_interp = np.linspace(
                        cycles[cyc]["t_QV"][0],
                        cycles[cyc]["t_QV"][-1],
                        n_points_per_curve,
                    ).astype("float32")
                    dV_dQ_x_interp = np.linspace(
                        cycles[cyc]["dV_dQ_x"][0],
                        cycles[cyc]["dV_dQ_x"][-1],
                        n_points_per_curve,
                    ).astype("float32")
                    dV_dQ_y_interp = np.interp(
                        t_exo_interp, cycles[cyc]["t_QV"], cycles[cyc]["dV_dQ"]
                    ).astype("float32")
                if data_ext_x is None:
                    try:
                        if target == "phis_c":
                            data_ext_x = t_interp
                            data_ext_y = phis_c_interp
                        elif target == "dQ_dV":
                            data_ext_x_exo = t_exo_interp
                            data_ext_x = dQ_dV_x_interp
                            data_ext_y = dQ_dV_y_interp
                        elif target == "dV_dQ":
                            data_ext_x_exo = t_exo_interp
                            data_ext_x = dV_dQ_x_interp
                            data_ext_y = dV_dQ_y_interp
                        print(f"{target} {data_ext_x.shape}")
                    except KeyError:
                        print(cellFolder)
                        print("target = ", target)
                        print("cyc = ", cyc)
                        print("neighbors = ", neighbors)
                        print("cycle_extract = ", cycle_extract)
                        print("cycles keys ", list(cycles.keys()))
                        logger.error("Missing cycles")
                        sys.exit()
                else:
                    if target == "phis_c":
                        data_ext_x = np.hstack((data_ext_x, t_interp))
                        data_ext_y = np.hstack((data_ext_y, phis_c_interp))
                    elif target == "dQ_dV":
                        data_ext_x = np.hstack((data_ext_x, dQ_dV_x_interp))
                        data_ext_x_exo = np.hstack(
                            (data_ext_x_exo, t_exo_interp)
                        )
                        data_ext_y = np.hstack((data_ext_y, dQ_dV_y_interp))
                    elif target == "dV_dQ":
                        data_ext_x = np.hstack((data_ext_x, dV_dQ_x_interp))
                        data_ext_x_exo = np.hstack(
                            (data_ext_x_exo, t_exo_interp)
                        )
                        data_ext_y = np.hstack((data_ext_y, dV_dQ_y_interp))
                    print(f"{target} {data_ext_x.shape}")
                if target == "phis_c":
                    plt.plot(
                        cycles[cyc]["t"],
                        cycles[cyc]["phis_c"],
                        color="k",
                        linewidth=3,
                    )
                if target == "dQ_dV":
                    plt.plot(
                        cycles[cyc]["dQ_dV_x"],
                        cycles[cyc]["dQ_dV"],
                        color="k",
                        linewidth=3,
                    )
                    # if cellFolder =="Cell30" and cycle_extract == 0:
                    #    breakpoint()
                if target == "dV_dQ":
                    plt.plot(
                        cycles[cyc]["dV_dQ_x"],
                        cycles[cyc]["dV_dQ"],
                        color="k",
                        linewidth=3,
                    )
            if target == "phis_c":
                pretty_labels(
                    "time",
                    r"$\phi_{s,+}$ [V]",
                    14,
                    title=f"{cellFolder}, cycles [{min(neighbors)},{max(neighbors)}]",
                )
            if target == "dQ_dV":
                pretty_labels(
                    r"$\phi_{s,+}$ [V]",
                    r"dQ/dV [mAh V$^{-1}$]",
                    14,
                    title=f"{cellFolder}, cycles [{min(neighbors)},{max(neighbors)}]",
                )
            if target == "dV_dQ":
                pretty_labels(
                    "Q [mAh]",
                    r"dV/dQ [V (mAh)$^{-1}$]",
                    14,
                    title=f"{cellFolder}, cycles [{min(neighbors)},{max(neighbors)}]",
                )

            if discharging:
                plt.savefig(
                    os.path.join(
                        figureFolder,
                        f"{target}_dis_cyc_{cycle_extract}_ave{len(neighbors)}.png",
                    )
                )
            elif charging and cc:
                plt.savefig(
                    os.path.join(
                        figureFolder,
                        f"{target}_chcc_cyc_{cycle_extract}_ave{len(neighbors)}.png",
                    )
                )
            elif charging and cccv:
                plt.savefig(
                    os.path.join(
                        figureFolder,
                        f"{target}_chcccv_cyc_{cycle_extract}_ave{len(neighbors)}.png",
                    )
                )
            plt.close()

            if target == "phis_c":
                print("phis_c shape data_ext_x = ", data_ext_x.shape)
                if discharging:
                    np.savez(
                        os.path.join(
                            dataFolderSave,
                            f"{target}_dis_cyc_{cycle_extract}_ave{len(neighbors)}.npz",
                        ),
                        t=data_ext_x,
                        phis_c=data_ext_y,
                    )
                elif charging and cc:
                    np.savez(
                        os.path.join(
                            dataFolderSave,
                            f"{target}_chcc_cyc_{cycle_extract}_ave{len(neighbors)}.npz",
                        ),
                        t=data_ext_x,
                        phis_c=data_ext_y,
                    )
                elif charging and cccv:
                    np.savez(
                        os.path.join(
                            dataFolderSave,
                            f"{target}_chcccv_cyc_{cycle_extract}_ave{len(neighbors)}.npz",
                        ),
                        t=data_ext_x,
                        phis_c=data_ext_y,
                    )
            elif target == "dQ_dV":
                print("dqdv shape data_ext_x = ", data_ext_x.shape)
                if discharging:
                    np.savez(
                        os.path.join(
                            dataFolderSave,
                            f"{target}_dis_cyc_{cycle_extract}_ave{len(neighbors)}.npz",
                        ),
                        V=data_ext_x,
                        t=data_ext_x_exo,
                        dQ_dV=data_ext_y,
                    )
                elif charging and cc:
                    np.savez(
                        os.path.join(
                            dataFolderSave,
                            f"{target}_chcc_cyc_{cycle_extract}_ave{len(neighbors)}.npz",
                        ),
                        V=data_ext_x,
                        t=data_ext_x_exo,
                        dQ_dV=data_ext_y,
                    )
                elif charging and cccv:
                    np.savez(
                        os.path.join(
                            dataFolderSave,
                            f"{target}_chcccv_cyc_{cycle_extract}_ave{len(neighbors)}.npz",
                        ),
                        V=data_ext_x,
                        t=data_ext_x_exo,
                        dQ_dV=data_ext_y,
                    )
            elif target == "dV_dQ":
                print("dv_dq shape data_ext_x = ", data_ext_x.shape)
                if discharging:
                    np.savez(
                        os.path.join(
                            dataFolderSave,
                            f"{target}_dis_cyc_{cycle_extract}_ave{len(neighbors)}.npz",
                        ),
                        Q=data_ext_x,
                        t=data_ext_x_exo,
                        dV_dQ=data_ext_y,
                    )
                elif charging and cc:
                    np.savez(
                        os.path.join(
                            dataFolderSave,
                            f"{target}_chcc_cyc_{cycle_extract}_ave{len(neighbors)}.npz",
                        ),
                        Q=data_ext_x,
                        t=data_ext_x_exo,
                        dV_dQ=data_ext_y,
                    )
                elif charging and cccv:
                    np.savez(
                        os.path.join(
                            dataFolderSave,
                            f"{target}_chcccv_cyc_{cycle_extract}_ave{len(neighbors)}.npz",
                        ),
                        Q=data_ext_x,
                        t=data_ext_x_exo,
                        dV_dQ=data_ext_y,
                    )

            if target == "phis_c":
                fig = plt.figure()
                for cyc in cycles:
                    plt.plot(
                        cycles[cyc]["t"],
                        cycles[cyc]["phis_c"],
                        color="k",
                        linewidth=1,
                    )
                pretty_labels(
                    "time [s]",
                    r"$\phi_{s,+}$ [V]",
                    14,
                    title=f"{cellFolder}, all cycles",
                )
            elif target == "dQ_dV":
                fig = plt.figure()
                for cyc in cycles:
                    plt.plot(
                        cycles[cyc]["dQ_dV_x"],
                        cycles[cyc]["dQ_dV"],
                        color="k",
                        linewidth=1,
                    )
                pretty_labels(
                    r"$\phi_{s,+}$ [V]",
                    r"dQ/dV [mAh V$^{-1}$]",
                    14,
                    title=f"{cellFolder}, all cycles",
                )
            elif target == "dV_dQ":
                fig = plt.figure()
                for cyc in cycles:
                    plt.plot(
                        cycles[cyc]["dV_dQ_x"],
                        cycles[cyc]["dV_dQ"],
                        color="k",
                        linewidth=1,
                    )
                pretty_labels(
                    r"Q [mAh]",
                    r"dV/dQ [V (mAh)$^{-1}$]",
                    14,
                    title=f"{cellFolder}, all cycles",
                )
            if discharging:
                plt.savefig(
                    os.path.join(figureFolder, f"{target}_dis_all_cyc.png")
                )
            elif charging and cc:
                plt.savefig(
                    os.path.join(figureFolder, f"{target}_chcc_all_cyc.png")
                )
            elif charging and cccv:
                plt.savefig(
                    os.path.join(figureFolder, f"{target}_chcccv_all_cyc.png")
                )

            plt.close()


def get_neighbor_cycles(cycle_id, n_window_min=5, n_window_max=10):
    neighbors = []
    if cycle_id == 0:
        for i in range(n_window_min):
            neighbors.append(cycle_id + i)
    elif cycle_id <= 125:
        for i in range(n_window_min):
            neighbors.append(cycle_id - n_window_min + 1 + i)
    elif cycle_id > 125:
        for i in range(n_window_max):
            neighbors.append(cycle_id - n_window_max + 1 + i)
    return neighbors


def obs_filename(
    obsFolder, cell_id, cycle_id, nave, cyc_mode="discharge", target="phis_c"
):
    if cyc_mode == "discharge":
        pref = f"{target}_dis"
    elif cyc_mode == "chargecc":
        pref = f"{target}_chcc"
    else:
        sys.exit(f"cyc_mode = {cyc_mode} not recognized")
    return os.path.join(
        obsFolder,
        f"cell{cell_id}",
        f"{pref}_cyc_{cycle_id}_ave{nave}.npz",
    )


def load_observation_data(filename, step_size, verbose=True):
    obs_data = {}
    data_t = {}
    data_t_dQ_dV = {}
    data_t_dV_dQ = {}
    data_t_dV_dQ_p = {}
    data_t_dV_dQ_m = {}
    data_phis_c = {}
    data_dV_dQ_x = {}
    data_dV_dQ_y = {}
    data_dQ_dV_x = {}
    data_dQ_dV_y = {}
    deg_param_truth = {}
    for key in filename:
        deg_param_truth[key] = None
        data_t[key] = np.load(filename[key]["phis_c"], allow_pickle=True)[
            "t"
        ].astype("float64")
        if verbose:
            print("loading ", filename[key]["phis_c"])
        data_phis_c[key] = np.load(filename[key]["phis_c"], allow_pickle=True)[
            "phis_c"
        ].astype("float64")
        if verbose:
            print("loading ", filename[key]["dV_dQ"])
        data_t_dV_dQ[key] = np.load(filename[key]["dV_dQ"], allow_pickle=True)[
            "t"
        ].astype("float64")
        data_t_dV_dQ_p[key] = (
            np.load(filename[key]["dV_dQ"], allow_pickle=True)["t"].astype(
                "float64"
            )
            + step_size
        )
        data_t_dV_dQ_m[key] = (
            np.load(filename[key]["dV_dQ"], allow_pickle=True)["t"].astype(
                "float64"
            )
            - step_size
        )
        data_dV_dQ_x[key] = np.load(filename[key]["dV_dQ"], allow_pickle=True)[
            "Q"
        ].astype("float64")
        data_dV_dQ_y[key] = np.load(filename[key]["dV_dQ"], allow_pickle=True)[
            "dV_dQ"
        ].astype("float64")
        if verbose:
            print("loading ", filename[key]["dQ_dV"])
        data_t_dQ_dV[key] = np.load(filename[key]["dQ_dV"], allow_pickle=True)[
            "t"
        ].astype("float64")
        data_dQ_dV_x[key] = np.load(filename[key]["dQ_dV"], allow_pickle=True)[
            "V"
        ].astype("float64")
        data_dQ_dV_y[key] = np.load(filename[key]["dQ_dV"], allow_pickle=True)[
            "dQ_dV"
        ].astype("float64")

    # Balance data if need be
    if len(data_t) > 1:
        minsamp = np.inf
        maxsamp = -np.inf
        for key in filename:
            if len(data_t[key]) < minsamp:
                minsamp = len(data_t[key])
            if len(data_t[key]) > maxsamp:
                maxsamp = len(data_t[key])
        for key in filename:
            ratio = len(data_t[key]) // minsamp
            if len(data_t[key]) > minsamp:
                data_t[key] = data_t[key][::ratio]
                data_phis_c[key] = data_phis_c[key][::ratio]
                data_t_dV_dQ[key] = data_t_dV_dQ[key][::ratio]
                data_t_dV_dQ_p[key] = data_t_dV_dQ_p[key][::ratio]
                data_t_dV_dQ_m[key] = data_t_dV_dQ_m[key][::ratio]
                data_dV_dQ_x[key] = data_dV_dQ_x[key][::ratio]
                data_dV_dQ_y[key] = data_dV_dQ_y[key][::ratio]
                data_t_dQ_dV[key] = data_t_dQ_dV[key][::ratio]
                data_dQ_dV_x[key] = data_dQ_dV_x[key][::ratio]
                data_dQ_dV_y[key] = data_dQ_dV_y[key][::ratio]

    return (
        data_t,
        data_phis_c,
        data_dV_dQ_x,
        data_dV_dQ_y,
        data_dQ_dV_x,
        data_dQ_dV_y,
        data_t_dQ_dV,
        data_t_dV_dQ,
        data_t_dV_dQ_m,
        data_t_dV_dQ_p,
        deg_param_truth,
    )


def collect_observation_files(args_cal, nave):
    cyc_mode = args_cal["cyc_mode"]
    obsFolder = args_cal["obsFolder"]
    cell_id = args_cal["cell_id"]
    cycle_id = args_cal["cycle_id"]
    filename = {}
    if cyc_mode == "discharge" or cyc_mode == "discharge-chargecc":
        filename["discharge"] = {}
        filename["discharge"]["phis_c"] = obs_filename(
            obsFolder,
            cell_id,
            cycle_id,
            nave,
            cyc_mode="discharge",
            target="phis_c",
        )
        filename["discharge"]["dQ_dV"] = obs_filename(
            obsFolder,
            cell_id,
            cycle_id,
            nave,
            cyc_mode="discharge",
            target="dQ_dV",
        )
        filename["discharge"]["dV_dQ"] = obs_filename(
            obsFolder,
            cell_id,
            cycle_id,
            nave,
            cyc_mode="discharge",
            target="dV_dQ",
        )
        # samp_key = 'discharge'
    if cyc_mode == "chargecc" or cyc_mode == "discharge-chargecc":
        filename["chargecc"] = {}
        filename["chargecc"]["phis_c"] = obs_filename(
            obsFolder,
            cell_id,
            cycle_id,
            nave,
            cyc_mode="chargecc",
            target="phis_c",
        )
        filename["chargecc"]["dQ_dV"] = obs_filename(
            obsFolder,
            cell_id,
            cycle_id,
            nave,
            cyc_mode="chargecc",
            target="dQ_dV",
        )
        filename["chargecc"]["dV_dQ"] = obs_filename(
            obsFolder,
            cell_id,
            cycle_id,
            nave,
            cyc_mode="chargecc",
            target="dV_dQ",
        )
        # samp_key = 'chargecc'

    return filename


def make_target_data(
    cyc_mode, target_list, data_phis_c, data_dV_dQ_y, data_dQ_dV_y
):
    if cyc_mode.lower() == "discharge":
        if (
            ("phis_c" in target_list)
            and ("dV_dQ" not in target_list)
            and ("dQ_dV" not in target_list)
        ):
            data_tar = jnp.hstack((jnp.array(data_phis_c["discharge"]),))
        elif (
            ("phis_c" not in target_list)
            and ("dV_dQ" in target_list)
            and ("dQ_dV" not in target_list)
        ):
            data_tar = jnp.hstack((jnp.array(data_dV_dQ_y["discharge"]),))
        elif (
            ("phis_c" not in target_list)
            and ("dV_dQ" not in target_list)
            and ("dQ_dV" in target_list)
        ):
            data_tar = jnp.hstack((jnp.array(data_dQ_dV_y["discharge"]),))
        elif (
            ("phis_c" in target_list)
            and ("dV_dQ" in target_list)
            and ("dQ_dV" not in target_list)
        ):
            data_tar = jnp.hstack(
                (
                    jnp.array(data_phis_c["discharge"]),
                    jnp.array(data_dV_dQ_y["discharge"]),
                )
            )
        elif (
            ("phis_c" in target_list)
            and ("dV_dQ" not in target_list)
            and ("dQ_dV" in target_list)
        ):
            data_tar = jnp.hstack(
                (
                    jnp.array(data_phis_c["discharge"]),
                    jnp.array(data_dQ_dV_y["discharge"]),
                )
            )
        elif (
            ("phis_c" not in target_list)
            and ("dV_dQ" in target_list)
            and ("dQ_dV" in target_list)
        ):
            data_tar = jnp.hstack(
                (
                    jnp.array(data_dV_dQ_y["discharge"]),
                    jnp.array(data_dQ_dV_y["discharge"]),
                )
            )
        elif (
            ("phis_c" in target_list)
            and ("dV_dQ" in target_list)
            and ("dQ_dV" in target_list)
        ):
            data_tar = jnp.hstack(
                (
                    jnp.array(data_phis_c["discharge"]),
                    jnp.array(data_dV_dQ_y["discharge"]),
                    jnp.array(data_dQ_dV_y["discharge"]),
                )
            )
    elif cyc_mode.lower() == "chargecc":
        if (
            ("phis_c" in target_list)
            and ("dV_dQ" not in target_list)
            and ("dQ_dV" not in target_list)
        ):
            data_tar = jnp.hstack((jnp.array(data_phis_c["chargecc"]),))
        elif (
            ("phis_c" not in target_list)
            and ("dV_dQ" in target_list)
            and ("dQ_dV" not in target_list)
        ):
            data_tar = jnp.hstack((jnp.array(data_dV_dQ_y["chargecc"]),))
        elif (
            ("phis_c" not in target_list)
            and ("dV_dQ" not in target_list)
            and ("dQ_dV" in target_list)
        ):
            data_tar = jnp.hstack((jnp.array(data_dQ_dV_y["chargecc"]),))
        elif (
            ("phis_c" in target_list)
            and ("dV_dQ" in target_list)
            and ("dQ_dV" not in target_list)
        ):
            data_tar = jnp.hstack(
                (
                    jnp.array(data_phis_c["chargecc"]),
                    jnp.array(data_dV_dQ_y["chargecc"]),
                )
            )
        elif (
            ("phis_c" in target_list)
            and ("dV_dQ" not in target_list)
            and ("dQ_dV" in target_list)
        ):
            data_tar = jnp.hstack(
                (
                    jnp.array(data_phis_c["chargecc"]),
                    jnp.array(data_dQ_dV_y["chargecc"]),
                )
            )
        elif (
            ("phis_c" not in target_list)
            and ("dV_dQ" in target_list)
            and ("dQ_dV" in target_list)
        ):
            data_tar = jnp.hstack(
                (
                    jnp.array(data_dV_dQ_y["chargecc"]),
                    jnp.array(data_dQ_dV_y["chargecc"]),
                )
            )
        elif (
            ("phis_c" in target_list)
            and ("dV_dQ" in target_list)
            and ("dQ_dV" in target_list)
        ):
            data_tar = jnp.hstack(
                (
                    jnp.array(data_phis_c["chargecc"]),
                    jnp.array(data_dV_dQ_y["chargecc"]),
                    jnp.array(data_dQ_dV_y["chargecc"]),
                )
            )
    elif cyc_mode.lower() == "discharge-chargecc":

        if (
            ("phis_c" in target_list)
            and ("dV_dQ" not in target_list)
            and ("dQ_dV" not in target_list)
        ):
            data_tar = jnp.hstack(
                (
                    jnp.array(data_phis_c["discharge"]),
                    jnp.array(data_phis_c["chargecc"]),
                )
            )
        elif (
            ("phis_c" not in target_list)
            and ("dV_dQ" in target_list)
            and ("dQ_dV" not in target_list)
        ):
            data_tar = jnp.hstack(
                (
                    jnp.array(data_dV_dQ_y["discharge"]),
                    jnp.array(data_dV_dQ_y["chargecc"]),
                )
            )
        elif (
            ("phis_c" not in target_list)
            and ("dV_dQ" not in target_list)
            and ("dQ_dV" in target_list)
        ):
            data_tar = jnp.hstack(
                (
                    jnp.array(data_dQ_dV_y["discharge"]),
                    jnp.array(data_dQ_dV_y["chargecc"]),
                )
            )
        elif (
            ("phis_c" in target_list)
            and ("dV_dQ" in target_list)
            and ("dQ_dV" not in target_list)
        ):
            data_tar = jnp.hstack(
                (
                    jnp.array(data_phis_c["discharge"]),
                    jnp.array(data_phis_c["chargecc"]),
                    jnp.array(data_dV_dQ_y["discharge"]),
                    jnp.array(data_dV_dQ_y["chargecc"]),
                )
            )
        elif (
            ("phis_c" in target_list)
            and ("dV_dQ" not in target_list)
            and ("dQ_dV" in target_list)
        ):
            data_tar = jnp.hstack(
                (
                    jnp.array(data_phis_c["discharge"]),
                    jnp.array(data_phis_c["chargecc"]),
                    jnp.array(data_dQ_dV_y["discharge"]),
                    jnp.array(data_dQ_dV_y["chargecc"]),
                )
            )
        elif (
            ("phis_c" not in target_list)
            and ("dV_dQ" in target_list)
            and ("dQ_dV" in target_list)
        ):
            data_tar = jnp.hstack(
                (
                    jnp.array(data_dV_dQ_y["discharge"]),
                    jnp.array(data_dV_dQ_y["chargecc"]),
                    jnp.array(data_dQ_dV_y["discharge"]),
                    jnp.array(data_dQ_dV_y["chargecc"]),
                )
            )
        elif (
            ("phis_c" in target_list)
            and ("dV_dQ" in target_list)
            and ("dQ_dV" in target_list)
        ):
            data_tar = jnp.hstack(
                (
                    jnp.array(data_phis_c["discharge"]),
                    jnp.array(data_phis_c["chargecc"]),
                    jnp.array(data_dV_dQ_y["discharge"]),
                    jnp.array(data_dV_dQ_y["chargecc"]),
                    jnp.array(data_dQ_dV_y["discharge"]),
                    jnp.array(data_dQ_dV_y["chargecc"]),
                )
            )
    return data_tar


def make_error_data(
    y_err, cyc_mode, target_list, data_phis_c, data_dV_dQ_y, data_dQ_dV_y
):
    if isinstance(y_err, float):
        val = y_err
        y_err = {}
        y_err["dis_phis_c"] = val
        y_err["dis_dV_dQ"] = val
        y_err["dis_dQ_dV"] = val
        y_err["chcc_phis_c"] = val
        y_err["chcc_dV_dQ"] = val
        y_err["chcc_dQ_dV"] = val
    if cyc_mode.lower() == "discharge":
        if (
            ("phis_c" in target_list)
            and ("dV_dQ" not in target_list)
            and ("dQ_dV" not in target_list)
        ):
            data_err = jnp.hstack(
                (
                    y_err["dis_phis_c"]
                    * jnp.ones(data_phis_c["discharge"].shape),
                )
            )
        elif (
            ("phis_c" not in target_list)
            and ("dV_dQ" in target_list)
            and ("dQ_dV" not in target_list)
        ):
            data_err = jnp.hstack(
                (
                    y_err["dis_dV_dQ"]
                    * jnp.ones(data_dV_dQ_y["discharge"].shape),
                )
            )
        elif (
            ("phis_c" not in target_list)
            and ("dV_dQ" not in target_list)
            and ("dQ_dV" in target_list)
        ):
            data_err = jnp.hstack(
                (
                    y_err["dis_dQ_dV"]
                    * jnp.ones(data_dQ_dV_y["discharge"].shape),
                )
            )
        elif (
            ("phis_c" in target_list)
            and ("dV_dQ" in target_list)
            and ("dQ_dV" not in target_list)
        ):
            data_err = jnp.hstack(
                (
                    y_err["dis_phis_c"]
                    * jnp.ones(data_phis_c["discharge"].shape),
                    y_err["dis_dV_dQ"]
                    * jnp.ones(data_dV_dQ_y["discharge"].shape),
                )
            )
        elif (
            ("phis_c" in target_list)
            and ("dV_dQ" not in target_list)
            and ("dQ_dV" in target_list)
        ):
            data_err = jnp.hstack(
                (
                    y_err["dis_phis_c"]
                    * jnp.ones(data_phis_c["discharge"].shape),
                    y_err["dis_dQ_dV"]
                    * jnp.ones(data_dQ_dV_y["discharge"].shape),
                )
            )
        elif (
            ("phis_c" not in target_list)
            and ("dV_dQ" in target_list)
            and ("dQ_dV" in target_list)
        ):
            data_err = jnp.hstack(
                (
                    y_err["dis_dV_dQ"]
                    * jnp.ones(data_dV_dQ_y["discharge"].shape),
                    y_err["dis_dQ_dV"]
                    * jnp.ones(data_dQ_dV_y["discharge"].shape),
                )
            )
        elif (
            ("phis_c" in target_list)
            and ("dV_dQ" in target_list)
            and ("dQ_dV" in target_list)
        ):
            data_err = jnp.hstack(
                (
                    y_err["dis_phis_c"]
                    * jnp.ones(data_phis_c["discharge"].shape),
                    y_err["dis_dV_dQ"]
                    * jnp.ones(data_dV_dQ_y["discharge"].shape),
                    y_err["dis_dQ_dV"]
                    * jnp.ones(data_dQ_dV_y["discharge"].shape),
                )
            )
    elif cyc_mode.lower() == "chargecc":
        if (
            ("phis_c" in target_list)
            and ("dV_dQ" not in target_list)
            and ("dQ_dV" not in target_list)
        ):
            data_err = y_err["chcc_phis_c"] * jnp.ones(
                data_phis_c["chargecc"].shape
            )
        elif (
            ("phis_c" not in target_list)
            and ("dV_dQ" in target_list)
            and ("dQ_dV" not in target_list)
        ):
            data_err = jnp.hstack(
                (
                    y_err["chcc_dV_dQ"]
                    * jnp.ones(data_dV_dQ_y["chargecc"].shape),
                )
            )
        elif (
            ("phis_c" not in target_list)
            and ("dV_dQ" not in target_list)
            and ("dQ_dV" in target_list)
        ):
            data_err = jnp.hstack(
                (
                    y_err["chcc_dQ_dV"]
                    * jnp.ones(data_dQ_dV_y["chargecc"].shape),
                )
            )
        elif (
            ("phis_c" in target_list)
            and ("dV_dQ" in target_list)
            and ("dQ_dV" not in target_list)
        ):
            data_err = jnp.hstack(
                (
                    y_err["chcc_phis_c"]
                    * jnp.ones(data_phis_c["chargecc"].shape),
                    y_err["chcc_dV_dQ"]
                    * jnp.ones(data_dV_dQ_y["chargecc"].shape),
                )
            )
        elif (
            ("phis_c" in target_list)
            and ("dV_dQ" not in target_list)
            and ("dQ_dV" in target_list)
        ):
            data_err = jnp.hstack(
                (
                    y_err["chcc_phis_c"]
                    * jnp.ones(data_phis_c["chargecc"].shape),
                    y_err["chcc_dQ_dV"]
                    * jnp.ones(data_dQ_dV_y["chargecc"].shape),
                )
            )
        elif (
            ("phis_c" not in target_list)
            and ("dV_dQ" in target_list)
            and ("dQ_dV" in target_list)
        ):
            data_err = jnp.hstack(
                (
                    y_err["chcc_dV_dQ"]
                    * jnp.ones(data_dV_dQ_y["chargecc"].shape),
                    y_err["chcc_dQ_dV"]
                    * jnp.ones(data_dQ_dV_y["chargecc"].shape),
                )
            )
        elif (
            ("phis_c" in target_list)
            and ("dV_dQ" in target_list)
            and ("dQ_dV" in target_list)
        ):
            data_err = jnp.hstack(
                (
                    y_err["chcc_phis_c"]
                    * jnp.ones(data_phis_c["chargecc"].shape),
                    y_err["chcc_dV_dQ"]
                    * jnp.ones(data_dV_dQ_y["chargecc"].shape),
                    y_err["chcc_dQ_dV"]
                    * jnp.ones(data_dQ_dV_y["chargecc"].shape),
                )
            )
    elif cyc_mode.lower() == "discharge-chargecc":

        if (
            ("phis_c" in target_list)
            and ("dV_dQ" not in target_list)
            and ("dQ_dV" not in target_list)
        ):
            data_err = jnp.hstack(
                (
                    y_err["dis_phis_c"]
                    * jnp.ones(data_phis_c["discharge"].shape),
                    y_err["chcc_phis_c"]
                    * jnp.ones(data_phis_c["chargecc"].shape),
                )
            )
        elif (
            ("phis_c" not in target_list)
            and ("dV_dQ" in target_list)
            and ("dQ_dV" not in target_list)
        ):
            data_err = jnp.hstack(
                (
                    y_err["dis_dV_dQ"]
                    * jnp.ones(data_dV_dQ_y["discharge"].shape),
                    y_err["chcc_dV_dQ"]
                    * jnp.ones(data_dV_dQ_y["chargecc"].shape),
                )
            )
        elif (
            ("phis_c" not in target_list)
            and ("dV_dQ" not in target_list)
            and ("dQ_dV" in target_list)
        ):
            data_err = jnp.hstack(
                (
                    y_err["dis_dQ_dV"]
                    * jnp.ones(data_dQ_dV_y["discharge"].shape),
                    y_err["chcc_dQ_dV"]
                    * jnp.ones(data_dQ_dV_y["chargecc"].shape),
                )
            )
        elif (
            ("phis_c" in target_list)
            and ("dV_dQ" in target_list)
            and ("dQ_dV" not in target_list)
        ):
            data_err = jnp.hstack(
                (
                    y_err["dis_phis_c"]
                    * jnp.ones(data_phis_c["discharge"].shape),
                    y_err["chcc_phis_c"]
                    * jnp.ones(data_phis_c["chargecc"].shape),
                    y_err["dis_dV_dQ"]
                    * jnp.ones(data_dV_dQ_y["discharge"].shape),
                    y_err["chcc_dV_dQ"]
                    * jnp.ones(data_dV_dQ_y["chargecc"].shape),
                )
            )
        elif (
            ("phis_c" in target_list)
            and ("dV_dQ" not in target_list)
            and ("dQ_dV" in target_list)
        ):
            data_err = jnp.hstack(
                (
                    y_err["dis_phis_c"]
                    * jnp.ones(data_phis_c["discharge"].shape),
                    y_err["chcc_phis_c"]
                    * jnp.ones(data_phis_c["chargecc"].shape),
                    y_err["dis_dQ_dV"]
                    * jnp.ones(data_dQ_dV_y["discharge"].shape),
                    y_err["chcc_dQ_dV"]
                    * jnp.ones(data_dQ_dV_y["chargecc"].shape),
                )
            )
        elif (
            ("phis_c" not in target_list)
            and ("dV_dQ" in target_list)
            and ("dQ_dV" in target_list)
        ):
            data_err = jnp.hstack(
                (
                    y_err["dis_dV_dQ"]
                    * jnp.ones(data_dV_dQ_y["discharge"].shape),
                    y_err["chcc_dV_dQ"]
                    * jnp.ones(data_dV_dQ_y["chargecc"].shape),
                    y_err["dis_dQ_dV"]
                    * jnp.ones(data_dQ_dV_y["discharge"].shape),
                    y_err["chcc_dQ_dV"]
                    * jnp.ones(data_dQ_dV_y["chargecc"].shape),
                )
            )
        elif (
            ("phis_c" in target_list)
            and ("dV_dQ" in target_list)
            and ("dQ_dV" in target_list)
        ):
            data_err = jnp.hstack(
                (
                    y_err["dis_phis_c"]
                    * jnp.ones(data_phis_c["discharge"].shape),
                    y_err["chcc_phis_c"]
                    * jnp.ones(data_phis_c["chargecc"].shape),
                    y_err["dis_dV_dQ"]
                    * jnp.ones(data_dV_dQ_y["discharge"].shape),
                    y_err["chcc_dV_dQ"]
                    * jnp.ones(data_dV_dQ_y["chargecc"].shape),
                    y_err["dis_dQ_dV"]
                    * jnp.ones(data_dQ_dV_y["discharge"].shape),
                    y_err["chcc_dQ_dV"]
                    * jnp.ones(data_dQ_dV_y["chargecc"].shape),
                )
            )
    return data_err


def read_sigma_from_file(filename):
    with open(filename, "r+") as f:
        lines = f.readlines()
    sigma = float(lines[0].split(" +/- ")[0])
    return sigma


def sigma_from_file(cyc_mode, target_list, cell_id, cycle_id, folder):
    y_err = {}
    if cyc_mode.lower() in ["discharge", "discharge-chargecc"]:
        if "phis_c" in target_list:
            filename = os.path.join(
                folder,
                f"phis_c_discharge_cell{cell_id}_cyc{cycle_id}_sigma.txt",
            )
            y_err["dis_phis_c"] = read_sigma_from_file(filename)
        if "dV_dQ" in target_list:
            filename = os.path.join(
                folder,
                f"dV_dQ_discharge_cell{cell_id}_cyc{cycle_id}_sigma.txt",
            )
            y_err["dis_dV_dQ"] = read_sigma_from_file(filename)
        if "dQ_dV" in target_list:
            filename = os.path.join(
                folder,
                f"dQ_dV_discharge_cell{cell_id}_cyc{cycle_id}_sigma.txt",
            )
            y_err["dis_dQ_dV"] = read_sigma_from_file(filename)

    if cyc_mode.lower() in ["chargecc", "discharge-chargecc"]:
        if "phis_c" in target_list:
            filename = os.path.join(
                folder,
                f"phis_c_chargecc_cell{cell_id}_cyc{cycle_id}_sigma.txt",
            )
            y_err["chcc_phis_c"] = read_sigma_from_file(filename)
        if "dV_dQ" in target_list:
            filename = os.path.join(
                folder, f"dV_dQ_chargecc_cell{cell_id}_cyc{cycle_id}_sigma.txt"
            )
            y_err["chcc_dV_dQ"] = read_sigma_from_file(filename)
        if "dQ_dV" in target_list:
            filename = os.path.join(
                folder, f"dQ_dV_chargecc_cell{cell_id}_cyc{cycle_id}_sigma.txt"
            )
            y_err["chcc_dQ_dV"] = read_sigma_from_file(filename)
    print("y_err = ", y_err)
    if cyc_mode.lower() in ["discharge", "discharge-chargecc"]:
        if not "dis_phis_c" in y_err:
            try:
                y_err["dis_phis_c"] = y_err["dis_dV_dQ"] / 4
            except KeyError:
                y_err["dis_phis_c"] = y_err["dis_dQ_dV"] / 100
        if not "dis_dV_dQ" in y_err:
            try:
                y_err["dis_dV_dQ"] = y_err["dis_phis_c"] * 4
            except KeyError:
                y_err["dis_dV_dQ"] = y_err["dis_dQ_dV"] * 4 / 100
        if not "dis_dQ_dV" in y_err:
            try:
                y_err["dis_dQ_dV"] = y_err["dis_phis_c"] * 100
            except KeyError:
                y_err["dis_dQ_dV"] = y_err["dis_dV_dQ"] * 100 / 4

    if cyc_mode.lower() in ["chargecc", "discharge-chargecc"]:
        if not "chcc_phis_c" in y_err:
            try:
                y_err["chcc_phis_c"] = y_err["chcc_dV_dQ"] / 4
            except KeyError:
                y_err["chcc_phis_c"] = y_err["chcc_dQ_dV"] / 100
        if not "chcc_dV_dQ" in y_err:
            try:
                y_err["chcc_dV_dQ"] = y_err["chcc_phis_c"] * 4
            except KeyError:
                y_err["chcc_dV_dQ"] = y_err["chcc_dQ_dV"] * 4 / 100
        if not "chcc_dQ_dV" in y_err:
            try:
                y_err["chcc_dQ_dV"] = y_err["chcc_phis_c"] * 100
            except KeyError:
                y_err["chcc_dQ_dV"] = y_err["chcc_dV_dQ"] * 100 / 4
    print("y_err_completed = ", y_err)
    return y_err


def perturb_val_dict(val_dict, sim_params_dict):
    factor = 0.001
    if "discharge" in sim_params_dict:
        for ipar, name in enumerate(
            sim_params_dict["discharge"]["deg_param_names"]
        ):
            amp = factor * (
                sim_params_dict["discharge"]["deg_" + name + "_max"]
                - sim_params_dict["discharge"]["deg_" + name + "_min"]
            )
            pert = np.random.uniform(-amp / 2, amp / 2)
            val_dict[name] += pert
            val_dict[name] = np.clip(
                val_dict[name],
                sim_params_dict["discharge"]["deg_" + name + "_min"]
                + amp / 100,
                sim_params_dict["discharge"]["deg_" + name + "_max"]
                - amp / 100,
            )
    if "chargecc" in sim_params_dict and "discharge" not in sim_params_dict:
        for ipar, name in enumerate(
            sim_params_dict["chargecc"]["deg_param_names"]
        ):
            amp = factor * (
                sim_params_dict["chargecc"]["deg_" + name + "_max"]
                - sim_params_dict["chargecc"]["deg_" + name + "_min"]
            )
            pert = np.random.uniform(-amp / 2, amp / 2)
            val_dict[name] += pert
            val_dict[name] = np.clip(
                val_dict[name],
                sim_params_dict["chargecc"]["deg_" + name + "_min"]
                + amp / 100,
                sim_params_dict["chargecc"]["deg_" + name + "_max"]
                - amp / 100,
            )
    if "chargecc" in sim_params_dict and "discharge" in sim_params_dict:
        val_dict_name = ["x0_a_charge", "x0_c_charge"]
        par_name = ["x0_a", "x0_c"]
        for pname, vname in zip(par_name, val_dict_name):
            amp = factor * (
                sim_params_dict["chargecc"]["deg_" + pname + "_max"]
                - sim_params_dict["chargecc"]["deg_" + pname + "_min"]
            )
            pert = np.random.uniform(-amp / 2, amp / 2)
            val_dict[vname] += pert
            val_dict[vname] = np.clip(
                val_dict[vname],
                sim_params_dict["chargecc"]["deg_" + pname + "_min"]
                + amp / 100,
                sim_params_dict["chargecc"]["deg_" + pname + "_max"]
                - amp / 100,
            )
    return val_dict


def mcmc_iter(
    y_err=0.1,
    mcmc_method="HMC",
    cyc_mode="discharge",
    num_chains=None,
    cal_sigma=False,
    read_sigma=False,
    min_sigma=None,
    max_sigma=None,
    nn_dict=None,
    target_list=None,
    data_phis_c=None,
    data_dV_dQ_y=None,
    data_dQ_dV_y=None,
    jax_func_dict=None,
    jax_params_dict=None,
    forward_dict=None,
    num_warmup=None,
    num_samples=None,
    save_sigma=False,
    cell_id=None,
    cycle_id=None,
    cons_LLI=False,
    variable_delete=None,
):
    rng_key = jax.random.PRNGKey(0)
    rng_key, rng_key_ = jax.random.split(rng_key)
    # Guess

    theta = []
    if cyc_mode == "discharge":
        for ipar, name in enumerate(
            sim_params_dict["discharge"]["deg_param_names"]
        ):
            theta.append(
                np.random.uniform(
                    sim_params_dict["discharge"]["deg_" + name + "_min"],
                    sim_params_dict["discharge"]["deg_" + name + "_max"],
                )
            )
    if cyc_mode == "chargecc":
        for ipar, name in enumerate(
            sim_params_dict["chargecc"]["deg_param_names"]
        ):
            theta.append(
                np.random.uniform(
                    sim_params_dict["chargecc"]["deg_" + name + "_min"],
                    sim_params_dict["chargecc"]["deg_" + name + "_max"],
                )
            )
    if cyc_mode == "discharge-chargecc":
        for ipar, name in enumerate(
            sim_params_dict["discharge"]["deg_param_names"]
        ):
            theta.append(
                np.random.uniform(
                    sim_params_dict["discharge"]["deg_" + name + "_min"],
                    sim_params_dict["discharge"]["deg_" + name + "_max"],
                )
            )
        theta.append(
            np.random.uniform(
                sim_params_dict["chargecc"]["deg_x0_a_min"],
                sim_params_dict["chargecc"]["deg_x0_a_max"],
            )
        )
        theta.append(
            np.random.uniform(
                sim_params_dict["chargecc"]["deg_x0_c_min"],
                sim_params_dict["chargecc"]["deg_x0_c_max"],
            )
        )

    if cal_sigma and read_sigma:
        logger.error("Cal sigma OR read sigma")
        sys.exit()
    if cal_sigma:
        theta.append(np.random.uniform(min_sigma, max_sigma))
    if read_sigma:
        y_err = sigma_from_file(
            cyc_mode=cyc_mode,
            target_list=target_list,
            cell_id=cell_id,
            cycle_id=cycle_id,
            folder="data_sigma",
        )
    if cyc_mode.lower() == "discharge":
        if cal_sigma:
            bayes_step = bayes_step_discharge_sigma
        elif read_sigma:
            bayes_step = bayes_step_discharge_read_sigma
        else:
            bayes_step = bayes_step_discharge
    elif cyc_mode.lower() == "chargecc":
        if cal_sigma:
            bayes_step = bayes_step_chargecc_sigma
        elif read_sigma:
            bayes_step = bayes_step_chargecc_read_sigma
        else:
            bayes_step = bayes_step_chargecc
    elif cyc_mode.lower() == "discharge-chargecc":
        if cal_sigma:
            bayes_step = bayes_step_discharge_chargecc_sigma
        elif read_sigma:
            bayes_step = bayes_step_discharge_chargecc_read_sigma
        else:
            bayes_step = bayes_step_discharge_chargecc
    data_tar = make_target_data(
        cyc_mode, target_list, data_phis_c, data_dV_dQ_y, data_dQ_dV_y
    )
    data_err = make_error_data(
        y_err, cyc_mode, target_list, data_phis_c, data_dV_dQ_y, data_dQ_dV_y
    )
    factor = 0.1
    while True:

        try:
            if save_sigma:
                num_warmup = num_warmup // 2
                num_samples = num_samples // 2
            # Hamiltonian Monte Carlo (HMC) with no u turn sampling (NUTS)
            if mcmc_method.lower() == "hmc":
                init_val = False
                if cyc_mode == "discharge":
                    if cell_id == 30:
                        init_val = True
                        val_dict = {
                            "i0_a": 3.88,
                            "ds_c": 9.41,
                            "x0_a": 0.8,
                            "x0_c": 1.15,
                            "i0_c": 0.37,
                            "eps_s_c_am": 0.92,
                        }
                    elif cell_id == 31:
                        init_val = True
                        val_dict = {
                            "i0_a": 3.87,
                            "ds_c": 9.44,
                            "x0_a": 0.81,
                            "x0_c": 1.14,
                            "i0_c": 0.51,
                            "eps_s_c_am": 0.92,
                        }
                    elif cell_id == 32:
                        init_val = True
                        val_dict = {
                            "i0_a": 3.84,
                            "ds_c": 9.34,
                            "x0_a": 0.78,
                            "x0_c": 1.18,
                            "i0_c": 0.42,
                            "eps_s_c_am": 0.91,
                        }
                    elif cell_id == 35:
                        init_val = True
                        val_dict = {
                            "i0_a": 3.9,
                            "ds_c": 9.56,
                            "x0_a": 0.78,
                            "x0_c": 1.17,
                            "i0_c": 0.58,
                            "eps_s_c_am": 0.91,
                        }
                    elif cell_id == 39:
                        init_val = True
                        val_dict = {
                            "i0_a": 3.89,
                            "ds_c": 9.59,
                            "x0_a": 0.84,
                            "x0_c": 1.09,
                            "i0_c": 0.35,
                            "eps_s_c_am": 0.93,
                        }
                    elif cell_id == 42:
                        init_val = True
                        val_dict = {
                            "i0_a": 3.91,
                            "ds_c": 9.7,
                            "x0_a": 0.84,
                            "x0_c": 1.09,
                            "i0_c": 0.42,
                            "eps_s_c_am": 0.93,
                        }
                    if cal_sigma:
                        val_dict["sigma"] = (
                            min_sigma + (max_sigma - min_sigma) * factor
                        )
                if cyc_mode == "chargecc":
                    if cell_id == 30:
                        init_val = True
                        val_dict = {
                            "i0_a": 0.22,
                            "ds_c": 0.46,
                            "x0_a": 0.45,
                            "x0_c": 1.07,
                            "i0_c": 0.11,
                            "eps_s_c_am": 0.99,
                        }
                    if cell_id == 31:
                        init_val = True
                        val_dict = {
                            "i0_a": 0.26,
                            "ds_c": 0.51,
                            "x0_a": 0.49,
                            "x0_c": 1.07,
                            "i0_c": 0.11,
                            "eps_s_c_am": 0.99,
                        }
                    if cell_id == 32:
                        init_val = True
                        val_dict = {
                            "i0_a": 0.14,
                            "ds_c": 0.54,
                            "x0_a": 0.35,
                            "x0_c": 1.07,
                            "i0_c": 0.11,
                            "eps_s_c_am": 0.99,
                        }
                    if cell_id == 35:
                        init_val = True
                        val_dict = {
                            "i0_a": 0.15,
                            "ds_c": 0.58,
                            "x0_a": 0.38,
                            "x0_c": 1.07,
                            "i0_c": 0.11,
                            "eps_s_c_am": 0.99,
                        }
                    if cell_id == 39:
                        init_val = True
                        val_dict = {
                            "i0_a": 0.11,
                            "ds_c": 0.62,
                            "x0_a": 0.31,
                            "x0_c": 1.06,
                            "i0_c": 0.11,
                            "eps_s_c_am": 0.99,
                        }
                    if cell_id == 42:
                        init_val = True
                        val_dict = {
                            "i0_a": 0.11,
                            "ds_c": 0.68,
                            "x0_a": 0.31,
                            "x0_c": 1.03,
                            "i0_c": 0.11,
                            "eps_s_c_am": 0.99,
                        }
                    if cal_sigma:
                        val_dict["sigma"] = (
                            min_sigma + (max_sigma - min_sigma) * factor
                        )
                if cyc_mode == "discharge-chargecc":
                    if cell_id == 30:
                        init_val = True

                        # val_dict = {"i0_a": 0.11, "ds_c": 0.52, "x0_a":0.8, "x0_c":1.11, "i0_c":0.21, "eps_s_c":0.99, "x0_a_charge":0.48, "x0_c_charge":1.07}
                        val_dict = {
                            "i0_a": 0.1401,
                            "ds_c": 8.727,
                            "x0_a": 1.077,
                            "x0_c": 1.176,
                            "i0_c": 0.1011,
                            "eps_s_c_am": 0.9167,
                            "x0_a_charge": 0.3054,
                            "x0_c_charge": 1.047,
                        }
                    elif cell_id == 31:
                        init_val = True
                        val_dict = {
                            "i0_a": 0.2192,
                            "ds_c": 9.154,
                            "x0_a": 1.077,
                            "x0_c": 1.18,
                            "i0_c": 0.1005,
                            "eps_s_c_am": 0.9281,
                            "x0_a_charge": 0.343,
                            "x0_c_charge": 1.041,
                        }
                        # val_dict = {"i0_a": 0.11, "ds_c": 0.52, "x0_a":0.8, "x0_c":1.11, "i0_c":0.21, "eps_s_c":0.99, "x0_a_charge":0.48, "x0_c_charge":1.07}
                    elif cell_id == 32:
                        init_val = True
                        val_dict = {
                            "i0_a": 0.11,
                            "ds_c": 0.53,
                            "x0_a": 0.77,
                            "x0_c": 1.15,
                            "i0_c": 0.14,
                            "eps_s_c_am": 0.99,
                            "x0_a_charge": 0.35,
                            "x0_c_charge": 1.07,
                        }
                    elif cell_id == 35:
                        init_val = True
                        val_dict = {
                            "i0_a": 0.11,
                            "ds_c": 0.56,
                            "x0_a": 0.77,
                            "x0_c": 1.15,
                            "i0_c": 0.15,
                            "eps_s_c_am": 0.99,
                            "x0_a_charge": 0.38,
                            "x0_c_charge": 1.07,
                        }
                    elif cell_id == 39:
                        init_val = True
                        val_dict = {
                            "i0_a": 0.11,
                            "ds_c": 0.62,
                            "x0_a": 0.84,
                            "x0_c": 1.05,
                            "i0_c": 0.11,
                            "eps_s_c_am": 0.99,
                            "x0_a_charge": 0.31,
                            "x0_c_charge": 1.06,
                        }
                    elif cell_id == 42:
                        init_val = True
                        val_dict = {
                            "i0_a": 0.11,
                            "ds_c": 0.69,
                            "x0_a": 0.83,
                            "x0_c": 1.06,
                            "i0_c": 0.11,
                            "eps_s_c_am": 0.99,
                            "x0_a_charge": 0.31,
                            "x0_c_charge": 1.03,
                        }
                    if cal_sigma:
                        val_dict["sigma"] = (
                            min_sigma + (max_sigma - min_sigma) * factor
                        )
                if init_val:
                    val_dict = perturb_val_dict(val_dict, nn_dict)
                    init_strategy = init_to_value(values=val_dict)
                    step_size = 0.0001
                else:
                    init_strategy = None
                    step_size = 1
                if save_sigma:
                    kernel = NUTS(
                        bayes_step,
                        target_accept_prob=0.65,
                        max_tree_depth=(5, 9),
                        init_strategy=init_strategy,
                        step_size=step_size,
                    )
                else:
                    # kernel = NUTS(bayes_step, target_accept_prob=0.65, max_tree_depth=(6,10), init_strategy=init_strategy, step_size=step_size)
                    kernel = NUTS(
                        bayes_step,
                        target_accept_prob=0.65,
                        max_tree_depth=(6, 10),
                    )
                # kernel = NUTS(bayes_step, target_accept_prob=0.65, max_tree_depth=(5,10), step_size=0.01)
                # kernel = NUTS(bayes_step, target_accept_prob=0.75, max_tree_depth=(5,10))
                # kernel = NUTS(bayes_step, target_accept_prob=0.75, max_tree_depth=(5,10), init_strategy=init_strategy, step_size=0.01)
            elif mcmc_method.lower() == "sa":
                kernel = SA(bayes_step)
            else:
                sys.exit(f"MCMC method {mcmc_method} unrecognized")

            mcmc = MCMC(
                kernel,
                num_chains=num_chains,
                num_warmup=num_warmup,
                num_samples=num_samples,
                # jit_model_args=True,
            )
            mcmc.run(
                rng_key_,
                y=data_tar,
                y_err=data_err,
                min_sigma=min_sigma,
                max_sigma=max_sigma,
                nn_dict=nn_dict,
                jax_func_dict=jax_func_dict,
                jax_params_dict=jax_params_dict,
                target_list=target_list,
                cons_LLI=cons_LLI,
                variable_delete=variable_delete,
            )
            break
        except RuntimeError as err:
            print(err)
            factor += 0.1
            print(f"Failed, resampling init parameters with factor = {factor}")
            if factor > 1:
                raise ValueError
    mcmc.print_summary()

    # Draw samples
    mcmc_samples = mcmc.get_samples()
    labels = list(mcmc_samples.keys())
    nsamples = len(mcmc_samples[labels[0]])
    nparams = len(labels)
    np_mcmc_samples = np.zeros((nsamples, nparams))
    labels_np = []
    if cyc_mode.lower() == "discharge":
        labels_np += sim_params_dict["discharge"]["deg_param_names"]
    elif cyc_mode.lower() == "chargecc":
        labels_np += sim_params_dict["chargecc"]["deg_param_names"]
    elif cyc_mode.lower() == "discharge-chargecc":
        labels_np += sim_params_dict["discharge"]["deg_param_names"] + [
            "x0_a_charge",
            "x0_c_charge",
        ]
    if cal_sigma:
        labels_np += ["sigma"]
    for ilabel, label in enumerate(labels):
        nplabel = labels_np.index(label)
        np_mcmc_samples[:, nplabel] = np.array(mcmc_samples[label])

    # Uncertainty propagation
    nsamples = np_mcmc_samples.shape[0]
    realization_phis_c = {}
    realization_dV_dQ = {}
    realization_dQ_dV = {}

    if cyc_mode.lower() == "discharge":
        realization_phis_c["discharge"] = []
        realization_dV_dQ["discharge"] = []
        realization_dQ_dV["discharge"] = []
        for i in range(min(nsamples, 40)):
            if cal_sigma:
                phis_c, dV_dQ, dQ_dV = forward_dict["discharge"](
                    np_mcmc_samples[i, :-1]
                )
            else:
                phis_c, dV_dQ, dQ_dV = forward_dict["discharge"](
                    np_mcmc_samples[i, :]
                )
            realization_phis_c["discharge"].append(phis_c)
            realization_dV_dQ["discharge"].append(dV_dQ)
            realization_dQ_dV["discharge"].append(dQ_dV)
        realization_phis_c["discharge"] = np.array(
            realization_phis_c["discharge"]
        )
        realization_dV_dQ["discharge"] = np.array(
            realization_dV_dQ["discharge"]
        )
        realization_dQ_dV["discharge"] = np.array(
            realization_dQ_dV["discharge"]
        )
    if cyc_mode.lower() == "chargecc":
        realization_phis_c["chargecc"] = []
        realization_dV_dQ["chargecc"] = []
        realization_dQ_dV["chargecc"] = []
        for i in range(min(nsamples, 40)):
            if cal_sigma:
                phis_c, dV_dQ, dQ_dV = forward_dict["chargecc"](
                    np_mcmc_samples[i, :-1]
                )
            else:
                phis_c, dV_dQ, dQ_dV = forward_dict["chargecc"](
                    np_mcmc_samples[i, :]
                )
            realization_phis_c["chargecc"].append(phis_c)
            realization_dV_dQ["chargecc"].append(dV_dQ)
            realization_dQ_dV["chargecc"].append(dQ_dV)
        realization_phis_c["chargecc"] = np.array(
            realization_phis_c["chargecc"]
        )
        realization_dV_dQ["chargecc"] = np.array(realization_dV_dQ["chargecc"])
        realization_dQ_dV["chargecc"] = np.array(realization_dQ_dV["chargecc"])
    if cyc_mode.lower() == "discharge-chargecc":
        realization_phis_c["discharge"] = []
        realization_dV_dQ["discharge"] = []
        realization_dQ_dV["discharge"] = []
        for i in range(min(nsamples, 40)):
            if cal_sigma:
                phis_c, dV_dQ, dQ_dV = forward_dict["discharge"](
                    np_mcmc_samples[i, :-3]
                )
            else:
                phis_c, dV_dQ, dQ_dV = forward_dict["discharge"](
                    np_mcmc_samples[i, :-2]
                )
            realization_phis_c["discharge"].append(phis_c)
            realization_dV_dQ["discharge"].append(dV_dQ)
            realization_dQ_dV["discharge"].append(dQ_dV)
        realization_phis_c["discharge"] = np.array(
            realization_phis_c["discharge"]
        )
        realization_dV_dQ["discharge"] = np.array(
            realization_dV_dQ["discharge"]
        )
        realization_dQ_dV["discharge"] = np.array(
            realization_dQ_dV["discharge"]
        )

        realization_phis_c["chargecc"] = []
        realization_dV_dQ["chargecc"] = []
        realization_dQ_dV["chargecc"] = []
        indc = list(range(sim_params_dict["chargecc"]["n_deg_params"]))
        indc[sim_params_dict["chargecc"]["ind_deg_x0_a"]] = nn_dict[
            "chargecc"
        ].params["n_deg_params"]
        indc[sim_params_dict["chargecc"]["ind_deg_x0_c"]] = (
            sim_params_dict["chargecc"]["n_deg_params"] + 1
        )
        for i in range(min(nsamples, 40)):
            if cal_sigma:
                phis_c, dV_dQ, dQ_dV = forward_dict["chargecc"](
                    np_mcmc_samples[i, indc]
                )
            else:
                phis_c, dV_dQ, dQ_dV = forward_dict["chargecc"](
                    np_mcmc_samples[i, indc]
                )
            realization_phis_c["chargecc"].append(phis_c)
            realization_dV_dQ["chargecc"].append(dV_dQ)
            realization_dQ_dV["chargecc"].append(dQ_dV)
        realization_phis_c["chargecc"] = np.array(
            realization_phis_c["chargecc"]
        )
        realization_dV_dQ["chargecc"] = np.array(realization_dV_dQ["chargecc"])
        realization_dQ_dV["chargecc"] = np.array(realization_dQ_dV["chargecc"])

    min_real = {}
    max_real = {}
    for key in forward_dict:
        min_real[key] = np.min(realization_phis_c[key], axis=0)
        max_real[key] = np.max(realization_phis_c[key], axis=0)

    results = {
        "samples": np_mcmc_samples,
        "labels_np": labels_np,
        "labels": labels,
    }

    if not cal_sigma:
        true_m95 = {}
        true_p95 = {}
        for key in forward_dict:
            true_m95[key] = data_phis_c[key] - 2 * y_err
            true_p95[key] = data_phis_c[key] + 2 * y_err

        for key in forward_dict:
            if (
                np.amax(true_m95[key] - min_real[key]) > 0
                or np.amin(true_p95[key] - max_real[key]) < 0
            ):
                print(
                    f" Increase STD  {np.amax(true_m95[key] - min_real[key])} - {np.amin(true_p95[key] - max_real[key])}"
                )
                return False, results
        else:
            return True, results

    else:
        return True, results


def postprocess(
    args_cal,
    nn_dict=None,
    np_mcmc_samples=None,
    labels_np=None,
    nparams=None,
    data_t=None,
    data_phis_c=None,
    data_dV_dQ_y=None,
    data_dQ_dV_y=None,
    data_t_dQ_dV=None,
    data_t_dV_dQ=None,
    forward_dict=None,
    target_list=None,
    use_p2d=False,
):

    figureFolderRoot = "Figures_cal"
    figureFolderRoot = os.path.join(figureFolderRoot, "pouch")

    figureFolder = os.path.join(figureFolderRoot, f"cell{args_cal.cell_id}")
    log_dir = Path(figureFolder)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Corner plot
    ranges = []
    if args_cal.cyc_mode.lower() == "discharge":
        for ipar, name in enumerate(
            sim_params_dict["discharge"]["deg_param_names"]
        ):
            ranges.append(
                (
                    sim_params_dict["discharge"]["deg_" + name + "_min"],
                    sim_params_dict["discharge"]["deg_" + name + "_max"],
                )
            )
    if args_cal.cyc_mode.lower() == "chargecc":
        for ipar, name in enumerate(
            sim_params_dict["chargecc"]["deg_param_names"]
        ):
            ranges.append(
                (
                    sim_params_dict["chargecc"]["deg_" + name + "_min"],
                    sim_params_dict["chargecc"]["deg_" + name + "_max"],
                )
            )
    if args_cal.cyc_mode.lower() == "discharge-chargecc":
        for ipar, name in enumerate(
            sim_params_dict["discharge"]["deg_param_names"]
        ):
            ranges.append(
                (
                    sim_params_dict["discharge"]["deg_" + name + "_min"],
                    sim_params_dict["discharge"]["deg_" + name + "_max"],
                )
            )
        ranges.append(
            (
                sim_params_dict["chargecc"]["deg_x0_a_min"],
                sim_params_dict["chargecc"]["deg_x0_a_max"],
            )
        )
        ranges.append(
            (
                sim_params_dict["chargecc"]["deg_x0_c_min"],
                sim_params_dict["chargecc"]["deg_x0_c_max"],
            )
        )
    if args_cal.calibrate_one_sigma:
        ranges.append((args_cal.minsigma, args_cal.maxsigma))
    truths = None

    try:
        fig = corner.corner(
            np_mcmc_samples,
            truths=truths,
            labels=labels_np,
            bins=50,
            range=ranges,
        )
    except:
        breakpoint()
    plt.savefig(os.path.join(figureFolder, f"corner{args_cal.cycle_id}.png"))
    plt.close()

    # Convergence MCMC sequence
    fig, axes = plt.subplots(nparams, sharex=True)
    for i in range(nparams):
        ax = axes[i]
        ax.plot(np_mcmc_samples[:, i], "k", alpha=0.3, rasterized=True)
        ax.set_ylabel(labels_np[i])
    plt.savefig(os.path.join(figureFolder, f"seq{args_cal.cycle_id}.png"))
    plt.close()

    # Uncertainty propagation
    for key in forward_dict:
        if key == "discharge":
            maxt = np.amax(data_t["discharge"])
            ranget = np.reshape(
                np.linspace(
                    np.amin(data_t["discharge"] + maxt / 250),
                    np.amax(data_t["discharge"] - maxt / 250),
                    250,
                ),
                (250, 1),
            )
            ranget_tens = tf.convert_to_tensor(ranget, dtype=tf.dtypes.float64)
            ranget_m_tens = tf.convert_to_tensor(
                ranget - args_cal.stepsize, dtype=tf.dtypes.float64
            )
            ranget_p_tens = tf.convert_to_tensor(
                ranget + args_cal.stepsize, dtype=tf.dtypes.float64
            )
            dummyR = sim_params_dict["discharge"]["Rs_c"] * np.ones(
                (ranget.shape[0], 1)
            )
            dummyR_tens = (
                tf.convert_to_tensor(dummyR, dtype=tf.dtypes.float64)
                / sim_params_dict["discharge"]["rescale_R"]
            )
            if use_p2d:
                dummyX = (
                    sim_params_dict["discharge"]["L_a"]
                    + sim_params_dict["discharge"]["L_s"]
                    + sim_params_dict["discharge"]["L_c"]
                ) * np.ones((ranget.shape[0], 1))
                dummyX_tens = tf.convert_to_tensor(
                    dummyX, dtype=tf.dtypes.float64
                )
                dummyX_tens_resc = (
                    tf.convert_to_tensor(dummyX, dtype=tf.dtypes.float64)
                    / sim_params_dict["discharge"]["rescale_x"]
                )
            ones_tf64 = tf.ones(tf.shape(ranget_tens), dtype=tf.dtypes.float64)

            @tf.function
            def forward_range_dis(p, ranget_tens):
                deg_par = [p[i] * ones_tf64 for i in range(len(p))]
                deg_par_resc = [
                    nn_dict["discharge"].rescale_param(p[i], i) * ones_tf64
                    for i in range(len(p))
                ]
                if use_p2d:
                    out = nn_dict["discharge"].model(
                        [
                            ranget_tens
                            / sim_params_dict["discharge"]["rescale_T"],
                            dummyX_tens_resc,
                            dummyR_tens,
                        ]
                        + deg_par_resc
                    )
                else:
                    out = nn_dict["discharge"].model(
                        [
                            ranget_tens
                            / sim_params_dict["discharge"]["rescale_T"],
                            dummyR_tens,
                        ]
                        + deg_par_resc
                    )
                if use_p2d:
                    out_m = nn_dict["discharge"].model(
                        [
                            ranget_m_tens
                            / sim_params_dict["discharge"]["rescale_T"],
                            dummyX_tens_resc,
                            dummyR_tens,
                        ]
                        + deg_par_resc
                    )
                    out_p = nn_dict["discharge"].model(
                        [
                            ranget_p_tens
                            / sim_params_dict["discharge"]["rescale_T"],
                            dummyX_tens_resc,
                            dummyR_tens,
                        ]
                        + deg_par_resc
                    )
                else:
                    out_m = nn_dict["discharge"].model(
                        [
                            ranget_m_tens
                            / sim_params_dict["discharge"]["rescale_T"],
                            dummyR_tens,
                        ]
                        + deg_par_resc
                    )
                    out_p = nn_dict["discharge"].model(
                        [
                            ranget_p_tens
                            / sim_params_dict["discharge"]["rescale_T"],
                            dummyR_tens,
                        ]
                        + deg_par_resc
                    )
                phis_c_unrescaled = out[nn_dict["discharge"].ind_phis_c]
                phis_c_m_unrescaled = out_m[nn_dict["discharge"].ind_phis_c]
                phis_c_p_unrescaled = out_p[nn_dict["discharge"].ind_phis_c]
                if use_p2d:
                    phis_c = nn_dict["discharge"].rescalePhis_c(
                        phis_c_unrescaled, ranget_tens, dummyX_tens, *deg_par
                    )[:, 0]
                    phis_c_m = nn_dict["discharge"].rescalePhis_c(
                        phis_c_m_unrescaled,
                        ranget_m_tens,
                        dummyX_tens,
                        *deg_par,
                    )[:, 0]
                    phis_c_p = nn_dict["discharge"].rescalePhis_c(
                        phis_c_p_unrescaled,
                        ranget_p_tens,
                        dummyX_tens,
                        *deg_par,
                    )[:, 0]
                else:
                    phis_c = nn_dict["discharge"].rescalePhis_c(
                        phis_c_unrescaled, ranget_tens, *deg_par
                    )[:, 0]
                    phis_c_m = nn_dict["discharge"].rescalePhis_c(
                        phis_c_m_unrescaled, ranget_m_tens, *deg_par
                    )[:, 0]
                    phis_c_p = nn_dict["discharge"].rescalePhis_c(
                        phis_c_p_unrescaled, ranget_p_tens, *deg_par
                    )[:, 0]
                dV_dQ = (phis_c_p - phis_c_m) / np.float64(
                    2 * args_cal.stepsize
                )
                dV_dQ_act = (
                    tf.clip_by_value(phis_c_p, 3, 4.1)
                    - tf.clip_by_value(phis_c_m, 3, 4.1)
                ) / np.float64(2 * args_cal.stepsize)
                dV_dQ /= abs(
                    sim_params_dict["discharge"]["I_discharge"]
                ) * np.float64(1000 / 3600)
                dV_dQ = tf.clip_by_value(
                    dV_dQ, np.float64(-1e6), np.float64(-1e-2)
                )
                dV_dQ_act /= abs(
                    sim_params_dict["discharge"]["I_discharge"]
                ) * np.float64(1000 / 3600)
                dV_dQ_act = tf.clip_by_value(
                    dV_dQ_act, np.float64(-1e6), np.float64(-1e-2)
                )
                dQ_dV = np.float64(1) / dV_dQ_act

                return phis_c, dV_dQ_act, dQ_dV

        if key == "chargecc":
            maxt = np.amax(data_t["chargecc"])
            ranget = np.reshape(
                np.linspace(
                    np.amin(data_t["chargecc"] + maxt / 250),
                    np.amax(data_t["chargecc"] - maxt / 250),
                    250,
                ),
                (250, 1),
            )
            ranget_tens = tf.convert_to_tensor(ranget, dtype=tf.dtypes.float64)
            ranget_m_tens = tf.convert_to_tensor(
                ranget - args_cal.stepsize, dtype=tf.dtypes.float64
            )
            ranget_p_tens = tf.convert_to_tensor(
                ranget + args_cal.stepsize, dtype=tf.dtypes.float64
            )
            dummyR = sim_params_dict["chargecc"]["Rs_c"] * np.ones(
                (ranget.shape[0], 1)
            )
            dummyR_tens = (
                tf.convert_to_tensor(dummyR, dtype=tf.dtypes.float64)
                / sim_params_dict["chargecc"]["rescale_R"]
            )
            if use_p2d:
                dummyX = (
                    sim_params_dict["chargecc"]["L_a"]
                    + sim_params_dict["chargecc"]["L_s"]
                    + sim_params_dict["chargecc"]["L_c"]
                ) * np.ones((ranget.shape[0], 1))
                dummyX_tens = tf.convert_to_tensor(
                    dummyX, dtype=tf.dtypes.float64
                )
                dummyX_tens_resc = (
                    tf.convert_to_tensor(dummyX, dtype=tf.dtypes.float64)
                    / sim_params_dict["chargecc"]["rescale_x"]
                )

            ones_tf64 = tf.ones(tf.shape(ranget_tens), dtype=tf.dtypes.float64)

            @tf.function
            def forward_range_ch(p, ranget_tens):
                deg_par = [p[i] * ones_tf64 for i in range(len(p))]
                deg_par_resc = [
                    nn_dict["chargecc"].rescale_param(p[i], i) * ones_tf64
                    for i in range(len(p))
                ]
                if use_p2d:
                    out = nn_dict["chargecc"].model(
                        [
                            ranget_tens
                            / sim_params_dict["chargecc"]["rescale_T"],
                            dummyX_tens_resc,
                            dummyR_tens,
                        ]
                        + deg_par_resc
                    )
                    out_m = nn_dict["chargecc"].model(
                        [
                            ranget_m_tens
                            / sim_params_dict["chargecc"]["rescale_T"],
                            dummyX_tens_resc,
                            dummyR_tens,
                        ]
                        + deg_par_resc
                    )
                    out_p = nn_dict["chargecc"].model(
                        [
                            ranget_p_tens
                            / sim_params_dict["chargecc"]["rescale_T"],
                            dummyX_tens_resc,
                            dummyR_tens,
                        ]
                        + deg_par_resc
                    )
                else:
                    out = nn_dict["chargecc"].model(
                        [
                            ranget_tens
                            / sim_params_dict["chargecc"]["rescale_T"],
                            dummyR_tens,
                        ]
                        + deg_par_resc
                    )
                    out_m = nn_dict["chargecc"].model(
                        [
                            ranget_m_tens
                            / sim_params_dict["chargecc"]["rescale_T"],
                            dummyR_tens,
                        ]
                        + deg_par_resc
                    )
                    out_p = nn_dict["chargecc"].model(
                        [
                            ranget_p_tens
                            / sim_params_dict["chargecc"]["rescale_T"],
                            dummyR_tens,
                        ]
                        + deg_par_resc
                    )
                phis_c_unrescaled = out[nn_dict["chargecc"].ind_phis_c]
                phis_c_m_unrescaled = out_m[nn_dict["chargecc"].ind_phis_c]
                phis_c_p_unrescaled = out_p[nn_dict["chargecc"].ind_phis_c]
                if use_p2d:
                    phis_c = nn_dict["chargecc"].rescalePhis_c(
                        phis_c_unrescaled, ranget_tens, dummyX_tens, *deg_par
                    )[:, 0]
                    phis_c_m = nn_dict["chargecc"].rescalePhis_c(
                        phis_c_m_unrescaled,
                        ranget_m_tens,
                        dummyX_tens,
                        *deg_par,
                    )[:, 0]
                    phis_c_p = nn_dict["chargecc"].rescalePhis_c(
                        phis_c_p_unrescaled,
                        ranget_p_tens,
                        dummyX_tens,
                        *deg_par,
                    )[:, 0]
                else:
                    phis_c = nn_dict["chargecc"].rescalePhis_c(
                        phis_c_unrescaled, ranget_tens, *deg_par
                    )[:, 0]
                    phis_c_m = nn_dict["chargecc"].rescalePhis_c(
                        phis_c_m_unrescaled, ranget_m_tens, *deg_par
                    )[:, 0]
                    phis_c_p = nn_dict["chargecc"].rescalePhis_c(
                        phis_c_p_unrescaled, ranget_p_tens, *deg_par
                    )[:, 0]
                # phis_c = tf.clip_by_value(phis_c, 3, 4.1)
                # phis_c_p = tf.clip_by_value(phis_c_p, 3, 4.1)
                # phis_c_m = tf.clip_by_value(phis_c_m, 3, 4.1)
                dV_dQ = (phis_c_p - phis_c_m) / np.float64(
                    2 * args_cal.stepsize
                )
                dV_dQ_act = (
                    tf.clip_by_value(phis_c_p, 3, 4.1)
                    - tf.clip_by_value(phis_c_m, 3, 4.1)
                ) / np.float64(2 * args_cal.stepsize)
                dV_dQ /= abs(
                    sim_params_dict["chargecc"]["I_discharge"]
                ) * np.float64(1000 / 3600)
                dV_dQ = tf.clip_by_value(
                    dV_dQ, np.float64(1e-3), np.float64(1e6)
                )
                dV_dQ_act /= abs(
                    sim_params_dict["chargecc"]["I_discharge"]
                ) * np.float64(1000 / 3600)
                dV_dQ_act = tf.clip_by_value(
                    dV_dQ_act, np.float64(1e-3), np.float64(1e6)
                )
                dQ_dV = np.float64(1) / dV_dQ_act

                return phis_c, dV_dQ_act, dQ_dV

        nsamples = np_mcmc_samples.shape[0]
        print("Num samples = ", nsamples)
        realization_phis_c = []
        realization_dV_dQ = []
        realization_dQ_dV = []

        if args_cal.cyc_mode.lower() in ["discharge", "chargecc"]:
            for i in range(nsamples):
                if args_cal.calibrate_one_sigma:
                    if args_cal.cyc_mode.lower() == "discharge":
                        phis_c, dV_dQ, dQ_dV = forward_range_dis(
                            np_mcmc_samples[i, :-1], ranget_tens
                        )
                    elif args_cal.cyc_mode.lower() == "chargecc":
                        phis_c, dV_dQ, dQ_dV = forward_range_ch(
                            np_mcmc_samples[i, :-1], ranget_tens
                        )

                else:
                    if args_cal.cyc_mode.lower() == "discharge":
                        phis_c, dV_dQ, dQ_dV = forward_range_dis(
                            np_mcmc_samples[i, :], ranget_tens
                        )
                    elif args_cal.cyc_mode.lower() == "chargecc":
                        phis_c, dV_dQ, dQ_dV = forward_range_ch(
                            np_mcmc_samples[i, :], ranget_tens
                        )
                realization_phis_c.append(phis_c)
                realization_dV_dQ.append(dV_dQ)
                realization_dQ_dV.append(dQ_dV)
        if args_cal.cyc_mode.lower() == "discharge-chargecc":
            if key == "discharge":
                for i in range(nsamples):
                    if args_cal.calibrate_one_sigma:
                        phis_c, dV_dQ, dQ_dV = forward_range_dis(
                            np_mcmc_samples[i, :-3], ranget_tens
                        )
                    else:
                        phis_c, dV_dQ, dQ_dV = forward_range_dis(
                            np_mcmc_samples[i, :-2], ranget_tens
                        )
                    realization_phis_c.append(phis_c)
                    realization_dV_dQ.append(dV_dQ)
                    realization_dQ_dV.append(dQ_dV)
            if key == "chargecc":
                indc = list(range(sim_params_dict["chargecc"]["n_deg_params"]))
                indc[sim_params_dict["chargecc"]["ind_deg_x0_a"]] = nn_dict[
                    "chargecc"
                ].params["n_deg_params"]
                indc[sim_params_dict["chargecc"]["ind_deg_x0_c"]] = (
                    sim_params_dict["chargecc"]["n_deg_params"] + 1
                )
                for i in range(nsamples):
                    if args_cal.calibrate_one_sigma:
                        phis_c, dV_dQ, dQ_dV = forward_range_ch(
                            np_mcmc_samples[i, indc], ranget_tens
                        )
                    else:
                        phis_c, dV_dQ, dQ_dV = forward_range_ch(
                            np_mcmc_samples[i, indc], ranget_tens
                        )
                    realization_phis_c.append(phis_c)
                    realization_dV_dQ.append(dV_dQ)
                    realization_dQ_dV.append(dQ_dV)

        mean_phis_c_real = np.mean(realization_phis_c, axis=0)
        min_phis_c_real = np.min(realization_phis_c, axis=0)
        max_phis_c_real = np.max(realization_phis_c, axis=0)
        std90_phis_c_real = np.percentile(realization_phis_c, 90, axis=0)
        std10_phis_c_real = np.percentile(realization_phis_c, 10, axis=0)
        mean_dV_dQ_real = np.mean(realization_dV_dQ, axis=0)
        min_dV_dQ_real = np.min(realization_dV_dQ, axis=0)
        max_dV_dQ_real = np.max(realization_dV_dQ, axis=0)
        std90_dV_dQ_real = np.percentile(realization_dV_dQ, 90, axis=0)
        std10_dV_dQ_real = np.percentile(realization_dV_dQ, 10, axis=0)
        mean_dQ_dV_real = np.mean(realization_dQ_dV, axis=0)
        min_dQ_dV_real = np.min(realization_dQ_dV, axis=0)
        max_dQ_dV_real = np.max(realization_dQ_dV, axis=0)
        std90_dQ_dV_real = np.percentile(realization_dQ_dV, 90, axis=0)
        std10_dQ_dV_real = np.percentile(realization_dQ_dV, 10, axis=0)

        y_err = None
        if args_cal.calibrate_one_sigma:
            sigma = np.mean(np_mcmc_samples[:, -1])

        if args_cal.read_sigma:
            y_err = sigma_from_file(
                cyc_mode=args_cal.cyc_mode,
                target_list=target_list,
                cell_id=args_cal.cell_id,
                cycle_id=args_cal.cycle_id,
                folder="data_sigma",
            )
            target_pref = "_".join(target_list)
            if key == "discharge":
                sigma_V = y_err["dis_phis_c"]
                sigma_dV_dQ = y_err["dis_dV_dQ"]
                sigma_dQ_dV = y_err["dis_dQ_dV"]
            elif key == "chargecc":
                sigma_V = y_err["chcc_phis_c"]
                sigma_dV_dQ = y_err["chcc_dV_dQ"]
                sigma_dQ_dV = y_err["chcc_dQ_dV"]
        elif (
            not args_cal.read_sigma
            and ("phis_c" in target_list)
            and ("dV_dQ" not in target_list)
            and ("dQ_dV" not in target_list)
        ):
            sigma_V = sigma
            sigma_dV_dQ = sigma * 4
            sigma_dQ_dV = sigma * 100
            target_pref = "tarV"
        elif (
            not args_cal.read_sigma
            and ("phis_c" not in target_list)
            and ("dV_dQ" in target_list)
            and ("dQ_dV" not in target_list)
        ):
            sigma_V = sigma / 4
            sigma_dV_dQ = sigma
            sigma_dQ_dV = sigma * 100 / 4
            target_pref = "tardVdQ"
        elif (
            not args_cal.read_sigma
            and ("phis_c" not in target_list)
            and ("dV_dQ" not in target_list)
            and ("dQ_dV" in target_list)
        ):
            sigma_V = sigma / 100
            sigma_dV_dQ = sigma * 4 / 100
            sigma_dQ_dV = sigma
            target_pref = "tardQdV"

        # if np.amin(data_phis_c - min_real) < 0 or np.amax(data_phis_c - max_real) > 0:
        #    print(f" Increase STD  {np.amin(data_phis_c - std10_real)} - {np.amax(data_phis_c - std90_real)}")

        fig = plt.figure()
        plt.plot(
            data_t[key],
            data_phis_c[key],
            "o",
            color="r",
            markersize=7,
            label="Data",
        )
        plt.plot(
            ranget,
            mean_phis_c_real,
            color="k",
            linewidth=3,
            label="mean degradation",
        )
        pretty_labels("t [s]", r"$\phi_{s,ca}$ [V]", 14)
        pretty_legend()
        plt.savefig(
            os.path.join(
                figureFolder,
                f"{target_pref}_Vnobar_{key}_forw{args_cal.cycle_id}.png",
            )
        )
        fig = plt.figure()
        plt.plot(
            data_t[key],
            data_phis_c[key],
            "o",
            color="r",
            markersize=7,
            label="Data",
        )
        plt.plot(
            ranget,
            mean_phis_c_real,
            color="k",
            linewidth=3,
            label="mean degradation",
        )
        plt.plot(
            ranget,
            std90_phis_c_real + sigma_V,
            "--",
            color="k",
            linewidth=3,
            label="10th and 90th percentile",
        )
        plt.plot(
            ranget, std10_phis_c_real - sigma_V, "--", color="k", linewidth=3
        )
        pretty_labels("t [s]", "phis_c", 14)
        pretty_legend()
        plt.savefig(
            os.path.join(
                figureFolder,
                f"{target_pref}_V_{key}_forw{args_cal.cycle_id}.png",
            )
        )

        rangeQ = (
            ranget
            * abs(nn_dict[key].params["I_discharge"])
            * np.float64(1000 / 3600)
        )
        fig = plt.figure()
        plt.plot(
            data_t_dQ_dV[key],
            data_dQ_dV_y[key],
            "o",
            color="r",
            markersize=7,
            label="Data",
        )
        plt.plot(
            ranget,
            mean_dQ_dV_real,
            color="k",
            linewidth=3,
            label="mean degradation",
        )
        plt.plot(
            ranget,
            std90_dQ_dV_real + sigma_dQ_dV,
            "--",
            color="k",
            linewidth=3,
            label="10th and 90th percentile",
        )
        plt.plot(
            ranget,
            std10_dQ_dV_real - sigma_dQ_dV,
            "--",
            color="k",
            linewidth=3,
        )
        pretty_labels("t [s]", r"dQ/dV [mAhV$^{-1}$]", 14)
        pretty_legend()
        plt.savefig(
            os.path.join(
                figureFolder,
                f"{target_pref}_dQ_dV_{key}_forw{args_cal.cycle_id}.png",
            )
        )
        plt.close()

        fig = plt.figure()
        plt.plot(
            data_t_dV_dQ[key],
            data_dV_dQ_y[key],
            "o",
            color="r",
            markersize=7,
            label="Data",
        )
        plt.plot(
            ranget,
            mean_dV_dQ_real,
            color="k",
            linewidth=3,
            label="mean degradation",
        )
        plt.plot(
            ranget,
            std90_dV_dQ_real + sigma_dV_dQ,
            "--",
            color="k",
            linewidth=3,
            label="10th and 90th percentile",
        )

        plt.plot(
            ranget,
            std10_dV_dQ_real - sigma_dV_dQ,
            "--",
            color="k",
            linewidth=3,
        )
        pretty_labels("t [s]", r"dV/dQ [V.(mAh)$^{-1}$]", 14)
        pretty_legend()
        plt.savefig(
            os.path.join(
                figureFolder,
                f"{target_pref}_dV_dQ_{key}_forw{args_cal.cycle_id}.png",
            )
        )
        plt.close()


def get_nchan(target_mode: str):

    if target_mode.lower() in ["phi", "dvdq", "dqdv"]:
        return 2
    elif target_mode.lower() in shuffle_substrings(
        "phi-dvdq"
    ) + shuffle_substrings("dvdq-dqdv") + shuffle_substrings("phi-dqdv"):
        return 3
    elif target_mode.lower() in shuffle_substrings("phi-dvdq-dqdv"):
        return 4
    else:
        raise NotImplementedError


def make_data_in(
    target_mode: str,
    cyc_mode: str,
    n_points: int,
    data_t_dV_dQ,
    data_dV_dQ_x,
    data_t,
    data_phis_c,
    data_dV_dQ_y,
    data_dQ_dV_y,
):
    n_chan = get_nchan(target_mode)
    if cyc_mode.lower() in ["discharge", "chargecc"]:
        list_cyc = [cyc_mode]
    else:
        list_cyc = ["discharge", "chargecc"]

    data_in = np.zeros((1, n_chan * len(list_cyc), n_points)).astype("float32")

    for icyc, cyc in enumerate(list_cyc):
        if target_mode.lower() == "phi":
            data_in[0, 0 + n_chan * icyc, :] = data_t[cyc]
            data_in[0, 1 + n_chan * icyc, :] = data_phis_c[cyc]
        else:
            t_int = data_t_dV_dQ[cyc][:]
            qt_ratio = np.mean(data_dV_dQ_x[cyc] / data_t_dV_dQ[cyc])

        if target_mode.lower() == "dvdq":
            data_in[0, 0 + n_chan * icyc, :] = t_int
            data_in[0, 1 + n_chan * icyc, :] = data_dV_dQ_y[cyc] * qt_ratio
        elif target_mode.lower() == "dqdv":
            data_in[0, 0 + n_chan * icyc, :] = t_int
            data_in[0, 1 + n_chan * icyc, :] = data_dQ_dV_y[cyc] * qt_ratio
        elif target_mode.lower() in shuffle_substrings("phi-dvdq"):
            data_in[0, 0 + n_chan * icyc, :] = t_int
            data_in[0, 1 + n_chan * icyc, :] = np.interp(
                t_int, data_t[cyc], data_phis_c[cyc]
            )
            data_in[0, 2 + n_chan * icyc, :] = data_dV_dQ_y[cyc] * qt_ratio
        elif target_mode.lower() in shuffle_substrings("phi-dvdq"):
            data_in[0, 0 + n_chan * icyc, :] = t_int
            data_in[0, 1 + n_chan * icyc, :] = np.interp(
                t_int, data_t[cyc], data_phis_c[cyc]
            )
            data_in[0, 2 + n_chan * icyc, :] = data_dQ_dV_y[cyc] / qt_ratio
        elif target_mode.lower() in shuffle_substrings("dvdq-dqdv"):
            data_in[0, 0 + n_chan * icyc, :] = t_int
            data_in[0, 1 + n_chan * icyc, :] = data_dV_dQ_y[cyc] * qt_ratio
            data_in[0, 2 + n_chan * icyc, :] = data_dQ_dV_y[cyc] / qt_ratio
        elif target_mode.lower() in shuffle_substrings("phi-dvdq-dqdv"):
            data_in[0, 0 + n_chan * icyc, :] = t_int
            data_in[0, 1 + n_chan * icyc, :] = np.interp(
                t_int, data_t[cyc], data_phis_c[cyc]
            )
            data_in[0, 2 + n_chan * icyc, :] = data_dV_dQ_y[cyc] * qt_ratio
            data_in[0, 3 + n_chan * icyc, :] = data_dQ_dV_y[cyc] / qt_ratio

    return data_in
