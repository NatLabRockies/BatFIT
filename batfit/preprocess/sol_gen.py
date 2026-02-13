import os
import pickle
import random
import sys
import time
from pathlib import Path

import bmlite as bm
import numpy as np
import pandas as pd

from batfit import BATFIT_EXP, logger
from batfit.preprocess.diff_cap import calc_dqdv_dvdq
from batfit.preprocess.pickledb import PickleDB


def mod_sim(sim, sim_params, deg_param_sample, cyc_mode, run_mode):

    # Discretization
    sim.an.Nr = sim_params["Nr_a"]
    sim.ca.Nr = sim_params["Nr_c"]
    if sim_params["model"].lower() == "p2d":
        sim.an.Nx = sim_params["Nx_a"]
        sim.ca.Nx = sim_params["Nx_c"]
        sim.sep.Nx = sim_params["Nx_s"]

    # Degradation parameters
    if cyc_mode.lower() == "discharge-chargecc":
        if run_mode.lower() == "discharge":
            if "cs0_c" in deg_param_sample:
                sim.ca.x_0 = sim_params["x0_c_dis"] * deg_param_sample["cs0_c"]
            else:
                sim.ca.x_0 = sim_params["x0_c_dis"]
            if "cs0_a" in deg_param_sample:
                sim.an.x_0 = sim_params["x0_a_dis"] * deg_param_sample["cs0_a"]
            else:
                sim.an.x_0 = sim_params["x0_a_dis"]
            C_rate = sim_params["C_dis"]
        elif run_mode.lower() == "chargecc":
            if "cs0_c_chcc" in deg_param_sample:
                sim.ca.x_0 = (
                    sim_params["x0_c_chcc"] * deg_param_sample["cs0_c_chcc"]
                )
            else:
                sim.ca.x_0 = sim_params["x0_c_chcc"]

            if "cs0_a_chcc" in deg_param_sample:
                sim.an.x_0 = (
                    sim_params["x0_a_chcc"] * deg_param_sample["cs0_a_chcc"]
                )
            else:
                sim.an.x_0 = sim_params["x0_a_chcc"]
            C_rate = sim_params["C_chcc"]
    elif cyc_mode.lower() in ["discharge", "chargecc", "rh", "lh"]:
        if "cs0_c" in deg_param_sample:
            sim.ca.x_0 = sim_params["x0_c"] * deg_param_sample["cs0_c"]
        else:
            sim.ca.x_0 = sim_params["x0_c"]
        if "cs0_a" in deg_param_sample:
            sim.an.x_0 = sim_params["x0_a"] * deg_param_sample["cs0_a"]
        else:
            sim.an.x_0 = sim_params["x0_a"]
        C_rate = None
    if cyc_mode.lower() in ["discharge", "chargecc"]:
        C_rate = sim_params["C"]
    if "ds_c" in deg_param_sample:
        sim.ca.Ds_deg = deg_param_sample["ds_c"]
    else:
        sim.ca.Ds_deg = 1.0
    if "ds_a" in deg_param_sample:
        sim.an.Ds_deg = deg_param_sample["ds_a"]
    else:
        sim.an.Ds_deg = 1.0
    if "i0_a" in deg_param_sample:
        sim.an.i0_deg = deg_param_sample["i0_a"]
    else:
        sim.an.i0_deg = 1.0
    if "i0_c" in deg_param_sample:
        sim.ca.i0_deg = deg_param_sample["i0_c"]
    else:
        sim.ca.i0_deg = 1.0

    if "eps_cbd_a" in deg_param_sample:
        sim.an.eps_CBD = (
            sim_params["eps_CBD_a"] * deg_param_sample["eps_cbd_a"]
        )
    else:
        sim.an.eps_CBD = sim_params["eps_CBD_a"]
    if "eps_cbd_c" in deg_param_sample:
        sim.ca.eps_CBD = (
            sim_params["eps_CBD_c"] * deg_param_sample["eps_cbd_c"]
        )
    else:
        sim.ca.eps_CBD = sim_params["eps_CBD_c"]
    if "eps_s_c" in deg_param_sample:
        sim.ca.eps_s = sim_params["eps_s_c"] * deg_param_sample["eps_s_c"]
    elif "eps_s_c_am" in deg_param_sample:
        # sim.ca.eps_s = sim.ca.eps_CBD + sim.ca.eps_AM * deg
        # sim.ca.eps_s = sim.ca.eps_CBD + (sim_params["eps_s_c"] - sim_params["eps_CBD_c"]) * deg
        sim.ca.eps_s = (
            sim.ca.eps_CBD
            + (sim_params["eps_s_c"] - sim_params["eps_CBD_c"])
            * deg_param_sample["eps_s_c_am"]
        )
    else:
        sim.ca.eps_s = sim_params["eps_s_c"]
    if "eps_s_a" in deg_param_sample:
        sim.an.eps_s = sim_params["eps_s_a"] * deg_param_sample["eps_s_a"]
    elif "eps_s_a_am" in deg_param_sample:
        # sim.an.eps_s = sim.an.eps_CBD + sim.an.eps_AM * deg
        # sim.an.eps_s = sim.an.eps_CBD + (sim_params["eps_s_a"] - sim_params["eps_CBD_a"]) * deg
        sim.an.eps_s = (
            sim.an.eps_CBD
            + (sim_params["eps_s_a"] - sim_params["eps_CBD_a"])
            * deg_param_sample["eps_s_a_am"]
        )
    else:
        sim.an.eps_s = sim_params["eps_s_a"]
    if "ce" in deg_param_sample:
        sim.el.Li_0 = sim_params["ce"] * deg_param_sample["ce"]
    else:
        sim.el.Li_0 = sim_params["ce"]
    if "eps_el_a" in deg_param_sample:
        sim.an.eps_el = sim_params["eps_el_a"] * deg_param_sample["eps_el_a"]
    else:
        sim.an.eps_el = sim_params["eps_el_a"]
    if "eps_el_c" in deg_param_sample:
        sim.ca.eps_el = sim_params["eps_el_c"] * deg_param_sample["eps_el_c"]
    else:
        sim.ca.eps_el = sim_params["eps_el_c"]
    if "area" in deg_param_sample:
        sim.bat.area = sim_params["area"] * deg_param_sample["area"]
    else:
        sim.bat.area = sim_params["area"]
    if "l_a" in deg_param_sample:
        sim.an.thick = sim_params["L_a"] * deg_param_sample["l_a"]
    else:
        sim.an.thick = sim_params["L_a"]
    if "l_c" in deg_param_sample:
        sim.ca.thick = sim_params["L_c"] * deg_param_sample["l_c"]
    else:
        sim.ca.thick = sim_params["L_c"]

    if "rs_a" in deg_param_sample:
        sim.an.R_s = sim_params["Rs_a"] * deg_param_sample["rs_a"]
    else:
        sim.an.R_s = sim_params["Rs_a"]

    if "rs_c" in deg_param_sample:
        sim.ca.R_s = sim_params["Rs_c"] * deg_param_sample["rs_c"]
    else:
        sim.ca.R_s = sim_params["Rs_c"]

    if sim_params["model"].lower() == "p2d":
        if "l_s" in deg_param_sample:
            sim.sep.thick = sim_params["L_s"] * deg_param_sample["l_s"]
        else:
            sim.sep.thick = sim_params["L_s"]

        if "eps_el" in deg_param_sample:
            sim.sep.eps_el = sim_params["eps_el"] * deg_param_sample["eps_el"]
        else:
            sim.sep.eps_el = sim_params["eps_el"]

        if "p_l" in deg_param_sample:
            sim.sep.p_liq = sim_params["p_l"] * deg_param_sample["p_l"]
        else:
            sim.sep.p_liq = sim_params["p_l"]

        if "p_s_a" in deg_param_sample:
            sim.an.p_sol = sim_params["p_s_a"] * deg_param_sample["p_s_a"]
        else:
            sim.an.p_sol = sim_params["p_s_a"]

        if "p_l_a" in deg_param_sample:
            sim.an.p_liq = sim_params["p_l_a"] * deg_param_sample["p_l_a"]
        else:
            sim.an.p_liq = sim_params["p_l_a"]

        if "p_s_c" in deg_param_sample:
            sim.ca.p_sol = sim_params["p_s_c"] * deg_param_sample["p_s_c"]
        else:
            sim.ca.p_sol = sim_params["p_s_c"]

        if "p_l_c" in deg_param_sample:
            sim.ca.p_liq = sim_params["p_l_c"] * deg_param_sample["p_l_c"]
        else:
            sim.ca.p_liq = sim_params["p_l_c"]

        if "de" in deg_param_sample:
            sim.el.D_deg = deg_param_sample["de"]
        else:
            sim.el.D_deg = 1.0

        if "t0" in deg_param_sample:
            sim.el.t0_deg = deg_param_sample["t0"]
        else:
            sim.el.t0_deg = 1.0

        if "kappa" in deg_param_sample:
            sim.el.kappa_deg = deg_param_sample["kappa"]
        else:
            sim.el.kappa_deg = 1.0

        if "gamma" in deg_param_sample:
            sim.el.gamma_deg = deg_param_sample["gamma"]
        else:
            sim.el.gamma_deg = 1.0

    return sim, C_rate


def print_an(sim):
    print("Anode")
    print(f"\tA_s = {sim.an.A_s}")
    print(f"\tLi_max = {sim.an.Li_max}")
    print(f"\tR_s = {sim.an.R_s}")
    print(f"\tmaterial = {sim.an.material}")
    print(f"\talpha_a = {sim.an.alpha_a}")
    print(f"\talpha_c = {sim.an.alpha_c}")
    print(f"\teps_AM = {sim.an.eps_AM}")
    print(f"\teps_CBD = {sim.an.eps_CBD}")
    print(f"\teps_el = {sim.an.eps_el}")
    print(f"\teps_s = {sim.an.eps_s}")
    print(f"\teps_void = {sim.an.eps_void}")
    print(f"\tx_0 = {sim.an.x_0}")


def print_ca(sim):
    print("Anode")
    print(f"\tA_s = {sim.ca.A_s}")
    print(f"\tLi_max = {sim.ca.Li_max}")
    print(f"\tR_s = {sim.ca.R_s}")
    print(f"\tmaterial = {sim.ca.material}")
    print(f"\talpha_a = {sim.ca.alpha_a}")
    print(f"\talpha_c = {sim.ca.alpha_c}")
    print(f"\teps_AM = {sim.ca.eps_AM}")
    print(f"\teps_CBD = {sim.ca.eps_CBD}")
    print(f"\teps_el = {sim.ca.eps_el}")
    print(f"\teps_s = {sim.ca.eps_s}")
    print(f"\teps_void = {sim.ca.eps_void}")
    print(f"\tDs_deg = {sim.ca.Ds_deg}")
    print(f"\ti0_deg = {sim.ca.i0_deg}")
    print(f"\tx_0 = {sim.ca.x_0}")


def remove_file(filename):
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass


def robust_LHRH(sim, df, charge, protocol, sim_params, bat_model):
    rootsol = None
    try:
        stmp = sim.run(charge, reset_state=False)
        assert all(stmp.success)
        for i in range(40):
            LHmax_tmp = bm.Experiment()
            for _, row in df.iterrows():
                dt, P_ratio = row["dt_s"], row["P_ratio"]
                P_scalar = 0.11  # Wh
                LHmax_tmp.add_step(
                    "power_W",
                    P_scalar * P_ratio,
                    (dt, 10.0),
                    limits=(
                        "voltage_V",
                        sim_params["vmin"],
                        "voltage_V",
                        sim_params["vmax"],
                    ),
                )
            lhtmp = sim.run(LHmax_tmp, reset_state=False, bar=False)
            assert all(lhtmp.success)
            if i == 0:
                all_solns = lhtmp._solns
            else:
                all_solns += lhtmp._solns
        s1 = sim.run(protocol)
        assert all(s1.success)
        all_solns += s1._solns
        if bat_model.lower() == "spm":
            sim_prot = bm.SPM.CycleSolution(*all_solns)
        elif bat_model.lower() == "p2d":
            sim_prot = bm.P2D.CycleSolution(*all_solns)
        else:
            sys.exit("ERROR: battery model not recognized")
        rootsol = sim_prot
        assert all(rootsol.success)
    except:
        for fact in [0.5, 0.1]:
            try:
                print(f"retrying with C = {fact:.2f}")
                expr_init = bm.Experiment()
                expr_init.add_step(
                    "current_C",
                    -1.0 * fact,
                    (7200.0 / fact, 60.0),
                    limits=("voltage_V", sim_params["vmax"]),
                )
                expr_init.add_step(
                    "voltage_V", sim_params["vmax"], (3600.0, 60.0)
                )
                sol_init = sim.run(expr_init)
                assert all(sol_init.success)
                for i in range(40):
                    LHmax_tmp = bm.Experiment()
                    for _, row in df.iterrows():
                        dt, P_ratio = row["dt_s"], row["P_ratio"]
                        P_scalar = 0.11  # Wh
                        LHmax_tmp.add_step(
                            "power_W",
                            P_scalar * P_ratio,
                            (dt, 10.0),
                            limits=(
                                "voltage_V",
                                sim_params["vmin"],
                                "voltage_V",
                                sim_params["vmax"],
                            ),
                        )
                    lhtmp = sim.run(LHmax_tmp, reset_state=False, bar=False)
                    assert all(lhtmp.success)
                    if i == 0:
                        all_solns = lhtmp._solns
                    else:
                        all_solns += lhtmp._solns
                s1 = sim.run(protocol)
                assert all(s1.success)
                all_solns += s1._solns
                if bat_model.lower() == "spm":
                    sim_prot = bm.SPM.CycleSolution(*all_solns)
                elif bat_model.lower() == "p2d":
                    sim_prot = bm.P2D.CycleSolution(*all_solns)
                else:
                    sys.exit("ERROR: battery model not recognized")
                rootsol = sim_prot
                assert all(rootsol.success)
                break
            except:
                # print(f"sim failed for {deg_param_sample}")
                pass
    return rootsol


def robust_CC(sim, C_rate, sim_params):

    t_step = (3600 / abs(C_rate), 10000)
    t_step_init = (10 / abs(C_rate), 150)

    expr = bm.Experiment()
    if C_rate > 0:
        # Discharge
        phis_c_min = sim_params["vmin"]
        phis_c_max = np.inf
        lims = ("voltage_V", phis_c_min)
        expr.add_step(
            "current_C",
            C_rate,
            t_step,
            limits=lims,
            atol=1e-10,
            rtol=1e-8,
        )

    elif C_rate < 0:
        # Charge
        phis_c_min = -np.inf
        phis_c_max = sim_params["vmax"]
        lims = ("voltage_V", phis_c_max)
        expr.add_step(
            "current_C",
            C_rate,
            t_step,
            limits=lims,
            atol=1e-10,
            rtol=1e-8,
        )

    rootsol = None
    try:
        rootsol = sim.run(expr)
        assert rootsol.success
    except:
        for fact in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
            try:
                print(f"retrying with C = {C_rate*fact:.2f}")
                expr_init = bm.Experiment()
                expr_init.add_step(
                    "current_C",
                    C_rate * fact,
                    t_step_init,
                    limits=lims,
                    atol=1e-10,
                    rtol=1e-8,
                )
                sol_init = sim.run(expr_init)
                assert sol_init.success
                sim._sv0 = sol_init.y[0, :]
                sim._svdot0 = sol_init.yp[0, :]
                rootsol = sim.run(expr)
                assert rootsol.success
                break
            except:
                # print(f"sim failed for {deg_param_sample}")
                pass
    return rootsol


def single_run(
    deg_param_sample,
    sim_params,
    count=None,
    nsim=None,
    parallel_env=None,
    run_mode=None,
):

    cyc_mode = sim_params["cyc_mode"]
    params_list = [
        deg_param_sample[key] for key in sim_params["deg_param_names"]
    ]
    param_string = from_param_list_to_str(params_list)

    bat_model = None
    if sim_params["model"] == "SPM":
        bat_model = "SPM"
        sim = bm.SPM.Simulation(sim_params["material"])
    elif sim_params["model"] == "P2D":
        bat_model = "P2D"
        sim = bm.P2D.Simulation(sim_params["material"])
    sim, C_rate = mod_sim(
        sim, sim_params, deg_param_sample, cyc_mode, run_mode=run_mode
    )
    # print_an(sim)
    # print_ca(sim)

    sim.pre()
    time_s = time.time()
    if cyc_mode.lower() in ["discharge", "chargecc", "discharge-chargecc"]:
        rootsol = robust_CC(sim=sim, C_rate=C_rate, sim_params=sim_params)
        if rootsol is None:
            print(f"All sim failed for {deg_param_sample}")
        else:
            print(f"Success for {deg_param_sample}")

    elif cyc_mode.lower() in ["rh", "lh"]:
        df = pd.read_csv(os.path.join(BATFIT_EXP, "LHmax.csv"))
        charge = bm.Experiment()
        charge.add_step(
            "current_C",
            -1.0,
            (7200.0, 60.0),
            limits=("voltage_V", sim_params["vmax"]),
        )
        charge.add_step("voltage_V", sim_params["vmax"], (3600.0, 30.0))
        # LHmax = bm.Experiment()
        # for i in range(40):
        #    for _, row in df.iterrows():
        #        dt, P_ratio = row["dt_s"], row["P_ratio"]
        #        P_scalar = 0.11  # Wh
        #        LHmax.add_step(
        #            "power_W",
        #            P_scalar * P_ratio,
        #            (dt, 10.0),
        #            limits=(
        #                "voltage_V",
        #                sim_params["vmin"],
        #                "voltage_V",
        #                sim_params["vmax"],
        #            ),
        #        )
        if cyc_mode.lower() == "rh":
            reg = bm.Experiment(max_step=1.0)
            reg.add_step(
                "current_C",
                -0.13,
                (3600.0 * 8.0, 10.0),
                limits=("voltage_V", sim_params["vmax"]),
            )
            reg.add_step(
                "voltage_V",
                sim_params["vmax"],
                (3600.0 * 8.0, 10.0),
                limits=("time_h", 8.0),
            )
            rootsol = robust_LHRH(
                sim=sim,
                df=df,
                charge=charge,
                protocol=reg,
                sim_params=sim_params,
                bat_model=bat_model,
            )
            if rootsol is None:
                print(f"All sim failed for {deg_param_sample}")
            else:
                print(f"Success for {deg_param_sample}")
        if cyc_mode.lower() == "lh":
            long = bm.Experiment(max_step=1.0)
            long.add_step(
                "current_C",
                -0.4,
                (3600.0 * 2.0, 10.0),
                limits=("voltage_V", sim_params["vmax"]),
            )
            long.add_step(
                "voltage_V",
                sim_params["vmax"],
                (3600.0 * 2.0, 10.0),
                limits=("time_h", 2.0),
            )
            rootsol = robust_LHRH(
                sim=sim,
                df=df,
                charge=charge,
                protocol=long,
                sim_params=sim_params,
                bat_model=bat_model,
            )
            if rootsol is None:
                print(f"All sim failed for {deg_param_sample}")
            else:
                print(f"Success for {deg_param_sample}")

    time_e = time.time()

    if parallel_env is None:
        if count is None or nsim is None:
            print(f"Elapsed time = {time_e-time_s:.2f}s")
        else:
            print(f"Elapsed time ({count+1}/{nsim}) = {time_e-time_s:.2f}s")
    else:
        if count is None or nsim is None:
            parallel_env.printAll(f"Elapsed time = {time_e-time_s:.2f}s")
        else:
            parallel_env.printAll(
                f"Elapsed time ({count+1}/{nsim}) = {time_e-time_s:.2f}s"
            )

    return params_list, rootsol


def reduce_npoints_dict(sol_dict, n_points_reduce=512):
    new_sol_dict = {}
    t_int = np.linspace(
        np.nanmin(sol_dict["t"]),
        np.nanmax(sol_dict["t"]),
        n_points_reduce,
    )
    phis_c_int = np.interp(t_int, sol_dict["t"], sol_dict["phis_c"])
    new_sol_dict["t"] = t_int
    new_sol_dict["phis_c"] = phis_c_int

    if "t_diff" in sol_dict:
        t_diff_int = np.linspace(
            np.nanmin(sol_dict["t_diff"]),
            np.nanmax(sol_dict["t_diff"]),
            n_points_reduce,
        )
        phis_c_diff_int = np.interp(
            t_diff_int, sol_dict["t_diff"], sol_dict["phis_c_diff"]
        )
        dvdq_int = np.interp(t_diff_int, sol_dict["t_diff"], sol_dict["dvdq"])
        dqdv_int = np.interp(t_diff_int, sol_dict["t_diff"], sol_dict["dqdv"])
        new_sol_dict["t_diff"] = t_diff_int
        new_sol_dict["phis_c_diff"] = phis_c_diff_int
        new_sol_dict["dvdq"] = dvdq_int
        new_sol_dict["dqdv"] = dqdv_int

    if "t_diff_crop" in sol_dict:
        t_diff_crop_int = np.linspace(
            np.nanmin(sol_dict["t_diff_crop"]),
            np.nanmax(sol_dict["t_diff_crop"]),
            n_points_reduce,
        )
        phis_c_diff_crop_int = np.interp(
            t_diff_crop_int,
            sol_dict["t_diff_crop"],
            sol_dict["phis_c_diff_crop"],
        )
        dvdq_crop_int = np.interp(
            t_diff_crop_int,
            sol_dict["t_diff_crop"],
            sol_dict["dvdq_crop"],
        )
        dqdv_crop_int = np.interp(
            t_diff_crop_int,
            sol_dict["t_diff_crop"],
            sol_dict["dqdv_crop"],
        )

        new_sol_dict["t_diff_crop"] = t_diff_crop_int
        new_sol_dict["phis_c_diff_crop"] = phis_c_diff_crop_int
        new_sol_dict["dvdq_crop"] = dvdq_crop_int
        new_sol_dict["dqdv_crop"] = dqdv_crop_int

    return new_sol_dict


def reduce_npoints_records(records, n_points_reduce=512):
    new_records = []
    for record in records:
        new_sol = reduce_npoints_dict(
            record["sol"], n_points_reduce=n_points_reduce
        )
        record["sol"] = new_sol
        new_records.append(record)

    return new_records


def single_run_save(
    params_list,
    rootsol,
    phis_c_min,
    phis_c_max,
    folder_save=".",
    bad_par_filename="bad_par.txt",
    bad_sol_filename="bad_sol.txt",
    only_phi_CC=True,
    n_points_reduce=512,
    cyc_mode="discharge",
    run_mode=None,
):

    param_string = from_param_list_to_str(params_list)
    if "p2d" in str(type(rootsol)).lower():
        run_p2d = True
        run_spm = False
    elif "spm" in str(type(rootsol)).lower():
        run_p2d = False
        run_spm = True
    elif rootsol is not None:
        sys.exit("ERROR: battery model not recognized")
    try:
        assert rootsol is not None
        if only_phi_CC:
            sol_dict = {}
            sol_dict["t"] = rootsol.vars["time_s"]
            if run_spm:
                sol_dict["phis_c"] = rootsol.vars["voltage_V"]
            else:
                sol_dict["phis_c"] = np.expand_dims(
                    rootsol.vars["voltage_V"], axis=1
                )
        else:
            sol_dict = rootsol.to_dict()
        if cyc_mode in ["discharge", "chargecc", "discharge-chargecc"]:
            if phis_c_min is not -np.inf:
                try:
                    ind_t_max = np.argwhere(sol_dict["phis_c"] < phis_c_min)[
                        0
                    ][0]
                except IndexError:
                    ind_t_max = None
            elif phis_c_max is not np.inf:
                try:
                    ind_t_max = np.argwhere(sol_dict["phis_c"] > phis_c_max)[
                        0
                    ][0]
                except IndexError:
                    ind_t_max = None
            else:
                ind_t_max = None
        else:
            ind_t_max = None

        save_dict = {}

        if run_spm:
            # SPM
            if only_phi_CC:
                save_dict["t"] = sol_dict["t"][:ind_t_max]
                save_dict["phis_c"] = sol_dict["phis_c"][:ind_t_max]
            else:
                save_dict["t"] = sol_dict["t"][:ind_t_max]
                save_dict["cs_a"] = sol_dict["cs_a"][:ind_t_max]
                save_dict["cs_c"] = sol_dict["cs_c"][:ind_t_max]
                save_dict["phie"] = sol_dict["phie"][:ind_t_max]
                save_dict["phis_c"] = sol_dict["phis_c"][:ind_t_max]
        elif run_p2d:
            # P2D
            if only_phi_CC:
                save_dict["t"] = sol_dict["t"][:ind_t_max]
                save_dict["phis_c"] = sol_dict["phis_c"][:ind_t_max, -1]
            else:
                save_dict["t"] = sol_dict["t"][:ind_t_max]
                save_dict["cs_a"] = sol_dict["cs_a"][:ind_t_max]
                save_dict["cs_c"] = sol_dict["cs_c"][:ind_t_max]
                save_dict["phie"] = sol_dict["phie"][:ind_t_max]
                save_dict["phis_c"] = sol_dict["phis_c"][:ind_t_max]

                save_dict["ce"] = sol_dict["ce"][:ind_t_max]
                save_dict["phis_a"] = sol_dict["phis_a"][:ind_t_max]
                save_dict["ie"] = sol_dict["ie"][:ind_t_max]
                save_dict["j_a"] = sol_dict["j_a"][:ind_t_max]
                save_dict["j_c"] = sol_dict["j_c"][:ind_t_max]

        t = sol_dict["t"]
        phis_c = sol_dict["phis_c"]
        assert np.amax(phis_c) - np.amin(phis_c) > 0.1

        if cyc_mode.lower() in ["discharge", "chargecc", "discharge-chargecc"]:
            if run_p2d:
                diff_dict = calc_dqdv_dvdq(t, phis_c[:, -1])
            elif run_spm:
                diff_dict = calc_dqdv_dvdq(t, phis_c)
        elif cyc_mode.lower() in ["rh", "lh"]:
            diff_dict = {}

        for key in diff_dict:
            if key in save_dict:
                raise ValueError(f"ERROR: save_dict already contains {key}")
            save_dict[key] = diff_dict[key]

        # Reduce if needed
        if len(save_dict["t"]) > n_points_reduce and only_phi_CC:
            t_int = np.linspace(
                np.nanmin(save_dict["t"]),
                np.nanmax(save_dict["t"]),
                n_points_reduce,
            )
            phis_c_int = np.interp(t_int, save_dict["t"], save_dict["phis_c"])
            save_dict["t"] = t_int
            save_dict["phis_c"] = phis_c_int

            if len(diff_dict) > 0:
                t_diff_int = np.linspace(
                    np.nanmin(save_dict["t_diff"]),
                    np.nanmax(save_dict["t_diff"]),
                    n_points_reduce,
                )
                phis_c_diff_int = np.interp(
                    t_diff_int, save_dict["t_diff"], save_dict["phis_c_diff"]
                )
                dvdq_int = np.interp(
                    t_diff_int, save_dict["t_diff"], save_dict["dvdq"]
                )
                dqdv_int = np.interp(
                    t_diff_int, save_dict["t_diff"], save_dict["dqdv"]
                )

                t_diff_crop_int = np.linspace(
                    np.nanmin(save_dict["t_diff_crop"]),
                    np.nanmax(save_dict["t_diff_crop"]),
                    n_points_reduce,
                )
                phis_c_diff_crop_int = np.interp(
                    t_diff_crop_int,
                    save_dict["t_diff_crop"],
                    save_dict["phis_c_diff_crop"],
                )
                dvdq_crop_int = np.interp(
                    t_diff_crop_int,
                    save_dict["t_diff_crop"],
                    save_dict["dvdq_crop"],
                )
                dqdv_crop_int = np.interp(
                    t_diff_crop_int,
                    save_dict["t_diff_crop"],
                    save_dict["dqdv_crop"],
                )

                save_dict["t_diff"] = t_diff_int
                save_dict["phis_c_diff"] = phis_c_diff_int
                save_dict["dvdq"] = dvdq_int
                save_dict["dqdv"] = dqdv_int

                save_dict["t_diff_crop"] = t_diff_crop_int
                save_dict["phis_c_diff_crop"] = phis_c_diff_crop_int
                save_dict["dvdq_crop"] = dvdq_crop_int
                save_dict["dqdv_crop"] = dqdv_crop_int

        return save_dict, param_string

    except (AssertionError, TypeError, AttributeError) as err:
        print(f"ERROR: {err}")
        with open(os.path.join(folder_save, bad_sol_filename), "a+") as f:
            f.write(f"solution{param_string}.npz\n")
        with open(os.path.join(folder_save, bad_par_filename), "a+") as f:
            string_par = ""
            for parameter in params_list:
                string_par += f"{parameter:g} "
            f.write(f"{string_par}\n")

        return None, None


def save_datapoint(
    params_list,
    rootsol,
    phis_c_min,
    phis_c_max,
    folder_save=".",
    save_separate_sols=True,
    save_combined_sols=True,
    db: PickleDB | None = None,
    bad_par_filename="bad_par.txt",
    bad_sol_filename="bad_sol.txt",
    only_phi_CC=True,
    n_points_reduce=512,
    cyc_mode="discharge",
    run_mode=None,
):

    if cyc_mode.lower() in ["discharge-chargecc", "discharge"]:
        if cyc_mode.lower() == "discharge-chargecc":
            p_list = params_list[0]
            rsol = rootsol[0]
        else:
            p_list = params_list
            rsol = rootsol
        save_dict_dis, param_string_dis = single_run_save(
            p_list,
            rsol,
            phis_c_min=phis_c_min,
            phis_c_max=phis_c_max,
            folder_save=folder_save,
            bad_par_filename=bad_par_filename,
            bad_sol_filename=bad_sol_filename,
            only_phi_CC=only_phi_CC,
            n_points_reduce=n_points_reduce,
            cyc_mode=cyc_mode,
            run_mode="discharge",
        )
        if save_dict_dis is None:
            save_separate_sols = False
            save_combined_sols = False

    if cyc_mode.lower() in ["discharge-chargecc", "chargecc"]:
        if cyc_mode.lower() == "discharge-chargecc":
            p_list = params_list[1]
            rsol = rootsol[1]
        else:
            p_list = params_list
            rsol = rootsol
        if cyc_mode.lower() == "discharge-chargecc" and (
            save_dict_dis is None
        ):
            save_dict_chcc = None
            param_string_chcc = None
        else:
            save_dict_chcc, param_string_chcc = single_run_save(
                p_list,
                rsol,
                phis_c_min=phis_c_min,
                phis_c_max=phis_c_max,
                folder_save=folder_save,
                bad_par_filename=bad_par_filename,
                bad_sol_filename=bad_sol_filename,
                only_phi_CC=only_phi_CC,
                n_points_reduce=n_points_reduce,
                cyc_mode=cyc_mode,
                run_mode="chargecc",
            )
        if save_dict_chcc is None:
            save_separate_sols = False
            save_combined_sols = False
    if cyc_mode.lower() in ["rh"]:
        p_list = params_list
        rsol = rootsol
        save_dict_rh, param_string_rh = single_run_save(
            p_list,
            rsol,
            phis_c_min=phis_c_min,
            phis_c_max=phis_c_max,
            folder_save=folder_save,
            bad_par_filename=bad_par_filename,
            bad_sol_filename=bad_sol_filename,
            only_phi_CC=only_phi_CC,
            n_points_reduce=n_points_reduce,
            cyc_mode=cyc_mode,
        )
        if save_dict_rh is None:
            save_separate_sols = False
            save_combined_sols = False
    if cyc_mode.lower() in ["lh"]:
        p_list = params_list
        rsol = rootsol
        save_dict_lh, param_string_lh = single_run_save(
            p_list,
            rsol,
            phis_c_min=phis_c_min,
            phis_c_max=phis_c_max,
            folder_save=folder_save,
            bad_par_filename=bad_par_filename,
            bad_sol_filename=bad_sol_filename,
            only_phi_CC=only_phi_CC,
            n_points_reduce=n_points_reduce,
            cyc_mode=cyc_mode,
        )
        if save_dict_lh is None:
            save_separate_sols = False
            save_combined_sols = False

    if cyc_mode.lower() not in [
        "lh",
        "rh",
        "discharge-chargecc",
        "discharge",
        "chargecc",
    ]:
        raise NotImplementedError

    if cyc_mode.lower() == "discharge-chargecc":
        assert param_string_chcc == param_string_dis
        param_string = param_string_chcc
        params_list = params_list[0]
    elif cyc_mode.lower() == "discharge":
        param_string = param_string_dis
        save_dict = save_dict_dis
    elif cyc_mode.lower() == "chargecc":
        param_string = param_string_chcc
        save_dict = save_dict_chcc
    elif cyc_mode.lower() == "rh":
        param_string = param_string_rh
        save_dict = save_dict_rh
    elif cyc_mode.lower() == "lh":
        param_string = param_string_lh
        save_dict = save_dict_lh

    if save_separate_sols:
        if cyc_mode.lower() == "discharge-chargecc":
            np.savez(
                os.path.join(
                    folder_save,
                    f"solution_discharge_{param_string}.npz",
                ),
                **save_dict_dis,
            )
            np.savez(
                os.path.join(
                    folder_save,
                    f"solution_chargecc_{param_string}.npz",
                ),
                **save_dict_chcc,
            )
        elif cyc_mode.lower() in ["discharge", "chargecc", "rh", "lh"]:
            np.savez(
                os.path.join(folder_save, f"solution{param_string}.npz"),
                **save_dict,
            )

    if save_combined_sols:
        assert db is not None
        sim_count = db.n_data
        sim_id = sim_count + 1

        combined_data = {"sim_id": int(sim_id)}
        ## Convert to SP
        # for key in save_dict:
        #    save_dict[key] = save_dict[key].astype("float32")
        # params_list = [np.float32(entry) for entry in params_list]

        combined_data["params"] = params_list
        if cyc_mode.lower() in ["discharge", "chargecc", "rh", "lh"]:
            combined_data["sol"] = save_dict
        elif cyc_mode.lower() == "discharge-chargecc":
            combined_data["sol_dis"] = save_dict_dis
            combined_data["sol_chcc"] = save_dict_chcc
        db.append(combined_data, max_try=10)


def from_param_list_to_str(params_list, params_name=None):
    param_string = ""
    if params_list is not None:
        if isinstance(params_list[0], str):
            params_list_val = [float(val) for val in params_list]
        else:
            params_list_val = params_list
        if params_name is None:
            for paramval in params_list_val:
                param_string += "_"
                param_string += f"{paramval:g}"
        else:
            for paramval, name in zip(params_list_val, params_name):
                param_string += f"_{name}_"
                param_string += f"{paramval:g}"
    return param_string


def from_param_list_to_dict(params_list, params):
    deg_dict = {}
    for ipar, name in enumerate(params["deg_param_names"]):
        if params_list is not None:
            if isinstance(params_list[0], str):
                deg_dict[name] = float(params_list[ipar])
            else:
                deg_dict[name] = params_list[ipar]
        else:
            deg_dict[name] = params["deg_" + name + "_ref"]
    return deg_dict


def from_param_list_to_str(params_list, params_name=None):
    param_string = ""
    if params_list is not None:
        if isinstance(params_list[0], str):
            params_list_val = [float(val) for val in params_list]
        else:
            params_list_val = params_list
        if params_name is None:
            for paramval in params_list_val:
                param_string += "_"
                param_string += f"{paramval:g}"
        else:
            for paramval, name in zip(params_list_val, params_name):
                param_string += f"_{name}_"
                param_string += f"{paramval:g}"
    return param_string


def from_param_list_to_dict(params_list, params):
    deg_dict = {}
    for ipar, name in enumerate(params["deg_param_names"]):
        if params_list is not None:
            if isinstance(params_list[0], str):
                deg_dict[name] = float(params_list[ipar])
            else:
                deg_dict[name] = params_list[ipar]
        else:
            deg_dict[name] = params["deg_" + name + "_ref"]
    return deg_dict


def clean_sol_par(
    folder_save=".",
    param_list_file="parameter_list.txt",
    sol_list_file="solution_list.txt",
    bad_par_file="bad_par.txt",
    bad_sol_file="bad_sol.txt",
    param_list_multi_file="parameter_list_multi.txt",
    sol_list_multi_file="solution_list_multi.txt",
):

    param_list_file = os.path.join(folder_save, param_list_file)
    sol_list_file = os.path.join(folder_save, sol_list_file)
    bad_par_list_file = os.path.join(folder_save, bad_par_list_file)
    bad_sol_list_file = os.path.join(folder_save, bad_sol_list_file)
    param_list_multi_file = os.path.join(folder_save, param_list_multi_file)
    sol_list_multi_file = os.path.join(folder_save, sol_list_multi_file)

    with open(param_list_file, "r+") as f:
        old_par_lines = f.readlines()
    with open(sol_list_file, "r+") as f:
        old_sol_lines = f.readlines()
    if not os.path.isfile(bad_par_file):
        with open(param_list_multi_file, "w+") as f:
            for line in old_par_lines:
                f.write(line)
        with open(sol_list_multi_file, "w+") as f:
            for line in old_sol_lines:
                f.write(line)
        return

    with open(bad_par_file, "r+") as f:
        bad_par_lines = f.readlines()
    with open(bad_sol_file, "r+") as f:
        bad_sol_lines = f.readlines()

    with open(param_list_multi_file, "w+") as f:
        count_remove = 0
        for line in old_par_lines:
            if line not in bad_par_lines:
                f.write(line)
            else:
                count_remove += 1
        print(f"Removed {count_remove} param")
    with open(sol_list_multi_file, "w+") as f:
        count_remove = 0
        for line in old_sol_lines:
            if line not in bad_sol_lines:
                f.write(line)
            else:
                count_remove += 1
        print(f"Removed {count_remove} sol")


def read_list_param(
    folder_save=".", param_list_file="parameter_list.txt", parameter_list=[]
):
    param_list_file = os.path.join(folder_save, param_list_file)
    if not os.path.isfile(param_list_file):
        return parameter_list
    with open(param_list_file, "r+") as f:
        lines = f.readlines()
    for line in lines:
        parameter_list.append([float(entry) for entry in line.split()])
    return parameter_list


def read_list_sol(
    folder_save=".", sol_list_file="solution_list.txt", solution_list=[]
):
    sol_list_file = os.path.join(folder_save, sol_list_file)
    if not os.path.isfile(sol_list_file):
        return solution_list
    with open(sol_list_file, "r+") as f:
        lines = f.readlines()
    for line in lines:
        solution_list.append(line[:-1])
    return solution_list


def check_degparamdict(deg_param_dict, sim_params, parallel_env=None):
    for deg_param_name in sim_params["deg_param_names"]:
        try:
            assert (
                deg_param_dict[deg_param_name]
                >= sim_params["deg_" + deg_param_name + "_min"]
            )
            assert (
                deg_param_dict[deg_param_name]
                <= sim_params["deg_" + deg_param_name + "_max"]
            )
        except AssertionError:
            msg = f"ERROR: In dict {deg_param_dict}\n\tParameter {deg_param_name} = {deg_param_dict[deg_param_name]} out of bounds ({sim_params['deg_' + deg_param_name + '_min']}-{sim_params['deg_' + deg_param_name + '_max']})"
            if parallel_env is None:
                sys.exit(msg)
            else:
                parallel_env.printAll(msg)
                parallel_env.comm.Abort()


def check_degparamlist(deg_param_list, sim_params, parallel_env=None):
    for deg_val, deg_param_name in zip(
        deg_param_list, sim_params["deg_param_names"]
    ):
        try:
            assert deg_val >= sim_params["deg_" + deg_param_name + "_min"]
            assert deg_val <= sim_params["deg_" + deg_param_name + "_max"]

        except AssertionError:
            msg = f"ERROR: In list {deg_param_list}\n\t"
            msg += f"Parameter {deg_param_name} = {deg_val} out of bounds"
            msg += f"({sim_params['deg_' + deg_param_name + '_min']}-"
            msg += f"{sim_params['deg_' + deg_param_name + '_max']})"
            if parallel_env is None:
                sys.exit(msg)
            else:
                parallel_env.printAll(msg)
                parallel_env.comm.Abort()


def from_degparamlist_to_degparamdict(
    deg_param_list, sim_params, parallel_env=None
):
    check_degparamlist(deg_param_list, sim_params, parallel_env)
    deg_param_dict = {}
    for deg_param_val, deg_param_name in zip(
        deg_param_list, sim_params["deg_param_names"]
    ):
        deg_param_dict[deg_param_name] = deg_param_val
    check_degparamdict(deg_param_dict, sim_params, parallel_env)
    return deg_param_dict


def from_degparamdict_to_degparamlist(
    deg_param_dict, sim_params, parallel_env=None
):
    check_degparamdict(deg_param_dict, sim_params, parallel_env)
    deg_param_list = []
    for deg_param_name in sim_params["deg_param_names"]:
        deg_param_list.append(deg_param_dict[deg_param_name])
    check_degparamlist(deg_param_list, sim_params, parallel_env)
    return deg_param_list


def multi_run_ser(
    sim_params,
    param_list_file="parameter_list.txt",
    sol_list_file="solution_list.txt",
    bad_par_file="bad_par.txt",
    bad_sol_file="bad_sol.txt",
    save_separate_sols=False,
    save_combined_sols=True,
    folder_save=".",
    only_phi_CC=True,
    n_points_reduce=512,
):

    cyc_mode = sim_params["cyc_mode"]
    log_dir = Path(folder_save)
    log_dir.mkdir(parents=True, exist_ok=True)
    remove_file(os.path.join(folder_save, bad_par_file))
    remove_file(os.path.join(folder_save, bad_sol_file))
    remove_file(os.path.join(folder_save, "sols.pkl"))

    deg_parameter_list = read_list_param(
        folder_save=folder_save, param_list_file=param_list_file
    )
    solution_list = read_list_sol(
        folder_save=folder_save, sol_list_file=sol_list_file
    )
    try:
        assert len(deg_parameter_list) == len(solution_list)
    except AssertionError:
        msg = f"ERROR: deg_parameter_list (len={len(deg_parameter_list)}) and"
        msg += f"solution_list (len={len(solution_list)}) are inconsistent"
        print(msg)
        sys.exit()

    nsim = len(solution_list)

    db = PickleDB(filename=os.path.join(folder_save, "sols.pkl"))

    for count, (deg_param_entry, solution_entry) in enumerate(
        zip(deg_parameter_list, solution_list)
    ):
        if cyc_mode.lower() in ["discharge", "chargecc", "rh", "lh"]:
            params_list, root_sol = single_run(
                sim_params=sim_params,
                deg_param_sample=from_degparamlist_to_degparamdict(
                    deg_param_entry, sim_params, parallel_env=None
                ),
                count=count,
                nsim=nsim,
            )
            save_datapoint(
                params_list,
                root_sol,
                phis_c_min=sim_params["vmin"],
                phis_c_max=sim_params["vmax"],
                folder_save=folder_save,
                save_separate_sols=save_separate_sols,
                save_combined_sols=save_combined_sols,
                db=db,
                bad_par_filename="bad_par.txt",
                bad_sol_filename="bad_sol.txt",
                only_phi_CC=only_phi_CC,
                cyc_mode=cyc_mode,
                n_points_reduce=n_points_reduce,
            )
        elif cyc_mode.lower() == "discharge-chargecc":
            params_list = []
            root_sol = []
            for run_mode in ["discharge", "chargecc"]:
                params_list_i, root_sol_i = single_run(
                    sim_params=sim_params,
                    deg_param_sample=from_degparamlist_to_degparamdict(
                        deg_param_entry, sim_params, parallel_env=None
                    ),
                    run_mode=run_mode,
                    count=count,
                    nsim=nsim,
                )
                params_list.append(params_list_i)
                root_sol.append(root_sol_i)

            save_datapoint(
                params_list,
                root_sol,
                phis_c_min=sim_params["vmin"],
                phis_c_max=sim_params["vmax"],
                folder_save=folder_save,
                save_separate_sols=save_separate_sols,
                save_combined_sols=save_combined_sols,
                db=db,
                bad_par_filename="bad_par.txt",
                bad_sol_filename="bad_sol.txt",
                only_phi_CC=only_phi_CC,
                cyc_mode=cyc_mode,
                n_points_reduce=n_points_reduce,
            )

    # Rewrite with the appropriate format
    records = db.read(max_try=10)
    sols = {}
    for record in records:
        sols[record["sim_id"]] = record

    remove_file(os.path.join(folder_save, "sols.pkl"))
    with open(os.path.join(folder_save, "sols.pkl"), "wb") as f:
        pickle.dump(sols, f)


def merge_badpar_badsol(
    sim_params,
    parallel_env=None,
    folder_save=".",
    bad_par_file="bad_par.txt",
    bad_sol_file="bad_sol.txt",
):
    parallel_env.comm.Barrier()

    if not parallel_env.irank == parallel_env.iroot:
        return
    param_list = []
    sol_list = []

    for rank in range(
        parallel_env.iroot, parallel_env.nProc + parallel_env.iroot
    ):
        param_list = read_list_param(
            folder_save=folder_save,
            param_list_file=f"bad_par_filename_{rank}.txt",
            parameter_list=param_list,
        )
        sol_list = read_list_sol(
            folder_save=folder_save,
            sol_list_file=f"bad_sol_filename_{rank}.txt",
            solution_list=sol_list,
        )

    with open(os.path.join(folder_save, bad_par_file), "w+") as f:
        for param_entry in param_list:
            string_par = ""
            for parameter in param_entry:
                string_par += f"{parameter:g} "
            f.write(f"{string_par}\n")
    with open(os.path.join(folder_save, bad_sol_file), "w+") as f:
        for sol_entry in sol_list:
            f.write(sol_entry + "\n")

    for rank in range(
        parallel_env.iroot, parallel_env.nProc + parallel_env.iroot
    ):
        remove_file(os.path.join(folder_save, f"bad_par_filename_{rank}.txt"))
        remove_file(os.path.join(folder_save, f"bad_sol_filename_{rank}.txt"))


def merge_combined_sols(
    sim_params,
    parallel_env=None,
    nProc=None,
    folder_save=".",
    combined_sols_filename="sols.pkl",
    n_points_reduce: int | None = None,
):

    if parallel_env is not None:
        parallel_env.comm.Barrier()

        if not parallel_env.irank == parallel_env.iroot:
            return

        logger.info("\n\nmerging all databases")

        sols = {}
        offset_arr = np.zeros(parallel_env.nProc, dtype=int)

        for rank in range(
            parallel_env.iroot, parallel_env.nProc + parallel_env.iroot
        ):
            file_to_merge = os.path.join(folder_save, f"sols_{rank}.pkl")
            try:
                db_ = PickleDB(filename=file_to_merge, read_from_existing=True)
                records_ = db_.read(max_try=10)
                offset_arr[rank - parallel_env.iroot] = db_.n_data
                if rank == parallel_env.iroot:
                    offset = 0
                else:
                    offset += offset_arr[rank - 1 - parallel_env.iroot]
                if n_points_reduce is not None:
                    records_ = reduce_npoints_records(
                        records_, n_points_reduce=n_points_reduce
                    )
                for record_ in records_:
                    sim_id = record_["sim_id"]
                    del record_["sim_id"]
                    sols[int(sim_id + offset - 1)] = record_
            except FileNotFoundError:
                logger.warning(f"{file_to_merge} was not found")

        logger.info("writing final database")
        for rank in range(
            parallel_env.iroot, parallel_env.nProc + parallel_env.iroot
        ):
            remove_file(os.path.join(folder_save, f"sols_{rank}.pkl"))
        remove_file(os.path.join(folder_save, combined_sols_filename))
        with open(
            os.path.join(folder_save, combined_sols_filename), "wb"
        ) as f:
            pickle.dump(sols, f)
    else:

        assert nProc is not None
        logger.info("\n\nmerging all databases")

        sols = {}
        offset_arr = np.zeros(nProc, dtype=int)

        for rank in range(1, nProc + 1):
            file_to_merge = os.path.join(folder_save, f"sols_{rank}.pkl")
            logger.info(f"Treating {file_to_merge}")
            try:
                db_ = PickleDB(filename=file_to_merge, read_from_existing=True)
                records_ = db_.read(max_try=10)
                offset_arr[rank - 1] = db_.n_data
                if rank == 1:
                    offset = 0
                else:
                    offset += offset_arr[rank - 1 - 1]
                if n_points_reduce is not None:
                    records_ = reduce_npoints_records(
                        records_, n_points_reduce=n_points_reduce
                    )
                for record_ in records_:
                    sim_id = record_["sim_id"]
                    del record_["sim_id"]
                    sols[int(sim_id + offset - 1)] = record_
            except FileNotFoundError:
                logger.warning(f"{file_to_merge} was not found")

        logger.info("writing final database")
        for rank in range(1, nProc + 1):
            remove_file(os.path.join(folder_save, f"sols_{rank}.pkl"))
        remove_file(os.path.join(folder_save, combined_sols_filename))
        with open(
            os.path.join(folder_save, combined_sols_filename), "wb"
        ) as f:
            pickle.dump(sols, f)


def multi_run(
    sim_params,
    param_list_file="parameter_list.txt",
    sol_list_file="solution_list.txt",
    bad_par_file="bad_par.txt",
    bad_sol_file="bad_sol.txt",
    save_separate_sols=False,
    save_combined_sols=True,
    folder_save=".",
    parallel_env=None,
    only_phi_CC=True,
    n_points_reduce=512,
):

    cyc_mode = sim_params["cyc_mode"]

    if parallel_env is None or parallel_env.nProc == 1:
        return multi_run_ser(
            sim_params=sim_params,
            param_list_file=param_list_file,
            sol_list_file=sol_list_file,
            bad_par_file=bad_par_file,
            bad_sol_file=bad_sol_file,
            folder_save=folder_save,
            n_points_reduce=n_points_reduce,
        )
    else:

        if parallel_env.irank == parallel_env.iroot:
            log_dir = Path(folder_save)
            log_dir.mkdir(parents=True, exist_ok=True)
            remove_file(os.path.join(folder_save, bad_par_file))
            remove_file(os.path.join(folder_save, bad_sol_file))
            for rank in range(
                parallel_env.iroot, parallel_env.nProc + parallel_env.iroot
            ):
                remove_file(
                    os.path.join(folder_save, f"bad_par_filename_{rank}.txt")
                )
                remove_file(
                    os.path.join(folder_save, f"bad_sol_filename_{rank}.txt")
                )

        parallel_env.comm.Barrier()

        deg_parameter_list = read_list_param(
            folder_save=folder_save, param_list_file=param_list_file
        )
        solution_list = read_list_sol(
            folder_save=folder_save, sol_list_file=sol_list_file
        )
        parallel_env.printRoot("INFO: partition data to simulate")
        nsim_, startSim_ = parallel_env.partitionData(len(deg_parameter_list))

        if nsim_ == 0:
            parallel_env.printAll("WARNING: Nothing to do")

        remove_file(
            os.path.join(folder_save, f"sols_{parallel_env.irank}.pkl")
        )

        db_ = PickleDB(
            filename=os.path.join(
                folder_save, f"sols_{parallel_env.irank}.pkl"
            )
        )

        for count_, (deg_param_entry_, solution_entry_) in enumerate(
            zip(
                deg_parameter_list[startSim_ : startSim_ + nsim_],
                solution_list[startSim_ : startSim_ + nsim_],
            )
        ):
            if cyc_mode.lower() in ["discharge", "chargecc", "rh", "lh"]:
                params_list, root_sol = single_run(
                    sim_params=sim_params,
                    deg_param_sample=from_degparamlist_to_degparamdict(
                        deg_param_entry_, sim_params, parallel_env
                    ),
                    count=count_,
                    nsim=nsim_,
                    parallel_env=parallel_env,
                )
                save_datapoint(
                    params_list,
                    root_sol,
                    phis_c_min=sim_params["vmin"],
                    phis_c_max=sim_params["vmax"],
                    folder_save=folder_save,
                    save_separate_sols=save_separate_sols,
                    save_combined_sols=save_combined_sols,
                    db=db_,
                    bad_par_filename=f"bad_par_filename_{parallel_env.irank}.txt",
                    bad_sol_filename=f"bad_sol_filename_{parallel_env.irank}.txt",
                    only_phi_CC=only_phi_CC,
                    cyc_mode=cyc_mode,
                    n_points_reduce=n_points_reduce,
                )
            elif cyc_mode.lower() == "discharge-chargecc":
                params_list = []
                root_sol = []
                for run_mode in ["discharge", "chargecc"]:
                    params_list_i, root_sol_i = single_run(
                        sim_params=sim_params,
                        deg_param_sample=from_degparamlist_to_degparamdict(
                            deg_param_entry_, sim_params, parallel_env=None
                        ),
                        count=count_,
                        nsim=nsim_,
                        parallel_env=parallel_env,
                        run_mode=run_mode,
                    )
                    params_list.append(params_list_i)
                    root_sol.append(root_sol_i)

                save_datapoint(
                    params_list,
                    root_sol,
                    phis_c_min=sim_params["vmin"],
                    phis_c_max=sim_params["vmax"],
                    folder_save=folder_save,
                    save_separate_sols=save_separate_sols,
                    save_combined_sols=save_combined_sols,
                    db=db_,
                    bad_par_filename=f"bad_par_filename_{parallel_env.irank}.txt",
                    bad_sol_filename=f"bad_sol_filename_{parallel_env.irank}.txt",
                    only_phi_CC=only_phi_CC,
                    cyc_mode=cyc_mode,
                    n_points_reduce=n_points_reduce,
                )

        merge_badpar_badsol(
            sim_params=sim_params,
            parallel_env=parallel_env,
            folder_save=folder_save,
            bad_par_file="bad_par.txt",
            bad_sol_file="bad_sol.txt",
        )
        if save_combined_sols:
            merge_combined_sols(
                sim_params=sim_params,
                parallel_env=parallel_env,
                folder_save=folder_save,
                combined_sols_filename="sols.pkl",
            )


if __name__ == "__main__":
    import argparse

    import batfit.utils.parallel as parallel_env
    from batfit import BATFIT_EXP
    from batfit.preprocess.sim_setup import make_params

    parser = argparse.ArgumentParser(description="dataset generator")
    parser.add_argument(
        "-sim_config",
        "--sim_config",
        type=str,
        metavar="",
        required=False,
        help="Sim config file",
        default=os.path.join(BATFIT_EXP, "spm_discharge_charge_C4.yaml"),
    )
    parser.add_argument(
        "-folder_save",
        "--folder_save",
        type=str,
        metavar="",
        required=False,
        help="Data folder",
        default=".",
    )
    parser.add_argument(
        "-cm",
        "--cyc_mode",
        type=str,
        metavar="",
        required=False,
        help="cycling mode",
        default="discharge-chargecc",
    )

    args, unknown = parser.parse_known_args()

    sim_params = make_params(args.sim_config, parallel_env=parallel_env)
    multi_run(
        sim_params=sim_params,
        parallel_env=parallel_env,
        folder_save=args.folder_save,
        n_points_reduce=512,
    )
