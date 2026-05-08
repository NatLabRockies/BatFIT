import os
import pickle
import random
import sys
import time
from pathlib import Path
import re
import bmlite as bm
import numpy as np
import pandas as pd

from batfit import BATFIT_EXP, logger
from batfit.preprocess.diff_cap import calc_dqdv_dvdq
from batfit.preprocess.pickledb import PickleDB

from .hdvolts_prot import (
    define_diffcap_experiment,
    define_hppc_experiment,
    define_post_hppc_experiment,
    define_pre_hppc_experiment,
)
from .sim_setup import *
from .utils import *


def mod_sim(
    sim: bm.SPM._simulation.Simulation | bm.P2D._simulation.Simulation,
    sim_params: dict,
    deg_param_sample: dict,
    cyc_mode: str,
    run_mode: str,
):
    """
    Modify the parameters of a BatMODS-lite simulation

    Parameters
    ----------
    sim: bm.SPM._simulation.Simulation | bm.P2D._simulation.Simulation
        BatMODS-lite simulation
    sim_params: dict
        Parameters values used by BatMODS-lite
    deg_param_sample: dict
        Degradation parameter values: scaling parameters
    """

    sim = set_discretization(sim=sim, sim_params=sim_params)
    C_rate, sim = set_interc(
        sim=sim,
        sim_params=sim_params,
        deg_param_sample=deg_param_sample,
        cyc_mode=cyc_mode,
        run_mode=run_mode,
    )

    if isinstance(sim, bm.P2D._simulation.Simulation):
        is_p2d = True
    else:
        is_p2d = False

    sim = set_battery(
        sim=sim,
        sim_params=sim_params,
        deg_param_sample=deg_param_sample,
        cyc_mode=cyc_mode,
        run_mode=run_mode,
        is_p2d=is_p2d,
    )
    sim = set_electrodes(
        sim=sim,
        sim_params=sim_params,
        deg_param_sample=deg_param_sample,
        cyc_mode=cyc_mode,
        run_mode=run_mode,
        is_p2d=is_p2d,
    )
    sim = set_electrolyte(
        sim=sim,
        sim_params=sim_params,
        deg_param_sample=deg_param_sample,
        cyc_mode=cyc_mode,
        run_mode=run_mode,
        is_p2d=is_p2d,
    )
    sim = set_separator(
        sim=sim,
        sim_params=sim_params,
        deg_param_sample=deg_param_sample,
        cyc_mode=cyc_mode,
        run_mode=run_mode,
        is_p2d=is_p2d,
    )

    return sim, C_rate


def robust_DiffCap(sim, sim_params, force_fail=False):
    if force_fail:
        return None

    sol = None
    try:
        exp = define_diffcap_experiment(sim_params)
        sol = sim.run(exp, reset_state=True, bar=False)
        assert all(sol.success)
    except:
        #for atol, max_step in zip(
        #    [1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13],
        #    [
        #        int(1e3),
        #        int(1e4),
        #        int(1e5),
        #        int(1e6),
        #        int(1e6),
        #        int(1e6),
        #        int(1e6),
        #        int(1e6),
        #    ],
        #):
        for atol, max_step in zip(
            [1e-6, 1e-12],
            [
                int(1e3),
                int(1e6),
            ],
        ):
            try:
                exp = define_diffcap_experiment(
                    sim_params, atol=atol, max_step=max_step
                )
                sol = sim.run(exp, reset_state=False, bar=False)
                assert all(sol.success)
                break
            except:
                pass
        pass
    return sol

def robust_preHPPC(sim, sim_params, force_fail=False):
    if force_fail:
        return None

    sol = None
    try:
        exp = define_pre_hppc_experiment(sim_params)
        sol = sim.run(exp, reset_state=True, bar=False)
        assert all(sol.success)
    except:
        for atol, max_step in zip(
            [1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13],
            [
                int(1e3),
                int(1e4),
                int(1e5),
                int(1e6),
                int(1e6),
                int(1e6),
                int(1e6),
                int(1e6),
            ],
        ):
            try:
                exp = define_pre_hppc_experiment(
                    sim_params, atol=atol, max_step=max_step
                )
                sol = sim.run(exp, reset_state=False, bar=False)
                assert all(sol.success)
                break
            except:
                pass
        pass
    return sol


def robust_HPPC(sim, sim_params, force_fail=False):
    if force_fail:
        return None
    sol = None
    try:
        exp = define_hppc_experiment(sim_params)
        sol = sim.run(exp, reset_state=True, bar=False)
        assert all(sol.success)
    except:
        counter = 0
        #for atol, max_step in zip(
        #    [1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13],
        #    [
        #        int(1e3),
        #        int(1e4),
        #        int(1e5),
        #        int(1e6),
        #        int(1e6),
        #        int(1e6),
        #        int(1e6),
        #        int(1e6),
        #    ],
        #):
        for atol, max_step in zip(
            [1e-6, 1e-12],
            [
                int(1e3),
                int(1e6),
            ],
        ):
            try:
                counter += 1
                exp = define_hppc_experiment(
                    sim_params, atol=atol, max_step=max_step
                )
                sol = sim.run(exp, reset_state=False, bar=False)
                assert all(sol.success)
                break
            except:
                pass
        pass
    return sol

def robust_postHPPC(sim, sim_params, force_fail=False):
    if force_fail:
        return None
    sol = None
    exp = define_post_hppc_experiment(sim_params)
    try:
        exp = define_post_hppc_experiment(sim_params)
        sol = sim.run(exp, reset_state=True, bar=False)
        assert all(sol.success)
    except:
        counter = 0
        #for atol, max_step in zip(
        #    [1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13],
        #    [
        #        int(1e3),
        #        int(1e4),
        #        int(1e5),
        #        int(1e6),
        #        int(1e6),
        #        int(1e6),
        #        int(1e6),
        #        int(1e6),
        #    ],
        #):
        for atol, max_step in zip(
            [1e-12],
            [
                int(1e3),
            ],
        ):
            try:
                counter += 1
                exp = define_post_hppc_experiment(
                    sim_params, atol=atol, max_step=max_step
                )
                sol = sim.run(exp, reset_state=False, bar=False)
                assert all(sol.success)
                break
            except:
                pass
        pass
    return sol

def robust_LHRH(
    sim, df, charge, protocol, sim_params, bat_model, force_fail=False
):
    raise NotImplementedError("timespan needs to be defined differently now")
    if force_fail:
        return None
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
            logger.error("Battery model not recognized")
            sys.exit()
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
                    logger.error("Battery model not recognized")
                    sys.exit()
                rootsol = sim_prot
                assert all(rootsol.success)
                break
            except:
                # print(f"sim failed for {deg_param_sample}")
                pass
    return rootsol


def robust_CC(sim, C_rate, sim_params, force_fail=False):
    raise NotImplementedError("timespan needs to be defined differently now")
    if force_fail:
        return None

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

    # if any exception occurred during pre, then fail
    force_fail = False
    try:
        sim.pre()
    except Exception:
        force_fail = True

    # print(deg_param_sample)
    # print_an(sim)
    # print_ca(sim)

    time_s = time.time()
    if cyc_mode.lower() in ["discharge", "chargecc", "discharge-chargecc"]:
        rootsol = robust_CC(
            sim=sim,
            C_rate=C_rate,
            sim_params=sim_params,
            force_fail=force_fail,
        )
        if rootsol is None:
            print(f"All sim failed for {deg_param_sample}")
        else:
            print(f"Success for {deg_param_sample}")

    elif cyc_mode.lower() in ["diffcap", "hppc", "prehppc", "posthppc"]:
        if cyc_mode.lower() == "diffcap":
            rootsol = robust_DiffCap(
                sim=sim,
                sim_params=sim_params,
                force_fail=force_fail,
            )
            if rootsol is None:
                print(f"All sim failed for {deg_param_sample}")
            else:
                print(f"Success for {deg_param_sample}")
        if cyc_mode.lower() == "hppc":
            rootsol = robust_HPPC(
                sim=sim,
                sim_params=sim_params,
                force_fail=force_fail,
            )
            if rootsol is None:
                print(f"All sim failed for {deg_param_sample}")
            else:
                print(f"Success for {deg_param_sample}")
        if cyc_mode.lower() == "posthppc":
            rootsol = robust_postHPPC(
                sim=sim,
                sim_params=sim_params,
                force_fail=force_fail,
            )
            if rootsol is None:
                print(f"All sim failed for {deg_param_sample}")
            else:
                print(f"Success for {deg_param_sample}")
        if cyc_mode.lower() == "prehppc":
            rootsol = robust_preHPPC(
                sim=sim,
                sim_params=sim_params,
                force_fail=force_fail,
            )
            if rootsol is None:
                print(f"All sim failed for {deg_param_sample}")
            else:
                print(f"Success for {deg_param_sample}")

    elif cyc_mode.lower() in ["rh", "lh", "lh2"]:
        df = pd.read_csv(os.path.join(BATFIT_EXP, "LHmax.csv"))
        charge = bm.Experiment()
        charge.add_step(
            "current_C",
            -1.0,
            (7200.0, 60.0),
            limits=("voltage_V", sim_params["vmax"]),
        )
        charge.add_step("voltage_V", sim_params["vmax"], (3600.0, 30.0))
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
                force_fail=force_fail,
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
                force_fail=force_fail,
            )
            if rootsol is None:
                print(f"All sim failed for {deg_param_sample}")
            else:
                print(f"Success for {deg_param_sample}")
        if cyc_mode.lower() == "lh2":
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
                force_fail=force_fail,
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




def single_run_save(
    params_list,
    rootsol,
    phis_c_min,
    phis_c_max,
    folder_save=".",
    bad_par_filename="bad_par.txt",
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
        logger.error("Battery model not recognized")
        sys.exit()
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
        elif cyc_mode.lower() in [
            "rh",
            "lh",
            "lh2",
            "hppc",
            "posthppc",
            "prehppc",
            "diffcap",
        ]:
            diff_dict = {}

        for key in diff_dict:
            if key in save_dict:
                raise ValueError(f"Save_dict already contains {key}")
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
                only_phi_CC=only_phi_CC,
                n_points_reduce=n_points_reduce,
                cyc_mode=cyc_mode,
                run_mode="chargecc",
            )
        if save_dict_chcc is None:
            save_separate_sols = False
            save_combined_sols = False
    if cyc_mode.lower() in ["diffcap", "hppc", "prehppc", "posthppc"]:
        p_list = params_list
        rsol = rootsol
        save_dict, param_string = single_run_save(
            p_list,
            rsol,
            phis_c_min=phis_c_min,
            phis_c_max=phis_c_max,
            folder_save=folder_save,
            bad_par_filename=bad_par_filename,
            only_phi_CC=only_phi_CC,
            n_points_reduce=n_points_reduce,
            cyc_mode=cyc_mode,
        )
        if save_dict is None:
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
            only_phi_CC=only_phi_CC,
            n_points_reduce=n_points_reduce,
            cyc_mode=cyc_mode,
        )
        if save_dict_lh is None:
            save_separate_sols = False
            save_combined_sols = False
    if cyc_mode.lower() in ["lh2"]:
        p_list = params_list
        rsol = rootsol
        save_dict_lh, param_string_lh = single_run_save(
            p_list,
            rsol,
            phis_c_min=phis_c_min,
            phis_c_max=phis_c_max,
            folder_save=folder_save,
            bad_par_filename=bad_par_filename,
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
        "hppc",
        "posthppc",
        "prehppc",
        "diffcap",
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
    elif cyc_mode.lower() == "lh2":
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
        elif cyc_mode.lower() in [
            "discharge",
            "chargecc",
            "rh",
            "lh",
            "lh2",
            "hppc",
            "posthppc",
            "prehppc",
            "diffcap",
        ]:
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
        if cyc_mode.lower() in [
            "discharge",
            "chargecc",
            "rh",
            "lh",
            "lh2",
            "hppc",
            "posthppc",
            "prehppc",
            "diffcap",
        ]:
            combined_data["sol"] = save_dict
        elif cyc_mode.lower() == "discharge-chargecc":
            combined_data["sol_dis"] = save_dict_dis
            combined_data["sol_chcc"] = save_dict_chcc
        db.append(combined_data, max_try=10)


def clean_sol_par(
    folder_save=".",
    param_list_file="parameter_list.txt",
    bad_par_file="bad_par.txt",
    param_list_multi_file="parameter_list_multi.txt",
):

    param_list_file = os.path.join(folder_save, param_list_file)
    bad_par_list_file = os.path.join(folder_save, bad_par_list_file)
    param_list_multi_file = os.path.join(folder_save, param_list_multi_file)

    with open(param_list_file, "r+") as f:
        old_par_lines = f.readlines()
    if not os.path.isfile(bad_par_file):
        with open(param_list_multi_file, "w+") as f:
            for line in old_par_lines:
                f.write(line)
        return
    with open(bad_par_file, "r+") as f:
        bad_par_lines = f.readlines()

    with open(param_list_multi_file, "w+") as f:
        count_remove = 0
        for line in old_par_lines:
            if line not in bad_par_lines:
                f.write(line)
            else:
                count_remove += 1
        print(f"Removed {count_remove} param")


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


def multi_run_ser(
    sim_params,
    param_list_file="parameter_list.txt",
    bad_par_file="bad_par.txt",
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
    remove_file(os.path.join(folder_save, "sols.pkl"))

    deg_parameter_list = read_list_param(
        folder_save=folder_save, param_list_file=param_list_file
    )
    nsim = len(deg_parameter_list)

    db = PickleDB(filename=os.path.join(folder_save, "sols.pkl"))

    for count, deg_param_entry in enumerate(deg_parameter_list):
        if cyc_mode.lower() in [
            "discharge",
            "chargecc",
            "rh",
            "lh",
            "lh2",
            "prehppc",
            "hppc",
            "posthppc",
            "diffcap",
        ]:
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
):
    if parallel_env is not None:
        parallel_env.comm.Barrier()

        if not parallel_env.irank == parallel_env.iroot:
            return
        param_list = []

        for rank in range(
            parallel_env.iroot, parallel_env.nProc + parallel_env.iroot
        ):
            param_list = read_list_param(
                folder_save=folder_save,
                param_list_file=f"bad_par_filename_{rank}.txt",
                parameter_list=param_list,
            )

        with open(os.path.join(folder_save, bad_par_file), "w+") as f:
            for param_entry in param_list:
                string_par = ""
                for parameter in param_entry:
                    string_par += f"{parameter:g} "
                f.write(f"{string_par}\n")

        for rank in range(
            parallel_env.iroot, parallel_env.nProc + parallel_env.iroot
        ):
            remove_file(os.path.join(folder_save, f"bad_par_filename_{rank}.txt"))
    else:
        def list_bad_par_files(folder_save="."):
            directory = Path(folder_save)
            files = list(directory.glob("bad_par_filename_*.txt"))
            def extract_rank(filepath):
                match = re.search(r'bad_par_filename_(\d+)\.txt', filepath.name)
                if match:
                    return int(match.group(1))
                return -1
            sorted_files = sorted(files, key=extract_rank)
            return sorted_files

        param_list = []
        sorted_files =  list_bad_par_files(folder_save=folder_save)
        for filename in sorted_files:
            param_list = read_list_param(
                folder_save=folder_save,
                param_list_file=filename,
                parameter_list=param_list,
            )

        with open(os.path.join(folder_save, bad_par_file), "w+") as f:
            for param_entry in param_list:
                string_par = ""
                for parameter in param_entry:
                    string_par += f"{parameter:g} "
                f.write(f"{string_par}\n")

        for filename in sorted_files:
            remove_file(os.path.join(folder_save, filename))


def merge_combined_sols(
    sim_params,
    parallel_env=None,
    folder_save=".",
    combined_sols_filename="sols.pkl",
    n_points_reduce: int | None = None,
):

    if parallel_env is not None:
        parallel_env.comm.Barrier()

        if not parallel_env.irank == parallel_env.iroot:
            return

        logger.info("Merging all databases")

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

        logger.info("Writing final database")
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
        def list_sols_files(folder_save="."):
            directory = Path(folder_save)
            files = list(directory.glob("sols_*.pkl"))
            def extract_rank(filepath):
                match = re.search(r'sols_(\d+)\.pkl', filepath.name)
                if match:
                    return int(match.group(1))
                return -1
            sorted_files = sorted(files, key=extract_rank)
            return sorted_files

        logger.info("Merging all databases")
        sols = {}
        sorted_files = list_sols_files(folder_save=folder_save)
        offset_arr = np.zeros(len(sorted_files), dtype=int)

        for ifile, filename in enumerate(sorted_files):
            rank = ifile+1
            file_to_merge = os.path.join(folder_save, filename)
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

       
        logger.info("Writing final database")
        for filename in sorted_files:
            remove_file(os.path.join(folder_save, filename))
        remove_file(os.path.join(folder_save, combined_sols_filename))
        with open(
            os.path.join(folder_save, combined_sols_filename), "wb"
        ) as f:
            pickle.dump(sols, f)


def multi_run(
    sim_params,
    param_list_file="parameter_list.txt",
    bad_par_file="bad_par.txt",
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
            bad_par_file=bad_par_file,
            folder_save=folder_save,
            n_points_reduce=n_points_reduce,
        )
    else:

        if parallel_env.irank == parallel_env.iroot:
            log_dir = Path(folder_save)
            log_dir.mkdir(parents=True, exist_ok=True)
            remove_file(os.path.join(folder_save, bad_par_file))
            for rank in range(
                parallel_env.iroot, parallel_env.nProc + parallel_env.iroot
            ):
                remove_file(
                    os.path.join(folder_save, f"bad_par_filename_{rank}.txt")
                )

        parallel_env.comm.Barrier()

        deg_parameter_list = read_list_param(
            folder_save=folder_save, param_list_file=param_list_file
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

        for count_, deg_param_entry_ in enumerate(
            deg_parameter_list[startSim_ : startSim_ + nsim_]
        ):
            if cyc_mode.lower() in [
                "discharge",
                "chargecc",
                "rh",
                "lh",
                "lh2",
                "prehppc",
                "hppc",
                "posthppc",
                "diffcap",
            ]:
                params_list, root_sol = single_run(
                    sim_params=sim_params,
                    deg_param_sample=from_degparamlist_to_degparamdict(
                        deg_param_entry_, sim_params, parallel_env=parallel_env
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
                    only_phi_CC=only_phi_CC,
                    cyc_mode=cyc_mode,
                    n_points_reduce=n_points_reduce,
                )

        merge_badpar_badsol(
            sim_params=sim_params,
            parallel_env=parallel_env,
            folder_save=folder_save,
            bad_par_file="bad_par.txt",
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
