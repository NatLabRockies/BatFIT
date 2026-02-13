import os
import sys

import numpy as np
from ruamel.yaml import YAML
from scipy.stats import qmc

from batfit import logger


def parse_input(filename, parallel_env=None):
    yaml = YAML()
    with open(filename, "r") as f:
        exp = yaml.load(f)
    cyc_mode = exp["cycling mode"]
    deg_param_names_min = list(exp["min degradation parameter"].keys())
    deg_param_names_max = list(exp["max degradation parameter"].keys())
    assert set(deg_param_names_min) == set(deg_param_names_max)
    deg_param_names = [
        entry.strip()
        for entry in exp["degradation parameter names"].split(",")
    ]
    assert deg_param_names == deg_param_names_min
    assert deg_param_names == deg_param_names_max
    # deg_param_names = list(set(deg_param_names_min))
    # deg_param_names.sort()
    # deg_param_names = ["i0_a", "ds_c", "cs0_a", "cs0_c", "i0_c", "eps_s_c"]

    if cyc_mode == "discharge-chargecc":
        deg_param_names_chcc_min = list(
            exp["min degradation parameter charge"].keys()
        )
        deg_param_names_chcc_max = list(
            exp["max degradation parameter charge"].keys()
        )
        assert set(deg_param_names_chcc_min) == set(deg_param_names_chcc_max)
        deg_param_names_chcc = list(set(deg_param_names_chcc_min))
        deg_param_names_chcc = ["cs0_a", "cs0_c"]
        deg_param_names_chcc_new = [
            f"{entry}_chcc" for entry in deg_param_names_chcc
        ]
    elif cyc_mode.lower() in ["discharge", "chargecc"]:
        pass
    elif cyc_mode.lower() in ["rh", "lh"]:
        pass
    else:
        raise NotImplementedError

    try:
        deg_param_min = [
            exp["min degradation parameter"][param_name]
            for param_name in deg_param_names
        ]
        deg_param_max = [
            exp["max degradation parameter"][param_name]
            for param_name in deg_param_names
        ]
    except KeyError:
        print("ERROR: mismatch of parameters")
        raise KeyError

    if cyc_mode == "discharge-chargecc":
        try:
            deg_param_min_chcc = [
                exp["min degradation parameter charge"][param_name]
                for param_name in deg_param_names_chcc
            ]
            deg_param_max_chcc = [
                exp["max degradation parameter charge"][param_name]
                for param_name in deg_param_names_chcc
            ]
        except KeyError:
            print("ERROR: mismatch of parameters")
            raise KeyError
        deg_param_names += deg_param_names_chcc_new
        deg_param_min += deg_param_min_chcc
        deg_param_max += deg_param_max_chcc

    assert np.amin(np.array(deg_param_max) - np.array(deg_param_min)) > 0

    phy_par = {}
    phy_par["cyc_mode"] = cyc_mode
    if cyc_mode.lower() in ["discharge", "chargecc", "rh", "lh"]:
        phy_par["model"] = exp["macroscopic"]["model"]
        phy_par["cap"] = exp["macroscopic"]["cap"]
        try:
            phy_par["material"] = exp["macroscopic"]["material"]
        except:
            phy_par["material"] = "graphite_nmc532"
        print(f"INFO : material = {phy_par['material']}")
        try:
            phy_par["C"] = exp["macroscopic"]["C"]
        except KeyError:
            phy_par["C"] = None
        phy_par["x0_a"] = exp["anode"]["x_0"]
        phy_par["x0_c"] = exp["cathode"]["x_0"]
        phy_par["eps_s_c"] = exp["cathode"]["eps_s"]
        phy_par["eps_s_a"] = exp["anode"]["eps_s"]
        phy_par["eps_el_a"] = exp["anode"]["eps_el"]
        phy_par["eps_el_c"] = exp["cathode"]["eps_el"]
        phy_par["ce"] = exp["electrolyte"]["Li_0"]
        phy_par["eps_CBD_c"] = exp["cathode"]["eps_CBD"]
        phy_par["eps_CBD_a"] = exp["anode"]["eps_CBD"]
        phy_par["csanmax"] = exp["anode"]["Li_max"]
        phy_par["cscamax"] = exp["cathode"]["Li_max"]
        phy_par["L_a"] = exp["anode"]["thick"]
        phy_par["L_s"] = exp["separator"]["thick"]
        phy_par["L_c"] = exp["cathode"]["thick"]
        phy_par["Rs_a"] = exp["anode"]["R_s"]
        phy_par["Rs_c"] = exp["cathode"]["R_s"]

        if phy_par["model"].lower() == "p2d":
            try:
                phy_par["Nx_a"] = exp["anode"]["Nx"]
            except KeyError:
                phy_par["Nx_a"] = 32
            try:
                phy_par["Nx_s"] = exp["separator"]["Nx"]
            except KeyError:
                phy_par["Nx_s"] = 32
            try:
                phy_par["Nx_c"] = exp["cathode"]["Nx"]
            except KeyError:
                phy_par["Nx_c"] = 32
        try:
            phy_par["Nr_a"] = exp["anode"]["Nr"]
        except KeyError:
            phy_par["Nr_a"] = 30
        try:
            phy_par["Nr_c"] = exp["cathode"]["Nr"]
        except KeyError:
            phy_par["Nr_c"] = 30

        phy_par["area"] = exp["macroscopic"]["area"]
        try:
            phy_par["vmin"] = exp["macroscopic"]["vmin"]
        except KeyError:
            logger.warning("Using default min v 3V")
            phy_par["vmin"] = 3
        try:
            phy_par["vmax"] = exp["macroscopic"]["vmax"]
        except KeyError:
            logger.warning("Using default max v 4.1V")
            phy_par["vmax"] = 4.1
        if phy_par["model"].lower() == "p2d":
            phy_par["eps_el"] = exp["separator"]["eps_el"]
            phy_par["p_l"] = exp["separator"]["p_liq"]
            phy_par["p_s_a"] = exp["anode"]["p_sol"]
            phy_par["p_l_a"] = exp["anode"]["p_liq"]
            phy_par["p_s_c"] = exp["cathode"]["p_sol"]
            phy_par["p_l_c"] = exp["cathode"]["p_liq"]
    elif cyc_mode.lower() == "discharge-chargecc":
        try:
            phy_par["material"] = exp["macroscopic charge"]["material"]
        except:
            phy_par["material"] = "graphite_nmc532"
        print(f"INFO : material = {phy_par['material']}")
        assert (
            exp["macroscopic charge"]["model"]
            == exp["macroscopic discharge"]["model"]
        )
        phy_par["model"] = exp["macroscopic charge"]["model"]
        assert (
            exp["macroscopic charge"]["cap"]
            == exp["macroscopic discharge"]["cap"]
        )
        phy_par["cap"] = exp["macroscopic charge"]["cap"]
        phy_par["C_chcc"] = exp["macroscopic charge"]["C"]
        phy_par["C_dis"] = exp["macroscopic discharge"]["C"]
        phy_par["x0_a_chcc"] = exp["anode charge"]["x_0"]
        phy_par["x0_c_chcc"] = exp["cathode charge"]["x_0"]
        phy_par["x0_a_dis"] = exp["anode discharge"]["x_0"]
        phy_par["x0_c_dis"] = exp["cathode discharge"]["x_0"]
        assert (
            exp["cathode charge"]["eps_s"] == exp["cathode discharge"]["eps_s"]
        )
        phy_par["eps_s_c"] = exp["cathode charge"]["eps_s"]
        assert exp["anode charge"]["eps_s"] == exp["anode discharge"]["eps_s"]
        phy_par["eps_s_a"] = exp["anode charge"]["eps_s"]
        assert (
            exp["cathode charge"]["eps_CBD"]
            == exp["cathode discharge"]["eps_CBD"]
        )
        phy_par["eps_CBD_c"] = exp["cathode charge"]["eps_CBD"]
        assert (
            exp["anode charge"]["eps_CBD"] == exp["anode discharge"]["eps_CBD"]
        )
        phy_par["eps_CBD_a"] = exp["anode charge"]["eps_CBD"]
        assert (
            exp["anode charge"]["Li_max"] == exp["anode discharge"]["Li_max"]
        )
        phy_par["csanmax"] = exp["anode charge"]["Li_max"]
        assert (
            exp["cathode charge"]["Li_max"]
            == exp["cathode discharge"]["Li_max"]
        )
        phy_par["cscamax"] = exp["cathode charge"]["Li_max"]
        assert exp["anode charge"]["thick"] == exp["anode discharge"]["thick"]
        phy_par["L_a"] = exp["anode discharge"]["thick"]
        assert (
            exp["separator charge"]["thick"]
            == exp["separator discharge"]["thick"]
        )
        phy_par["L_s"] = exp["separator discharge"]["thick"]
        assert (
            exp["cathode charge"]["thick"] == exp["cathode discharge"]["thick"]
        )
        phy_par["L_c"] = exp["cathode discharge"]["thick"]
        assert (
            exp["anode charge"]["eps_el"] == exp["anode discharge"]["eps_el"]
        )
        phy_par["eps_el_a"] = exp["anode discharge"]["eps_el"]
        assert (
            exp["cathode charge"]["eps_el"]
            == exp["cathode discharge"]["eps_el"]
        )
        phy_par["eps_el_c"] = exp["cathode discharge"]["eps_el"]
        assert (
            exp["cathode charge"]["eps_el"]
            == exp["cathode discharge"]["eps_el"]
        )
        assert (
            exp["anode charge"]["eps_el"] == exp["anode discharge"]["eps_el"]
        )
        assert (
            exp["electrolyte charge"]["Li_0"]
            == exp["electrolyte discharge"]["Li_0"]
        )
        phy_par["ce"] = exp["electrolyte discharge"]["Li_0"]
        assert (
            exp["macroscopic charge"]["area"]
            == exp["macroscopic discharge"]["area"]
        )
        phy_par["area"] = exp["macroscopic discharge"]["area"]
        assert exp["anode charge"]["R_s"] == exp["anode discharge"]["R_s"]
        phy_par["Rs_a"] = exp["anode discharge"]["R_s"]
        assert exp["cathode charge"]["R_s"] == exp["cathode discharge"]["R_s"]
        phy_par["Rs_c"] = exp["cathode discharge"]["R_s"]
        try:
            phy_par["vmin"] = exp["macroscopic"]["vmin"]
        except KeyError:
            print("WARNING: Using default min v 3V")
            phy_par["vmin"] = 3
        try:
            phy_par["vmax"] = exp["macroscopic"]["vmax"]
        except KeyError:
            print("WARNING: Using default max v 4.1V")
            phy_par["vmax"] = 4.1
        try:
            phy_par["Nr_a"] = exp["anode discharge"]["Nr"]
        except KeyError:
            phy_par["Nr_a"] = 30
        if "Nr" in exp["anode discharge"]:
            assert exp["anode discharge"]["Nr"] == exp["anode charge"]["Nr"]
        try:
            phy_par["Nr_c"] = exp["cathode discharge"]["Nr"]
        except KeyError:
            phy_par["Nr_c"] = 30
        if "Nr" in exp["cathode discharge"]:
            assert (
                exp["cathode discharge"]["Nr"] == exp["cathode charge"]["Nr"]
            )

        if phy_par["model"].lower() == "p2d":
            assert (
                exp["separator charge"]["eps_el"]
                == exp["separator discharge"]["eps_el"]
            )
            phy_par["eps_el"] = exp["separator discharge"]["eps_el"]
            assert (
                exp["separator charge"]["p_liq"]
                == exp["separator discharge"]["p_liq"]
            )
            phy_par["p_l"] = exp["separator discharge"]["p_liq"]
            assert (
                exp["anode charge"]["p_sol"] == exp["anode discharge"]["p_sol"]
            )
            phy_par["p_s_a"] = exp["anode discharge"]["p_sol"]
            assert (
                exp["anode charge"]["p_liq"] == exp["anode discharge"]["p_liq"]
            )
            phy_par["p_l_a"] = exp["anode discharge"]["p_liq"]
            assert (
                exp["cathode charge"]["p_sol"]
                == exp["cathode discharge"]["p_sol"]
            )
            phy_par["p_s_c"] = exp["cathode discharge"]["p_sol"]
            assert (
                exp["cathode charge"]["p_liq"]
                == exp["cathode discharge"]["p_liq"]
            )
            phy_par["p_l_c"] = exp["cathode discharge"]["p_liq"]
            try:
                phy_par["Nx_a"] = exp["anode discharge"]["Nx"]
            except KeyError:
                phy_par["Nx_a"] = 32
            if "Nx" in exp["anode discharge"]:
                assert (
                    exp["anode discharge"]["Nx"] == exp["anode charge"]["Nx"]
                )
            try:
                phy_par["Nx_s"] = exp["separator discharge"]["Nx"]
            except KeyError:
                phy_par["Nx_s"] = 32
            if "Nx" in exp["separator discharge"]:
                assert (
                    exp["separator discharge"]["Nx"]
                    == exp["separator charge"]["Nx"]
                )
            try:
                phy_par["Nx_c"] = exp["cathode discharge"]["Nx"]
            except KeyError:
                phy_par["Nx_c"] = 32
            if "Nx" in exp["cathode discharge"]:
                assert (
                    exp["cathode discharge"]["Nx"]
                    == exp["cathode charge"]["Nx"]
                )

    if parallel_env is None:
        print("deg param names = ", deg_param_names)
    else:
        parallel_env.printAll("deg param names = " + str(deg_param_names))
    return deg_param_names, deg_param_min, deg_param_max, phy_par


def make_params(filename, parallel_env=None):
    deg_param_names, deg_param_min, deg_param_max, phy_par = parse_input(
        filename, parallel_env=parallel_env
    )

    params = {}
    params["deg_param_names"] = deg_param_names
    for param_name in deg_param_names:
        params["deg_" + param_name + "_min"] = deg_param_min[
            deg_param_names.index(param_name)
        ]
        params["deg_" + param_name + "_max"] = deg_param_max[
            deg_param_names.index(param_name)
        ]
    params["n_params"] = len(deg_param_names)
    for key in phy_par:
        params[key] = phy_par[key]

    return params


if __name__ == "__main__":
    import argparse

    from batfit import BATFIT_EXP
    from batfit.preprocess.param_sampling import *

    parser = argparse.ArgumentParser(description="Parameter sampling")
    parser.add_argument(
        "-n_int",
        "--n_int",
        type=int,
        metavar="",
        required=False,
        help="Number of interior samples",
        default=10,
    )
    parser.add_argument(
        "-n_bound",
        "--n_bound",
        type=int,
        metavar="",
        required=False,
        help="Number of boundary samples",
        default=10,
    )
    args, unknown = parser.parse_known_args()

    n_int = args.n_int
    n_bound = args.n_bound
    sim_params = make_params(
        os.path.join(BATFIT_EXP, "spm_discharge_charge_C4.yaml")
    )
    deg_param_names = None
    int_samples = get_samples(
        n_int=n_int, deg_param_names=deg_param_names, sim_params=sim_params
    )
    bound_samples = get_bounding_samples(
        n_bound=n_bound, deg_param_names=deg_param_names, sim_params=sim_params
    )
    if n_bound == 0 and n_int > 0:
        write_exec(
            int_samples,
            deg_param_names=deg_param_names,
            sim_params=sim_params,
        )
    elif n_bound > 0 and n_int == 0:
        write_exec(
            bound_samples,
            deg_param_names=deg_param_names,
            sim_params=sim_params,
        )
    elif n_bound > 0 and n_int > 0:
        write_exec(
            np.vstack((int_samples, bound_samples)),
            deg_param_names=deg_param_names,
            sim_params=sim_params,
        )
    else:
        print("WARNING: No sample parameter requested")
