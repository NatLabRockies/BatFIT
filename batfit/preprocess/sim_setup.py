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
    # deg_param_names = ["i0_a", "ds_c", "x0_a", "x0_c", "i0_c", "eps_s_c"]

    if cyc_mode == "discharge-chargecc":
        deg_param_names_chcc_min = list(
            exp["min degradation parameter charge"].keys()
        )
        deg_param_names_chcc_max = list(
            exp["max degradation parameter charge"].keys()
        )
        assert set(deg_param_names_chcc_min) == set(deg_param_names_chcc_max)
        deg_param_names_chcc = list(set(deg_param_names_chcc_min))
        deg_param_names_chcc = ["x0_a", "x0_c"]
        deg_param_names_chcc_new = [
            f"{entry}_chcc" for entry in deg_param_names_chcc
        ]
    elif cyc_mode.lower() in ["discharge", "chargecc"]:
        pass
    elif cyc_mode.lower() in ["rh", "lh", "lh2"]:
        pass
    elif cyc_mode.lower() in ["diffcap", "hppc", "prehppc", "posthppc"]:
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
        logger.error("Mismatch of parameters")
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
            logger.error("Mismatch of parameters")
            raise KeyError
        deg_param_names += deg_param_names_chcc_new
        deg_param_min += deg_param_min_chcc
        deg_param_max += deg_param_max_chcc

    assert np.amin(np.array(deg_param_max) - np.array(deg_param_min)) > 0

    phy_par = {}
    phy_par["cyc_mode"] = cyc_mode
    if cyc_mode.lower() in [
        "discharge",
        "chargecc",
        "rh",
        "lh",
        "lh2",
        "hppc",
        "posthppc",
        "diffcap",
        "prehppc",
    ]:
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
        # posthppc/hppc only: skip the CV-hold step of a pulse when the
        # regen pulse never reached vmax (see sol_gen)
        try:
            phy_par["skip_degenerate_cv"] = bool(
                exp["macroscopic"]["skip_degenerate_cv"]
            )
        except KeyError:
            phy_par["skip_degenerate_cv"] = True
        if phy_par["model"].lower() == "p2d":
            phy_par["eps_el_s"] = exp["separator"]["eps_el"]
            phy_par["p_l_s"] = exp["separator"]["p_liq"]
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
            phy_par["eps_el_s"] = exp["separator discharge"]["eps_el"]
            assert (
                exp["separator charge"]["p_liq"]
                == exp["separator discharge"]["p_liq"]
            )
            phy_par["p_l_s"] = exp["separator discharge"]["p_liq"]
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


def set_discretization(sim, sim_params: dict):
    sim.an.Nr = sim_params["Nr_a"]
    sim.ca.Nr = sim_params["Nr_c"]
    if sim_params["model"].lower() == "p2d":
        sim.an.Nx = sim_params["Nx_a"]
        sim.ca.Nx = sim_params["Nx_c"]
        sim.sep.Nx = sim_params["Nx_s"]

    return sim


def read_deg_param(key: str, deg_param_sample: dict):
    if key in deg_param_sample:
        return deg_param_sample[key]
    else:
        return 1.0


def set_interc_disconnected_discharge(
    sim, sim_params: dict, deg_param_sample: dict
):
    sim.ca.x_0 = sim_params["x0_c_dis"] * read_deg_param(
        key="x0_c", deg_param_sample=deg_param_sample
    )
    sim.an.x_0 = sim_params["x0_a_dis"] * read_deg_param(
        key="x0_a", deg_param_sample=deg_param_sample
    )
    C_rate = sim_params["C_dis"]
    return sim, C_rate


def set_interc_disconnected_charge(
    sim, sim_params: dict, deg_param_sample: dict
):
    sim.ca.x_0 = sim_params["x0_c_chcc"] * read_deg_param(
        key="x0_c_chcc", deg_param_sample=deg_param_sample
    )
    sim.an.x_0 = sim_params["x0_a_chcc"] * read_deg_param(
        key="x0_a_chcc", deg_param_sample=deg_param_sample
    )
    C_rate = sim_params["C_chcc"]
    return sim, C_rate


def set_interc_connected(
    sim, sim_params: dict, deg_param_sample: dict, cyc_mode: str
):
    sim.ca.x_0 = sim_params["x0_c"] * read_deg_param(
        key="x0_c", deg_param_sample=deg_param_sample
    )
    sim.an.x_0 = sim_params["x0_a"] * read_deg_param(
        key="x0_a", deg_param_sample=deg_param_sample
    )
    # C rate if constant current cycle
    if cyc_mode.lower() in ["discharge", "chargecc"]:
        C_rate = sim_params["C"]
    else:
        C_rate = None

    return C_rate, sim


def set_interc(
    sim, sim_params: dict, deg_param_sample: dict, cyc_mode: str, run_mode: str
):
    # CC charge and discharge specific parameters if disconnected charge and discharge
    if cyc_mode.lower() == "discharge-chargecc":
        if run_mode.lower() == "discharge":
            C_rate, sim = set_interc_disconnected_discharge(
                sim=sim,
                sim_params=sim_params,
                deg_param_sample=deg_param_sample,
            )

        elif run_mode.lower() == "chargecc":
            C_rate, sim = set_interc_disconnected_charge(
                sim=sim,
                sim_params=sim_params,
                deg_param_sample=deg_param_sample,
            )
    # Any other cycle is not disconnected
    else:
        C_rate, sim = set_interc_connected(
            sim=sim,
            sim_params=sim_params,
            deg_param_sample=deg_param_sample,
            cyc_mode=cyc_mode,
        )
    return C_rate, sim


def set_separator(
    sim,
    sim_params: dict,
    deg_param_sample: dict,
    cyc_mode: str,
    run_mode: str,
    is_p2d: bool,
):
    if not is_p2d:
        return sim
    else:
        sim.sep.thick = sim_params["L_s"] * read_deg_param(
            key="l_s", deg_param_sample=deg_param_sample
        )
        sim.sep.eps_el = sim_params["eps_el_s"] * read_deg_param(
            key="eps_el_s", deg_param_sample=deg_param_sample
        )
        sim.sep.p_liq = sim_params["p_l_s"] * read_deg_param(
            key="p_l_s", deg_param_sample=deg_param_sample
        )

        return sim


def set_battery(
    sim,
    sim_params: dict,
    deg_param_sample: dict,
    cyc_mode: str,
    run_mode: str,
    is_p2d: bool,
):
    sim.bat.area = sim_params["area"] * read_deg_param(
        key="area", deg_param_sample=deg_param_sample
    )
    return sim


def set_electrodes(
    sim,
    sim_params: dict,
    deg_param_sample: dict,
    cyc_mode: str,
    run_mode: str,
    is_p2d: bool,
):
    for elec, suffix, long_name in zip(
        [sim.ca, sim.an], ["c", "a"], ["ca", "an"]
    ):
        # Ds and i0 are set through degradation parameters
        elec.Ds_deg = read_deg_param(
            key=f"ds_{suffix}", deg_param_sample=deg_param_sample
        )
        elec.i0_deg = read_deg_param(
            key=f"i0_{suffix}", deg_param_sample=deg_param_sample
        )

        elec.Li_max = sim_params[f"cs{long_name}max"] * read_deg_param(
            key=f"cs{long_name}max", deg_param_sample=deg_param_sample
        )

        if f"eps_cbd_{suffix}" in deg_param_sample:
            # logger.warning(
            #    "Changing CBD independently of AM is not recommended"
            # )
            elec.eps_CBD = (
                sim_params[f"eps_CBD_{suffix}"]
                * deg_param_sample[f"eps_cbd_{suffix}"]
            )
        else:
            elec.eps_CBD = sim_params[f"eps_CBD_{suffix}"]

        if f"eps_s_{suffix}" in deg_param_sample:
            # logger.warning(
            #    "Changing eps_s independently of AM is not recommended"
            # )
            elec.eps_s = (
                sim_params[f"eps_s_{suffix}"]
                * deg_param_sample[f"eps_s_{suffix}"]
            )
        elif f"eps_s_{suffix}_am" in deg_param_sample:
            # elec.eps_s = elec.eps_CBD + elec.eps_AM * deg
            # elec.eps_s = elec.eps_CBD + (sim_params[f"eps_s_{suffix}"] - sim_params[f"eps_CBD_{suffix}"]) * deg
            elec.eps_s = (
                elec.eps_CBD
                + (sim_params[f"eps_s_{suffix}"] - elec.eps_CBD)
                * deg_param_sample[f"eps_s_{suffix}_am"]
            )
        else:
            elec.eps_s = sim_params[f"eps_s_{suffix}"]

        elec.eps_el = sim_params[f"eps_el_{suffix}"] * read_deg_param(
            key=f"eps_el_{suffix}", deg_param_sample=deg_param_sample
        )
        elec.thick = sim_params[f"L_{suffix}"] * read_deg_param(
            key=f"l_{suffix}", deg_param_sample=deg_param_sample
        )
        elec.R_s = sim_params[f"Rs_{suffix}"] * read_deg_param(
            key=f"rs_{suffix}", deg_param_sample=deg_param_sample
        )
        if is_p2d:
            elec.p_sol = sim_params[f"p_s_{suffix}"] * read_deg_param(
                key=f"p_s_{suffix}", deg_param_sample=deg_param_sample
            )
            elec.p_liq = sim_params[f"p_l_{suffix}"] * read_deg_param(
                key=f"p_l_{suffix}", deg_param_sample=deg_param_sample
            )

    return sim


def set_electrolyte(
    sim,
    sim_params: dict,
    deg_param_sample: dict,
    cyc_mode: str,
    run_mode: str,
    is_p2d: bool,
):
    sim.el.Li_0 = sim_params["ce"] * read_deg_param(
        key="ce", deg_param_sample=deg_param_sample
    )
    if is_p2d:
        sim.el.D_deg = read_deg_param(
            key="de", deg_param_sample=deg_param_sample
        )
        sim.el.t0_deg = read_deg_param(
            key="t0", deg_param_sample=deg_param_sample
        )
        sim.el.kappa_deg = read_deg_param(
            key="kappa", deg_param_sample=deg_param_sample
        )
        sim.el.gamma_deg = read_deg_param(
            key="gamma", deg_param_sample=deg_param_sample
        )

    return sim


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
    print("Cathode")
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
