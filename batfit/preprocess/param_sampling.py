import os
import sys
from pathlib import Path

import numpy as np
from ruamel.yaml import YAML
from scipy.stats import qmc

from batfit import logger
from batfit.preprocess.sim_setup import read_deg_param
from batfit.preprocess.utils import from_degparamlist_to_degparamdict


def round_samples(samples):
    samples_rounded = samples.copy()
    for i in range(len(samples_rounded)):
        for j in range(len(samples_rounded[i])):
            samples_rounded[i][j] = round(samples_rounded[i][j], 5)
    return samples_rounded


def _get_LI(
    deg_par, deg_param_names, sim_params, cyc_mode=None, electrode=None
):
    assert cyc_mode.lower() in ["charge", "discharge"]
    assert electrode.lower() in ["anode", "cathode"]

    if cyc_mode.lower() == "charge":
        cyc_suffix = "chcc"
    elif cyc_mode.lower() == "discharge":
        cyc_suffix = "dis"
    if electrode.lower() == "anode":
        elec_suffix = "a"
        elec_long_suffix = "an"
    elif electrode.lower() == "cathode":
        elec_suffix = "c"
        elec_long_suffix = "ca"

    deg_par_dict = from_degparamlist_to_degparamdict(
        deg_param_list=deg_par, sim_params=sim_params
    )
    x0 = sim_params[f"x0_{elec_suffix}_{cyc_suffix}"] * read_deg_param(
        f"x0_{elec_suffix}_{cyc_suffix}", deg_par_dict
    )
    csmax = sim_params[f"cs{elec_long_suffix}max"] * read_deg_param(
        f"cs{elec_long_suffix}max", deg_par_dict
    )
    l = sim_params[f"L_{elec_suffix}"] * read_deg_param(
        f"l_{elec_suffix}", deg_par_dict
    )
    eps_cbd = sim_params[f"eps_CBD_{elec_suffix}"] * read_deg_param(
        f"eps_cbd_{elec_suffix}", deg_par_dict
    )
    if f"eps_s_{elec_suffix}" in deg_param_names:
        eps_s = sim_params[f"eps_s_{elec_suffix}"] * read_deg_param(
            f"eps_s_{elec_suffix}", deg_par_dict
        )
    elif f"eps_s_{elec_suffix}_am" in deg_param_names:
        eps_s = eps_cbd + (
            sim_params[f"eps_s_{elec_suffix}"] - eps_cbd
        ) * read_deg_param(f"eps_s_{elec_suffix}_am", deg_par_dict)
    eps_am = eps_s - eps_cbd

    LI_ch = x0 * csmax * l * eps_am
    return LI_ch


def get_LI_ch(deg_par, deg_param_names, sim_params):
    return _get_LI(
        deg_par,
        deg_param_names,
        sim_params,
        cyc_mode="charge",
        electrode="anode",
    ) + _get_LI(
        deg_par,
        deg_param_names,
        sim_params,
        cyc_mode="charge",
        electrode="cathode",
    )


def get_LI_dis(deg_par, deg_param_names, sim_params):
    return _get_LI(
        deg_par,
        deg_param_names,
        sim_params,
        cyc_mode="discharge",
        electrode="anode",
    ) + _get_LI(
        deg_par,
        deg_param_names,
        sim_params,
        cyc_mode="discharge",
        electrode="cathode",
    )


def backout_x0_a(deg_par, deg_param_names, l_bounds, u_bounds, sim_params):
    # We want LI ch = LI dis_a + Li dis_c
    LI_ch = get_LI_ch(deg_par, deg_param_names, sim_params)
    LI_dis_c = _get_LI(
        deg_par,
        deg_param_names,
        sim_params,
        cyc_mode="discharge",
        electrode="cathode",
    )
    LI_dis_a = _get_LI_dis_a(
        deg_par,
        deg_param_names,
        sim_params,
        cyc_mode="discharge",
        electrode="anode",
    )

    if "x0_a" in deg_param_names:
        ind = deg_param_names.index("x0_a")
        corr_factor = (LI_ch - LI_dis_c) / LI_dis_a
        min_val = l_bounds[ind]
        max_val = u_bounds[ind]
        new_deg = deg_par[ind] * corr_factor
        if new_deg > max_val or new_deg < min_val:
            return False, None
        else:
            logger.info(f"Adjusted x0_a from {deg_par[ind]} to {new_deg}")
            return True, new_deg
    else:
        return False, None


def backout_x0_c(deg_par, deg_param_names, l_bounds, u_bounds, sim_params):
    # We want LI ch = LI dis_a + Li dis_c
    LI_ch = get_LI_ch(deg_par, deg_param_names, sim_params)
    LI_dis_c = get_LI_dis_c(deg_par, deg_param_names, sim_params)
    LI_dis_a = get_LI_dis_a(deg_par, deg_param_names, sim_params)

    if "x0_c" in deg_param_names:
        ind = deg_param_names.index("x0_c")
        corr_factor = (LI_ch - LI_dis_a) / LI_dis_c
        min_val = l_bounds[ind]
        max_val = u_bounds[ind]
        new_deg = deg_par[ind] * corr_factor
        if new_deg > max_val or new_deg < min_val:
            return False, None
        else:
            logger.info(f"Adjusted x0_c from {deg_par[ind]} to {new_deg}")
            return True, new_deg
    else:
        return False, None


def backout_x0_a_chcc(
    deg_par, deg_param_names, l_bounds, u_bounds, sim_params
):
    # We want LI dis = LI ch_a + Li ch_c
    LI_dis = get_LI_dis(deg_par, deg_param_names, sim_params)
    LI_ch_c = _get_LI(
        deg_par,
        deg_param_names,
        sim_params,
        cyc_mode="charge",
        electrode="cathode",
    )
    LI_ch_a = _get_LI(
        deg_par,
        deg_param_names,
        sim_params,
        cyc_mode="charge",
        electrode="anode",
    )

    if "x0_a_chcc" in deg_param_names:
        ind = deg_param_names.index("x0_a_chcc")
        corr_factor = (LI_dis - LI_ch_c) / LI_ch_a
        min_val = l_bounds[ind]
        max_val = u_bounds[ind]
        new_deg = deg_par[ind] * corr_factor
        if new_deg > max_val or new_deg < min_val:
            return False, None
        else:
            logger.info(f"Adjusted x0_a_chcc from {deg_par[ind]} to {new_deg}")
            return True, new_deg
    else:
        return False, None


def backout_x0_c_chcc(
    deg_par, deg_param_names, l_bounds, u_bounds, sim_params
):
    # We want LI dis = LI ch_a + Li ch_c
    LI_dis = get_LI_dis(deg_par, deg_param_names, sim_params)
    LI_ch_c = _get_LI(
        deg_par,
        deg_param_names,
        sim_params,
        cyc_mode="charge",
        electrode="cathode",
    )
    LI_ch_a = _get_LI(
        deg_par,
        deg_param_names,
        sim_params,
        cyc_mode="charge",
        electrode="anode",
    )

    if "x0_c_chcc" in deg_param_names:
        ind = deg_param_names.index("x0_c_chcc")
        corr_factor = (LI_dis - LI_ch_a) / LI_ch_c
        min_val = l_bounds[ind]
        max_val = u_bounds[ind]
        new_deg = deg_par[ind] * corr_factor
        if new_deg > max_val or new_deg < min_val:
            return False, None
        else:
            logger.info(f"Adjusted x0_c_chcc from {deg_par[ind]} to {new_deg}")
            return True, new_deg
    else:
        return False, None


def enforce_li_conservation(
    sample_scaled, deg_param_names, l_bounds, u_bounds, sim_params
):
    if (
        ("x0_a" not in deg_param_names)
        or ("x0_c" not in deg_param_names)
        or ("x0_a_chcc" not in deg_param_names)
        or ("x0_c_chcc" not in deg_param_names)
    ):
        return sample_scaled

    ind_x0_a = deg_param_names.index("x0_a")
    ind_x0_c = deg_param_names.index("x0_c")
    ind_x0_a_chcc = deg_param_names.index("x0_a_chcc")
    ind_x0_c_chcc = deg_param_names.index("x0_c_chcc")

    index_to_remove = []

    for isamp in range(sample_scaled.shape[0]):
        success = False
        for name_enf in ["x0_c_chcc", "x0_a_chcc", "x0_a", "x0_c"]:
            if not success and name_enf.lower() == "x0_c_chcc":
                success, val = backout_x0_c_chcc(
                    sample_scaled[isamp, :],
                    deg_param_names,
                    l_bounds,
                    u_bounds,
                    sim_params,
                )
                if success:
                    sample_scaled[isamp, ind_x0_c_chcc] = val
            if not success and name_enf.lower() == "x0_a_chcc":
                success, val = backout_x0_a_chcc(
                    sample_scaled[isamp, :],
                    deg_param_names,
                    l_bounds,
                    u_bounds,
                    sim_params,
                )
                if success:
                    sample_scaled[isamp, ind_x0_a_chcc] = val
            if not success and name_enf.lower() == "x0_a":
                success, val = backout_x0_a(
                    sample_scaled[isamp, :],
                    deg_param_names,
                    l_bounds,
                    u_bounds,
                    sim_params,
                )
                if success:
                    sample_scaled[isamp, ind_x0_a] = val
            if not success and name_enf.lower() == "x0_c":
                success, val = backout_x0_c(
                    sample_scaled[isamp, :],
                    deg_param_names,
                    l_bounds,
                    u_bounds,
                    sim_params,
                )
                if success:
                    sample_scaled[isamp, ind_x0_c] = val

        if not success:
            index_to_remove.append(isamp)

    if len(index_to_remove) > 0:
        logger.warning(f"Li cons {len(index_to_remove)} Failed index")
        sample_scaled = np.delete(sample_scaled, index_to_remove, axis=0)

    return sample_scaled


# def backout_eps_s_a(
#    deg_par, deg_param_names, eps_s, eps_el, l_bounds, u_bounds, sim_params
# ):
#    ind_eps_s_a = deg_param_names.index("eps_s_a")
#    eps_s = 1 - eps_el
#    deg_eps_s_a = eps_s / sim_params["eps_s_a"]
#    min_val = l_bounds[ind_eps_s_a]
#    max_val = u_bounds[ind_eps_s_a]
#    if deg_eps_s_a > max_val or deg_eps_s_a < min_val:
#        return False, None
#    else:
#        return True, deg_eps_s_a
#
#
# def backout_eps_el_a(
#    deg_par, deg_param_names, eps_s, eps_el, l_bounds, u_bounds, sim_params
# ):
#    ind_eps_el_a = deg_param_names.index("eps_el_a")
#    eps_el = 1 - eps_s
#    deg_eps_el_a = eps_el / sim_params["eps_el_a"]
#    min_val = l_bounds[ind_eps_el_a]
#    max_val = u_bounds[ind_eps_el_a]
#    if deg_eps_el_a > max_val or deg_eps_el_a < min_val:
#        return False, None
#    else:
#        return True, deg_eps_el_a
#
#
# def backout_eps_s_c(
#    deg_par, deg_param_names, eps_s, eps_el, l_bounds, u_bounds, sim_params
# ):
#    ind_eps_s_c = deg_param_names.index("eps_s_c")
#    eps_s = 1 - eps_el
#    deg_eps_s_c = eps_s / sim_params["eps_s_c"]
#    min_val = l_bounds[ind_eps_s_c]
#    max_val = u_bounds[ind_eps_s_c]
#    if deg_eps_s_c > max_val or deg_eps_s_c < min_val:
#        return False, None
#    else:
#        return True, deg_eps_s_c
#
#
# def backout_eps_el_c(
#    deg_par, deg_param_names, eps_s, eps_el, l_bounds, u_bounds, sim_params
# ):
#    ind_eps_el_c = deg_param_names.index("eps_el_c")
#    eps_el = 1 - eps_s
#    deg_eps_el_c = eps_el / sim_params["eps_el_c"]
#    min_val = l_bounds[ind_eps_el_c]
#    max_val = u_bounds[ind_eps_el_c]
#    if deg_eps_el_c > max_val or deg_eps_el_c < min_val:
#        return False, None
#    else:
#        return True, deg_eps_el_c


def enforce_pos_void_a(
    sample_scaled, deg_param_names, l_bounds, u_bounds, sim_params
):

    index_to_remove = []
    try:
        ind_eps_el_a = deg_param_names.index("eps_el_a")
    except ValueError:
        ind_eps_el_a = -1

    try:
        ind_eps_s_a = deg_param_names.index("eps_s_a")
    except ValueError:
        ind_eps_s_a = -1

    try:
        ind_eps_s_a_am = deg_param_names.index("eps_s_a_am")
    except ValueError:
        ind_eps_s_a_am = -1

    try:
        ind_eps_cbd_a = deg_param_names.index("eps_cbd_a")
    except ValueError:
        ind_eps_cbd_a = -1

    for isamp in range(sample_scaled.shape[0]):
        if ind_eps_cbd_a > -1:
            eps_cbd = (
                sim_params["eps_CBD_a"] * sample_scaled[isamp, ind_eps_cbd_a]
            )
        else:
            eps_cbd = sim_params["eps_CBD_a"]
        if ind_eps_s_a > -1:
            eps_s = sim_params["eps_s_a"] * sample_scaled[isamp, ind_eps_s_a]
        elif ind_eps_s_a_am > -1:
            eps_s = (
                eps_cbd
                + (sim_params[f"eps_s_a"] - eps_cbd)
                * sample_scaled[isamp, ind_eps_s_a_am]
            )
        else:
            eps_s = sim_params["eps_s_a"]
        if ind_eps_el_a > -1:
            eps_el = (
                sim_params["eps_el_a"] * sample_scaled[isamp, ind_eps_el_a]
            )
        else:
            eps_el = sim_params["eps_el_a"]

        void = 1 - eps_s - eps_el

        if void < -np.finfo(float).eps or eps_cbd > eps_s:
            index_to_remove.append(isamp)

    if len(index_to_remove) > 0:
        logger.warning(
            f"Positive void anode {len(index_to_remove)} Failed index"
        )
        sample_scaled = np.delete(sample_scaled, index_to_remove, axis=0)

    return sample_scaled


def enforce_pos_void_c(
    sample_scaled, deg_param_names, l_bounds, u_bounds, sim_params
):

    index_to_remove = []
    try:
        ind_eps_el_c = deg_param_names.index("eps_el_c")
    except ValueError:
        ind_eps_el_c = -1

    try:
        ind_eps_s_c = deg_param_names.index("eps_s_c")
    except ValueError:
        ind_eps_s_c = -1

    try:
        ind_eps_s_c_am = deg_param_names.index("eps_s_c_am")
    except ValueError:
        ind_eps_s_c_am = -1

    try:
        ind_eps_cbd_c = deg_param_names.index("eps_cbd_c")
    except ValueError:
        ind_eps_cbd_c = -1

    for isamp in range(sample_scaled.shape[0]):
        if ind_eps_cbd_c > -1:
            eps_cbd = (
                sim_params["eps_CBD_c"] * sample_scaled[isamp, ind_eps_cbd_c]
            )
        else:
            eps_cbd = sim_params["eps_CBD_c"]
        if ind_eps_s_c > -1:
            eps_s = sim_params["eps_s_c"] * sample_scaled[isamp, ind_eps_s_c]
        elif ind_eps_s_c_am > -1:
            eps_s = (
                eps_cbd
                + (sim_params[f"eps_s_c"] - eps_cbd)
                * sample_scaled[isamp, ind_eps_s_c_am]
            )
        else:
            eps_s = sim_params["eps_s_c"]
        if ind_eps_el_c > -1:
            eps_el = (
                sim_params["eps_el_c"] * sample_scaled[isamp, ind_eps_el_c]
            )
        else:
            eps_el = sim_params["eps_el_c"]

        void = 1 - eps_s - eps_el

        if void < -np.finfo(float).eps or eps_cbd > eps_s:
            index_to_remove.append(isamp)

    if len(index_to_remove) > 0:
        logger.warning(
            f"Positive void cathode {len(index_to_remove)} Failed index"
        )
        sample_scaled = np.delete(sample_scaled, index_to_remove, axis=0)

    return sample_scaled


def enforce_stoich_a(
    sample_scaled, deg_param_names, l_bounds, u_bounds, sim_params
):

    index_to_remove = []
    try:
        ind_x0_a = deg_param_names.index("x0_a")
    except ValueError:
        ind_x0_a = -1

    for isamp in range(sample_scaled.shape[0]):
        if ind_x0_a > -1:
            x0_a = sim_params["x0_a"] * sample_scaled[isamp, ind_x0_a]
        else:
            x0_a = sim_params["x0_a"]

        if x0_a > 1.0 or x0_a < 0:
            index_to_remove.append(isamp)

    if len(index_to_remove) > 0:
        logger.warning(
            f"Stoichiometry anode {len(index_to_remove)} Failed index"
        )
        sample_scaled = np.delete(sample_scaled, index_to_remove, axis=0)

    return sample_scaled


def enforce_stoich_c(
    sample_scaled, deg_param_names, l_bounds, u_bounds, sim_params
):

    index_to_remove = []
    try:
        ind_x0_c = deg_param_names.index("x0_c")
    except ValueError:
        ind_x0_c = -1

    for isamp in range(sample_scaled.shape[0]):
        if ind_x0_c > -1:
            x0_c = sim_params["x0_c"] * sample_scaled[isamp, ind_x0_c]
        else:
            x0_c = sim_params["x0_c"]

        if x0_c > 1.0 or x0_c < 0:
            index_to_remove.append(isamp)

    if len(index_to_remove) > 0:
        logger.warning(
            f"Stoichiometry anode {len(index_to_remove)} Failed index"
        )
        sample_scaled = np.delete(sample_scaled, index_to_remove, axis=0)

    return sample_scaled


def get_samples(
    n_int=25,
    deg_param_names=None,
    prot_param_names=None,
    sim_params=None,
    li_cons=False,
    uniform=False,
):
    if not sim_params["cyc_mode"].lower() == "discharge-chargecc":
        li_cons = False

    if deg_param_names is None:
        deg_param_names = sim_params["deg_param_names"]
    if prot_param_names is None:
        try:
            prot_param_names = sim_params["prot_param_names"]
        except KeyError:
            prot_param_names = None

    n_deg_params = len(deg_param_names)
    deg_l_bounds = []
    deg_u_bounds = []
    for par_name in deg_param_names:
        deg_l_bounds.append(sim_params["deg_" + par_name + "_min"])
        deg_u_bounds.append(sim_params["deg_" + par_name + "_max"])

    if sim_params["cyc_mode"].lower() in ["chirp"]:
        n_prot_params = len(prot_param_names)
        prot_l_bounds = []
        prot_u_bounds = []
        for par_name in prot_param_names:
            prot_l_bounds.append(sim_params["prot_" + par_name + "_min"])
            prot_u_bounds.append(sim_params["prot_" + par_name + "_max"])
    else:
        n_prot_params = 0
        prot_l_bounds = []
        prot_u_bounds = []

    if uniform:
        deg_sample = np.random.uniform(size=(n_int, n_deg_params))
        prot_sample = np.random.uniform(size=(n_int, n_prot_params))
    else:
        sampler = qmc.LatinHypercube(d=n_deg_params + n_prot_params)
        sample = sampler.random(n=n_int)
        deg_sample = sample[:,:n_deg_params]
        prot_sample = sample[:,n_deg_params:n_prot_params+n_deg_params]

    deg_sample_scaled = qmc.scale(deg_sample, deg_l_bounds, deg_u_bounds)
    if n_prot_params > 0:
        prot_sample_scaled = qmc.scale(prot_sample, prot_l_bounds, prot_u_bounds)
        sample_scaled = round_samples(np.hstack((deg_sample_scaled, prot_sample_scaled)))
    else:
        prot_sample_scaled = prot_sample
        sample_scaled = deg_sample_scaled


    sample_scaled = enforce_pos_void_a(
        sample_scaled, deg_param_names, deg_l_bounds + prot_l_bounds, deg_u_bounds + prot_u_bounds, sim_params
    )
    sample_scaled = enforce_pos_void_c(
        sample_scaled, deg_param_names, deg_l_bounds + prot_l_bounds, deg_u_bounds + prot_u_bounds, sim_params
    )
    sample_scaled = enforce_stoich_a(
        sample_scaled, deg_param_names, deg_l_bounds + prot_l_bounds, deg_u_bounds + prot_u_bounds, sim_params
    )
    sample_scaled = enforce_stoich_c(
        sample_scaled, deg_param_names, deg_l_bounds + prot_l_bounds, deg_u_bounds + prot_u_bounds, sim_params
    )
    if li_cons:
        sample_scaled = enforce_li_conservation(
            sample_scaled, deg_param_names, deg_l_bounds + prot_l_bounds, deg_u_bounds + prot_u_bounds, sim_params
        )

    return sample_scaled[:,:n_deg_params], sample_scaled[:,n_deg_params:n_deg_params+n_prot_params]


def hypercube_combinations(val_list):
    if val_list:
        for el in val_list[0]:
            for combination in hypercube_combinations(val_list[1:]):
                yield [el] + combination
    else:
        yield []


def get_bounding_samples(
    n_bound=None, deg_param_names=None, prot_param_names=None, sim_params=None, li_cons=False
):

    if not sim_params["cyc_mode"].lower() == "discharge-chargecc":
        li_cons = False

    if deg_param_names is None:
        deg_param_names = sim_params["deg_param_names"]
    if prot_param_names is None:
        try:
            prot_param_names = sim_params["prot_param_names"]
        except KeyError:
            prot_param_names = []

    n_deg_params = len(deg_param_names)
    n_prot_params = len(prot_param_names)

    deg_l_bounds = []
    deg_u_bounds = []
    for par_name in deg_param_names:
        deg_l_bounds.append(sim_params["deg_" + par_name + "_min"])
        deg_u_bounds.append(sim_params["deg_" + par_name + "_max"])
    prot_l_bounds = []
    prot_u_bounds = []
    for par_name in prot_param_names:
        prot_l_bounds.append(sim_params["prot_" + par_name + "_min"])
        prot_u_bounds.append(sim_params["prot_" + par_name + "_max"])

    if n_bound == 0:
        return np.empty((0, n_deg_params + n_prot_params))

    verts = [[0, 1] for _ in range(n_deg_params + n_prot_params)]
    combs = hypercube_combinations(verts)
    deg_samples = []
    combs_list = list(combs)

    if n_bound is None:
        n_bound = len(combs_list)
    n_bound = min(n_bound, len(combs_list))
    for comb in combs_list[:n_bound]:
        deg_par_list = []
        for ipar, name in enumerate(deg_param_names):
            if comb[ipar] == 0:
                deg_par_list.append(sim_params["deg_" + name + "_min"])
            elif comb[ipar] == 1:
                deg_par_list.append(sim_params["deg_" + name + "_max"])
        deg_samples.append(deg_par_list)
        prot_par_list = []
        for ipar, name in enumerate(prot_param_names):
            if comb[ipar] == 0:
                prot_par_list.append(sim_params["prot_" + name + "_min"])
            elif comb[ipar] == 1:
                prot_par_list.append(sim_params["prot_" + name + "_max"])
        prot_samples.append(prot_par_list)

    if n_prot_params > 0:
        samples = np.hstack((np.array(deg_samples), np.array(prot_samples)))
    else:
        samples = np.array(deg_samples)

    if li_cons:
        samples = enforce_li_conservation(
            samples, deg_param_names, deg_l_bounds + prot_l_bounds, deg_u_bounds + prot_u_bounds, sim_params
        )

    return samples[:,:n_deg_params], samples[:,n_deg_params:n_deg_params+n_prot_params]


def write_exec(
    deg_samples,
    deg_param_names=None,
    prot_samples=None,
    prot_param_names=None,
    folder_save=".",
    deg_param_list_file="parameter_list.txt",
    prot_param_list_file="protocol_parameter_list.txt",
    sim_params=None,
):

    if deg_param_names is None:
        deg_param_names = sim_params["deg_param_names"]
    n_deg_params = len(deg_param_names)
    if prot_param_names is None:
        try:
            prot_param_names = sim_params["prot_param_names"]
        except KeyError:
            prot_param_names = []
    n_prot_params = len(prot_param_names)

    id_deg_param = []
    for name in deg_param_names:
        id_deg_param.append(sim_params["deg_param_names"].index(name))
    id_prot_param = []
    for name in prot_param_names:
        id_prot_param.append(sim_params["prot_param_names"].index(name))


    log_dir = Path(folder_save)
    log_dir.mkdir(parents=True, exist_ok=True)
    # os.makedirs(folder_save, exist_ok=True)
    deg_param_list_file = os.path.join(folder_save, deg_param_list_file)
    prot_param_list_file = os.path.join(folder_save, prot_param_list_file)
     
    with open(deg_param_list_file, "w+") as f:
        for sample in deg_samples:
            str_par = ""
            sample_aug = [1] * sim_params["n_deg_params"]
            for i, id_p in enumerate(id_deg_param):
                sample_aug[id_p] = sample[i]
            for s in sample_aug:
                str_par += f"{s:g} "
            str_par += "\n"
            f.write(str_par)
   
    if prot_samples is not None:
        with open(prot_param_list_file, "w+") as f:
            for sample in prot_samples:
                str_par = ""
                sample_aug = [1] * sim_params["n_prot_params"]
                for i, id_p in enumerate(id_prot_param):
                    sample_aug[id_p] = sample[i]
                for s in sample_aug:
                    str_par += f"{s:g} "
                str_par += "\n"
                f.write(str_par)


if __name__ == "__main__":
    import argparse

    from batfit import BATFIT_EXP
    from batfit.preprocess.sim_setup import make_params

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
    parser.add_argument(
        "-sim_config",
        "--sim_config",
        type=str,
        metavar="",
        required=False,
        help="Sim config file",
        default=os.path.join(BATFIT_EXP, "spm_discharge.yaml"),
    )
    parser.add_argument(
        "-folder_save",
        "--folder_save",
        type=str,
        metavar="",
        required=False,
        help="data folder",
        default=".",
    )
    args, unknown = parser.parse_known_args()

    n_int = args.n_int
    n_bound = args.n_bound

    # params = make_params(os.path.join(BATFIT_EXP, "spm_discharge.yaml"))
    # params = make_params(os.path.join(BATFIT_EXP, "spm_charge_C4.yaml"))
    sim_params = make_params(args.sim_config)
    deg_param_names = None
    int_samples = get_samples(
        n_int=n_int, deg_param_names=deg_param_names, sim_params=sim_params
    )
    bound_samples = get_bounding_samples(
        n_bound=n_bound, deg_param_names=deg_param_names, sim_params=sim_params
    )
    if n_bound == 0 and n_int > 0:
        write_exec(int_samples)
    elif n_bound > 0 and n_int == 0:
        write_exec(bound_samples)
    elif n_bound > 0 and n_int > 0:
        write_exec(
            np.vstack((int_samples, bound_samples)),
            deg_param_names=deg_param_names,
            folder_save=args.folder_save,
            sim_params=sim_params,
        )
    else:
        logger.warning("No sample parameter requested")
