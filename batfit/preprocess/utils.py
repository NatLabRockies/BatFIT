import os

import numpy as np


def remove_file(filename):
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass


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
    deg_param_list: list[float],
    sim_params: dict,
    deg_param_names=None,
    parallel_env=None,
):

    if deg_param_names is not None:
        try:
            assert sim_params["deg_param_names"] == deg_param_names
        except AssertionError:
            msg = f"ERROR: sim_params['deg_param_names'] and deg_param_names do not match\n"
            msg += f"\tsim_params['deg_param_names'] = {sim_params['deg_param_names']}"
            msg += f"\tdeg_param_names = {deg_param_names}"
            if parallel_env is None:
                sys.exit(msg)
            else:
                parallel_env.printAll(msg)
                parallel_env.comm.Abort()

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
