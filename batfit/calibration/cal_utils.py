import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import scipy
from numpyro.infer import MCMC, NUTS, SA, init_to_value
from prettyPlot.plotting import *

from batfit import logger
from batfit.utils.text_utils import shuffle_substrings

from .data_utils import make_error_data, make_target_data, perturb_val_dict


def bayes_step_discharge(
    y=None,
    y_err=0.1,
    min_sigma=None,
    max_sigma=None,
    sim_params_dict=None,
    jax_func_dict=None,
    jax_params_dict=None,
    target_list=None,
):
    # define parameters (incl. prior ranges)
    deg_params = []
    for ipar, name in enumerate(
        sim_params_dict["discharge"]["deg_param_names"]
    ):
        deg_params.append(
            numpyro.sample(
                name,
                dist.Uniform(
                    sim_params_dict["discharge"]["deg_" + name + "_min"],
                    sim_params_dict["discharge"]["deg_" + name + "_max"],
                ),
            )
        )
    # implement the model
    # needs jax numpy for differentiability here
    (phis_c1, dV_dQ1, dQ_dV1), _ = jax_func_dict["discharge"](
        jnp.array(deg_params)
    )
    if (
        ("phis_c" in target_list)
        and ("dV_dQ" not in target_list)
        and ("dQ_dV" not in target_list)
    ):
        y_model = phis_c1
    elif (
        ("phis_c" not in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" not in target_list)
    ):
        y_model = dV_dQ1
    elif (
        ("phis_c" not in target_list)
        and ("dV_dQ" not in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = dQ_dV1
    elif (
        ("phis_c" in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" not in target_list)
    ):
        y_model = jnp.hstack((phis_c1, dV_dQ1))
    elif (
        ("phis_c" in target_list)
        and ("dV_dQ" not in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = jnp.hstack((phis_c1, dQ_dV1))
    elif (
        ("phis_c" not in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = jnp.hstack((dV_dQ1, dQ_dV1))
    elif (
        ("phis_c" in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = jnp.hstack((phis_c1, dV_dQ1, dQ_dV1))

    std_obs = jnp.ones(y_model.shape[0]) * y_err
    numpyro.sample("obs", dist.Normal(y_model, std_obs), obs=y)


def bayes_step_discharge_sigma(
    y=None,
    y_err=0.1,
    min_sigma=None,
    max_sigma=None,
    sim_params_dict=None,
    jax_func_dict=None,
    jax_params_dict=None,
    target_list=None,
):
    # define parameters (incl. prior ranges)
    deg_params = []
    for ipar, name in enumerate(
        sim_params_dict["discharge"]["deg_param_names"]
    ):
        deg_params.append(
            numpyro.sample(
                name,
                dist.Uniform(
                    sim_params_dict["discharge"]["deg_" + name + "_min"],
                    sim_params_dict["discharge"]["deg_" + name + "_max"],
                ),
            )
        )
    sigma = numpyro.sample("sigma", dist.Uniform(min_sigma, max_sigma))
    # implement the model
    # needs jax numpy for differentiability here
    (phis_c1, dV_dQ1, dQ_dV1), _ = jax_func_dict["discharge"](
        jnp.array(deg_params)
    )
    if (
        ("phis_c" in target_list)
        and ("dV_dQ" not in target_list)
        and ("dQ_dV" not in target_list)
    ):
        y_model = phis_c1
    elif (
        ("phis_c" not in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" not in target_list)
    ):
        y_model = dV_dQ1
    elif (
        ("phis_c" not in target_list)
        and ("dV_dQ" not in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = dQ_dV1
    elif (
        ("phis_c" in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" not in target_list)
    ):
        y_model = jnp.hstack((phis_c1, dV_dQ1))
    elif (
        ("phis_c" in target_list)
        and ("dV_dQ" not in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = jnp.hstack((phis_c1, dQ_dV1))
    elif (
        ("phis_c" not in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = jnp.hstack((dV_dQ1, dQ_dV1))
    elif (
        ("phis_c" in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = jnp.hstack((phis_c1, dV_dQ1, dQ_dV1))

    std_obs = jnp.ones(y_model.shape[0]) * sigma
    numpyro.sample("obs", dist.Normal(y_model, std_obs), obs=y)


def bayes_step_chargecc(
    y=None,
    y_err=0.1,
    min_sigma=None,
    max_sigma=None,
    sim_params_dict=None,
    jax_func_dict=None,
    jax_params_dict=None,
    target_list=None,
):
    # define parameters (incl. prior ranges)
    deg_params = []
    for ipar, name in enumerate(
        sim_params_dict["chargecc"]["deg_param_names"]
    ):
        deg_params.append(
            numpyro.sample(
                name,
                dist.Uniform(
                    sim_params_dict["chargecc"]["deg_" + name + "_min"],
                    sim_params_dict["chargecc"]["deg_" + name + "_max"],
                ),
            )
        )
    # implement the model
    # needs jax numpy for differentiability here
    (phis_c2, dV_dQ2, dQ_dV2), _ = jax_func_dict["chargecc"](
        jnp.array(deg_params)
    )
    if (
        ("phis_c" in target_list)
        and ("dV_dQ" not in target_list)
        and ("dQ_dV" not in target_list)
    ):
        y_model = phis_c2
    elif (
        ("phis_c" not in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" not in target_list)
    ):
        y_model = dV_dQ2
    elif (
        ("phis_c" not in target_list)
        and ("dV_dQ" not in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = dQ_dV2
    elif (
        ("phis_c" in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" not in target_list)
    ):
        y_model = jnp.hstack((phis_c2, dV_dQ2))
    elif (
        ("phis_c" in target_list)
        and ("dV_dQ" not in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = jnp.hstack((phis_c2, dQ_dV2))
    elif (
        ("phis_c" not in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = jnp.hstack((dV_dQ2, dQ_dV2))
    elif (
        ("phis_c" in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = jnp.hstack((phis_c2, dV_dQ2, dQ_dV2))

    std_obs = jnp.ones(y_model.shape[0]) * y_err
    numpyro.sample("obs", dist.Normal(y_model, std_obs), obs=y)


def bayes_step_chargecc_sigma(
    y=None,
    y_err=0.1,
    min_sigma=None,
    max_sigma=None,
    sim_params_dict=None,
    jax_func_dict=None,
    jax_params_dict=None,
    target_list=None,
):
    # define parameters (incl. prior ranges)
    deg_params = []
    for ipar, name in enumerate(
        sim_params_dict["chargecc"]["deg_param_names"]
    ):
        deg_params.append(
            numpyro.sample(
                name,
                dist.Uniform(
                    sim_params_dict["chargecc"]["deg_" + name + "_min"],
                    sim_params_dict["chargecc"]["deg_" + name + "_max"],
                ),
            )
        )
    sigma = numpyro.sample("sigma", dist.Uniform(min_sigma, max_sigma))
    # implement the model
    # needs jax numpy for differentiability here
    (phis_c2, dV_dQ2, dQ_dV2), _ = jax_func_dict["chargecc"](
        jnp.array(deg_params)
    )
    if (
        ("phis_c" in target_list)
        and ("dV_dQ" not in target_list)
        and ("dQ_dV" not in target_list)
    ):
        y_model = phis_c2
    elif (
        ("phis_c" not in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" not in target_list)
    ):
        y_model = dV_dQ2
    elif (
        ("phis_c" not in target_list)
        and ("dV_dQ" not in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = dQ_dV2
    elif (
        ("phis_c" in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" not in target_list)
    ):
        y_model = jnp.hstack((phis_c2, dV_dQ2))
    elif (
        ("phis_c" in target_list)
        and ("dV_dQ" not in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = jnp.hstack((phis_c2, dQ_dV2))
    elif (
        ("phis_c" not in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = jnp.hstack((dV_dQ2, dQ_dV2))
    elif (
        ("phis_c" in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = jnp.hstack((phis_c2, dV_dQ2, dQ_dV2))

    std_obs = jnp.ones(y_model.shape[0]) * sigma
    numpyro.sample("obs", dist.Normal(y_model, std_obs), obs=y)


def bayes_step_discharge_chargecc(
    y=None,
    y_err=0.1,
    min_sigma=None,
    max_sigma=None,
    sim_params_dict=None,
    jax_func_dict=None,
    jax_params_dict=None,
    target_list=None,
):
    # define parameters (incl. prior ranges)
    deg_params = []
    for ipar, name in enumerate(
        sim_params_dict["discharge"]["deg_param_names"]
    ):
        deg_params.append(
            numpyro.sample(
                name,
                dist.Uniform(
                    sim_params_dict["discharge"]["deg_" + name + "_min"],
                    sim_params_dict["discharge"]["deg_" + name + "_max"],
                ),
            )
        )
    deg_params.append(
        numpyro.sample(
            "cs0_a_charge",
            dist.Uniform(
                sim_params_dict["chargecc"]["deg_cs0_a_min"],
                sim_params_dict["chargecc"]["deg_cs0_a_max"],
            ),
        )
    )
    deg_params.append(
        numpyro.sample(
            "cs0_c_charge",
            dist.Uniform(
                sim_params_dict["chargecc"]["deg_cs0_c_min"],
                sim_params_dict["chargecc"]["deg_cs0_c_max"],
            ),
        )
    )
    deg_params_discharge = deg_params[:-2]
    deg_params_charge = [
        deg_params[0],
        deg_params[1],
        deg_params[6],
        deg_params[7],
        deg_params[4],
        deg_params[5],
    ]
    # implement the model
    # needs jax numpy for differentiability here
    (phis_c1, dV_dQ1, dQ_dV1), _ = jax_func_dict["discharge"](
        jnp.array(deg_params_discharge)
    )
    (phis_c2, dV_dQ2, dQ_dV2), _ = jax_func_dict["chargecc"](
        jnp.array(deg_params_charge)
    )
    if (
        ("phis_c" in target_list)
        and ("dV_dQ" not in target_list)
        and ("dQ_dV" not in target_list)
    ):
        y_model = jnp.hstack((phis_c1, phis_c2))
    elif (
        ("phis_c" not in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" not in target_list)
    ):
        y_model = jnp.hstack((dV_dQ1, dV_dQ2))
    elif (
        ("phis_c" not in target_list)
        and ("dV_dQ" not in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = jnp.hstack((dQ_dV1, dQ_dV2))
    elif (
        ("phis_c" in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" not in target_list)
    ):
        y_model = jnp.hstack((phis_c1, phis_c2, dV_dQ1, dV_dQ2))
    elif (
        ("phis_c" in target_list)
        and ("dV_dQ" not in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = jnp.hstack((phis_c1, phis_c2, dQ_dV1, dQ_dV2))
    elif (
        ("phis_c" not in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = jnp.hstack((dV_dQ1, dV_dQ2, dQ_dV1, dQ_dV2))
    elif (
        ("phis_c" in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = jnp.hstack(
            (phis_c1, phis_c2, dV_dQ1, dV_dQ2, dQ_dV1, dQ_dV2)
        )

    std_obs = jnp.ones(y_model.shape[0]) * y_err
    numpyro.sample("obs", dist.Normal(y_model, std_obs), obs=y)


def bayes_step_discharge_chargecc_sigma(
    y=None,
    y_err=0.1,
    min_sigma=None,
    max_sigma=None,
    sim_params_dict=None,
    jax_func_dict=None,
    jax_params_dict=None,
    target_list=None,
):
    # define parameters (incl. prior ranges)
    deg_params = []
    for ipar, name in enumerate(
        sim_params_dict["discharge"]["deg_param_names"]
    ):
        deg_params.append(
            numpyro.sample(
                name,
                dist.Uniform(
                    sim_params_dict["discharge"]["deg_" + name + "_min"],
                    sim_params_dict["discharge"]["deg_" + name + "_max"],
                ),
            )
        )
    deg_params.append(
        numpyro.sample(
            "cs0_a_charge",
            dist.Uniform(
                sim_params_dict["chargecc"]["deg_cs0_a_min"],
                sim_params_dict["chargecc"]["deg_cs0_a_max"],
            ),
        )
    )
    deg_params.append(
        numpyro.sample(
            "cs0_c_charge",
            dist.Uniform(
                sim_params_dict["chargecc"]["deg_cs0_c_min"],
                sim_params_dict["chargecc"]["deg_cs0_c_max"],
            ),
        )
    )
    sigma = numpyro.sample("sigma", dist.Uniform(min_sigma, max_sigma))
    deg_params_discharge = deg_params[:-2]
    deg_params_charge = [
        deg_params[0],
        deg_params[1],
        deg_params[6],
        deg_params[7],
        deg_params[4],
        deg_params[5],
    ]
    # implement the model
    # needs jax numpy for differentiability here
    (phis_c1, dV_dQ1, dQ_dV1), _ = jax_func_dict["discharge"](
        jnp.array(deg_params_discharge)
    )
    (phis_c2, dV_dQ2, dQ_dV2), _ = jax_func_dict["chargecc"](
        jnp.array(deg_params_charge)
    )
    if (
        ("phis_c" in target_list)
        and ("dV_dQ" not in target_list)
        and ("dQ_dV" not in target_list)
    ):
        y_model = jnp.hstack((phis_c1, phis_c2))
    elif (
        ("phis_c" not in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" not in target_list)
    ):
        y_model = jnp.hstack((dV_dQ1, dV_dQ2))
    elif (
        ("phis_c" not in target_list)
        and ("dV_dQ" not in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = jnp.hstack((dQ_dV1, dQ_dV2))
    elif (
        ("phis_c" in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" not in target_list)
    ):
        y_model = jnp.hstack((phis_c1, phis_c2, dV_dQ1, dV_dQ2))
    elif (
        ("phis_c" in target_list)
        and ("dV_dQ" not in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = jnp.hstack((phis_c1, phis_c2, dQ_dV1, dQ_dV2))
    elif (
        ("phis_c" not in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = jnp.hstack((dV_dQ1, dV_dQ2, dQ_dV1, dQ_dV2))
    elif (
        ("phis_c" in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = jnp.hstack(
            (phis_c1, phis_c2, dV_dQ1, dV_dQ2, dQ_dV1, dQ_dV2)
        )

    std_obs = jnp.ones(y_model.shape[0]) * sigma
    numpyro.sample("obs", dist.Normal(y_model, std_obs), obs=y)


def collect_observation_files(args_cal, nave):
    filename = {}
    if (
        args_cal.cyc_mode == "discharge"
        or args_cal.cyc_mode == "discharge-chargecc"
    ):
        filename["discharge"] = {}
        filename["discharge"]["phis_c"] = obs_filename(
            args_cal.obsFolder,
            args_cal.cell_id,
            args_cal.cycle_id,
            nave,
            cyc_mode="discharge",
            target="phis_c",
        )
        filename["discharge"]["dQ_dV"] = obs_filename(
            args_cal.obsFolder,
            args_cal.cell_id,
            args_cal.cycle_id,
            nave,
            cyc_mode="discharge",
            target="dQ_dV",
        )
        filename["discharge"]["dV_dQ"] = obs_filename(
            args_cal.obsFolder,
            args_cal.cell_id,
            args_cal.cycle_id,
            nave,
            cyc_mode="discharge",
            target="dV_dQ",
        )
        # samp_key = 'discharge'
    if (
        args_cal.cyc_mode == "chargecc"
        or args_cal.cyc_mode == "discharge-chargecc"
    ):
        filename["chargecc"] = {}
        filename["chargecc"]["phis_c"] = obs_filename(
            args_cal.obsFolder,
            args_cal.cell_id,
            args_cal.cycle_id,
            nave,
            cyc_mode="chargecc",
            target="phis_c",
        )
        filename["chargecc"]["dQ_dV"] = obs_filename(
            args_cal.obsFolder,
            args_cal.cell_id,
            args_cal.cycle_id,
            nave,
            cyc_mode="chargecc",
            target="dQ_dV",
        )
        filename["chargecc"]["dV_dQ"] = obs_filename(
            args_cal.obsFolder,
            args_cal.cell_id,
            args_cal.cycle_id,
            nave,
            cyc_mode="chargecc",
            target="dV_dQ",
        )
        # samp_key = 'chargecc'

    return filename


def bayes_step_discharge(
    y=None,
    y_err=0.1,
    min_sigma=None,
    max_sigma=None,
    sim_params_dict=None,
    jax_func_dict=None,
    jax_params_dict=None,
    target_list=None,
):
    # define parameters (incl. prior ranges)
    deg_params = []
    for ipar, name in enumerate(
        sim_params_dict["discharge"]["deg_param_names"]
    ):
        deg_params.append(
            numpyro.sample(
                name,
                dist.Uniform(
                    sim_params_dict["discharge"]["deg_" + name + "_min"],
                    sim_params_dict["discharge"]["deg_" + name + "_max"],
                ),
            )
        )
    # implement the model
    # needs jax numpy for differentiability here
    (phis_c1, dV_dQ1, dQ_dV1), _ = jax_func_dict["discharge"](
        jnp.array(deg_params)
    )
    if (
        ("phis_c" in target_list)
        and ("dV_dQ" not in target_list)
        and ("dQ_dV" not in target_list)
    ):
        y_model = phis_c1
    elif (
        ("phis_c" not in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" not in target_list)
    ):
        y_model = dV_dQ1
    elif (
        ("phis_c" not in target_list)
        and ("dV_dQ" not in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = dQ_dV1
    elif (
        ("phis_c" in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" not in target_list)
    ):
        y_model = jnp.hstack((phis_c1, dV_dQ1))
    elif (
        ("phis_c" in target_list)
        and ("dV_dQ" not in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = jnp.hstack((phis_c1, dQ_dV1))
    elif (
        ("phis_c" not in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = jnp.hstack((dV_dQ1, dQ_dV1))
    elif (
        ("phis_c" in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = jnp.hstack((phis_c1, dV_dQ1, dQ_dV1))

    std_obs = jnp.ones(y_model.shape[0]) * y_err
    numpyro.sample("obs", dist.Normal(y_model, std_obs), obs=y)


def bayes_step_discharge_sigma(
    y=None,
    y_err=0.1,
    min_sigma=None,
    max_sigma=None,
    sim_params_dict=None,
    jax_func_dict=None,
    jax_params_dict=None,
    target_list=None,
):
    # define parameters (incl. prior ranges)
    deg_params = []
    for ipar, name in enumerate(
        sim_params_dict["discharge"]["deg_param_names"]
    ):
        deg_params.append(
            numpyro.sample(
                name,
                dist.Uniform(
                    sim_params_dict["discharge"]["deg_" + name + "_min"],
                    sim_params_dict["discharge"]["deg_" + name + "_max"],
                ),
            )
        )
    sigma = numpyro.sample("sigma", dist.Uniform(min_sigma, max_sigma))
    # implement the model
    # needs jax numpy for differentiability here
    # (phis_c1, dV_dQ1, dQ_dV1), _ = jax_func_dict["discharge"](jnp.array(deg_params))
    phis_c1 = jax_func_dict["discharge"](jnp.array(deg_params))
    if (
        ("phis_c" in target_list)
        and ("dV_dQ" not in target_list)
        and ("dQ_dV" not in target_list)
    ):
        y_model = phis_c1
    elif (
        ("phis_c" not in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" not in target_list)
    ):
        y_model = dV_dQ1
    elif (
        ("phis_c" not in target_list)
        and ("dV_dQ" not in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = dQ_dV1
    elif (
        ("phis_c" in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" not in target_list)
    ):
        y_model = jnp.hstack((phis_c1, dV_dQ1))
    elif (
        ("phis_c" in target_list)
        and ("dV_dQ" not in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = jnp.hstack((phis_c1, dQ_dV1))
    elif (
        ("phis_c" not in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = jnp.hstack((dV_dQ1, dQ_dV1))
    elif (
        ("phis_c" in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = jnp.hstack((phis_c1, dV_dQ1, dQ_dV1))

    std_obs = jnp.ones(y_model.shape[0]) * sigma
    numpyro.sample("obs", dist.Normal(y_model, std_obs), obs=y)


def bayes_step_chargecc(
    y=None,
    y_err=0.1,
    min_sigma=None,
    max_sigma=None,
    sim_params_dict=None,
    jax_func_dict=None,
    jax_params_dict=None,
    target_list=None,
):
    # define parameters (incl. prior ranges)
    deg_params = []
    for ipar, name in enumerate(
        sim_params_dict["chargecc"]["deg_param_names"]
    ):
        deg_params.append(
            numpyro.sample(
                name,
                dist.Uniform(
                    sim_params_dict["chargecc"]["deg_" + name + "_min"],
                    sim_params_dict["chargecc"]["deg_" + name + "_max"],
                ),
            )
        )
    # implement the model
    # needs jax numpy for differentiability here
    (phis_c2, dV_dQ2, dQ_dV2), _ = jax_func_dict["chargecc"](
        jnp.array(deg_params)
    )
    if (
        ("phis_c" in target_list)
        and ("dV_dQ" not in target_list)
        and ("dQ_dV" not in target_list)
    ):
        y_model = phis_c2
    elif (
        ("phis_c" not in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" not in target_list)
    ):
        y_model = dV_dQ2
    elif (
        ("phis_c" not in target_list)
        and ("dV_dQ" not in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = dQ_dV2
    elif (
        ("phis_c" in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" not in target_list)
    ):
        y_model = jnp.hstack((phis_c2, dV_dQ2))
    elif (
        ("phis_c" in target_list)
        and ("dV_dQ" not in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = jnp.hstack((phis_c2, dQ_dV2))
    elif (
        ("phis_c" not in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = jnp.hstack((dV_dQ2, dQ_dV2))
    elif (
        ("phis_c" in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = jnp.hstack((phis_c2, dV_dQ2, dQ_dV2))

    std_obs = jnp.ones(y_model.shape[0]) * y_err
    numpyro.sample("obs", dist.Normal(y_model, std_obs), obs=y)


def bayes_step_chargecc_sigma(
    y=None,
    y_err=0.1,
    min_sigma=None,
    max_sigma=None,
    sim_params_dict=None,
    jax_func_dict=None,
    jax_params_dict=None,
    target_list=None,
):
    # define parameters (incl. prior ranges)
    deg_params = []
    for ipar, name in enumerate(
        sim_params_dict["chargecc"]["deg_param_names"]
    ):
        deg_params.append(
            numpyro.sample(
                name,
                dist.Uniform(
                    sim_params_dict["chargecc"]["deg_" + name + "_min"],
                    sim_params_dict["chargecc"]["deg_" + name + "_max"],
                ),
            )
        )
    sigma = numpyro.sample("sigma", dist.Uniform(min_sigma, max_sigma))
    # implement the model
    # needs jax numpy for differentiability here
    # (phis_c2, dV_dQ2, dQ_dV2), _ = jax_func_dict["chargecc"](
    #    jnp.array(deg_params)
    # )
    phis_c2 = jax_func_dict["chargecc"](jnp.array(deg_params))
    if (
        ("phis_c" in target_list)
        and ("dV_dQ" not in target_list)
        and ("dQ_dV" not in target_list)
    ):
        y_model = phis_c2
    elif (
        ("phis_c" not in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" not in target_list)
    ):
        y_model = dV_dQ2
    elif (
        ("phis_c" not in target_list)
        and ("dV_dQ" not in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = dQ_dV2
    elif (
        ("phis_c" in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" not in target_list)
    ):
        y_model = jnp.hstack((phis_c2, dV_dQ2))
    elif (
        ("phis_c" in target_list)
        and ("dV_dQ" not in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = jnp.hstack((phis_c2, dQ_dV2))
    elif (
        ("phis_c" not in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = jnp.hstack((dV_dQ2, dQ_dV2))
    elif (
        ("phis_c" in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = jnp.hstack((phis_c2, dV_dQ2, dQ_dV2))

    std_obs = jnp.ones(y_model.shape[0]) * sigma
    numpyro.sample("obs", dist.Normal(y_model, std_obs), obs=y)


def bayes_step_discharge_chargecc(
    y=None,
    y_err=0.1,
    min_sigma=None,
    max_sigma=None,
    sim_params_dict=None,
    jax_func_dict=None,
    jax_params_dict=None,
    target_list=None,
):
    # define parameters (incl. prior ranges)
    deg_params = []
    for ipar, name in enumerate(
        sim_params_dict["discharge"]["deg_param_names"]
    ):
        deg_params.append(
            numpyro.sample(
                name,
                dist.Uniform(
                    sim_params_dict["discharge"]["deg_" + name + "_min"],
                    sim_params_dict["discharge"]["deg_" + name + "_max"],
                ),
            )
        )
    deg_params.append(
        numpyro.sample(
            "cs0_a_charge",
            dist.Uniform(
                sim_params_dict["chargecc"]["deg_cs0_a_min"],
                sim_params_dict["chargecc"]["deg_cs0_a_max"],
            ),
        )
    )
    deg_params.append(
        numpyro.sample(
            "cs0_c_charge",
            dist.Uniform(
                sim_params_dict["chargecc"]["deg_cs0_c_min"],
                sim_params_dict["chargecc"]["deg_cs0_c_max"],
            ),
        )
    )
    deg_params_discharge = deg_params[:-2]
    deg_params_charge = [
        deg_params[0],
        deg_params[1],
        deg_params[6],
        deg_params[7],
        deg_params[4],
        deg_params[5],
    ]
    # implement the model
    # needs jax numpy for differentiability here
    # (phis_c1, dV_dQ1, dQ_dV1), _ = jax_func_dict["discharge"](
    #    jnp.array(deg_params_discharge)
    # )
    # (phis_c2, dV_dQ2, dQ_dV2), _ = jax_func_dict["chargecc"](
    #    jnp.array(deg_params_charge)
    # )
    phis_c1 = jax_func_dict["discharge"](jnp.array(deg_params_discharge))
    phis_c2 = jax_func_dict["chargecc"](jnp.array(deg_params_charge))
    if (
        ("phis_c" in target_list)
        and ("dV_dQ" not in target_list)
        and ("dQ_dV" not in target_list)
    ):
        y_model = jnp.hstack((phis_c1, phis_c2))
    elif (
        ("phis_c" not in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" not in target_list)
    ):
        y_model = jnp.hstack((dV_dQ1, dV_dQ2))
    elif (
        ("phis_c" not in target_list)
        and ("dV_dQ" not in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = jnp.hstack((dQ_dV1, dQ_dV2))
    elif (
        ("phis_c" in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" not in target_list)
    ):
        y_model = jnp.hstack((phis_c1, phis_c2, dV_dQ1, dV_dQ2))
    elif (
        ("phis_c" in target_list)
        and ("dV_dQ" not in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = jnp.hstack((phis_c1, phis_c2, dQ_dV1, dQ_dV2))
    elif (
        ("phis_c" not in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = jnp.hstack((dV_dQ1, dV_dQ2, dQ_dV1, dQ_dV2))
    elif (
        ("phis_c" in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = jnp.hstack(
            (phis_c1, phis_c2, dV_dQ1, dV_dQ2, dQ_dV1, dQ_dV2)
        )

    std_obs = jnp.ones(y_model.shape[0]) * y_err
    numpyro.sample("obs", dist.Normal(y_model, std_obs), obs=y)


def bayes_step_discharge_chargecc_sigma(
    y=None,
    y_err=0.1,
    min_sigma=None,
    max_sigma=None,
    sim_params_dict=None,
    jax_func_dict=None,
    jax_params_dict=None,
    target_list=None,
):
    # define parameters (incl. prior ranges)
    deg_params = []
    for ipar, name in enumerate(
        sim_params_dict["discharge"]["deg_param_names"]
    ):
        deg_params.append(
            numpyro.sample(
                name,
                dist.Uniform(
                    sim_params_dict["discharge"]["deg_" + name + "_min"],
                    sim_params_dict["discharge"]["deg_" + name + "_max"],
                ),
            )
        )
    deg_params.append(
        numpyro.sample(
            "cs0_a_charge",
            dist.Uniform(
                sim_params_dict["chargecc"]["deg_cs0_a_min"],
                sim_params_dict["chargecc"]["deg_cs0_a_max"],
            ),
        )
    )
    deg_params.append(
        numpyro.sample(
            "cs0_c_charge",
            dist.Uniform(
                sim_params_dict["chargecc"]["deg_cs0_c_min"],
                sim_params_dict["chargecc"]["deg_cs0_c_max"],
            ),
        )
    )
    sigma = numpyro.sample("sigma", dist.Uniform(min_sigma, max_sigma))
    deg_params_discharge = deg_params[:-2]
    deg_params_charge = [
        deg_params[0],
        deg_params[1],
        deg_params[6],
        deg_params[7],
        deg_params[4],
        deg_params[5],
    ]
    # implement the model
    # needs jax numpy for differentiability here
    # (phis_c1, dV_dQ1, dQ_dV1), _ = jax_func_dict["discharge"](
    #    jnp.array(deg_params_discharge)
    # )
    # (phis_c2, dV_dQ2, dQ_dV2), _ = jax_func_dict["chargecc"](
    #    jnp.array(deg_params_charge)
    # )
    phis_c1 = jax_func_dict["discharge"](jnp.array(deg_params_discharge))
    phis_c2 = jax_func_dict["chargecc"](jnp.array(deg_params_charge))
    if (
        ("phis_c" in target_list)
        and ("dV_dQ" not in target_list)
        and ("dQ_dV" not in target_list)
    ):
        y_model = jnp.hstack((phis_c1, phis_c2))
    elif (
        ("phis_c" not in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" not in target_list)
    ):
        y_model = jnp.hstack((dV_dQ1, dV_dQ2))
    elif (
        ("phis_c" not in target_list)
        and ("dV_dQ" not in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = jnp.hstack((dQ_dV1, dQ_dV2))
    elif (
        ("phis_c" in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" not in target_list)
    ):
        y_model = jnp.hstack((phis_c1, phis_c2, dV_dQ1, dV_dQ2))
    elif (
        ("phis_c" in target_list)
        and ("dV_dQ" not in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = jnp.hstack((phis_c1, phis_c2, dQ_dV1, dQ_dV2))
    elif (
        ("phis_c" not in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = jnp.hstack((dV_dQ1, dV_dQ2, dQ_dV1, dQ_dV2))
    elif (
        ("phis_c" in target_list)
        and ("dV_dQ" in target_list)
        and ("dQ_dV" in target_list)
    ):
        y_model = jnp.hstack(
            (phis_c1, phis_c2, dV_dQ1, dV_dQ2, dQ_dV1, dQ_dV2)
        )

    std_obs = jnp.ones(y_model.shape[0]) * sigma
    numpyro.sample("obs", dist.Normal(y_model, std_obs), obs=y)


def mcmc_iter(
    y_err=0.1,
    mcmc_method="HMC",
    cyc_mode="discharge",
    num_chains=None,
    cal_sigma=False,
    read_sigma=False,
    min_sigma=None,
    max_sigma=None,
    sim_params_dict=None,
    target_list=["phis_c"],
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
            sim_params_dict["discharge"]["deg_param_names"]
        ):
            theta.append(
                np.random.uniform(
                    sim_params_dict["discharge"]["deg_" + name + "_min"],
                    sim_params_dict["discharge"]["deg_" + name + "_max"],
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
                sim_params_dict["chargecc"]["deg_cs0_a_min"],
                sim_params_dict["chargecc"]["deg_cs0_a_max"],
            )
        )
        theta.append(
            np.random.uniform(
                sim_params_dict["chargecc"]["deg_cs0_c_min"],
                sim_params_dict["chargecc"]["deg_cs0_c_max"],
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
                            "cs0_a": 0.8,
                            "cs0_c": 1.15,
                            "i0_c": 0.37,
                            "eps_s_c_am": 0.92,
                        }
                    elif cell_id == 31:
                        init_val = True
                        val_dict = {
                            "i0_a": 3.87,
                            "ds_c": 9.44,
                            "cs0_a": 0.81,
                            "cs0_c": 1.14,
                            "i0_c": 0.51,
                            "eps_s_c_am": 0.92,
                        }
                    elif cell_id == 32:
                        init_val = True
                        val_dict = {
                            "i0_a": 3.84,
                            "ds_c": 9.34,
                            "cs0_a": 0.78,
                            "cs0_c": 1.18,
                            "i0_c": 0.42,
                            "eps_s_c_am": 0.91,
                        }
                    elif cell_id == 35:
                        init_val = True
                        val_dict = {
                            "i0_a": 3.9,
                            "ds_c": 9.56,
                            "cs0_a": 0.78,
                            "cs0_c": 1.17,
                            "i0_c": 0.58,
                            "eps_s_c_am": 0.91,
                        }
                    elif cell_id == 39:
                        init_val = True
                        val_dict = {
                            "i0_a": 3.89,
                            "ds_c": 9.59,
                            "cs0_a": 0.84,
                            "cs0_c": 1.09,
                            "i0_c": 0.35,
                            "eps_s_c_am": 0.93,
                        }
                    elif cell_id == 42:
                        init_val = True
                        val_dict = {
                            "i0_a": 3.91,
                            "ds_c": 9.7,
                            "cs0_a": 0.84,
                            "cs0_c": 1.09,
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
                            "cs0_a": 0.45,
                            "cs0_c": 1.07,
                            "i0_c": 0.11,
                            "eps_s_c_am": 0.99,
                        }
                    if cell_id == 31:
                        init_val = True
                        val_dict = {
                            "i0_a": 0.26,
                            "ds_c": 0.51,
                            "cs0_a": 0.49,
                            "cs0_c": 1.07,
                            "i0_c": 0.11,
                            "eps_s_c_am": 0.99,
                        }
                    if cell_id == 32:
                        init_val = True
                        val_dict = {
                            "i0_a": 0.14,
                            "ds_c": 0.54,
                            "cs0_a": 0.35,
                            "cs0_c": 1.07,
                            "i0_c": 0.11,
                            "eps_s_c_am": 0.99,
                        }
                    if cell_id == 35:
                        init_val = True
                        val_dict = {
                            "i0_a": 0.15,
                            "ds_c": 0.58,
                            "cs0_a": 0.38,
                            "cs0_c": 1.07,
                            "i0_c": 0.11,
                            "eps_s_c_am": 0.99,
                        }
                    if cell_id == 39:
                        init_val = True
                        val_dict = {
                            "i0_a": 0.11,
                            "ds_c": 0.62,
                            "cs0_a": 0.31,
                            "cs0_c": 1.06,
                            "i0_c": 0.11,
                            "eps_s_c_am": 0.99,
                        }
                    if cell_id == 42:
                        init_val = True
                        val_dict = {
                            "i0_a": 0.11,
                            "ds_c": 0.68,
                            "cs0_a": 0.31,
                            "cs0_c": 1.03,
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

                        # val_dict = {"i0_a": 0.11, "ds_c": 0.52, "cs0_a":0.8, "cs0_c":1.11, "i0_c":0.21, "eps_s_c_am":0.99, "cs0_a_charge":0.48, "cs0_c_charge":1.07}
                        val_dict = {
                            "i0_a": 0.1401,
                            "ds_c": 8.727,
                            "cs0_a": 1.077,
                            "cs0_c": 1.176,
                            "i0_c": 0.1011,
                            "eps_s_c_am": 0.9167,
                            "cs0_a_charge": 0.3054,
                            "cs0_c_charge": 1.047,
                        }
                    elif cell_id == 31:
                        init_val = True
                        val_dict = {
                            "i0_a": 0.2192,
                            "ds_c": 9.154,
                            "cs0_a": 1.077,
                            "cs0_c": 1.18,
                            "i0_c": 0.1005,
                            "eps_s_c_am": 0.9281,
                            "cs0_a_charge": 0.343,
                            "cs0_c_charge": 1.041,
                        }
                        # val_dict = {"i0_a": 0.11, "ds_c": 0.52, "cs0_a":0.8, "cs0_c":1.11, "i0_c":0.21, "eps_s_c_am":0.99, "cs0_a_charge":0.48, "cs0_c_charge":1.07}
                    elif cell_id == 32:
                        init_val = True
                        val_dict = {
                            "i0_a": 0.11,
                            "ds_c": 0.53,
                            "cs0_a": 0.77,
                            "cs0_c": 1.15,
                            "i0_c": 0.14,
                            "eps_s_c_am": 0.99,
                            "cs0_a_charge": 0.35,
                            "cs0_c_charge": 1.07,
                        }
                    elif cell_id == 35:
                        init_val = True
                        val_dict = {
                            "i0_a": 0.11,
                            "ds_c": 0.56,
                            "cs0_a": 0.77,
                            "cs0_c": 1.15,
                            "i0_c": 0.15,
                            "eps_s_c_am": 0.99,
                            "cs0_a_charge": 0.38,
                            "cs0_c_charge": 1.07,
                        }
                    elif cell_id == 39:
                        init_val = True
                        val_dict = {
                            "i0_a": 0.11,
                            "ds_c": 0.62,
                            "cs0_a": 0.84,
                            "cs0_c": 1.05,
                            "i0_c": 0.11,
                            "eps_s_c_am": 0.99,
                            "cs0_a_charge": 0.31,
                            "cs0_c_charge": 1.06,
                        }
                    elif cell_id == 42:
                        init_val = True
                        val_dict = {
                            "i0_a": 0.11,
                            "ds_c": 0.69,
                            "cs0_a": 0.83,
                            "cs0_c": 1.06,
                            "i0_c": 0.11,
                            "eps_s_c_am": 0.99,
                            "cs0_a_charge": 0.31,
                            "cs0_c_charge": 1.03,
                        }
                    if cal_sigma:
                        val_dict["sigma"] = (
                            min_sigma + (max_sigma - min_sigma) * factor
                        )
                if init_val:
                    val_dict = perturb_val_dict(val_dict, sim_params_dict)
                    init_strategy = init_to_value(values=val_dict)
                    step_size = 0.01
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
                sim_params_dict=sim_params_dict,
                jax_func_dict=jax_func_dict,
                jax_params_dict=jax_params_dict,
                target_list=target_list,
                # cons_LLI=cons_LLI,
                # variable_delete=variable_delete,
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
            "cs0_a_charge",
            "cs0_c_charge",
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
        # realization_dV_dQ["discharge"] = []
        # realization_dQ_dV["discharge"] = []
        for i in range(min(nsamples, 40)):
            if cal_sigma:
                # phis_c, dV_dQ, dQ_dV = forward_dict["discharge"](
                #    np_mcmc_samples[i, :-1]
                # )
                phis_c = forward_dict["discharge"](
                    np_mcmc_samples[i, :-1].astype("float32")
                )
            else:
                # phis_c, dV_dQ, dQ_dV = forward_dict["discharge"](
                #    np_mcmc_samples[i, :]
                # )
                phis_c = forward_dict["discharge"](
                    np_mcmc_samples[i, :].astype("float32")
                )
            realization_phis_c["discharge"].append(phis_c.detach())
            # realization_dV_dQ["discharge"].append(dV_dQ)
            # realization_dQ_dV["discharge"].append(dQ_dV)
        realization_phis_c["discharge"] = np.array(
            realization_phis_c["discharge"]
        )
        # realization_dV_dQ["discharge"] = np.array(
        #    realization_dV_dQ["discharge"]
        # )
        # realization_dQ_dV["discharge"] = np.array(
        #    realization_dQ_dV["discharge"]
        # )

    if cyc_mode.lower() == "chargecc":
        realization_phis_c["chargecc"] = []
        # realization_dV_dQ["chargecc"] = []
        # realization_dQ_dV["chargecc"] = []
        for i in range(min(nsamples, 40)):
            if cal_sigma:
                # phis_c, dV_dQ, dQ_dV = forward_dict["chargecc"](
                #    np_mcmc_samples[i, :-1]
                # )
                phis_c = forward_dict["chargecc"](np_mcmc_samples[i, :-1])
            else:
                # phis_c, dV_dQ, dQ_dV = forward_dict["chargecc"](
                #    np_mcmc_samples[i, :]
                # )
                phis_c = forward_dict["chargecc"](np_mcmc_samples[i, :])
            realization_phis_c["chargecc"].append(phis_c)
            # realization_dV_dQ["chargecc"].append(dV_dQ)
            # realization_dQ_dV["chargecc"].append(dQ_dV)
        realization_phis_c["chargecc"] = np.array(
            realization_phis_c["chargecc"]
        )
        # realization_dV_dQ["chargecc"] = np.array(realization_dV_dQ["chargecc"])
        # realization_dQ_dV["chargecc"] = np.array(realization_dQ_dV["chargecc"])
    if cyc_mode.lower() == "discharge-chargecc":
        realization_phis_c["discharge"] = []
        # realization_dV_dQ["discharge"] = []
        # realization_dQ_dV["discharge"] = []
        for i in range(min(nsamples, 40)):
            if cal_sigma:
                # phis_c, dV_dQ, dQ_dV = forward_dict["discharge"](
                #    np_mcmc_samples[i, :-3]
                # )
                phis_c = forward_dict["discharge"](np_mcmc_samples[i, :-3])
            else:
                # phis_c, dV_dQ, dQ_dV = forward_dict["discharge"](
                #    np_mcmc_samples[i, :-2]
                # )
                phis_c = forward_dict["discharge"](np_mcmc_samples[i, :-2])
            realization_phis_c["discharge"].append(phis_c)
            # realization_dV_dQ["discharge"].append(dV_dQ)
            # realization_dQ_dV["discharge"].append(dQ_dV)
        realization_phis_c["discharge"] = np.array(
            realization_phis_c["discharge"]
        )
        # realization_dV_dQ["discharge"] = np.array(
        #    realization_dV_dQ["discharge"]
        # )
        # realization_dQ_dV["discharge"] = np.array(
        #    realization_dQ_dV["discharge"]
        # )

        realization_phis_c["chargecc"] = []
        # realization_dV_dQ["chargecc"] = []
        # realization_dQ_dV["chargecc"] = []
        indc = list(range(sim_params_dict["chargecc"]["n_params"]))
        indc[sim_params_dict["chargecc"]["ind_deg_cs0_a"]] = sim_params_dict[
            "chargecc"
        ]["n_params"]
        indc[sim_params_dict["chargecc"]["ind_deg_cs0_c"]] = (
            sim_params_dict["chargecc"]["n_params"] + 1
        )
        for i in range(min(nsamples, 40)):
            if cal_sigma:
                # phis_c, dV_dQ, dQ_dV = forward_dict["chargecc"](
                #    np_mcmc_samples[i, indc]
                # )
                phis_c = forward_dict["chargecc"](np_mcmc_samples[i, indc])
            else:
                # phis_c, dV_dQ, dQ_dV = forward_dict["chargecc"](
                #    np_mcmc_samples[i, indc]
                # )
                phis_c = forward_dict["chargecc"](np_mcmc_samples[i, indc])
            realization_phis_c["chargecc"].append(phis_c)
            # realization_dV_dQ["chargecc"].append(dV_dQ)
            # realization_dQ_dV["chargecc"].append(dQ_dV)
        realization_phis_c["chargecc"] = np.array(
            realization_phis_c["chargecc"]
        )
        # realization_dV_dQ["chargecc"] = np.array(realization_dV_dQ["chargecc"])
        # realization_dQ_dV["chargecc"] = np.array(realization_dQ_dV["chargecc"])

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


def mcmc_iter_synth(
    y_err=0.1,
    mcmc_method="HMC",
    cyc_mode="discharge",
    num_chains=None,
    cal_sigma=True,
    min_sigma=None,
    max_sigma=None,
    sim_params_dict=None,
    target_list=["phis_c"],
    data_phis_c=None,
    data_dV_dQ_y=None,
    data_dQ_dV_y=None,
    jax_func_dict=None,
    jax_params_dict=None,
    forward_dict=None,
    num_warmup=None,
    num_samples=None,
    save_sigma=False,
    cons_LLI=False,
    variable_delete=None,
    parallel_env=None,
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
                sim_params_dict["chargecc"]["deg_cs0_a_min"],
                sim_params_dict["chargecc"]["deg_cs0_a_max"],
            )
        )
        theta.append(
            np.random.uniform(
                sim_params_dict["chargecc"]["deg_cs0_c_min"],
                sim_params_dict["chargecc"]["deg_cs0_c_max"],
            )
        )

    if cal_sigma:
        theta.append(np.random.uniform(min_sigma, max_sigma))
    if cyc_mode.lower() == "discharge":
        if cal_sigma:
            bayes_step = bayes_step_discharge_sigma
        else:
            bayes_step = bayes_step_discharge
    elif cyc_mode.lower() == "chargecc":
        if cal_sigma:
            bayes_step = bayes_step_chargecc_sigma
        else:
            bayes_step = bayes_step_chargecc
    elif cyc_mode.lower() == "discharge-chargecc":
        if cal_sigma:
            bayes_step = bayes_step_discharge_chargecc_sigma
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

            if parallel_env is None or parallel_env.nProc == 1:
                chain_method = "parallel"
            else:
                chain_method = "sequential"

            mcmc = MCMC(
                kernel,
                num_chains=num_chains,
                num_warmup=num_warmup,
                num_samples=num_samples,
                chain_method=chain_method,
                # jit_model_args=True,
            )
            mcmc.run(
                rng_key_,
                y=data_tar,
                y_err=data_err,
                min_sigma=min_sigma,
                max_sigma=max_sigma,
                sim_params_dict=sim_params_dict,
                jax_func_dict=jax_func_dict,
                jax_params_dict=jax_params_dict,
                target_list=target_list,
                # cons_LLI=cons_LLI,
                # variable_delete=variable_delete,
            )
            break
        except RuntimeError as err:
            print(err)
            factor += 0.1
            print(f"Failed, resampling init parameters with factor = {factor}")
            if factor > 1:
                raise ValueError
    if parallel_env is None or parallel_env.nProc == 1:
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
            "cs0_a_charge",
            "cs0_c_charge",
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
        # realization_dV_dQ["discharge"] = []
        # realization_dQ_dV["discharge"] = []
        for i in range(min(nsamples, 40)):
            if cal_sigma:
                # phis_c, dV_dQ, dQ_dV = forward_dict["discharge"](
                #    np_mcmc_samples[i, :-1]
                # )
                phis_c = forward_dict["discharge"](
                    np_mcmc_samples[i, :-1].astype("float32")
                )
            else:
                # phis_c, dV_dQ, dQ_dV = forward_dict["discharge"](
                #    np_mcmc_samples[i, :]
                # )
                phis_c = forward_dict["discharge"](
                    np_mcmc_samples[i, :].astype("float32")
                )
            realization_phis_c["discharge"].append(phis_c.detach())
            # realization_dV_dQ["discharge"].append(dV_dQ)
            # realization_dQ_dV["discharge"].append(dQ_dV)
        realization_phis_c["discharge"] = np.array(
            realization_phis_c["discharge"]
        )
        # realization_dV_dQ["discharge"] = np.array(
        #    realization_dV_dQ["discharge"]
        # )
        # realization_dQ_dV["discharge"] = np.array(
        #    realization_dQ_dV["discharge"]
        # )

    if cyc_mode.lower() == "chargecc":
        realization_phis_c["chargecc"] = []
        # realization_dV_dQ["chargecc"] = []
        # realization_dQ_dV["chargecc"] = []
        for i in range(min(nsamples, 40)):
            if cal_sigma:
                # phis_c, dV_dQ, dQ_dV = forward_dict["chargecc"](
                #    np_mcmc_samples[i, :-1]
                # )
                phis_c = forward_dict["chargecc"](
                    np_mcmc_samples[i, :-1].astype("float32")
                )
            else:
                # phis_c, dV_dQ, dQ_dV = forward_dict["chargecc"](
                #    np_mcmc_samples[i, :]
                # )
                phis_c = forward_dict["chargecc"](
                    np_mcmc_samples[i, :].astype("float32")
                )
            realization_phis_c["chargecc"].append(phis_c.detach())
            # realization_dV_dQ["chargecc"].append(dV_dQ)
            # realization_dQ_dV["chargecc"].append(dQ_dV)
        realization_phis_c["chargecc"] = np.array(
            realization_phis_c["chargecc"]
        )
        # realization_dV_dQ["chargecc"] = np.array(realization_dV_dQ["chargecc"])
        # realization_dQ_dV["chargecc"] = np.array(realization_dQ_dV["chargecc"])
    if cyc_mode.lower() == "discharge-chargecc":
        realization_phis_c["discharge"] = []
        # realization_dV_dQ["discharge"] = []
        # realization_dQ_dV["discharge"] = []
        for i in range(min(nsamples, 40)):
            if cal_sigma:
                # phis_c, dV_dQ, dQ_dV = forward_dict["discharge"](
                #    np_mcmc_samples[i, :-3]
                # )
                phis_c = forward_dict["discharge"](
                    np_mcmc_samples[i, :-3].astype("float32")
                )
            else:
                # phis_c, dV_dQ, dQ_dV = forward_dict["discharge"](
                #    np_mcmc_samples[i, :-2]
                # )
                phis_c = forward_dict["discharge"](
                    np_mcmc_samples[i, :-2].astype("float32")
                )
            realization_phis_c["discharge"].append(phis_c.detach())
            # realization_dV_dQ["discharge"].append(dV_dQ)
            # realization_dQ_dV["discharge"].append(dQ_dV)
        realization_phis_c["discharge"] = np.array(
            realization_phis_c["discharge"]
        )
        # realization_dV_dQ["discharge"] = np.array(
        #    realization_dV_dQ["discharge"]
        # )
        # realization_dQ_dV["discharge"] = np.array(
        #    realization_dQ_dV["discharge"]
        # )

        realization_phis_c["chargecc"] = []
        # realization_dV_dQ["chargecc"] = []
        # realization_dQ_dV["chargecc"] = []
        indc = list(range(sim_params_dict["chargecc"]["n_params"]))
        indc[sim_params_dict["chargecc"]["ind_deg_cs0_a"]] = sim_params_dict[
            "chargecc"
        ]["n_params"]
        indc[sim_params_dict["chargecc"]["ind_deg_cs0_c"]] = (
            sim_params_dict["chargecc"]["n_params"] + 1
        )
        for i in range(min(nsamples, 40)):
            if cal_sigma:
                # phis_c, dV_dQ, dQ_dV = forward_dict["chargecc"](
                #    np_mcmc_samples[i, indc]
                # )
                phis_c = forward_dict["chargecc"](
                    np_mcmc_samples[i, indc].astype("float32")
                )
            else:
                # phis_c, dV_dQ, dQ_dV = forward_dict["chargecc"](
                #    np_mcmc_samples[i, indc]
                # )
                phis_c = forward_dict["chargecc"](
                    np_mcmc_samples[i, indc].astype("float32")
                )
            realization_phis_c["chargecc"].append(phis_c.detach())
            # realization_dV_dQ["chargecc"].append(dV_dQ)
            # realization_dQ_dV["chargecc"].append(dQ_dV)
        realization_phis_c["chargecc"] = np.array(
            realization_phis_c["chargecc"]
        )
        # realization_dV_dQ["chargecc"] = np.array(realization_dV_dQ["chargecc"])
        # realization_dQ_dV["chargecc"] = np.array(realization_dQ_dV["chargecc"])

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
    sim_params_dict=None,
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
                sim_params_dict["chargecc"]["deg_cs0_a_min"],
                sim_params_dict["chargecc"]["deg_cs0_a_max"],
            )
        )
        ranges.append(
            (
                sim_params_dict["chargecc"]["deg_cs0_c_min"],
                sim_params_dict["chargecc"]["deg_cs0_c_max"],
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
                indc = list(range(sim_params_dict["chargecc"]["n_params"]))
                indc[sim_params_dict["chargecc"]["ind_deg_cs0_a"]] = nn_dict[
                    "chargecc"
                ].params["n_params"]
                indc[sim_params_dict["chargecc"]["ind_deg_cs0_c"]] = (
                    sim_params_dict["chargecc"]["n_params"] + 1
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
