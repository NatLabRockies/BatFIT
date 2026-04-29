import numpy as np
from bmlite import Experiment


def define_chirp_experiment(sim_params, chirp_params, expr=None, atol=1e-9, max_step=1000):
    if expr is None:
        expr = Experiment(atol=atol, max_step=max_step)

    chirp_beg_time = chirp_params["time_start"]
    chirp_amp = chirp_params["amplitude"]
    chirp_length = chirp_params["length"]

    assert chirp_amp < 1.0

    expr.add_step(
        "current_C",
        -1.0,
        (min(3600.0, chirp_beg_time), 30.0),
        limits=(
            "voltage_V",
            sim_params["vmax"],
        ),
    )
    expr.add_step(
        "current_C",
        -1.0*(1.0+chirp_amp),
        (chirp_length, chirp_length/25.0),
        limits=(
            "voltage_V",
            sim_params["vmax"],
        ),
    )
    expr.add_step(
        "current_C",
        -1.0*(1.0-chirp_amp),
        (chirp_length, chirp_length/25.0),
        limits=(
            "voltage_V",
            sim_params["vmax"],
        ),
    )
    expr.add_step(
        "current_C",
        -1.0,
        (3600.0, 30.0),
        limits=(
            "voltage_V",
            sim_params["vmax"],
        ),
    )

    return expr

