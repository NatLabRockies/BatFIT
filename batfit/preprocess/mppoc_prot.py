import numpy as np
from bmlite import Experiment

def define_chirp_experiment(
    sim_params, chirp_params, background_C_rate, expr=None, atol=1e-12, max_step=1000000
):
    if expr is None:
        expr = Experiment(atol=atol, max_step=max_step, rtol=1e-8)

    chirp_beg_time = chirp_params["time_start"]
    chirp_amp = chirp_params["amplitude"]
    chirp_length = chirp_params["length"]

    #assert chirp_amp < 1.0
    # Only for charge
    assert background_C_rate < 0.0

    # Run until chirp
    t_fin = min(3600.0/abs(background_C_rate), chirp_beg_time)
    t_step = t_fin/100.0
    expr.add_step(
        "current_C",
        background_C_rate,
        (t_fin, t_step),
        limits=(
            "voltage_V",
            sim_params["vmax"],
        ),
    )
    
    # Do chirp high
    t_fin = chirp_length/2
    t_step = t_fin/100.0
    expr.add_step(
        "current_C",
        min(background_C_rate * (1.0 + chirp_amp), 0.0),
        (t_fin, t_step),
        limits=(
            "voltage_V",
            sim_params["vmax"],
        ),
    )

    
    # Do chirp low
    t_fin = chirp_length/2
    t_step = t_fin/100.0
    expr.add_step(
        "current_C",
        min(background_C_rate * (1.0 - chirp_amp), 0.0),
        (t_fin, t_step),
        limits=(
            "voltage_V",
            sim_params["vmax"],
        ),
    )

    # Finish charge
    t_fin = 3600.0/abs(background_C_rate)
    t_step = t_fin/100.0
    expr.add_step(
        "current_C",
        background_C_rate,
        (t_fin, t_step),
        limits=(
            "voltage_V",
            sim_params["vmax"],
        ),
    )

    return expr

def define_ramp_chirp_experiment(
    sim_params, chirp_params, background_C_rate, expr=None, atol=1e-12, max_step=1000000
):
    if expr is None:
        expr = Experiment(atol=atol, max_step=max_step, rtol=1e-8)

    chirp_beg_time = chirp_params["time_start"]
    chirp_amp = chirp_params["amplitude"]
    chirp_length = chirp_params["length"]

    #assert chirp_amp < 1.0
    # Only for charge
    assert background_C_rate < 0.0

    # Run until chirp
    t_fin = min(3600.0/abs(background_C_rate), chirp_beg_time)
    t_step = t_fin/100.0
    expr.add_step(
        "current_C",
        background_C_rate,
        (t_fin, t_step),
        limits=(
            "voltage_V",
            sim_params["vmax"],
        ),
    )
    
    # Do chirp rampup
    ramp_length1 = chirp_length/4
    c_start1 =  background_C_rate
    c_end1 =  min(background_C_rate * (1.0 + chirp_amp), 0.0)
    ramp = lambda t:  c_start1 + (c_end1-c_start1) * t / ramp_length1
    t_fin = ramp_length1
    t_step = t_fin/100.0
    expr.add_step(
        "current_C",
        ramp,
        (t_fin, t_step),
        limits=(
            "current_C",
            c_end1,
        ),
    )

    # Do chirp high
    t_fin = chirp_length/4
    t_step = t_fin/100.0
    expr.add_step(
        "current_C",
        c_end1,
        (t_fin, t_step),
        limits=(
            "voltage_V",
            sim_params["vmax"],
        ),
    )
    
    # Do chirp rampdown
    ramp_length2 = chirp_length/2
    c_start2 =   min(background_C_rate * (1.0 + chirp_amp), 0.0)
    c_end2 =  min(background_C_rate * (1.0 - chirp_amp), 0.0)
    ramp2 = lambda t:  c_start2 + (c_end2-c_start2) * t/ramp_length2
    t_fin = ramp_length2
    t_step = t_fin/100.0
    expr.add_step(
        "current_C",
        ramp2,
        (t_fin, t_step),
        limits=(
            "current_C",
            c_end2,
        ),
    )

    # Do chirp low
    t_fin = chirp_length/4
    t_step = t_fin/100.0
    expr.add_step(
        "current_C",
        c_end2,
        (t_fin, t_step),
        #limits=(
        #    "voltage_V",
        #    sim_params["vmax"],
        #),
    )

    # Do chirp rampup
    ramp_length3 = chirp_length/4
    c_start3 =  min(background_C_rate * (1.0 - chirp_amp), 0.0)
    c_end3 = background_C_rate
    ramp3 = lambda t:  c_start3 + (c_end3-c_start3) * t/ramp_length3
    t_fin = ramp_length3
    t_step = t_fin/100.0
    expr.add_step(
        "current_C",
        ramp3,
        (t_fin, t_step),
        limits=(
            "current_C",
            c_end3,
        ),
    )

    # Finish charge
    t_fin = 3600.0/abs(background_C_rate)
    t_step = t_fin/100.0
    expr.add_step(
        "current_C",
        background_C_rate,
        (t_fin, t_step),
        limits=(
            "voltage_V",
            sim_params["vmax"],
        ),
    )

    return expr
