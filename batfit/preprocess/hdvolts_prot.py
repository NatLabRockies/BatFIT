import numpy as np
from bmlite import Experiment


def define_diffcap_experiment(sim_params, atol=1e-9, max_step=1000):
    diffcap = Experiment(atol=atol, max_step=max_step)
    diffcap.add_step(
        "current_A",
        -0.2277978276,
        (2000 * 60.0, 1000),
        limits=(
            "voltage_V",
            sim_params["vmax"],
        ),
    )
    diffcap.add_step("current_A", 0.0, (3600.0, 40))
    diffcap.add_step(
        "current_A",
        0.2277980605,
        (2000 * 60.0, 1000),
        limits=(
            "voltage_V",
            sim_params["vmin"],
        ),
    )
    diffcap.add_step("current_A", 0.0, (3600.0, 40))

    return diffcap


def define_pre_hppc_experiment(sim_params, atol=1e-9, max_step=1000):
    pre_hppc = Experiment(atol=atol, max_step=max_step)
    pre_hppc.add_step(
        "current_A",
        -0.9117096352,
        (400 * 60.0, 200),
        limits=(
            "voltage_V",
            sim_params["vmax"],
        ),
    )
    pre_hppc.add_step(
        "voltage_V",
        sim_params["vmax"],
        (100 * 60, 40),
        limits=(
            "current_A",
            0.23,
        ),
    )
    pre_hppc.add_step("current_A", 0.0, (60 * 60, 40))
    pre_hppc.add_step(
        "current_A",
        0.9117092553,
        (400 * 60.0, 200),
        limits=(
            "voltage_V",
            sim_params["vmin"],
        ),
    )
    pre_hppc.add_step("current_A", 0.0, (60 * 60, 40))

    return pre_hppc


def define_hppc_experiment(sim_params, atol=1e-9, max_step=1000):
    hppc = define_pre_hppc_experiment(sim_params, atol=atol, max_step=max_step)
    hppc.add_step(
        "current_A",
        -0.9117085305,
        (400 * 60.0, 200),
        limits=(
            "voltage_V",
            sim_params["vmax"],
        ),
    )
    hppc.add_step(
        "voltage_V",
        sim_params["vmax"],
        (100 * 60.0, 200),
        limits=(
            "current_A",
            0.23,
        ),
    )
    step18_current = {
        1: 4.559677946,
        2: 4.559687375,
        3: 4.559681632,
        4: 4.559679232,
        5: 4.559676403,
        6: 4.559678374,
        7: 4.559683775,
    }
    step21_current = {
        1: 1.400344099,
        2: 2.829861821,
        3: 3.328936251,
        4: 3.418400583,
        5: 3.419835874,
        6: 3.419834331,
        7: 3.419830473,
    }
    step21_currentstd = {
        1: 0.35531,
        2: 0.66846,
        3: 0.23621,
        4: 0.014597,
        5: 0.00041942,
        6: 0.00042797,
        7: 0.00043285,
    }
    step23_time = {
        1: 27.76903539,
        2: 28.02600449,
        3: 28.11602416,
        4: 28.13209663,
        5: 28.13250393,
        6: 28.1325882,
        7: 28.13263258,
    }
    step23_timestd = {
        1: 0.064009,
        2: 0.1203,
        3: 0.042056,
        4: 0.0023804,
        5: 4.0171e-05,
        6: 8.8851e-05,
        7: 7.6841e-05,
    }
    for pulse in range(7):
        hppc.add_step("current_A", 0.0, (60.0 * 60, 40))
        hppc.add_step(
            "current_A",
            step18_current[pulse + 1],
            (30.0, 100),
            limits=(
                "voltage_V",
                sim_params["vmin"],
            ),
        )
        hppc.add_step("current_A", 0.0, (40, 100))
        hppc.add_step(
            "current_A",
            -np.random.normal(
                step21_current[pulse + 1], step21_currentstd[pulse + 1]
            ),
            (10, 100),
        )
        hppc.add_step(
            "current_A",
            0.9117080293,
            (
                60
                * np.random.normal(
                    step23_time[pulse + 1], step23_timestd[pulse + 1]
                ),
                200,
            ),
            limits=(
                "voltage_V",
                sim_params["vmin"],
            ),
        )

    return hppc
