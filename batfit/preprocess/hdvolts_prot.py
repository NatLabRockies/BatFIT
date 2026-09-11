from bmlite import Experiment


def define_diffcap_experiment(sim_params, expr=None, atol=1e-9, max_step=1000):
    if expr is None:
        expr = Experiment(atol=atol, max_step=max_step, rtol=1e-6)
    expr.add_step(
        "current_A",
        -0.2277978276,
        (2000 * 60.0, 120.0),
        limits=(
            "voltage_V",
            sim_params["vmax"],
        ),
    )
    expr.add_step("current_A", 0.0, (3600.0, 90.0))
    expr.add_step(
        "current_A",
        0.2277980605,
        (2000 * 60.0, 120.0),
        limits=(
            "voltage_V",
            sim_params["vmin"],
        ),
    )
    expr.add_step("current_A", 0.0, (3600.0, 90.0))

    return expr


def define_pre_hppc_experiment(
    sim_params, expr=None, atol=1e-9, max_step=1000
):
    if expr is None:
        expr = Experiment(atol=atol, max_step=max_step, rtol=1e-6)
    expr.add_step(
        "current_A",
        -0.9117096352,
        (400 * 60.0, 60.0),
        limits=(
            "voltage_V",
            sim_params["vmax"],
        ),
    )
    expr.add_step(
        "voltage_V",
        sim_params["vmax"],
        (100 * 60, 50.0),
        limits=(
            "current_A",
            -0.23,
        ),
    )
    expr.add_step("current_A", 0.0, (60 * 60, 90.0))
    expr.add_step(
        "current_A",
        0.9117092553,
        (400 * 60.0, 60.0),
        limits=(
            "voltage_V",
            sim_params["vmin"],
        ),
    )
    expr.add_step("current_A", 0.0, (60 * 60, 90.0))
    return expr


def define_post_hppc_experiment(
    sim_params, expr=None, atol=1e-9, max_step=1000
):
    if expr is None:
        # max_num_steps is the IDA internal step-count budget between
        # outputs (default 500); stiff rests need more headroom
        expr = Experiment(
            atol=atol, max_step=max_step, rtol=1e-6,
            max_num_steps=int(1e4),
        )
    expr.add_step(
        "current_A",
        -0.9117085305,
        (400 * 60.0, 60.0),
        limits=(
            "voltage_V",
            sim_params["vmax"],
        ),
    )
    expr.add_step(
        "voltage_V",
        sim_params["vmax"],
        (100 * 60.0, 50.0),
        limits=(
            "current_A",
            -0.23,
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
        1: 3.418400583,
        2: 3.418400583,
        3: 3.418400583,
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
        expr.add_step("current_A", 0.0, (60.0 * 60, 90.0))
        expr.add_step(
            "current_A",
            step18_current[pulse + 1],
            (30.0, 0.3),
            limits=(
                "voltage_V",
                sim_params["vmin"],
            ),
        )
        expr.add_step("current_A", 0.0, (40, 0.4), reset_capacity=False)
        expr.add_step(
            "current_A", -step21_current[pulse + 1],
            (10.0, 0.1),
            reset_capacity=False,
            limits=(
                "voltage_V",
                sim_params["vmax"],
            ),
        )
        expr.add_step(
            "voltage_V", sim_params["vmax"],
            (10.0, 0.1),
            reset_capacity=False,
            reset_timer=False,
            limits=(
                "phase_time_s",
                10.0
            ),
        )
        expr.add_step(
            "current_A",
            0.9117080293,
            (60 * 60.0, 8.0),
            limits=(
                "capacity_Ah",
                0.456,
                "voltage_V",
                sim_params["vmin"],
            ),
            reset_capacity=False,
        )

    return expr


def define_hppc_experiment(sim_params, expr=None, atol=1e-9, max_step=1000):
    if expr is None:
        expr = Experiment(
            atol=atol, max_step=max_step, rtol=1e-6,
            max_num_steps=int(1e4),
        )
    expr = define_pre_hppc_experiment(
        sim_params, expr=expr, atol=atol, max_step=max_step
    )
    expr = define_post_hppc_experiment(
        sim_params, expr=expr, atol=atol, max_step=max_step
    )

    return expr
