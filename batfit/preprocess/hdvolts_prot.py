from bmlite import Experiment


def define_diffcap_experiment(sim_params):
    diffcap = Experiment()
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


def define_pre_hppc_experiment(sim_params):
    pre_hppc = Experiment()
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


def define_hppc_experiment(sim_params):
    hppc = define_pre_hppc_experiment(sim_params)
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
        (100 * 60.0, 100),
        limits=(
            "current_A",
            0.23,
        ),
    )
    for _ in range(7):
        hppc.add_step("current_A", 0.0, (60 * 60, 40))
        hppc.add_step(
            "current_A",
            4.559680677,
            (30.0, 100),
            limits=(
                "voltage_V",
                sim_params["vmin"],
            ),
        )
        hppc.add_step("current_A", 0.0, (60 * 0.6664989567, 40))
        hppc.add_step(
            "current_A",
            -3.419829906,
            (60 * 0.1658784912, 100),
        )
        #### Need to Handle CV
        hppc.add_step(
            "current_A",
            0.9117080293,
            (60 * 28.06298363, 200),
            limits=(
                "voltage_V",
                sim_params["vmin"],
            ),
        )

    return hppc
