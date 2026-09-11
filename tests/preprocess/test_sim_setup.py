import numpy as np

from batfit.preprocess.sim_setup import set_interc


class _FakeElectrode:
    """Minimal stand-in for a BatMODS-lite electrode object."""

    def __init__(self):
        self.x_0 = None


class _FakeSim:
    """Minimal stand-in for a BatMODS-lite simulation object."""

    def __init__(self):
        self.ca = _FakeElectrode()
        self.an = _FakeElectrode()


def test_set_interc():
    # Connected modes (discharge, chargecc, chirp): C rate comes from "C"
    for cyc_mode in ["discharge", "chargecc", "chirp"]:
        sim = _FakeSim()
        sim_params = {"x0_c": 0.9, "x0_a": 0.8, "C": -1.0}
        deg_param_sample = {"x0_c": 0.5, "x0_a": 1.1}
        C_rate, sim = set_interc(
            sim=sim,
            sim_params=sim_params,
            deg_param_sample=deg_param_sample,
            cyc_mode=cyc_mode,
            run_mode=None,
        )
        assert isinstance(C_rate, float)
        assert np.isclose(C_rate, -1.0)
        assert np.isclose(sim.ca.x_0, 0.9 * 0.5)
        assert np.isclose(sim.an.x_0, 0.8 * 1.1)

    # Connected mode without a CC/chirp cycle: no C rate
    sim = _FakeSim()
    sim_params = {"x0_c": 0.9, "x0_a": 0.8}
    C_rate, sim = set_interc(
        sim=sim,
        sim_params=sim_params,
        deg_param_sample={},
        cyc_mode="hppc",
        run_mode=None,
    )
    assert C_rate is None
    assert np.isclose(sim.ca.x_0, 0.9)
    assert np.isclose(sim.an.x_0, 0.8)

    # Disconnected discharge-chargecc: C rate depends on run_mode
    sim_params = {
        "x0_c_dis": 0.9,
        "x0_a_dis": 0.1,
        "x0_c_chcc": 0.4,
        "x0_a_chcc": 0.7,
        "C_dis": 1.0,
        "C_chcc": -0.25,
    }
    deg_param_sample = {
        "x0_c": 0.5,
        "x0_a": 1.1,
        "x0_c_chcc": 0.8,
        "x0_a_chcc": 1.2,
    }

    sim = _FakeSim()
    C_rate, sim = set_interc(
        sim=sim,
        sim_params=sim_params,
        deg_param_sample=deg_param_sample,
        cyc_mode="discharge-chargecc",
        run_mode="discharge",
    )
    assert isinstance(C_rate, float)
    assert np.isclose(C_rate, 1.0)
    assert np.isclose(sim.ca.x_0, 0.9 * 0.5)
    assert np.isclose(sim.an.x_0, 0.1 * 1.1)

    sim = _FakeSim()
    C_rate, sim = set_interc(
        sim=sim,
        sim_params=sim_params,
        deg_param_sample=deg_param_sample,
        cyc_mode="discharge-chargecc",
        run_mode="chargecc",
    )
    assert isinstance(C_rate, float)
    assert np.isclose(C_rate, -0.25)
    assert np.isclose(sim.ca.x_0, 0.4 * 0.8)
    assert np.isclose(sim.an.x_0, 0.7 * 1.2)
