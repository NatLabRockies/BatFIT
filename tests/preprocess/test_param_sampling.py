import os
import tempfile

import numpy as np

from batfit import BATFIT_EXP
from batfit.preprocess.param_sampling import (
    enforce_stoich_a,
    enforce_stoich_c,
    get_samples,
    hypercube_combinations,
    round_samples,
    write_exec,
)
from batfit.preprocess.sim_setup import make_params


def test_round_samples():
    result = round_samples([[1.123456789, 2.987654321]])
    assert result[0][0] == round(1.123456789, 5)
    assert result[0][1] == round(2.987654321, 5)


def test_hypercube_combinations():
    # n params → 2^n corner combinations
    result = list(hypercube_combinations([[0, 1], [0, 1]]))
    assert len(result) == 4
    assert [0, 0] in result
    assert [0, 1] in result
    assert [1, 0] in result
    assert [1, 1] in result

    # single param
    result_1d = list(hypercube_combinations([[0, 1]]))
    assert result_1d == [[0], [1]]


def test_enforce_stoich():
    # Make sure we remove the samples that go out of bounds for intercalation frac

    sim_params = {"x0_a": 0.9}
    deg_param_names = ["x0_a"]
    # 0.9 * 0.8 = 0.72 → valid; 0.9 * 1.2 = 1.08 → invalid
    samples = np.array([[0.8], [1.2]])
    result = enforce_stoich_a(
        samples, deg_param_names, [0.5], [1.5], sim_params
    )
    assert result.shape == (1, 1)
    assert np.isclose(result[0, 0], 0.8)

    sim_params = {"x0_c": 0.5}
    deg_param_names = ["x0_c"]
    samples = np.array([[1.5], [2.5], [-0.5]])
    result = enforce_stoich_c(
        samples, deg_param_names, [0.5], [2.0], sim_params
    )
    assert result.shape == (1, 1)
    assert np.isclose(result[0, 0], 1.5)


def test_get_samples():
    sim_params = make_params(os.path.join(BATFIT_EXP, "spm_discharge.yaml"))
    n_int = 20
    deg_samples, prot_samples = get_samples(n_int=n_int, sim_params=sim_params)
    n_deg = len(sim_params["deg_param_names"])
    # samples may be fewer than n_int after enforce filtering
    assert deg_samples.shape[0] <= n_int
    assert deg_samples.shape[1] == n_deg
    assert prot_samples.shape[1] == 0

    # all samples must be within declared bounds
    for i, name in enumerate(sim_params["deg_param_names"]):
        assert np.all(deg_samples[:, i] >= sim_params[f"deg_{name}_min"])
        assert np.all(deg_samples[:, i] <= sim_params[f"deg_{name}_max"])


def test_write_exec():
    sim_params = make_params(os.path.join(BATFIT_EXP, "spm_discharge.yaml"))
    deg_param_names = sim_params["deg_param_names"][:2]
    deg_samples = np.array([[0.8, 1.2], [1.1, 0.9]])

    with tempfile.TemporaryDirectory() as tmp_dir:
        write_exec(
            deg_samples,
            deg_param_names=deg_param_names,
            folder_save=tmp_dir,
            sim_params=sim_params,
        )
        param_file = os.path.join(tmp_dir, "parameter_list.txt")
        assert os.path.exists(param_file)
        with open(param_file) as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        assert len(lines) == 2  # number of samples
        # values for the two sampled params should appear in the first line
        vals = [float(v) for v in lines[0].split()]
        assert np.isclose(vals[deg_param_names.index(deg_param_names[0])], 0.8)
        assert np.isclose(vals[deg_param_names.index(deg_param_names[1])], 1.2)
