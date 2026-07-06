import os
import tempfile

import numpy as np

from batfit.utils.raw_sol_utils import (
    from_name_to_params,
    get_max_time,
    get_sol_list,
)


def test_get_sol_list():
    with tempfile.TemporaryDirectory() as tmp_dir:
        for name in [
            "solution_1.0_2.0.npz",
            "solution_3.0.npz",
            "not_a_solution.npz",
            "solution_4.0.txt",
        ]:
            open(os.path.join(tmp_dir, name), "w").close()
        list_files = get_sol_list(tmp_dir)

    assert sorted(list_files) == [
        "solution_1.0_2.0.npz",
        "solution_3.0.npz",
    ]


def test_from_name_to_params():
    filename = "solution_2.002_6.95014_1.04652.npz"
    params = from_name_to_params(filename)
    assert params == [2.002, 6.95014, 1.04652]


def test_get_max_time():
    with tempfile.TemporaryDirectory() as tmp_dir:
        np.savez(
            os.path.join(tmp_dir, "solution_1.0.npz"), t=np.array([0, 1, 2])
        )
        np.savez(
            os.path.join(tmp_dir, "solution_2.0.npz"), t=np.array([0, 1, 5])
        )
        max_t = get_max_time(tmp_dir)

    # smallest of the per-file maxima: min(2, 5) == 2
    assert max_t == 2
