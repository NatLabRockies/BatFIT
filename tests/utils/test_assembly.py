import os
import pickle
import tempfile

import numpy as np
import pytest

from batfit.utils.assembly import (
    assemble_all_data,
    assemble_surrogate_data,
    augment_data,
    check_assembled_data_shape,
    check_assembled_surrogate_data_shape,
    from_combined_sols_to_data,
    from_param_to_surrogate_data,
    from_sol_dict_to_xy,
    passes_quality_filters,
)


def _make_sol(n_t: int = 20, t_max: float = 10.0) -> dict:
    t = np.linspace(0, t_max, n_t)
    phis_c = np.sin(t)
    return {"t": t, "phis_c": phis_c}


def test_passes_quality_filters():
    combined_sols = {
        "good": {"sol": _make_sol(n_t=20, t_max=10.0)},
        "too_short": {"sol": _make_sol(n_t=3, t_max=10.0)},
        "too_early": {"sol": _make_sol(n_t=20, t_max=1.0)},
        "combo_good": {
            "sol_dis": _make_sol(n_t=20, t_max=10.0),
            "sol_chcc": _make_sol(n_t=20, t_max=10.0),
        },
        "combo_one_bad": {
            "sol_dis": _make_sol(n_t=20, t_max=10.0),
            "sol_chcc": _make_sol(n_t=3, t_max=10.0),
        },
    }

    assert passes_quality_filters(
        combined_sols, "good", "discharge", n_points_min=5, t_max_min=5.0
    )
    assert not passes_quality_filters(
        combined_sols, "too_short", "discharge", n_points_min=5, t_max_min=5.0
    )
    assert not passes_quality_filters(
        combined_sols, "too_early", "discharge", n_points_min=5, t_max_min=5.0
    )
    assert passes_quality_filters(
        combined_sols,
        "combo_good",
        "discharge-chargecc",
        n_points_min=5,
        t_max_min=5.0,
    )
    assert not passes_quality_filters(
        combined_sols,
        "combo_one_bad",
        "discharge-chargecc",
        n_points_min=5,
        t_max_min=5.0,
    )
    with pytest.raises(NotImplementedError):
        passes_quality_filters(combined_sols, "good", "unknown_mode")


def test_from_sol_dict_to_xy():
    n_points = 15
    sol_dict = _make_sol(n_t=20, t_max=10.0)
    combined_sols = {"key1": {"params": [1.0, 2.0, 3.0]}}

    x, y = from_sol_dict_to_xy(
        sol_dict, combined_sols, "key1", n_points, target_mode="phi"
    )
    assert x.shape == (2, n_points)
    assert np.allclose(x[0], np.linspace(0, 10.0, n_points))
    assert y == [1.0, 2.0, 3.0]

    # diff_cap branch, all three channels selected
    sol_dict_diff = {
        "t_diff": np.linspace(0, 5.0, 10),
        "phis_c_diff": np.linspace(3.0, 4.0, 10),
        "dvdq": np.linspace(0.1, 0.2, 10),
        "dqdv": np.linspace(1.0, 2.0, 10),
    }
    x_diff, y_diff = from_sol_dict_to_xy(
        sol_dict_diff,
        combined_sols,
        "key1",
        n_points,
        target_mode="phi_dvdq_dqdv",
        diff_cap=True,
    )
    assert x_diff.shape == (4, n_points)


def test_from_combined_sols_to_data():
    n_points = 12
    combined_sols = {
        "key1": {
            "sol": _make_sol(n_t=20, t_max=10.0),
            "params": [1.0, 2.0],
        },
        "key2": {
            "sol_dis": _make_sol(n_t=20, t_max=10.0),
            "sol_chcc": _make_sol(n_t=20, t_max=10.0),
            "params": [3.0, 4.0],
        },
    }

    x, y = from_combined_sols_to_data(
        combined_sols, "key1", n_points, "phi", "discharge"
    )
    assert x.shape == (2, n_points)
    assert y == [1.0, 2.0]

    x2, y2 = from_combined_sols_to_data(
        combined_sols, "key2", n_points, "phi", "discharge-chargecc"
    )
    assert x2.shape == (4, n_points)
    assert y2 == [3.0, 4.0]

    with pytest.raises(NotImplementedError):
        from_combined_sols_to_data(
            combined_sols, "key1", n_points, "phi", "unknown_mode"
        )


def test_check_assembled_data_shape():
    n_points, N, n_params = 10, 5, 3
    X_data = np.random.randn(N, 2, n_points).astype("float32")
    Y_data = np.random.randn(N, n_params).astype("float32")

    with tempfile.TemporaryDirectory() as tmp_dir:
        np.savez(
            os.path.join(tmp_dir, "assembled_data.npz"),
            X_data=X_data,
            Y_data=Y_data,
        )
        tmp = check_assembled_data_shape(
            data_root_folder=tmp_dir, n_points=n_points, save_path=tmp_dir
        )
        assert tmp["X_data"].shape == X_data.shape

        with pytest.raises(NotImplementedError):
            check_assembled_data_shape(
                data_root_folder=tmp_dir,
                n_points=n_points,
                target_mode="encoded",
                save_path=tmp_dir,
            )


def test_assemble_all_data():
    n_points, n_params = 8, 2
    combined_sols = {
        f"solution_{i}.npz": {
            "sol": _make_sol(n_t=20, t_max=10.0),
            "params": [float(i), float(i) + 0.5],
            "prot_params": [0.1 * i, 0.2 * i],
        }
        for i in range(4)
    }
    # one solution that should be rejected by the quality filter
    combined_sols["solution_bad.npz"] = {
        "sol": _make_sol(n_t=2, t_max=10.0),
        "params": [9.0, 9.0],
        "prot_params": [0.9, 0.9],
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        with open(os.path.join(tmp_dir, "sols.pkl"), "wb") as f:
            pickle.dump(combined_sols, f)

        X_data, Y_data = assemble_all_data(
            tmp_dir,
            n_points=n_points,
            n_points_min=5,
            combined_pickle_file="sols.pkl",
            target_mode="phi",
            save_data=True,
            cyc_mode="discharge",
            save_path=tmp_dir,
        )
        # the "bad" solution (2 points) is filtered out, leaving 4
        assert X_data.shape == (4, 2, n_points)
        assert Y_data.shape == (4, n_params)
        assert os.path.isfile(os.path.join(tmp_dir, "assembled_data.npz"))

        # cache-hit: second call loads from assembled_data.npz
        X_data2, Y_data2 = assemble_all_data(
            tmp_dir, n_points=n_points, save_path=tmp_dir
        )
        assert np.allclose(X_data, X_data2)

    # return_prot_params=True
    with tempfile.TemporaryDirectory() as tmp_dir:
        with open(os.path.join(tmp_dir, "sols.pkl"), "wb") as f:
            pickle.dump(combined_sols, f)

        X_data, P_data, Y_data = assemble_all_data(
            tmp_dir,
            n_points=n_points,
            n_points_min=5,
            combined_pickle_file="sols.pkl",
            target_mode="phi",
            save_data=True,
            cyc_mode="discharge",
            save_path=tmp_dir,
            return_prot_params=True,
        )
        assert P_data.shape == (4, 2)

    # requires a combined pickle file when there's no cache
    with tempfile.TemporaryDirectory() as tmp_dir:
        with pytest.raises(AssertionError):
            assemble_all_data(tmp_dir, n_points=n_points, save_path=tmp_dir)

    # n_sol_max caps the number of assembled (post-filter) entries, even
    # though 4 solutions pass the filter
    with tempfile.TemporaryDirectory() as tmp_dir:
        with open(os.path.join(tmp_dir, "sols.pkl"), "wb") as f:
            pickle.dump(combined_sols, f)

        X_data, Y_data = assemble_all_data(
            tmp_dir,
            n_points=n_points,
            n_points_min=5,
            combined_pickle_file="sols.pkl",
            target_mode="phi",
            save_data=True,
            cyc_mode="discharge",
            save_path=tmp_dir,
            n_sol_max=2,
        )
        assert X_data.shape == (2, 2, n_points)
        assert Y_data.shape == (2, n_params)


def test_check_assembled_surrogate_data_shape():
    n_points, n_param_pred, N = 10, 3, 50
    X_data = np.random.randn(N, n_param_pred + 1).astype("float32")
    Y_data = np.random.randn(N, 1).astype("float32")

    with tempfile.TemporaryDirectory() as tmp_dir:
        np.savez(
            os.path.join(tmp_dir, "assembled_surrogate_data.npz"),
            X_data=X_data,
            Y_data=Y_data,
        )
        tmp = check_assembled_surrogate_data_shape(
            data_root_folder=tmp_dir,
            n_points=n_points,
            n_param_pred=n_param_pred,
            save_path=tmp_dir,
        )
        assert tmp["X_data"].shape == X_data.shape


def test_assemble_surrogate_data():
    n_points, n_param_pred = 5, 2
    combined_sols = {
        f"solution_{i}.npz": {
            "sol": _make_sol(n_t=20, t_max=10.0),
            "params": [float(i), float(i) + 0.5],
        }
        for i in range(3)
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        with open(os.path.join(tmp_dir, "sols.pkl"), "wb") as f:
            pickle.dump(combined_sols, f)

        X_data, Y_data = assemble_surrogate_data(
            tmp_dir,
            n_points=n_points,
            n_param_pred=n_param_pred,
            combined_pickle_file="sols.pkl",
            cyc_mode="discharge",
            save_data=True,
            save_path=tmp_dir,
        )
        assert X_data.shape == (3 * n_points, n_param_pred + 1)
        assert Y_data.shape == (3 * n_points, 1)
        assert os.path.isfile(
            os.path.join(tmp_dir, "assembled_surrogate_data.npz")
        )


def test_from_param_to_surrogate_data():
    N, T, n_params = 5, 20, 3
    t = np.linspace(0, 1, T).astype("float32")
    X_data = np.stack(
        [
            np.tile(t, (N, 1)),  # channel 0: time
            np.random.randn(N, T).astype("float32"),  # channel 1: voltage
        ],
        axis=1,
    )  # (N, 2, T)
    Y_data = np.random.randn(N, n_params).astype("float32")
    new_x, new_y = from_param_to_surrogate_data(X_data, Y_data)
    assert new_x.shape == (N * T, n_params + 1)
    assert new_y.shape == (N * T, 1)
    # first column of new_x must be the time values from X_data channel 0
    assert np.allclose(new_x[:, 0], X_data[:, 0, :].reshape(-1))
    # new_y must be the voltage channel from X_data
    assert np.allclose(new_y[:, 0], X_data[:, 1, :].reshape(-1))


def test_augment_data():
    N, n_chan, T, n_params = 10, 2, 50, 3
    X = np.random.randn(N, n_chan, T).astype("float32")
    Y = np.random.randn(N, n_params).astype("float32")
    new_ds = 3
    X_aug, Y_aug = augment_data(X, Y, new_ds=new_ds, noise_level=0.001)
    assert X_aug.shape == ((new_ds + 1) * N, n_chan, T)
    assert Y_aug.shape == ((new_ds + 1) * N, n_params)
    # original samples must be preserved exactly in first N rows
    assert np.allclose(X_aug[:N], X)
    assert np.allclose(Y_aug[:N], Y)
