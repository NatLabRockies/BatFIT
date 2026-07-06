"""Assemble raw BatMODS-lite simulation output (a combined ``sols.pkl``) into
the ``(X, [P], Y)`` numpy arrays consumed by surrogate/NPE training.

Filtering is kept explicitly separate from assembling: :func:`passes_quality_filters`
implements a rejection-sampling prior that decides whether a raw solution is
usable at all (e.g. too few points, or a simulation that ended too early),
independent of :func:`from_combined_sols_to_data`/:func:`from_sol_dict_to_xy`,
which only ever convert an already-accepted solution into ``(x, y)`` arrays.
"""

import gc
import os
import pickle

import numpy as np
from prettyPlot.progressBar import print_progress_bar

from batfit import logger

_SINGLE_SOL_CYC_MODES = [
    "discharge",
    "chargecc",
    "rh",
    "lh",
    "diffcap",
    "hppc",
    "prehppc",
    "posthppc",
    "chirp",
]


def _sol_passes_filters(
    sol_dict: dict,
    n_points_min: int,
    t_max_min: float,
    max_start_phi: float,
) -> bool:
    """Reject a single raw solution dict if it's too short, ends too early,
    or starts at too high a voltage."""
    if sol_dict["phis_c"].shape[0] < n_points_min:
        logger.warning(
            f"Found and removed solution with {sol_dict['phis_c'].shape[0]} points"
        )
        return False
    if sol_dict["t"].max() < t_max_min:
        logger.warning(
            f"Found and removed solution with max t {sol_dict['t'].max()}"
        )
        return False
    if sol_dict["phis_c"][0] > max_start_phi:
        logger.warning(
            f"Found and removed solution with start phi {sol_dict['phis_c'][0]}"
        )
        return False
    return True


def passes_quality_filters(
    combined_sols: dict,
    key: str,
    cyc_mode: str,
    n_points_min: int = 0,
    t_max_min: float = 0,
    max_start_phi: float = 1e10,
) -> bool:
    """Rejection-sampling prior: decide whether a raw solution is usable.

    Rejects simulations with fewer than ``n_points_min`` timesteps, whose max
    time is below ``t_max_min``, or whose first recorded voltage exceeds
    ``max_start_phi`` — an implicit prior that excludes buggy or
    out-of-range physics solutions from training, independent of how an
    accepted solution is later converted into ``(x, y)`` arrays.

    :param combined_sols: the loaded ``sols.pkl`` dict
    :param key: the simulation key (raw solution filename) to check
    :param cyc_mode: cycling mode, selects which solution(s) under ``key``
        to check (``discharge-chargecc`` has two: ``sol_dis``/``sol_chcc``)
    """
    if cyc_mode.lower() in _SINGLE_SOL_CYC_MODES:
        return _sol_passes_filters(
            combined_sols[key]["sol"], n_points_min, t_max_min, max_start_phi
        )
    elif cyc_mode.lower() == "discharge-chargecc":
        return _sol_passes_filters(
            combined_sols[key]["sol_dis"],
            n_points_min,
            t_max_min,
            max_start_phi,
        ) and _sol_passes_filters(
            combined_sols[key]["sol_chcc"],
            n_points_min,
            t_max_min,
            max_start_phi,
        )
    else:
        raise NotImplementedError


def from_sol_dict_to_xy(
    sol_dict: dict,
    combined_sols: dict,
    key: str,
    n_points: int,
    target_mode: str,
    diff_cap: bool = False,
) -> tuple[np.ndarray, list]:
    """Interpolate one raw solution onto a fixed time grid and pair it with its params."""
    if diff_cap:
        min_t = np.amin(sol_dict["t_diff"])
        max_t = np.amax(sol_dict["t_diff"])
        t_grid = np.linspace(min_t, max_t, n_points)
        x = np.reshape(t_grid, (1, -1))
        if "phi" in target_mode.lower():
            phi_grid = np.interp(
                t_grid, sol_dict["t_diff"], sol_dict["phis_c_diff"]
            )
            x = np.vstack((x, np.reshape(phi_grid, (1, -1))))
        if "dvdq" in target_mode.lower():
            dvdq_grid = np.interp(t_grid, sol_dict["t_diff"], sol_dict["dvdq"])
            x = np.vstack((x, np.reshape(dvdq_grid, (1, -1))))
        if "dqdv" in target_mode.lower():
            dqdv_grid = np.interp(t_grid, sol_dict["t_diff"], sol_dict["dqdv"])
            x = np.vstack((x, np.reshape(dqdv_grid, (1, -1))))
        x = x.astype("float32")
        y = combined_sols[key]["params"]
    else:
        min_t = np.amin(sol_dict["t"])
        max_t = np.amax(sol_dict["t"])
        t_grid = np.linspace(min_t, max_t, n_points)
        x = np.reshape(t_grid, (1, -1))
        if "phi" in target_mode.lower():
            phi_grid = np.interp(t_grid, sol_dict["t"], sol_dict["phis_c"])
            x = np.vstack((x, np.reshape(phi_grid, (1, -1))))
        x = x.astype("float32")
        y = combined_sols[key]["params"]

    return x, y


def from_combined_sols_to_data(
    combined_sols: dict,
    key: str,
    n_points: int,
    target_mode: str,
    cyc_mode: str,
) -> tuple[np.ndarray, list]:
    """Convert one already-accepted raw solution into ``(x, y)`` arrays.

    Assumes the caller has already applied :func:`passes_quality_filters` to
    ``key``; this function only assembles, it never rejects.
    """
    if cyc_mode.lower() in _SINGLE_SOL_CYC_MODES:
        sol_dict = combined_sols[key]["sol"]
        return from_sol_dict_to_xy(
            sol_dict, combined_sols, key, n_points, target_mode
        )
    elif cyc_mode.lower() == "discharge-chargecc":
        sol_dis_dict = combined_sols[key]["sol_dis"]
        x_dis, y_dis = from_sol_dict_to_xy(
            sol_dis_dict, combined_sols, key, n_points, target_mode
        )
        sol_chcc_dict = combined_sols[key]["sol_chcc"]
        x_chcc, y_chcc = from_sol_dict_to_xy(
            sol_chcc_dict, combined_sols, key, n_points, target_mode
        )
        assert y_dis == y_chcc
        return np.vstack((x_dis, x_chcc)), y_dis
    else:
        raise NotImplementedError


def check_assembled_data_shape(
    data_root_folder,
    n_points,
    combined_pickle_file=None,
    target_mode="phi",
    save_data=True,
    cyc_mode="discharge",
    save_path=".",
):
    """Validate a cached ``assembled_data.npz`` against the requested shape."""
    assembled_data_filename = os.path.join(save_path, "assembled_data.npz")
    tmp = np.load(assembled_data_filename)

    if target_mode != "encoded":
        assert tmp["X_data"].shape[2] == n_points
        assert tmp["X_data"].shape[0] == tmp["Y_data"].shape[0]
    else:
        raise NotImplementedError

    # Don't check this, we might be in a situation where we post processed the assembled data
    # if combined_pickle_file is not None:
    #    with open(combined_pickle_file, "rb") as f:
    #        sols = pickle.load(f)
    #        assert len(sols) == tmp["X_data"].shape[0]

    return tmp


def assemble_all_data(
    data_root_folder: str,
    n_points: int = 100,
    n_points_min: int = 0,
    t_max_min: float = 0,
    max_start_phi: float = 1e10,
    combined_pickle_file: str | None = None,
    target_mode: str = "phi",
    save_data: bool = True,
    cyc_mode: str = "discharge",
    save_path: str = ".",
    return_prot_params: bool = False,
    n_sol_max: int | None = None,
):
    """Assemble raw simulation data (from a combined ``sols.pkl``) into ``(X, Y)`` arrays.

    :param n_points_min: reject solutions with fewer timesteps than this
        (see :func:`passes_quality_filters`).
    :param t_max_min: reject solutions whose max time is below this.
    :param max_start_phi: reject solutions whose first recorded voltage
        exceeds this.
    :param combined_pickle_file: filename of the combined ``sols.pkl`` (relative
        to ``data_root_folder``); required.
    :param return_prot_params: when True, also extract per-simulation protocol
        parameters from the combined sols and return ``(X_data, P_data, Y_data)``.
        ``P_data`` has shape ``(N, n_prot_params)``.
    :param n_sol_max: stop once this many solutions have passed quality
        filtering and been assembled, even if more raw solutions remain in
        ``sols.pkl`` (e.g. cap a 2M-entry ``sols.pkl`` down to 1M assembled
        entries). ``None`` assembles every solution that passes filtering.
    """
    assembled_data_filename = os.path.join(save_path, "assembled_data.npz")
    if os.path.isfile(assembled_data_filename):
        tmp = check_assembled_data_shape(
            data_root_folder,
            n_points,
            combined_pickle_file,
            target_mode,
            save_data,
            cyc_mode,
            save_path,
        )
        if return_prot_params:
            assert (
                "P_data" in tmp
            ), "P_data not found in assembled_data.npz; delete the cache and re-run"
            return tmp["X_data"], tmp["P_data"], tmp["Y_data"]
        return tmp["X_data"], tmp["Y_data"]

    logger.info("Assembling raw dataset")
    assert (
        combined_pickle_file is not None
    ), "assemble_all_data requires a combined sols.pkl file"
    with open(os.path.join(data_root_folder, combined_pickle_file), "rb") as f:
        combined_sols = pickle.load(f)
    list_files = list(combined_sols.keys())
    n_sol_files = len(list_files)

    assert n_sol_files > 1

    print_progress_bar(
        0,
        n_sol_files,
        prefix=f"Create DS File 0 / {n_sol_files} ",
        suffix="Complete",
        length=50,
    )

    X_data = []
    Y_data = []
    P_data = []
    for ifile, file in enumerate(list_files):
        if passes_quality_filters(
            combined_sols,
            file,
            cyc_mode,
            n_points_min,
            t_max_min,
            max_start_phi,
        ):
            x, y = from_combined_sols_to_data(
                combined_sols, file, n_points, target_mode, cyc_mode
            )
            X_data.append(x)
            Y_data.append(y)
            if return_prot_params:
                P_data.append(
                    np.array(
                        combined_sols[file]["prot_params"], dtype="float32"
                    )
                )
        print_progress_bar(
            ifile + 1,
            n_sol_files,
            prefix=f"Create DS File {ifile+1} / {n_sol_files} ",
            suffix="Complete",
            length=50,
        )
        if n_sol_max is not None and len(X_data) >= n_sol_max:
            logger.info(f"Reached n_sol_max={n_sol_max}, stopping early")
            break

    del combined_sols
    gc.collect()

    X_data = np.array(X_data).astype("float32")
    Y_data = np.array(Y_data).astype("float32")

    if save_data:
        save_kwargs: dict = dict(X_data=X_data, Y_data=Y_data)
        if return_prot_params:
            save_kwargs["P_data"] = np.array(P_data).astype("float32")
        np.savez(assembled_data_filename, **save_kwargs)

    if return_prot_params:
        return X_data, np.array(P_data).astype("float32"), Y_data
    return X_data, Y_data


def check_assembled_surrogate_data_shape(
    data_root_folder,
    n_points,
    n_param_pred,
    combined_pickle_file=None,
    cyc_mode="discharge",
    save_data=True,
    save_path=".",
):
    """Validate a cached ``assembled_surrogate_data.npz`` against the requested shape."""
    assembled_data_filename = os.path.join(
        save_path, "assembled_surrogate_data.npz"
    )
    tmp = np.load(assembled_data_filename)

    assert len(tmp["X_data"].shape) == 2
    assert tmp["X_data"].shape[1] == n_param_pred + 1
    assert tmp["X_data"].shape[0] == tmp["Y_data"].shape[0]

    if combined_pickle_file is not None:
        with open(combined_pickle_file, "rb") as f:
            sols = pickle.load(f)
            assert len(sols) * n_points == tmp["X_data"].shape[0]

    return tmp


def assemble_surrogate_data(
    data_root_folder,
    n_points,
    n_param_pred,
    combined_pickle_file=None,
    cyc_mode="discharge",
    save_data=True,
    save_path=".",
    n_sol_max: int | None = None,
):
    """Assemble a surrogate ``(time+params -> voltage)`` dataset from raw simulations.

    :param n_sol_max: forwarded to :func:`assemble_all_data` to cap how many
        raw solutions get assembled.
    """
    assembled_data_filename = os.path.join(
        save_path, "assembled_surrogate_data.npz"
    )
    if os.path.isfile(assembled_data_filename):
        try:
            tmp = check_assembled_surrogate_data_shape(
                data_root_folder,
                n_points,
                n_param_pred,
                combined_pickle_file,
                cyc_mode,
                save_data,
                save_path,
            )
            return tmp["X_data"], tmp["Y_data"]
        except AssertionError as err:
            logger.warning(
                f"Tried to load the assembled surrogate data instead of regenerating it, but something was inconsistent\n\t{err}"
            )
            pass

    logger.info("Assembling raw surrogate dataset")
    X_data, Y_data = assemble_all_data(
        data_root_folder,
        n_points,
        combined_pickle_file=combined_pickle_file,
        target_mode="phi",
        save_data=save_data,
        cyc_mode=cyc_mode,
        save_path=save_path,
        n_sol_max=n_sol_max,
    )
    new_x_data, new_y_data = from_param_to_surrogate_data(X_data, Y_data)

    if save_data:
        np.savez(
            assembled_data_filename,
            X_data=new_x_data.astype("float32"),
            Y_data=new_y_data.astype("float32"),
        )

    return new_x_data, new_y_data


def from_param_to_surrogate_data(
    X_data: np.ndarray, Y_data: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Reshape a ``(time, voltage)`` NPE dataset into per-timestep surrogate rows.

    :param X_data: shape ``(N, 2, n_points)`` — channel 0 is time, channel 1 voltage.
    :param Y_data: shape ``(N, n_param_pred)`` degradation parameters.
    :return: ``(new_x_data, new_y_data)`` of shape ``(N*n_points, n_param_pred+1)``
        and ``(N*n_points, 1)``, where each row is ``(time, *params) -> voltage``.
    """
    X_data = np.array(X_data).astype("float32")  # (N, 2, npoints)
    Y_data = np.array(Y_data).astype("float32")  # (N, n_param_pred)

    assert len(X_data.shape) == 3
    assert X_data.shape[1] == 2
    assert len(Y_data.shape) == 2

    # Make data surrogate style
    n_param_pred = Y_data.shape[1]
    Y_data = Y_data[:, np.newaxis, :]
    Y_data = np.repeat(Y_data, X_data.shape[2], axis=1)
    Y_data = np.reshape(Y_data, (-1, n_param_pred))  # (N*npoints,n_param_pred)
    t_data = np.reshape(X_data[:, 0, :], (-1, 1))  # (N*npoints,n_param_pred)

    new_x_data = np.hstack((t_data, Y_data))  # (N*npoints,n_param_pred+1)
    new_y_data = np.reshape(X_data[:, 1, :], (-1, 1))

    return new_x_data, new_y_data


def augment_data(
    X_data: np.ndarray,
    Y_data: np.ndarray,
    new_ds: int = 4,
    noise_level: float = 0.003,
) -> tuple[np.ndarray, np.ndarray]:
    """Augment an assembled ``(X, Y)`` dataset with noisy copies of every sample.

    Prestores ``new_ds`` noisy duplicates of the dataset (uniform noise on
    ``X``), as an alternative to adding noise online at train time.

    :return: ``(X_data, Y_data)`` with ``new_ds + 1`` times as many rows,
        the original samples preserved in the first ``N`` rows.
    """
    logger.info(
        f"Augmenting dataset by a factor {new_ds+1} with noise {noise_level}"
    )
    X_data_orig = X_data.copy()
    Y_data_orig = Y_data.copy()
    logger.info(f"\tOld data {X_data_orig.shape}, {Y_data_orig.shape}")

    for ds in range(new_ds):
        X_data = np.vstack(
            (
                X_data,
                X_data_orig
                + np.random.uniform(0, noise_level, size=X_data_orig.shape),
            )
        )
        Y_data = np.vstack((Y_data, Y_data_orig))
    logger.info(f"\tNew data {X_data.shape}, {Y_data.shape}")

    return X_data.astype("float32"), Y_data.astype("float32")
