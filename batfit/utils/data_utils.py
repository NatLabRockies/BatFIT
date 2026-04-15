import os
import pickle

import numpy as np
from prettyPlot.progressBar import print_progress_bar
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from batfit import BATFIT_DIR, logger


def get_sol_list(data_root_folder):
    list_files = os.listdir(data_root_folder)
    ind_remove = []
    for ifile, file in enumerate(list_files):
        if not file.startswith("solution") or not file.endswith(".npz"):
            ind_remove.append(ifile)

    for indr in reversed(ind_remove):
        list_files.pop(indr)

    return list_files


def from_name_to_params(filename):
    if filename.startswith("solution") and filename.endswith(".npz"):
        filename_par = filename[:-4].split("_")
    params = []
    filename_par.pop(0)
    for parstr in filename_par:
        params.append(float(parstr))
    return params


def get_max_time(data_root_folder):
    list_files = get_sol_list(data_root_folder)
    max_t = [
        np.amax(np.load(os.path.join(data_root_folder, file))["t"])
        for file in list_files
    ]
    max_t = np.array(max_t)
    max_t = np.amin(max_t)
    return max_t


def from_sol_to_data(data_root_folder, filename, n_points):
    logger.error("Use pkl file option instead")
    raise NotImplementedError
    sol = np.load(os.path.join(data_root_folder, filename))
    min_t = np.amin(sol["t"])
    max_t = np.amax(sol["t"])
    t_grid = np.linspace(min_t, max_t, n_points)
    y_grid = np.interp(t_grid, sol["t"], sol["phis_c"])
    params = from_name_to_params(filename)
    x = np.vstack((t_grid, y_grid))
    y = params
    return x, y


def from_sol_dict_to_xy(
    sol_dict, combined_sols, key, n_points, target_mode, diff_cap=True
):
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
        y = combined_sols[key]["params"]
    else:
        min_t = np.amin(sol_dict["t"])
        max_t = np.amax(sol_dict["t"])
        t_grid = np.linspace(min_t, max_t, n_points)
        x = np.reshape(t_grid, (1, -1))
        if "phi" in target_mode.lower():
            phi_grid = np.interp(t_grid, sol_dict["t"], sol_dict["phis_c"])
            x = np.vstack((x, np.reshape(phi_grid, (1, -1))))
        y = combined_sols[key]["params"]

    return x, y


def from_combined_sols_to_data(
    combined_sols, key, n_points, target_mode, cyc_mode, n_points_min=0
):
    if cyc_mode.lower() in ["discharge", "chargecc"]:
        sol_dict = combined_sols[key]["sol"]
        if sol_dict["phis_c"].shape[0] < n_points_min:
            logger.warning(
                f"Found and removed solution with {sol_dict['phis_c'].shape[0]} points"
            )
            return None, None
        x, y = from_sol_dict_to_xy(
            sol_dict, combined_sols, key, n_points, target_mode
        )
        return x, y
    elif cyc_mode.lower() in ["rh", "lh", "diffcap", "hppc", "prehppc", "posthppc"]:
        sol_dict = combined_sols[key]["sol"]
        if sol_dict["phis_c"].shape[0] < n_points_min:
            logger.warning(
                f"Found and removed solution with {sol_dict['phis_c'].shape[0]} points"
            )
            return None, None
        x, y = from_sol_dict_to_xy(
            sol_dict, combined_sols, key, n_points, target_mode, diff_cap=False
        )
        return x, y
    elif cyc_mode.lower() == "discharge-chargecc":
        sol_dis_dict = combined_sols[key]["sol_dis"]
        if sol_dis_dict["phis_c"].shape[0] < n_points_min:
            logger.warning(
                f"Found and removed solution with {sol_dis_dict['phis_c'].shape[0]} points"
            )
            return None, None
        x_dis, y_dis = from_sol_dict_to_xy(
            sol_dis_dict, combined_sols, key, n_points, target_mode
        )
        sol_chcc_dict = combined_sols[key]["sol_chcc"]
        if sol_chcc_dict["phis_c"].shape[0] < n_points_min:
            logger.warning(
                f"Found and removed solution with {sol_chcc_dict['phis_c'].shape[0]} points"
            )
            return None, None
        x_chcc, y_chcc = from_sol_dict_to_xy(
            sol_chcc_dict, combined_sols, key, n_points, target_mode
        )
        assert y_dis == y_chcc
        return np.vstack((x_dis, x_chcc)), y_dis
    else:
        raise NotImplementedError


def check_assembled_data(
    data_root_folder,
    n_points,
    combined_pickle_file=None,
    target_mode="phi",
    save_data=True,
    cyc_mode="discharge",
    save_path=".",
):
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


def check_assembled_surrogate_data(
    data_root_folder,
    n_points,
    n_param_pred,
    combined_pickle_file=None,
    cyc_mode="discharge",
    save_data=True,
    save_path=".",
):
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


def assemble_all_data(
    data_root_folder,
    n_points=100,
    n_points_min=0,
    combined_pickle_file=None,
    target_mode="phi",
    save_data=True,
    cyc_mode="discharge",
    save_path=".",
):

    assembled_data_filename = os.path.join(save_path, "assembled_data.npz")
    if os.path.isfile(assembled_data_filename):
        try:
            tmp = check_assembled_data(
                data_root_folder,
                n_points,
                combined_pickle_file,
                target_mode,
                save_data,
                cyc_mode,
                save_path,
            )
            return tmp["X_data"], tmp["Y_data"]
        except AssertionError as err:
            logger.warning(
                f"Tried to load the assembled data instead of regenerating it, but something was inconsistent\n\t{err}"
            )
            pass

    logger.info("Assembling raw dataset")
    if combined_pickle_file is None:
        combined = False
        list_files = get_sol_list(data_root_folder)
        n_sol_files = len(list_files)
    else:
        combined = True
        with open(
            os.path.join(data_root_folder, combined_pickle_file), "rb"
        ) as f:
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
    for ifile, file in enumerate(list_files):
        if not combined:
            x, y = from_sol_to_data(
                data_root_folder,
                file,
                n_points,
                target_mode,
                n_points_min=n_points_min,
            )
        else:
            x, y = from_combined_sols_to_data(
                combined_sols,
                file,
                n_points,
                target_mode,
                cyc_mode,
                n_points_min=n_points_min,
            )
        if x is not None and y is not None:
            X_data.append(x)
            Y_data.append(y)
        print_progress_bar(
            ifile + 1,
            n_sol_files,
            prefix=f"Create DS File {ifile+1} / {n_sol_files} ",
            suffix="Complete",
            length=50,
        )

    X_data = np.array(X_data).astype("float32")
    Y_data = np.array(Y_data).astype("float32")

    if save_data:
        np.savez(
            assembled_data_filename,
            X_data=X_data,
            Y_data=Y_data,
        )

    return X_data, Y_data


def from_param_to_surrogate_data(X_data, Y_data):
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


def assemble_surrogate_data(
    data_root_folder,
    n_points,
    n_param_pred,
    combined_pickle_file=None,
    cyc_mode="discharge",
    save_data=True,
    save_path=".",
):

    assembled_data_filename = os.path.join(
        save_path, "assembled_surrogate_data.npz"
    )
    if os.path.isfile(assembled_data_filename):
        try:
            tmp = check_assembled_surrogate_data(
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
        combined_pickle_file,
        target_mode="phi",
        save_data=save_data,
        cyc_mode=cyc_mode,
        save_path=save_path,
    )
    new_x_data, new_y_data = from_param_to_surrogate_data(X_data, Y_data)

    if save_data:
        np.savez(
            assembled_data_filename,
            X_data=new_x_data.astype("float32"),
            Y_data=new_y_data.astype("float32"),
        )

    return new_x_data, new_y_data


def assemble_all_data_diff_chan(data_root_folder, n_points=100):
    logger.info(
        "Assembling raw dataset with differential capacity as additional channels"
    )
    list_files = get_sol_list(data_root_folder)
    n_sol_files = len(list_files)
    assert n_sol_files > 1
    print_progress_bar(
        0,
        n_sol_files,
        prefix=f"Create DS File 0 / {n_sol_files} ",
        suffix="Complete",
        length=50,
    )
    for ifile, file in enumerate(list_files):
        sol = np.load(os.path.join(data_root_folder, file))
        min_t = np.amin(sol["t_diff"])
        max_t = np.amax(sol["t_diff"])
        t_grid = np.linspace(min_t, max_t, n_points)
        y0_grid = np.interp(t_grid, sol["t_diff"], sol["phis_c_diff"])
        y1_grid = np.interp(t_grid, sol["t_diff"], sol["dvdq"])
        y2_grid = np.interp(t_grid, sol["t_diff"], sol["dqdv"])
        params = from_name_to_params(file)
        x = np.vstack((t_grid, y0_grid, y1_grid, y2_grid))
        y = params
        if ifile == 0:
            X_data = np.reshape(x, (1, 4, -1))
            Y_data = np.reshape(y, (1, -1))
        else:
            X_data = np.vstack((X_data, np.reshape(x, (1, 4, -1))))
            Y_data = np.vstack((Y_data, np.reshape(y, (1, -1))))
        print_progress_bar(
            ifile + 1,
            n_sol_files,
            prefix=f"Create DS File {ifile+1} / {n_sol_files} ",
            suffix="Complete",
            length=50,
        )

    return X_data.astype("float32"), Y_data.astype("float32")


def augment_data(X_data, Y_data, new_ds=4, noise_level=0.003):
    logger.info(
        f"Augmenting dataset by a factor {new_ds+1} with noise {noise_level}"
    )
    X_data_orig = X_data.copy()
    Y_data_orig = Y_data.copy()
    print(f"\tOld data {X_data_orig.shape}, {Y_data_orig.shape}")

    for ds in range(new_ds):
        X_data = np.vstack(
            (
                X_data,
                X_data_orig
                + np.random.uniform(0, noise_level, size=X_data_orig.shape),
            )
        )
        Y_data = np.vstack((Y_data, Y_data_orig))
    print(f"\tNew data {X_data.shape}, {Y_data.shape}")

    return X_data.astype("float32"), Y_data.astype("float32")


class CustomScaler:
    def __init__(self, means, stds):
        self.means = means
        self.stds = stds

    def transform(self, data):
        assert len(data.shape) == len(self.means.shape)
        assert len(data.shape) == len(self.stds.shape)
        if self.stds.shape[1] == 2 and data.shape[1] == 1:
            transformed_data = (data - self.means[:, 1, :]) / self.stds[
                :, 1, :
            ]
        else:
            transformed_data = (data - self.means) / self.stds
        assert transformed_data.shape == data.shape
        return transformed_data

    def inverse_transform(self, transformed_data):
        assert len(transformed_data.shape) == len(self.means.shape)
        assert len(transformed_data.shape) == len(self.stds.shape)
        if self.stds.shape[1] == 2 and transformed_data.shape[1] == 1:
            data = transformed_data * self.stds[:, 1, :] + self.means[:, 1, :]
        else:
            data = transformed_data * self.stds + self.means
        assert transformed_data.shape == data.shape
        return data


def scale_input_from_scaler(
    X_data: np.ndarray[np.float32],
    scaler_X_file,
):
    assert len(X_data.shape) in [2, 3]

    with open(scaler_X_file, "rb") as f:
        scaler_X = pickle.load(f)

    return scaler_X.transform(X_data)


def scale_output_from_scaler(
    Y_data: np.ndarray[np.float32],
    scaler_Y_file,
):
    assert len(Y_data.shape) == 2

    with open(scaler_Y_file, "rb") as f:
        scaler_Y = pickle.load(f)

    return scaler_Y.transform(Y_data)


def scale_dataset_from_scaler(
    X_data: np.ndarray[np.float32],
    Y_data: np.ndarray[np.float32],
    scaler_X_file,
    scaler_Y_file,
):
    X_scaled = scale_input_from_scaler(X_data, scaler_X_file)
    Y_scaled = scale_output_from_scaler(Y_data, scaler_Y_file)

    return X_scaled, Y_scaled


def unscale_input_from_scaler(
    X_data: np.ndarray[np.float32],
    scaler_X_file,
):
    assert len(X_data.shape) == 3

    if scaler_X_file is None:
        return X_data

    try:
        with open(scaler_X_file, "rb") as f:
            scaler_X = pickle.load(f)
    except FileNotFoundError:
        return X_data

    return scaler_X.inverse_transform(X_data)


def unscale_output_from_scaler(
    Y_data: np.ndarray[np.float32],
    scaler_Y_file,
):
    assert len(Y_data.shape) == 2

    if scaler_Y_file is None:
        return Y_data

    try:
        with open(scaler_Y_file, "rb") as f:
            scaler_Y = pickle.load(f)
    except FileNotFoundError:
        return Y_data

    return scaler_Y.inverse_transform(Y_data)


def unscale_dataset_from_scaler(
    X_data: np.ndarray[np.float32],
    Y_data: np.ndarray[np.float32],
    scaler_X_file,
    scaler_Y_file,
):
    X_data_unscaled = unscale_input_from_scaler(X_data, scaler_X_file)
    Y_data_unscaled = unscale_output_from_scaler(Y_data, scaler_Y_file)

    return X_data_unscaled, Y_data_unscaled


def unscale_pred_from_scaler(
    Y_data: np.ndarray[np.float32],
    scaler_Y_file: str | None = None,
):
    assert len(Y_data.shape) == 2

    if scaler_Y_file is None:
        return Y_data

    try:
        with open(scaler_Y_file, "rb") as f:
            scaler_Y = pickle.load(f)
    except FileNotFoundError:
        return Y_data

    return scaler_Y.inverse_transform(Y_data)


def unscale_inp_from_scaler(
    X_data: np.ndarray[np.float32],
    scaler_X_file: str | None = None,
):
    assert len(X_data.shape) == 3

    if scaler_X_file is None:
        return X_data

    try:
        with open(scaler_X_file, "rb") as f:
            scaler_X = pickle.load(f)
    except FileNotFoundError:
        return X_data

    return scaler_X.inverse_transform(X_data)


def scale_dataset_from_np(
    X_train: np.ndarray[np.float32],
    X_test: np.ndarray[np.float32],
    Y_train: np.ndarray[np.float32],
    Y_test: np.ndarray[np.float32],
    save_path: str = ".",
    save_scaled=True,
    scale_y=False,
):

    scaler_x_filename = os.path.join(save_path, "scaler_X.pkl")
    data_scaled_filename = os.path.join(save_path, "data_scaled.npz")
    scaler_y_filename = os.path.join(save_path, "scaler_Y.pkl")

    if os.path.isfile(scaler_x_filename) and os.path.isfile(
        data_scaled_filename
    ):
        if (not scale_y) or (scale_y and os.path.isfile(scaler_y_filename)):
            logger.warning("Data already scaled, loading scaler and data")
            tmp = np.load(data_scaled_filename)
            return (
                tmp["X_train"],
                tmp["Y_train"],
                tmp["X_test"],
                tmp["Y_test"],
            )

    logger.info("Scaling the data")
    # Scale data
    means_X = np.mean(X_train, axis=(0, 2), keepdims=True)
    stds_X = np.std(X_train, axis=(0, 2), keepdims=True)
    scaler_X = CustomScaler(means_X, stds_X)
    X_train_scaled = scaler_X.transform(X_train).astype("float32")
    X_test_scaled = scaler_X.transform(X_test).astype("float32")
    logger.info(f"Dumping scaler X at {scaler_x_filename}")
    with open(scaler_x_filename, "wb") as f:
        pickle.dump(scaler_X, f)

    if scale_y:
        scaler_Y = preprocessing.StandardScaler().fit(Y_train)
        Y_train_scaled = scaler_Y.transform(Y_train).astype("float32")
        Y_test_scaled = scaler_Y.transform(Y_test).astype("float32")
        logger.info(f"Dumping scaler Y at {scaler_y_filename}")
        with open(scaler_y_filename, "wb") as f:
            pickle.dump(scaler_Y, f)
    else:
        Y_train_scaled = Y_train
        Y_test_scaled = Y_test

    if save_scaled:
        logger.info(f"Saving scaled data at {data_scaled_filename}")
        np.savez(
            data_scaled_filename,
            X_train=X_train_scaled,
            Y_train=Y_train_scaled,
            X_test=X_test_scaled,
            Y_test=Y_test_scaled,
        )

    return (
        X_train_scaled,
        Y_train_scaled,
        X_test_scaled,
        Y_test_scaled,
    )


def scale_surrogate_dataset_from_np(
    X_train: np.ndarray[np.float32],
    X_test: np.ndarray[np.float32],
    Y_train: np.ndarray[np.float32],
    Y_test: np.ndarray[np.float32],
    save_path: str = ".",
    save_scaled=True,
    scale_y=False,
):

    scaler_x_filename = os.path.join(save_path, "scaler_surrogate_X.pkl")
    data_scaled_filename = os.path.join(save_path, "data_surrogate_scaled.npz")
    scaler_y_filename = os.path.join(save_path, "scaler_surrogate_Y.pkl")

    if os.path.isfile(scaler_x_filename) and os.path.isfile(
        data_scaled_filename
    ):
        if (not scale_y) or (scale_y and os.path.isfile(scaler_y_filename)):
            logger.warning(
                "Data surrogate already scaled, loading scaler and data"
            )
            tmp = np.load(data_scaled_filename)
            return (
                tmp["X_train"],
                tmp["Y_train"],
                tmp["X_test"],
                tmp["Y_test"],
            )

    logger.info("Scaling the data")
    # Scale data
    means_X = np.mean(X_train, axis=0, keepdims=True)
    stds_X = np.std(X_train, axis=0, keepdims=True)
    scaler_X = CustomScaler(means_X, stds_X)
    X_train_scaled = scaler_X.transform(X_train).astype("float32")
    X_test_scaled = scaler_X.transform(X_test).astype("float32")
    logger.info(f"Dumping scaler X at {scaler_x_filename}")
    with open(scaler_x_filename, "wb") as f:
        pickle.dump(scaler_X, f)

    if scale_y:
        scaler_Y = preprocessing.StandardScaler().fit(Y_train)
        Y_train_scaled = scaler_Y.transform(Y_train).astype("float32")
        Y_test_scaled = scaler_Y.transform(Y_test).astype("float32")
        logger.info(f"Dumping scaler Y at {scaler_y_filename}")
        with open(scaler_y_filename, "wb") as f:
            pickle.dump(scaler_Y, f)
    else:
        Y_train_scaled = Y_train
        Y_test_scaled = Y_test

    if save_scaled:
        logger.info(f"Saving scaled surrogate data at {data_scaled_filename}")
        np.savez(
            data_scaled_filename,
            X_train=X_train_scaled,
            Y_train=Y_train_scaled,
            X_test=X_test_scaled,
            Y_test=Y_test_scaled,
        )

    return (
        X_train_scaled,
        Y_train_scaled,
        X_test_scaled,
        Y_test_scaled,
    )


def split_dataset_from_np(
    np_data: np.ndarray[np.float32] | None = None,
    np_data_label: np.ndarray[np.float32] | None = None,
    test_split: float = 0.1,
    save: bool = True,
    save_path: str = ".",
):

    # If data is already splitted, don't do it again
    data_split_filename = os.path.join(save_path, "data_split.npz")
    if os.path.isfile(data_split_filename):
        logger.warning("Data already splitted, loading it only")
        tmp = np.load(data_split_filename)
        return (
            tmp["X_train"],
            tmp["Y_train"],
            tmp["X_test"],
            tmp["Y_test"],
        )

    logger.info(
        f"Splitting the data with train/test split ({1-test_split:.2f}/{test_split:.2f})"
    )

    assert np_data is not None
    assert np_data_label is not None

    # Split in train and test
    X_train, X_test, Y_train, Y_test = train_test_split(
        np_data, np_data_label, test_size=test_split, shuffle=True
    )

    if save:
        logger.info(f"Saving data at {data_split_filename}")
        np.savez(
            data_split_filename,
            X_train=X_train.astype("float32"),
            Y_train=Y_train.astype("float32"),
            X_test=X_test.astype("float32"),
            Y_test=Y_test.astype("float32"),
        )

    return (
        X_train.astype("float32"),
        Y_train.astype("float32"),
        X_test.astype("float32"),
        Y_test.astype("float32"),
    )


def split_surrogate_dataset_from_np(
    np_data: np.ndarray[np.float32] | None = None,
    np_data_label: np.ndarray[np.float32] | None = None,
    test_split: float = 0.1,
    save: bool = True,
    save_path: str = ".",
):

    # If data is already splitted, don't do it again
    data_split_filename = os.path.join(save_path, "data_surrogate_split.npz")
    if os.path.isfile(data_split_filename):
        logger.warning("Data already splitted, loading it only")
        tmp = np.load(data_split_filename)
        return (
            tmp["X_train"],
            tmp["Y_train"],
            tmp["X_test"],
            tmp["Y_test"],
        )

    logger.info(
        f"Splitting the data with train/test split ({1-test_split:.2f}/{test_split:.2f})"
    )

    assert np_data is not None
    assert np_data_label is not None

    # Split in train and test
    X_train, X_test, Y_train, Y_test = train_test_split(
        np_data, np_data_label, test_size=test_split, shuffle=True
    )

    if save:
        logger.info(f"Saving data at {data_split_filename}")
        np.savez(
            data_split_filename,
            X_train=X_train.astype("float32"),
            Y_train=Y_train.astype("float32"),
            X_test=X_test.astype("float32"),
            Y_test=Y_test.astype("float32"),
        )

    return (
        X_train.astype("float32"),
        Y_train.astype("float32"),
        X_test.astype("float32"),
        Y_test.astype("float32"),
    )


if __name__ == "__main__":
    data_root_folder = os.path.join(
        BATFIT_DIR, "..", "dataset", "spm_discharge_extb"
    )
    filename = "solution_2.002_6.95014_1.04652_1.30786_1.0417_0.796.npz"
    # list_files = get_sol_list(data_root_folder)
    # print(list_files)
    # max_t = get_max_time(data_root_folder)
    # print(max_t)
    # from_sol_to_data(data_root_folder, filename, 100)
    X_data, Y_data = assemble_all_data(data_root_folder, n_points=100)
    X_train, Y_train, X_test, Y_test = split_dataset_from_np(
        X_data, Y_data, 0.1
    )

    X_train_scaled, Y_train_scaled, X_test_scaled, Y_test_scaled = (
        scale_dataset_from_np(X_train, X_test, Y_train, Y_test)
    )
    # filename = "solution_2.002_6.95014_1.04652_1.30786_1.0417_0.796.npz"
    # print(from_name_to_params(filename))
