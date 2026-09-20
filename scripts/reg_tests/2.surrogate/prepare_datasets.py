import os
import sys

from batfit.basicutilityc import ReadInput as ri
from batfit.utils.data_utils import *
from batfit.utils.torch_utils import *


def pre_proc_data(data_root_folder, cyc_mode, n_points):
    """
    Build the NPE and surrogate train/test/val splits for a data folder.

    Both datasets are built from the same whole-curve data so they share the
    battery-level split (``data_split.npz``); the surrogate then explodes each
    split into per-timestep rows (split-then-slice).
    """
    X_npe_data, Y_npe_data = assemble_all_data(
        data_root_folder,
        n_points=n_points,
        combined_pickle_file=os.path.join(data_root_folder, "sols.pkl"),
        target_mode="phi",
        save_data=True,
        cyc_mode=cyc_mode,
        save_path=data_root_folder,
    )
    n_curves = Y_npe_data.shape[0]
    batch_size = min(inp.batch_size, max(1, int(n_curves * 0.8)))
    # NPE dataset: creates the battery-level data_split.npz
    make_dataset_from_np(
        batch_size=batch_size,
        np_data=X_npe_data,
        np_data_label=Y_npe_data,
        scale=True,
        scale_y=False,
        save_path=data_root_folder,
    )
    # Surrogate dataset: split-then-slice, reusing the same battery split
    make_surrogate_dataset_from_np(
        batch_size=inp.batch_size,
        np_data=X_npe_data,
        np_data_label=Y_npe_data,
        scale=True,
        scale_y=False,
        save_path=data_root_folder,
    )


if __name__ == "__main__":
    inp = ri.basic_input(sys.argv[1])
    data_root_folder = inp.data_path
    data_root_folder_val = inp.data_val_path
    n_points = inp.n_points
    cyc_mode = inp.cyc_mode
    pre_proc_data(data_root_folder, cyc_mode, n_points)
    pre_proc_data(data_root_folder_val, cyc_mode, n_points)
