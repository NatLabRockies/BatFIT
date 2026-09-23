import os
import sys

from batfit.basicutilityc import ReadInput as ri
from batfit.preprocess.sim_setup import make_params
from batfit.utils.data_utils import *
from batfit.utils.torch_utils import *


def pre_proc_data(data_root_folder, cyc_mode, n_points):
    """
    Build the surrogate dataset of a data folder.

    The battery-level train/test/val split (``data_split.npz``) is created once
    by ``1.gen_data`` and reused here; each split is exploded into per-timestep
    rows (split-then-slice) and cached in ``data_surrogate_split.npz``.
    """
    X_npe_data, Y_npe_data = assemble_all_data(
        data_root_folder,
        n_points=n_points,
        combined_pickle_file="sols.pkl",
        target_mode="phi",
        save_data=True,
        cyc_mode=cyc_mode,
        save_path=data_root_folder,
    )
    # Surrogate dataset: split-then-slice, reusing the battery split
    make_surrogate_dataset_from_np(
        make_params(inp.sim_config),
        np_data=X_npe_data,
        np_data_label=Y_npe_data,
        batch_size=inp.batch_size,
        save_path=data_root_folder,
    )


if __name__ == "__main__":
    inp = ri.basic_input(sys.argv[1])
    data_root_folder = inp.data_path
    n_points = inp.n_points
    cyc_mode = inp.cyc_mode
    pre_proc_data(data_root_folder, cyc_mode, n_points)
