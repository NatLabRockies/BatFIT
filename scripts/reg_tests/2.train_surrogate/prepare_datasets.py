import os
import sys
from batfit.basicutilityc import ReadInput as ri
from batfit.utils.data_utils import *
from batfit.utils.torch_utils import *
from batfit import logger

def pre_proc_data(data_root_folder, cyc_mode, n_points):
    '''
    Split data consistently between the surrogate data and the NPE data
    '''

    # Split data and save the temporal coherence (useful for NPE)
    X_npe_data, Y_npe_data = assemble_all_data(
        data_root_folder,
        n_points=n_points,
        combined_pickle_file=os.path.join(data_root_folder, "sols.pkl"),
        target_mode="phi",
        save_data=True,
        cyc_mode=cyc_mode,
        save_path=data_root_folder,
    )
    tmp = np.load(os.path.join(data_root_folder, "assembled_data.npz"))
    BATCH_SIZE = min(inp.batch_size, int(Y_npe_data.shape[0] * 0.9))
    _, _ = make_dataset_from_np(
        batch_size=BATCH_SIZE,
        np_data=X_npe_data,
        np_data_label=Y_npe_data,
        scale=True,
        scale_y=False,
        save_path=data_root_folder,
    )
    # Split data without saving the temporal coherence (useful for surrogate)
    X_data, Y_data = assemble_surrogate_data(
        data_root_folder,
        n_points=n_points,
        n_param_pred=n_param_pred,
        combined_pickle_file=os.path.join(data_root_folder, "sols.pkl"),
        cyc_mode=cyc_mode,
        save_data=True,
        save_path=data_root_folder,
    )
    tmp = np.load(
        os.path.join(data_root_folder, "assembled_surrogate_data.npz")
    )
    BATCH_SIZE = min(inp.batch_size, int(Y_data.shape[0] * 0.9))
    _, _ = make_surrogate_dataset_from_np(
        batch_size=BATCH_SIZE,
        np_data=X_data,
        np_data_label=Y_data,
        scale=True,
        scale_y=False,
        save_path=data_root_folder,
    )


if __name__ == "__main__":
    inp = ri.basic_input(sys.argv[1])
    data_root_folder = inp.data_path
    data_root_folder_val = inp.data_val_path
    n_points = inp.n_points
    n_param_pred = inp.n_param_pred
    cyc_mode = inp.cyc_mode
    pre_proc_data(data_root_folder, cyc_mode, n_points)
    pre_proc_data(data_root_folder_val, cyc_mode, n_points)

