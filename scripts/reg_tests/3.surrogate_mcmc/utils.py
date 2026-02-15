import os
import pickle
from batfit.basicutilityc import ReadInput as ri
from batfit.model.surrogateNN import *
from batfit.utils.data_utils import *
from batfit.utils.torch_utils import *

def define_model(inp):
    data_root_folder = inp.data_path
    n_points = inp.n_points
    n_param_pred = inp.n_param_pred
    cyc_mode = inp.cyc_mode

    model = SurrogateFCNN(
        fc_list=inp.fc_units,
        loss_fn=mae_loss,
        n_param_pred=n_param_pred,
        sim_config=inp.sim_config,
        cyc_mode=cyc_mode,
        constrain_output=inp.constrain_output,
    )
    num_parameters = get_num_parameters(model)
    print(f"No. Trainable Parameters: {num_parameters}")

    with open(
        os.path.join(inp.data_path, "scaler_surrogate_X.pkl"), "rb"
    ) as f:
        scaler_X = pickle.load(f)

    return model, scaler_X

def get_model_it(model_dirs: str) -> np.ndarray:
    """
    Get list of model checkpoints except for the final
    """
    filenames = os.listdir(model_dirs)
    iterations = []
    for filename in filenames:
        if (
            filename.startswith("model_")
            and filename.endswith(".pt")
            and "final" not in filename
        ):
            ind_end = filename.index(".pt")
            ind_start = 6
            iterations.append(int(filename[ind_start:ind_end]))
    return np.array(iterations)


def read_test_loss(model_dirs: str) -> str | int:
    """
    Get step at which best test loss was achieved
    """
    filename = os.path.join(model_dirs, "test_loss.csv")
    vals = np.loadtxt(filename, delimiter=";", skiprows=1)
    best_ind = np.argmin(vals[:, 1])
    if best_ind == vals.shape[0] - 1 and os.path.isfile(
        os.path.join(model_dirs, "model_final.pt")
    ):
        return "final"
    else:
        return vals[best_ind, 0]


def find_best_model_file(model_dirs: str) -> str:
    """
    Get model checkpoint that correspond to the best test loss
    """
    best_iter = read_test_loss(model_dirs)
    if best_iter == "final":
        return os.path.join(model_dirs, "model_final.pt")
    else:
        iterations = get_model_it(model_dirs)
        if len(iterations) == 0:
            return os.path.join(model_dirs, f"model_final.pt")
        else:
            ind = np.argmin(abs(iterations - best_iter))
            return os.path.join(model_dirs, f"model_{iterations[ind]}.pt")

