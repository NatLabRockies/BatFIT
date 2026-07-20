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
