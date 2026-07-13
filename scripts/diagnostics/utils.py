import os
import pickle

from batfit import logger
from batfit.model.paramNN import *
from batfit.utils.data_utils import scale_input_from_scaler
from batfit.utils.torch_utils import find_best_model_file, get_device_type


def define_model(inp):
    n_points = inp.n_points
    target_mode = inp.target_mode
    cyc_mode = inp.cyc_mode
    n_param_pred = inp.n_param_pred
    if target_mode != "encoded":
        input_shape = (2, inp.n_points)

    model = ProbParamCNN(
        input_shape=input_shape,
        chan_list=[inp.num_channels] * inp.num_convs,
        fc_list=[inp.num_fc_units] * inp.num_fc_hidden,
        fc_mu_list=[inp.num_fc_gamma_mu_units] * inp.num_fc_gamma_mu_hidden,
        fc_gamma_list=[inp.num_fc_gamma_mu_units] * inp.num_fc_gamma_mu_hidden,
        loss_fn=independent_normal_loss,
        cyc_mode=cyc_mode,
        n_param_pred=n_param_pred,
        constrain_output=True,
        dependent_outputs=False,
        sim_config=inp.sim_config,
    )
    num_parameters = get_num_parameters(model)
    logger.info(f"No. Trainable Parameters: {num_parameters}")

    with open(inp.scaler_path, "rb") as f:
        scaler_X = pickle.load(f)

    return model, scaler_X


def single_forward_pass(
    time_input: np.ndarray,
    phis_c_input: np.ndarray,
    scaler_file: str,
    model: ProbParamCNN,
):
    assert isinstance(scaler_file, str)
    assert os.path.exists(scaler_file)
    assert isinstance(time_input, np.ndarray)
    assert isinstance(phis_c_input, np.ndarray)
    assert len(time_input.shape) == 1
    assert len(phis_c_input.shape) == 1
    assert phis_c_input.shape[0] == time_input.shape[0]

    input_array = np.stack((time_input, phis_c_input))[np.newaxis, ...]
    assert input_array.shape == (1, 2, phis_c_input.shape[0])

    device = torch.device(get_device_type())

    with torch.no_grad():
        input_scaled = scale_input_from_scaler(input_array, scaler_file)
        tmpmu_preds, tmpsigma_preds = model(
            torch.Tensor(input_scaled).to(device)
        )
        if model.constrain_output:
            tmpmu_preds = model.inv_transform_mu(
                tmpmu_preds.cpu(), model.min_par.numpy(), model.amp_par.numpy()
            )
            tmpsigma_preds = model.inv_transform_gamma(
                tmpsigma_preds.cpu(), model.amp_par.numpy()
            )
        tmpsigma_preds = tmpsigma_preds.numpy()
        tmpmu_preds = tmpmu_preds.numpy()

    return tmpmu_preds, tmpsigma_preds
