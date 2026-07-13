import os
import pickle

from batfit import logger
from batfit.model.param_utils.train_utils import create_model_from_log
from batfit.model.paramNN import *
from batfit.utils.data_utils import (
    scale_input_from_scaler,
    unscale_output_from_scaler,
)
from batfit.utils.torch_utils import get_device_type


def load_fm_model(inp) -> ProbParamFM:
    """Load the trained flow matching NPE from its training run directory.

    ``train_fm_model`` pickles the full model object (including the
    ``Y_prior`` buffer used for prior matching) to ``models_dir/model.pkl``,
    so no architecture reconstruction from recipe fields is needed: the
    pickle carries the architecture and ``load_state_dict`` only overwrites
    the weights with the best checkpoint.

    :param inp: parsed recipe; uses ``inp.models_dir``
    :return: trained ProbParamFM in eval mode (not yet moved to device)
    """
    best_model_file = find_best_model_file(inp.models_dir)
    logger.info(f"Loading {best_model_file}")
    model = create_model_from_log(
        os.path.join(inp.models_dir, "model.pkl"), best_model_file
    )
    model.eval()
    return model


def single_forward_pass(
    time_input: np.ndarray,
    phis_c_input: np.ndarray,
    scaler_file: str,
    scaler_Y_file: str,
    model: ProbParamFM,
    n_samples: int = 1000,
    n_ode_steps: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw FM posterior samples for one observed (time, voltage) curve.

    The model samples in z-scored parameter space (FM training uses
    ``scale_y=True``), so every sample is inverse-transformed with the
    ``scaler_Y`` fitted at training time; mu and sigma are then the mean and
    standard deviation of the physical-space samples.

    :param time_input: observation times, shape (n_points,)
    :param phis_c_input: observed voltage, shape (n_points,)
    :param scaler_file: path to the signal scaler ``scaler_X.pkl``
    :param scaler_Y_file: path to the parameter scaler ``scaler_Y.pkl``
    :param model: trained ProbParamFM (already on its device)
    :param n_samples: number of posterior samples to draw
    :param n_ode_steps: number of ODE integration steps for sampling
    :return: (samples, mu, sigma) in physical parameter space, with shapes
        (n_samples, n_param_pred), (n_param_pred,), (n_param_pred,)
    """
    assert isinstance(scaler_file, str)
    assert os.path.exists(scaler_file)
    assert isinstance(scaler_Y_file, str)
    assert os.path.exists(scaler_Y_file)
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
        samples_z = model.sample(
            torch.Tensor(input_scaled).to(device),
            n_samples=n_samples,
            n_steps=n_ode_steps,
        )
    # (1, n_samples, n_param_pred) -> (n_samples, n_param_pred)
    samples_z = samples_z.cpu().numpy()[0]
    samples = unscale_output_from_scaler(samples_z, scaler_Y_file)
    mu = samples.mean(axis=0)
    sigma = samples.std(axis=0)

    return samples, mu, sigma


def get_model_it(model_dirs):
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


def read_test_loss(model_dirs):
    filename = os.path.join(model_dirs, "test_loss.csv")
    vals = np.loadtxt(filename, delimiter=";", skiprows=1)
    best_ind = np.argmin(vals[:, 1])
    if best_ind == vals.shape[0] - 1:
        return "final"
    else:
        return vals[best_ind, 0]


def find_best_model_file(model_dirs):
    filename = None
    best_iter = read_test_loss(model_dirs)
    if best_iter == "final":
        filename = os.path.join(model_dirs, "model_final.pt")
    else:
        iterations = get_model_it(model_dirs)
        if len(iterations) > 0:
            ind = np.argmin(abs(iterations - best_iter))
            filename_try = os.path.join(
                model_dirs, f"model_{iterations[ind]}.pt"
            )
            if os.path.exists(filename_try):
                filename = filename_try
    if filename is None:
        return os.path.join(model_dirs, "model_final.pt")
    else:
        return filename


def figure_org(n_params: int):

    logger.info(f"Organizing figure for n_param = {n_params}")
    if n_params == 10:
        rows = 4
        cols = 4
        ax_id = 11
    elif n_params == 12:
        rows = 4
        cols = 5
        ax_id = 14
    elif n_params == 17:
        rows = 5
        cols = 5
        ax_id = 19
    elif n_params == 19:
        rows = 5
        cols = 5
        ax_id = 19
    elif n_params == 23:
        rows = 5
        cols = 6
        ax_id = 24
    elif n_params == 29:
        rows = 5
        cols = 7
        ax_id = 29
    elif n_params == 31:
        rows = 4
        cols = 9
        ax_id = 31
    else:
        raise NotImplementedError

    return {"rows": rows, "cols": cols, "ax_id": ax_id}
