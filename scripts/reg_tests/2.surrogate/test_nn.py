import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable MPS fallback

import numpy as np
import torch
from prettyPlot.plotting import *
from train_nn import define_model

from batfit import BATFIT_DIR, BATFIT_EXP, logger
from batfit.basicutilityc import ReadInput as ri
from batfit.utils.data_utils import *
from batfit.utils.torch_utils import *


def load_model(inp):
    """Build the surrogate and load its best checkpoint (weights + scalers)."""
    model = define_model(inp)
    best_model_file = find_best_model_file(inp.models_dir)
    logger.info(f"Loading {best_model_file}")
    model.load_state_dict(torch.load(best_model_file, weights_only=True))
    device = torch.device(get_device_type())
    model.to(device)
    model.eval()
    return model, device


def predict_voltage(model, device, X_rows):
    """Predict volts for physical surrogate rows ``(time, deg_params...)``."""
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(X_rows)),
        batch_size=512 * 256,
        shuffle=False,
    )
    preds = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            voltage = model.predict_physical(batch[:, :1], batch[:, 1:])
            preds.append(voltage.cpu().numpy())
    return np.vstack(preds)


def test_perf(inp):
    data_path = inp.data_path
    if not os.path.isfile(os.path.join(data_path, "data_surrogate_split.npz")):
        if os.path.isfile(os.path.join(data_path, "data_split.npz")):
            tmp = np.load(os.path.join(data_path, "data_split.npz"))
            X_val, Y_val = from_param_to_surrogate_data(
                tmp["X_val"], tmp["Y_val"]
            )
        else:
            return
    else:
        # physical (time, deg_params...) rows and voltages
        A_split = np.load(os.path.join(data_path, "data_surrogate_split.npz"))
        X_val = A_split["X_val"]
        Y_val = A_split["Y_val"]

    model, device = load_model(inp)
    mu_preds = predict_voltage(model, device, X_val)
    err = abs(mu_preds - Y_val)

    mean_err = np.mean(err, axis=0)
    rmse = np.sqrt(np.mean(err**2, axis=0))
    post_file = "post_val"

    with open(os.path.join(inp.models_dir, f"{post_file}.txt"), "w+") as f:
        f.write(f"MAE: {mean_err*1000} mV\n")
        f.write(f"RMSE: {rmse*1000} mV\n")
    np.savez(os.path.join(inp.models_dir, f"{post_file}.npz"), err=err)


def plot_perf(inp):
    data_path = inp.data_path
    if not os.path.isfile(os.path.join(data_path, "data_split.npz")):
        return

    # Physical (time, deg_params...) rows of the validation batteries
    A_split = np.load(os.path.join(data_path, "data_split.npz"))
    X_rows, _ = from_param_to_surrogate_data(
        A_split["X_val"], A_split["Y_val"]
    )

    model, device = load_model(inp)
    mu_preds = predict_voltage(model, device, X_rows)
    mu_preds = mu_preds.reshape((-1, inp.n_points))

    figure_folder = os.path.join(inp.models_dir, "Figures")
    os.makedirs(figure_folder, exist_ok=True)

    fig, axs = plt.subplots(3, 3, figsize=(8, 8))
    for i in range(min(9, A_split["X_val"].shape[0])):
        ix = i // 3
        iy = i % 3
        axs[ix, iy].plot(
            A_split["X_val"][i, 0, :],
            A_split["X_val"][i, 1, :],
            label="True",
        )
        axs[ix, iy].plot(
            A_split["X_val"][i, 0, :], mu_preds[i, :], label="pred"
        )
    plt.tight_layout()
    fig_file = "surr_preds"
    plt.savefig(os.path.join(inp.models_dir, "Figures", f"{fig_file}.pdf"))
    plt.close()


if __name__ == "__main__":
    import shutil
    import sys

    inp = ri.basic_input(sys.argv[1])
    test_perf(inp)
    plot_perf(inp)
