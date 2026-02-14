import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable MPS fallback
import pickle

import numpy as np
import torch
from prettyPlot.plotting import *
from train_nn import define_model, pre_inp

from batfit import BATFIT_DIR, BATFIT_EXP, logger
from batfit.basicutilityc import ReadInput as ri
from batfit.model.paramNN import *
from batfit.utils.data_utils import *
from batfit.utils.data_utils import scale_input_from_scaler
from batfit.utils.torch_utils import *
from batfit.utils.torch_utils import get_device_type


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
    Get model checkoint that correspond to the best test loss
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


def test_perf(inp, mode="test"):
    data_path = inp.data_path
    if not os.path.isfile(os.path.join(data_path, "data_surrogate_split.npz")):
        if os.path.isfile(os.path.join(data_path, "data_split.npz")):
            tmp = np.load(os.path.join(data_path, "data_split.npz"))
            X_test, Y_test = from_param_to_surrogate_data(
                tmp["X_test"], tmp["Y_test"]
            )
        else:
            return
    else:
        # Make dataset
        A_split = np.load(os.path.join(data_path, "data_surrogate_split.npz"))
        X_test = A_split["X_test"]
        Y_test = A_split["Y_test"]
    X_scaled = scale_input_from_scaler(
        X_test, os.path.join(inp.data_path, "scaler_surrogate_X.pkl")
    )
    input_data = torch.Tensor(X_scaled)
    output_data = torch.Tensor(Y_test)
    shape_in = input_data[0].shape
    test_data_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(input_data, output_data),
        batch_size=512 * 256,
        shuffle=False,
    )

    # Make model
    model, scaler = define_model(inp)
    best_model_file = find_best_model_file(inp.models_dir)
    logger.info(f"Loading {best_model_file}")
    model.load_state_dict(torch.load(best_model_file, weights_only=True))

    device = torch.device(get_device_type())
    model.to(device)
    model.eval()

    # Forward pass
    with torch.no_grad():
        for ibatch, batch in enumerate(test_data_loader):
            tmpmu_preds = model(batch[0].to(device))
            if model.constrain_output:
                tmpmu_preds = model.inv_transform_output(
                    tmpmu_preds.cpu(), model.min_v, model.amp_v
                )
            else:
                tmpmu_preds = tmpmu_preds.cpu()
            tmpmu_preds = tmpmu_preds.numpy()

            tmptruth = batch[1].cpu().numpy()

            tmperr = abs(tmpmu_preds - tmptruth)
            tmpmu_preds = tmpmu_preds

            if ibatch == 0:
                mu_preds = tmpmu_preds
                err = tmperr
                truth = tmptruth
            else:
                mu_preds = np.vstack((mu_preds, tmpmu_preds))
                err = np.vstack((err, tmperr))
                truth = np.vstack((truth, tmptruth))
    mean_err = np.mean(err, axis=0)
    rmse = np.sqrt(np.mean(err**2, axis=0))
    post_file = "post_surrogate"

    with open(os.path.join(inp.models_dir, f"{post_file}.txt"), "w+") as f:
        f.write(f"MAE: {mean_err*1000} mV\n")
        f.write(f"RMSE: {rmse*1000} mV\n")
    np.savez(os.path.join(inp.models_dir, f"{post_file}.npz"), err=err)


def plot_perf(inp, mode="test"):
    data_path = inp.data_path
    if not os.path.isfile(os.path.join(data_path, "data_split.npz")):
        return

    # Make dataset
    A_split = np.load(os.path.join(data_path, "data_split.npz"))
    X_data = A_split["X_test"]
    Y_data = A_split["Y_test"]
    n_param_pred = Y_data.shape[1]
    Y_data = Y_data[:, np.newaxis, :]
    Y_data = np.repeat(Y_data, X_data.shape[2], axis=1)
    Y_data = np.reshape(Y_data, (-1, n_param_pred))  # (N*npoints,n_param_pred)
    t_data = np.reshape(X_data[:, 0, :], (-1, 1))  # (N*npoints,n_param_pred)
    new_x_data = np.hstack((t_data, Y_data))  # (N*npoints,n_param_pred+1)
    new_y_data = np.reshape(X_data[:, 1, :], (-1, 1))

    X_scaled = scale_input_from_scaler(
        new_x_data, os.path.join(inp.data_path, "scaler_surrogate_X.pkl")
    )
    Y_scaled = new_y_data

    input_data = torch.Tensor(X_scaled)
    output_data = torch.Tensor(Y_scaled)
    shape_in = input_data[0].shape
    test_data_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(input_data, output_data),
        batch_size=512 * 256,
        shuffle=False,
    )

    # Make model
    model, scaler = define_model(inp)
    best_model_file = find_best_model_file(inp.models_dir)
    logger.info(f"Loading {best_model_file}")
    model.load_state_dict(torch.load(best_model_file, weights_only=True))

    device = torch.device(get_device_type())
    model.to(device)
    model.eval()

    # Forward pass
    with torch.no_grad():
        for ibatch, batch in enumerate(test_data_loader):
            tmpmu_preds = model(batch[0].to(device))
            if model.constrain_output:
                tmpmu_preds = model.inv_transform_output(
                    tmpmu_preds.cpu(), model.min_v, model.amp_v
                )
            else:
                tmpmu_preds = tmpmu_preds.cpu()
            tmpmu_preds = tmpmu_preds.numpy()
            if ibatch == 0:
                mu_preds = tmpmu_preds
            else:
                mu_preds = np.vstack((mu_preds, tmpmu_preds))

    mu_preds = mu_preds.reshape((-1, inp.n_points))

    figure_folder = os.path.join(inp.models_dir, "Figures")
    os.makedirs(figure_folder, exist_ok=True)

    fig, axs = plt.subplots(3, 3, figsize=(8, 8))
    for i in range(min(9, A_split["X_test"].shape[0])):
        ix = i // 3
        iy = i % 3
        axs[ix, iy].plot(
            A_split["X_test"][i, 0, :],
            A_split["X_test"][i, 1, :],
            label="True",
        )
        axs[ix, iy].plot(
            A_split["X_test"][i, 0, :], mu_preds[i, :], label="pred"
        )
    plt.tight_layout()
    fig_file = "surr_preds"
    plt.savefig(os.path.join(inp.models_dir, "Figures", f"{fig_file}.pdf"))
    plt.close()


if __name__ == "__main__":
    import shutil
    import sys

    inp = ri.basic_input(sys.argv[1])
    inp = pre_inp(inp)
    test_perf(inp, mode="normal")
    plot_perf(inp)
