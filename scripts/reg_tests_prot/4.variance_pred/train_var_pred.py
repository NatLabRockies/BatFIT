"""
Train the amortized variance estimator (VariancePredFCNN) on the dataset
produced by gen_var_dataset.py.

Inputs  (scaled): P_train, Mu_train   — MinMax-scaled protocol and deg-param mean
Target: Sigma_train — NPE sigma averaged over noise realisations, in the
parameterisation chosen at dataset generation (see detect_sigma_mode):
  amp_par     — physical sigma; Sigmoid output converted via
                inv_transform_gamma before the MSE (historical default)
  scale_sigma — MinMax-scaled sigma; Sigmoid output is the direct target
  log_sigma   — z-scored log sigma; linear output is the direct target

Outputs written to inp.models_dir:
  model.pkl, model_<step>.pt, optimizer_<step>.pt
  train_loss.csv, test_loss.csv
  recipe.yml  (copy of the recipe used)
"""

import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import shutil
import sys

import numpy as np
import torch
import torch.nn as nn
from prettyPlot.progressBar import print_progress_bar

from batfit import logger
from batfit.basicutilityc import ReadInput as ri
from batfit.model.varianceNN import VariancePredFCNN
from batfit.utils.torch_utils import (
    get_device_type,
    get_num_parameters,
    log_training,
    prepare_log,
    save_model,
)


def detect_sigma_mode(var_pred_save_path: str) -> str:
    """Detect how sigma targets were parameterised by gen_var_dataset.py.

    The mode is inferred from which scaler file sits next to
    var_pred_dataset.npz, so a dataset directory is self-describing.

    :param var_pred_save_path: directory holding var_pred_dataset.npz
    :return: "log_sigma" (z-scored log sigma, linear head), "scale_sigma"
        (MinMax-scaled sigma, Sigmoid head), or "amp_par" (physical sigma,
        Sigmoid head + inv_transform_gamma)
    """
    has_log = os.path.isfile(
        os.path.join(var_pred_save_path, "scaler_logsigma.pkl")
    )
    has_minmax = os.path.isfile(
        os.path.join(var_pred_save_path, "scaler_sigma.pkl")
    )
    assert not (has_log and has_minmax), (
        f"Both scaler_logsigma.pkl and scaler_sigma.pkl found in "
        f"{var_pred_save_path}: target parameterisation is ambiguous. "
        "Regenerate the dataset in a fresh var_pred_save_path."
    )
    if has_log:
        return "log_sigma"
    if has_minmax:
        return "scale_sigma"
    return "amp_par"


def _lr_schedule(
    epoch: int, epoch_end: int, lr_beg: float, lr_end: float
) -> float:
    """Piecewise linear LR decay matching the NPE training schedule."""
    epoch_delay = epoch_end // 10
    if epoch < epoch_delay:
        return lr_beg
    return lr_beg * (lr_end / lr_beg) ** (
        min((epoch - epoch_delay) / epoch_end, 1.0)
    )


def make_data_loaders(
    inp,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Load the variance predictor dataset and build train/test DataLoaders.

    :return: (train_loader, test_loader) — each batch is (P, Mu, Sigma)
    """
    dataset_file = os.path.join(inp.var_pred_save_path, "var_pred_dataset.npz")
    assert os.path.isfile(dataset_file), (
        f"var_pred_dataset.npz not found at {dataset_file}; "
        "run gen_var_dataset.py first"
    )
    A = np.load(dataset_file)
    assert "P_train" in A, "P_train missing from var_pred_dataset.npz"
    assert "Mu_train" in A, "Mu_train missing from var_pred_dataset.npz"
    assert "Sigma_train" in A, "Sigma_train missing from var_pred_dataset.npz"
    assert "P_test" in A, "P_test missing from var_pred_dataset.npz"
    assert "Mu_test" in A, "Mu_test missing from var_pred_dataset.npz"
    assert "Sigma_test" in A, "Sigma_test missing from var_pred_dataset.npz"

    def _loader(p, mu, sigma, shuffle):
        ds = torch.utils.data.TensorDataset(
            torch.from_numpy(p),
            torch.from_numpy(mu),
            torch.from_numpy(sigma),
        )
        return torch.utils.data.DataLoader(
            ds,
            batch_size=min(inp.batch_size, p.shape[0]),
            shuffle=shuffle,
            drop_last=shuffle,
        )

    train_loader = _loader(
        A["P_train"], A["Mu_train"], A["Sigma_train"], shuffle=True
    )
    test_loader = _loader(
        A["P_test"], A["Mu_test"], A["Sigma_test"], shuffle=False
    )
    logger.info(
        f"Train: {A['P_train'].shape[0]} samples  |  "
        f"Test: {A['P_test'].shape[0]} samples"
    )
    return train_loader, test_loader


def define_model(inp) -> VariancePredFCNN:
    """Instantiate VariancePredFCNN from recipe parameters.

    :return: model (on CPU; moved to device inside train_model)
    """
    sigma_mode = detect_sigma_mode(inp.var_pred_save_path)
    model = VariancePredFCNN(
        n_prot=inp.n_prot_params,
        n_deg=inp.n_param_pred,
        hidden_list=inp.hidden_list,
        sim_config=inp.sim_config,
        output_activation=(
            "linear" if sigma_mode == "log_sigma" else "sigmoid"
        ),
    )
    logger.info(f"Trainable parameters: {get_num_parameters(model)}")
    return model


def train_model(
    inp,
    model: VariancePredFCNN,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
) -> None:
    """Train the variance predictor with Adamax and piecewise LR decay.

    The loss target follows detect_sigma_mode: in "scale_sigma" and
    "log_sigma" modes the raw network output is compared directly to the
    (already transformed) dataset targets; in "amp_par" mode
    inv_transform_gamma is applied to the Sigmoid output before computing
    MSE in physical sigma space.

    :param inp: recipe object
    :param model: VariancePredFCNN instance
    :param train_loader: DataLoader yielding (P, Mu, Sigma) batches
    :param test_loader: DataLoader for test-loss logging
    """
    device_type = get_device_type(enable_cuda=True, enable_mps=True)
    device = torch.device(device_type)
    model = model.to(device)
    amp_par = model.amp_par.to(device)

    # In scale_sigma/log_sigma modes the dataset targets are already
    # transformed — the raw network output is the direct prediction target.
    sigma_mode = detect_sigma_mode(inp.var_pred_save_path)
    logger.info(f"sigma_mode={sigma_mode}")

    mse = nn.MSELoss()
    lr_end = inp.lr / 100.0
    optimizer = torch.optim.Adamax(
        model.parameters(), lr=inp.lr, weight_decay=1e-5
    )

    prepare_log(inp.models_dir)
    save_model(
        step=0,
        model=model,
        log_folder=inp.models_dir,
        save_model_obj=True,
        save_model_weights=False,
        save_model_opt=False,
    )

    num_batch = len(train_loader)
    total_steps = num_batch * inp.epochs
    log_freq = max(total_steps // 1000, 1)
    save_freq = max(total_steps // 70, 1)  # ~70 checkpoints like the NPE

    print_progress_bar(
        0,
        total_steps,
        prefix=f"Loss = ? Step 0 / {total_steps} ",
        suffix="Complete",
        length=50,
    )

    current_step = 0
    model.train()
    for epoch in range(inp.epochs):
        for param_group in optimizer.param_groups:
            param_group["lr"] = _lr_schedule(
                epoch, inp.epochs * 3 // 4, inp.lr, lr_end
            )

        for p_batch, mu_batch, sigma_batch in train_loader:
            current_step += 1
            optimizer.zero_grad()

            sigma_out = model(p_batch.to(device), mu_batch.to(device))
            if sigma_mode == "amp_par":
                sigma_pred = model.inv_transform_gamma(sigma_out, amp_par)
            else:
                sigma_pred = sigma_out
            loss = mse(sigma_pred, sigma_batch.to(device))

            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                optimizer.step()

            logged = False
            if current_step % save_freq == 0:
                logged = True
                log_training(
                    current_step,
                    loss,
                    inp.models_dir,
                    filename="train_loss.csv",
                )
                save_model(
                    step=current_step,
                    model=model,
                    optimizer=optimizer,
                    device_type=device_type,
                    log_folder=inp.models_dir,
                )
            elif current_step % log_freq == 0 and not logged:
                log_training(
                    current_step,
                    loss,
                    inp.models_dir,
                    filename="train_loss.csv",
                )

            print_progress_bar(
                current_step,
                total_steps,
                prefix=f"Loss = {loss.item():.4g} Step {current_step} / {total_steps} ",
                suffix="Complete",
                length=50,
            )

        # Test loss at end of each epoch
        model.eval()
        test_loss_acc, n_test = 0.0, 0
        with torch.no_grad():
            for p_batch, mu_batch, sigma_batch in test_loader:
                sigma_out = model(p_batch.to(device), mu_batch.to(device))
                if sigma_mode == "amp_par":
                    sigma_pred = model.inv_transform_gamma(
                        sigma_out, amp_par
                    )
                else:
                    sigma_pred = sigma_out
                b = p_batch.shape[0]
                test_loss_acc += (
                    mse(sigma_pred, sigma_batch.to(device)).item() * b
                )
                n_test += b
        log_training(
            current_step,
            test_loss_acc / n_test,
            inp.models_dir,
            filename="test_loss.csv",
        )
        model.train()

    save_model(
        step=total_steps,
        model=model,
        optimizer=optimizer,
        device_type=device_type,
        log_folder=inp.models_dir,
        bypass="final",
    )
    logger.info(f"Training complete. Model saved to {inp.models_dir}")


if __name__ == "__main__":
    inp = ri.basic_input(sys.argv[1])
    train_loader, test_loader = make_data_loaders(inp)
    model = define_model(inp)
    train_model(inp, model, train_loader, test_loader)
    shutil.copy(sys.argv[1], os.path.join(inp.models_dir, "recipe.yml"))
