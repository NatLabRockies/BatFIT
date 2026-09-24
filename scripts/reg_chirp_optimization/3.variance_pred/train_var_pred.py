"""
Train the amortized variance estimator (VariancePredFCNN) on the dataset
produced by gen_var_dataset.py.
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
from batfit.utils.scalers import ZScoreScaler
from batfit.utils.torch_utils import (
    get_device_type,
    get_num_parameters,
    load_model,
    log_training,
    prepare_log,
    read_restart_position,
    restart_requested,
    save_model,
    update_best_model,
)


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


def load_dataset(inp) -> dict[str, np.ndarray]:
    """Load the physical (P, mu, sigma) splits from var_pred_dataset.npz."""
    dataset_file = os.path.join(inp.var_pred_save_path, "var_pred_dataset.npz")
    assert os.path.isfile(dataset_file), (
        f"var_pred_dataset.npz not found at {dataset_file}; "
        "run gen_var_dataset.py first"
    )
    A = np.load(dataset_file)
    for split in ("train", "test"):
        for name in ("P", "Mu", "Sigma"):
            key = f"{name}_{split}"
            assert key in A, f"{key} missing from var_pred_dataset.npz"
    return {key: A[key] for key in A.files}


def make_data_loaders(
    inp, data: dict[str, np.ndarray], model: VariancePredFCNN
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Scale the dataset with the model's scalers and build DataLoaders.

    P and mu are scaled to [0, 1] from the config bounds; the targets are the
    z-scored log sigma.
    """

    def _loader(split: str, shuffle: bool) -> torch.utils.data.DataLoader:
        # in place on the loaded arrays: no copy of P and mu
        p = model.scaler_P.transform_(data[f"P_{split}"])
        mu = model.scaler_Y.transform_(data[f"Mu_{split}"])
        log_sigma = np.log(data[f"Sigma_{split}"])
        sigma_target = model.scaler_logsigma.transform_(log_sigma)
        ds = torch.utils.data.TensorDataset(
            torch.from_numpy(p),
            torch.from_numpy(mu),
            torch.from_numpy(sigma_target),
        )
        return torch.utils.data.DataLoader(
            ds,
            batch_size=min(inp.batch_size, p.shape[0]),
            shuffle=shuffle,
            drop_last=shuffle,
        )

    train_loader = _loader("train", shuffle=True)
    test_loader = _loader("test", shuffle=False)
    logger.info(
        f"Train: {data['P_train'].shape[0]} samples  |  "
        f"Test: {data['P_test'].shape[0]} samples"
    )
    return train_loader, test_loader


def define_model(
    inp, scaler_logsigma: ZScoreScaler | None = None
) -> VariancePredFCNN:
    """Instantiate VariancePredFCNN; scaler_logsigma=None leaves a placeholder
    that load_state_dict fills from the checkpoint."""
    model = VariancePredFCNN(
        hidden_list=inp.hidden_list,
        sim_config=inp.sim_config,
        scaler_logsigma=scaler_logsigma,
    )
    logger.info(f"Trainable parameters: {get_num_parameters(model)}")
    return model


def train_model(
    inp,
    model: VariancePredFCNN,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
) -> None:
    device_type = get_device_type(enable_cuda=True, enable_mps=True)
    device = torch.device(device_type)
    model = model.to(device)

    # loss on the z-scored log sigma: MSE then acts as a relative error
    mse = nn.MSELoss()
    lr_end = inp.lr / 100.0
    optimizer = torch.optim.Adamax(
        model.parameters(), lr=inp.lr, weight_decay=1e-5
    )

    # Save the model config (architecture) before training.
    save_model(
        step=0,
        model=model,
        log_folder=inp.models_dir,
        save_model_obj=True,
        save_model_weights=False,
        save_model_opt=False,
    )

    def _eval_test_loss() -> float:
        """Return the mean test-set MSE on the z-scored log sigma."""
        model.eval()
        test_loss_acc, n_test = 0.0, 0
        with torch.no_grad():
            for p_batch, mu_batch, sigma_batch in test_loader:
                sigma_pred = model(p_batch.to(device), mu_batch.to(device))
                b = p_batch.shape[0]
                test_loss_acc += (
                    mse(sigma_pred, sigma_batch.to(device)).item() * b
                )
                n_test += b
        model.train()
        return test_loss_acc / n_test

    num_batch = len(train_loader)
    save_freq = 1_000_000  # periodic checkpoints effectively disabled

    # Optionally restart from a checkpoint (weights only; the optimizer and LR
    # schedule start fresh). Epoch/step counters resume from an existing loss
    # CSV so the appended log stays monotonic, and the best test loss is seeded
    # by re-evaluating the loaded checkpoint.
    restart_from = getattr(inp, "restart_from", None) or None
    best_test_loss = float("inf")
    if restart_requested(restart_from):
        model = load_model(model, restart_from, device_type=device_type)
        start_epoch, start_step = read_restart_position(inp.models_dir)
        prepare_log(inp.models_dir, append=start_epoch > 0)
        best_test_loss = update_best_model(
            _eval_test_loss(),
            best_test_loss,
            model,
            device_type=device_type,
            log_folder=inp.models_dir,
        )
    else:
        start_epoch, start_step = 0, 0
        prepare_log(inp.models_dir)

    total_steps = start_step + num_batch * inp.epochs
    log_freq = max(total_steps // 1000, 1)

    print_progress_bar(
        0,
        total_steps,
        prefix=f"Loss = ? Step 0 / {total_steps} ",
        suffix="Complete",
        length=50,
    )

    current_step = start_step
    model.train()
    for epoch in range(start_epoch, start_epoch + inp.epochs):
        # On a restart the LR schedule restarts from inp.lr and decays over the
        # additional run, keyed on the local index ``epoch - start_epoch``.
        for param_group in optimizer.param_groups:
            param_group["lr"] = _lr_schedule(
                epoch - start_epoch, inp.epochs * 3 // 4, inp.lr, lr_end
            )

        for p_batch, mu_batch, sigma_batch in train_loader:
            current_step += 1
            optimizer.zero_grad()

            sigma_pred = model(p_batch.to(device), mu_batch.to(device))
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
        test_loss = _eval_test_loss()
        log_training(
            current_step,
            test_loss,
            inp.models_dir,
            filename="test_loss.csv",
        )
        best_test_loss = update_best_model(
            test_loss,
            best_test_loss,
            model,
            device_type=device_type,
            log_folder=inp.models_dir,
        )

    save_model(
        step=total_steps,
        model=model,
        device_type=device_type,
        log_folder=inp.models_dir,
        bypass="final",
    )
    logger.info(f"Training complete. Model saved to {inp.models_dir}")


if __name__ == "__main__":
    inp = ri.basic_input(sys.argv[1])
    data = load_dataset(inp)
    # the log-sigma z-score is fitted on the training sigmas
    scaler_logsigma = ZScoreScaler.fit(np.log(data["Sigma_train"]), axis=0)
    model = define_model(inp, scaler_logsigma=scaler_logsigma)
    train_loader, test_loader = make_data_loaders(inp, data, model)
    train_model(inp, model, train_loader, test_loader)
    shutil.copy(sys.argv[1], os.path.join(inp.models_dir, "recipe.yml"))
