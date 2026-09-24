import pickle

import numpy as np
import optuna
import torch
from prettyPlot.progressBar import print_progress_bar

from batfit import logger
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


def create_model_from_log(model_obj_file, model_state_dict_file, verbose=True):
    if verbose:
        logger.info(
            f"loading model from \n\t{model_obj_file} and {model_state_dict_file}"
        )
    with open(model_obj_file, "rb") as f:
        model = pickle.load(f)
    num_parameters = get_num_parameters(model)
    if verbose:
        print(f"\tNo. Trainable Parameters: {num_parameters}")
    if model_state_dict_file is not None:
        model = load_model(
            model, model_state_dict_file, enable_cuda=False, enable_mps=False
        )
    return model


def learning_rate_schedule(epoch, epoch_end, lr_beg, lr_end):
    # a run shorter than 2 epochs gives epoch_end = 0 (num_epochs * 3 // 4)
    epoch_end = max(epoch_end, 1)
    epoch_delay = epoch_end // 10
    if epoch < epoch_delay:
        return lr_beg
    else:
        return lr_beg * (lr_end / lr_beg) ** (
            min((epoch - epoch_delay) / epoch_end, 1.0)
        )


def train_model(
    model: torch.nn.Module,
    train_data_loader: torch.utils.data.DataLoader,
    learning_rate: float,
    num_epochs: int | None,
    learning_rate_end: float | None = None,
    test_data_loader: torch.utils.data.DataLoader | None = None,
    num_steps: int | None = None,
    num_steps_test: int | None = None,
    log_folder: str = "train_log",
    log_freq: int = 100,
    save_freq: int = 1_000_000,
    restart_from: str | None = None,
    enable_cuda: bool = True,
    enable_mps: bool = True,
    trial=None,
):
    # Device set up
    device_type = get_device_type(
        enable_cuda=enable_cuda, enable_mps=enable_mps
    )
    device = torch.device(device_type)

    # Save the model config
    save_model(
        step=0,
        model=model,
        log_folder=log_folder,
        save_model_obj=True,
        save_model_weights=False,
        save_model_opt=False,
    )

    if learning_rate_end is None:
        learning_rate_end = learning_rate / 100.0

    print("Device = ", device)
    model = model.to(device)

    loss_hist = np.array([])
    optimizer = torch.optim.Adamax(
        model.parameters(), lr=learning_rate, weight_decay=1e-5
    )

    num_batch = len(train_data_loader)

    # Optionally restart from a checkpoint (weights only; the optimizer and LR
    # schedule start fresh). Epoch/step counters resume from an existing loss
    # CSV so the appended log stays monotonic, and the best test loss is seeded
    # by re-evaluating the loaded checkpoint so a worse epoch cannot overwrite a
    # good ``model_best.pt``.
    best_test_loss = float("inf")
    if restart_requested(restart_from):
        model = load_model(model, restart_from, device_type=device_type)
        start_epoch, start_step = read_restart_position(log_folder)
        prepare_log(log_folder, append=start_epoch > 0)
        if test_data_loader is not None:
            seed_loss = compute_test_loss(
                model=model,
                test_data_loader=test_data_loader,
                num_steps=num_steps_test,
                enable_cuda=enable_cuda,
                enable_mps=enable_mps,
                verbose=False,
            )
            best_test_loss = update_best_model(
                seed_loss,
                best_test_loss,
                model,
                device_type=device_type,
                log_folder=log_folder,
            )
    else:
        start_epoch, start_step = 0, 0
        prepare_log(log_folder)

    model.train()

    if num_steps is not None:
        num_epochs = num_steps // num_batch + 1
        total_steps = start_epoch * num_batch + num_steps
    else:
        total_steps = (start_epoch + num_epochs) * num_batch
    end_epoch = start_epoch + num_epochs
    # train
    print_progress_bar(
        0,
        total_steps,
        prefix=f"Loss = ? Step 0 / {total_steps} ",
        suffix="Complete",
        length=50,
    )

    current_step = start_step
    for epoch in range(start_epoch, end_epoch):
        # Set LR for this epoch. On a restart the schedule restarts from
        # ``learning_rate`` and decays over the additional run, keyed on the
        # local index ``epoch - start_epoch``.
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate_schedule(
                epoch - start_epoch,
                num_epochs * 3 // 4,
                learning_rate,
                learning_rate_end,
            )

        for step, batch in enumerate(train_data_loader):
            current_step = epoch * num_batch + (step + 1)
            # Reinitialize grads
            optimizer.zero_grad()
            batch_in = batch[0]
            # Compute loss
            try:
                # loss in the scaled voltage space
                pred = model(batch_in.to(device))
                loss = model.loss_fn(pred, batch[1].to(device))
                # Do backprop and optimizer step
                if ~(torch.isnan(loss) | torch.isinf(loss)):
                    loss.backward()
                    optimizer.step()
            except (torch.OutOfMemoryError, RuntimeError) as err:
                if trial is not None:
                    # Make sure hyper par tuning can proceed
                    raise optuna.exceptions.TrialPruned()
                else:
                    raise err

            # Log loss
            loss_hist = np.append(loss_hist, loss.detach().to("cpu").numpy())
            logged = False
            if current_step % save_freq == 0:
                logged = True
                log_training(
                    current_step, loss, log_folder, filename="train_loss.csv"
                )
                save_model(
                    step=current_step,
                    model=model,
                    device_type=device_type,
                    log_folder=log_folder,
                )
            elif current_step % log_freq == 0 and not logged:
                log_training(
                    current_step, loss, log_folder, filename="train_loss.csv"
                )

            logged = False

            print_progress_bar(
                current_step,
                total_steps,
                prefix=f"Loss = {loss.item():.4g} Step {current_step} / {total_steps} ",
                suffix="Complete",
                length=50,
            )

            if current_step >= total_steps:
                break
            if trial is not None:
                # Handle pruning based on the intermediate value.
                if (
                    trial.should_prune()
                    or np.isnan(loss.item())
                    or np.isinf(loss.item())
                ):
                    raise optuna.exceptions.TrialPruned()

        if test_data_loader is not None:
            test_loss = compute_test_loss(
                model=model,
                test_data_loader=test_data_loader,
                num_steps=num_steps_test,
                enable_cuda=enable_cuda,
                enable_mps=enable_mps,
                verbose=False,
            )
            log_training(
                current_step,
                test_loss,
                log_folder,
                filename="test_loss.csv",
            )
            best_test_loss = update_best_model(
                test_loss,
                best_test_loss,
                model,
                device_type=device_type,
                log_folder=log_folder,
            )
            model.train()
        else:
            test_loss = None
        if trial is not None:
            if test_loss is not None:
                trial.report(test_loss, epoch)
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    save_model(
        step=total_steps,
        model=model,
        device_type=device_type,
        log_folder=log_folder,
        bypass="final",
    )
    return model, loss_hist


def compute_test_loss(
    model,
    test_data_loader: torch.utils.data.DataLoader,
    num_steps: int | None = None,
    enable_cuda: bool = True,
    enable_mps: bool = True,
    verbose=True,
):
    # Device set up
    device_type = get_device_type(
        enable_cuda=enable_cuda, enable_mps=enable_mps
    )
    device = torch.device(device_type)
    if verbose:
        print("Device = ", device)

    model = model.to(device)
    num_batch_test = len(test_data_loader)

    model.eval()
    if num_steps is not None:
        total_steps = num_steps
    else:
        total_steps = num_batch_test
    # eval loop
    if verbose:
        print_progress_bar(
            0,
            total_steps,
            prefix=f"Test Loss = ? Step 0 / {total_steps} ",
            suffix="Complete",
            length=50,
        )

    loss_ave = 0
    num_el = 0
    with torch.no_grad():
        for step, batch in enumerate(test_data_loader):
            current_step = step + 1
            batch_in = batch[0]
            # Compute loss
            # loss in the scaled voltage space
            pred = model(batch_in.to(device))
            loss = model.loss_fn(pred, batch[1].to(device))
            loss_ave += loss.item() * batch_in.shape[0]
            num_el += batch_in.shape[0]
            if verbose:
                print_progress_bar(
                    current_step,
                    total_steps,
                    prefix=f"Test loss = {loss_ave/current_step:.4g} Step {current_step} / {total_steps} ",
                    suffix="Complete",
                    length=50,
                )
            if current_step >= total_steps:
                break
        loss_ave /= num_el
    return loss_ave
