import pickle
import sys

import numpy as np
import optuna
import torch
import torch.distributions as dist
from prettyPlot.progressBar import print_progress_bar

from batfit import logger
from batfit.utils.data_utils import (
    scale_input_from_scaler,
    unscale_dataset_from_scaler,
)
from batfit.utils.torch_utils import (
    get_device_type,
    get_num_parameters,
    load_model,
    log_training,
    prepare_log,
    save_model,
)

from batfit.model.ae import AECNN
from .losses import mse_loss
from .metrics import *
from .noise_utils import apply_noise
from batfit.model.paramNN import ProbParamCNN, ProbParamFCNN
from batfit.model.vae import VAECNN

def create_model_from_log(model_obj_file, model_state_dict_file, verbose=True):
    if verbose:
        logger.info(
            f"loading model from \n\t{model_obj_file} and {model_state_dict_file}"
        )
    with open(model_obj_file, "rb") as f:
        model = pickle.load(f)
    if not hasattr(model, "dependent_outputs"):
        model.dependent_outputs = False
    num_parameters = get_num_parameters(model)
    if verbose:
        print(f"\tNo. Trainable Parameters: {num_parameters}")
    if model_state_dict_file is not None:
        model = load_model(
            model, model_state_dict_file, enable_cuda=False, enable_mps=False
        )
    return model


def forward_pass(model, np_data_in, scaler_X_file, scaler_Y_file, scale_y):
    model.eval()
    model.to("cpu")

    X_scaled = scale_input_from_scaler(np_data_in, scaler_X_file)
    with torch.no_grad():
        if isinstance(model, ProbParamCNN) or isinstance(model, ProbParamFCNN):
            pred_scaled, gamma_scaled = model(torch.from_numpy(X_scaled))
            if model.constrain_output and not model.dependent_outputs:
                pred_unscaled, gamma_unscaled = model.inv_transform_output(
                    pred_scaled,
                    gamma_scaled,
                    model.min_par.to("cpu"),
                    model.amp_par.to("cpu"),
                )

            elif model.constrain_output and model.dependent_outputs:
                pred_unscaled = model.inv_transform_mu(
                    pred_scaled,
                    model.min_par.to("cpu"),
                    model.amp_par.to("cpu"),
                )
                # gamma_unscaled = gamma_scaled
                gamma_unscaled = torch.sqrt(
                    gamma_scaled.diagonal(dim1=1, dim2=2)
                )
            elif not scale_y:
                pred_unscaled = pred_scaled
                gamma_unscaled = gamma_scaled
            elif scale_y:
                raise NotImplementedError
            else:
                raise NotImplementedError
            pred_unscaled = pred_unscaled.numpy()
            gamma_unscaled = gamma_unscaled.numpy()
            inp_unscaled, _ = unscale_dataset_from_scaler(
                X_scaled, pred_scaled, scaler_X_file, scaler_Y_file
            )
            probabilistic = True
        else:
            raise NotImplementedError

    if probabilistic:
        return (pred_unscaled, gamma_unscaled)
    else:
        return pred_unscaled

def learning_rate_schedule(epoch, epoch_end, lr_beg, lr_end):
    epoch_delay = epoch_end // 10
    if epoch < epoch_delay:
        return lr_beg
    else:
        return lr_beg * (lr_end / lr_beg) ** (
            min((epoch - epoch_delay) / epoch_end, 1.0)
        )


def temp_schedule(epoch, epoch_beg, epoch_end, val_beg, val_end):
    return val_beg + min(
        (epoch - epoch_beg) / (epoch_end - epoch_beg), 1.0
    ) * (val_end - val_beg)


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
    save_freq: int = 1000,
    optimizer_state_dict_filename: str | None = None,
    enable_cuda: bool = True,
    enable_mps: bool = True,
    trial=None,
    noise_levels: torch.Tensor | None = torch.tensor([0, 0.010, 0.04, 1]),
    bias_tensor: torch.Tensor | None = None,
    scaler_X=None,
    a_min: torch.Tensor | None = torch.tensor(
        [-torch.inf, 3, -torch.inf, -torch.inf]
    ),
    a_max: torch.Tensor | None = torch.tensor([torch.inf, 4.1, 0, 0]),
    target_mode: None | str = None,
    prior=None,
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

    if target_mode != "encoded":
        if len(noise_levels.shape) == 1:
            noise_levels = torch.reshape(
                noise_levels, (1, noise_levels.shape[0], 1)
            )
        if len(a_min.shape) == 1:
            a_min = torch.reshape(a_min, (1, a_min.shape[0], 1))
        if len(a_max.shape) == 1:
            a_max = torch.reshape(a_max, (1, a_max.shape[0], 1))

    if learning_rate_end is None:
        learning_rate_end = learning_rate / 100.0

    print("Device = ", device)
    model = model.to(device)
    if model.encoder_model is not None:
        model.encoder_model = model.encoder_model.to(device)

    loss_hist = np.array([])
    optimizer = torch.optim.Adamax(
        model.parameters(), lr=learning_rate, weight_decay=1e-5
    )
    if optimizer_state_dict_filename is not None:
        optimizer.load_state_dict(
            torch.load(optimizer_state_dict_filename, weights_only=True)
        )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #    optimizer, num_epochs, 0
    # )

    num_batch = len(train_data_loader)
    model.train()

    prepare_log(log_folder)
    if num_steps is not None:
        total_steps = num_steps
        num_epochs = num_steps // num_batch + 1
    else:
        total_steps = num_batch * num_epochs
    # train
    print_progress_bar(
        0,
        total_steps,
        prefix=f"Loss = ? Step 0 / {total_steps} ",
        suffix="Complete",
        length=50,
    )

    current_step=0
    for epoch in range(num_epochs):
        # Set LR for this epoch
        temp = temp_schedule(epoch, 0, num_epochs * 3 // 4, 0.1, 1.0)
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate_schedule(
                epoch, num_epochs * 3 // 4, learning_rate, learning_rate_end
            )
       
        for step, batch in enumerate(train_data_loader):
            current_step = epoch * num_batch + (step + 1)
            # Reinitialize grads
            optimizer.zero_grad()
            # Add noise to batch

            if target_mode != "encoded":
                batch_in = apply_noise(
                    batch_in=batch[0],
                    scaler_X=scaler_X,
                    noise_levels=noise_levels,
                    a_min=a_min,
                    a_max=a_max,
                    bias=bias_tensor,
                )
            else:
                batch_in = batch[0]
            if model.encoder_model is not None:
                if isinstance(model.encoder_model, VAECNN):
                    batch_in, _, _ = model.encoder_model.encode(
                        batch_in.to(device)
                    )
                elif isinstance(model.encoder_model, AECNN):
                    batch_in = model.encoder_model.encode(batch_in.to(device))

            # Compute loss
            try:
                if isinstance(model, ProbParamCNN) or isinstance(
                    model, ProbParamFCNN
                ):
                    mu, gamma = model(batch_in.to(device))
                    if model.constrain_output and model.dependent_outputs:
                        mu = model.inv_transform_mu(
                            mu,
                            model.min_par.to(device),
                            model.amp_par.to(device),
                        )
                    elif (
                        model.constrain_output and not model.dependent_outputs
                    ):
                        mu, gamma = model.inv_transform_output(
                            mu,
                            gamma,
                            model.min_par.to(device),
                            model.amp_par.to(device),
                        )
                    loss = model.loss_fn(mu, gamma, batch[1].to(device))
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
                    optimizer=optimizer,
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
                noise_levels=noise_levels,
                scaler_X=scaler_X,
                a_min=a_min,
                a_max=a_max,
                target_mode=target_mode,
                prior=prior,
            )
            log_training(
                current_step,
                test_loss,
                log_folder,
                filename="test_loss.csv",
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
        optimizer=optimizer,
        device_type=device_type,
        log_folder=log_folder,
        bypass="final",
    )
    return model, loss_hist


def compute_test_loss(
    model: ProbParamCNN,
    test_data_loader: torch.utils.data.DataLoader,
    num_steps: int | None = None,
    enable_cuda: bool = True,
    enable_mps: bool = True,
    verbose=True,
    noise_levels: torch.Tensor | None = torch.tensor([0, 0.010, 0.04, 1]),
    bias_tensor: torch.Tensor | None = None,
    scaler_X=None,
    a_min: torch.Tensor | None = torch.tensor(
        [-torch.inf, 3, -torch.inf, -torch.inf]
    ),
    a_max: torch.Tensor | None = torch.tensor([torch.inf, 4.1, 0, 0]),
    target_mode: None | str = None,
    prior=None,
):
    # Device set up
    device_type = get_device_type(
        enable_cuda=enable_cuda, enable_mps=enable_mps
    )
    device = torch.device(device_type)
    if verbose:
        print("Device = ", device)

    if target_mode != "encoded":
        if len(noise_levels.shape) == 1:
            noise_levels = torch.reshape(
                noise_levels, (1, noise_levels.shape[0], 1)
            )
        if len(a_min.shape) == 1:
            a_min = torch.reshape(a_min, (1, a_min.shape[0], 1))
        if len(a_max.shape) == 1:
            a_max = torch.reshape(a_max, (1, a_max.shape[0], 1))

    model = model.to(device)
    if model.encoder_model is not None:
        model.encoder_model = model.encoder_model.to(device)
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
            # Add noise to batch
            if target_mode != "encoded":
                batch_in = apply_noise(
                    batch_in=batch[0],
                    scaler_X=scaler_X,
                    noise_levels=noise_levels,
                    a_min=a_min,
                    a_max=a_max,
                    bias=bias_tensor,
                )
            else:
                batch_in = batch[0]
            if model.encoder_model is not None:
                if isinstance(model.encoder_model, AECNN):
                    batch_in = model.encoder_model.encode(batch_in.to(device))
                elif isinstance(model.encoder_model, VAECNN):
                    batch_in, _, _ = model.encoder_model.encode(
                        batch_in.to(device)
                    )
            # Compute loss
            if isinstance(model, ProbParamCNN) or isinstance(
                model, ProbParamFCNN
            ):
                mu, gamma = model(batch_in.to(device))
                if model.constrain_output and model.dependent_outputs:
                    mu = model.inv_transform_mu(
                        mu,
                        model.min_par.to(device),
                        model.amp_par.to(device),
                    )
                elif model.constrain_output and not model.dependent_outputs:
                    mu, gamma = model.inv_transform_output(
                        mu,
                        gamma,
                        model.min_par.to(device),
                        model.amp_par.to(device),
                    )

                loss = model.loss_fn(mu, gamma, batch[1].to(device))
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


def compute_post(
    model: ProbParamCNN,
    test_data_loader: torch.utils.data.DataLoader,
    num_steps: int | None = None,
    enable_cuda: bool = True,
    enable_mps: bool = True,
    verbose=True,
    noise_levels: torch.Tensor | None = torch.tensor([0, 0.010, 0.04, 1]),
    scaler_X=None,
    a_min: torch.Tensor | None = torch.tensor(
        [-torch.inf, 3, -torch.inf, -torch.inf]
    ),
    a_max: torch.Tensor | None = torch.tensor([torch.inf, 4.1, 0, 0]),
    post_fn=rel_accuracy,
    target_mode: str | None = None,
):
    # Device set up
    device_type = get_device_type(
        enable_cuda=enable_cuda, enable_mps=enable_mps
    )
    device = torch.device(device_type)
    if verbose:
        print("Device = ", device)

    if target_mode != "encoded":
        if len(noise_levels.shape) == 1:
            noise_levels = torch.reshape(
                noise_levels, (1, noise_levels.shape[0], 1)
            )
        if len(a_min.shape) == 1:
            a_min = torch.reshape(a_min, (1, a_min.shape[0], 1))
        if len(a_max.shape) == 1:
            a_max = torch.reshape(a_max, (1, a_max.shape[0], 1))

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
            prefix=f"Post = ? Step 0 / {total_steps} ",
            suffix="Complete",
            length=50,
        )

    post_val_ave = 0
    num_el = 0
    with torch.no_grad():
        for step, batch in enumerate(test_data_loader):
            current_step = step + 1
            if target_mode != "encoded":
                # Add noise to batch
                batch_in = apply_noise(
                    batch_in=batch[0],
                    scaler_X=scaler_X,
                    noise_levels=noise_levels,
                    a_min=a_min,
                    a_max=a_max,
                )
            else:
                batch_in = batch[0]
            # Compute loss
            if isinstance(model, ProbParamCNN) or isinstance(
                model, ProbParamFCNN
            ):
                mu, gamma = model(batch_in.to(device))
                if model.constrain_output and model.dependent_outputs:
                    mu = model.inv_transform_mu(
                        mu,
                        model.min_par.to(device),
                        model.amp_par.to(device),
                    )
                elif model.constrain_output and not model.dependent_outputs:
                    mu, gamma = model.inv_transform_output(
                        mu,
                        gamma,
                        model.min_par.to(device),
                        model.amp_par.to(device),
                    )
                if post_fn in [accuracy, rel_accuracy]:
                    post_val = post_fn(mu, batch[1].to(device))
                elif post_fn in [identifiability]:
                    post_val = 1.0 / post_fn(gamma)
            post_val_ave += post_val * batch_in.shape[0]
            num_el += batch_in.shape[0]
            if verbose:
                print_progress_bar(
                    current_step,
                    total_steps,
                    prefix=f"Post, Step {current_step} / {total_steps} ",
                    suffix="Complete",
                    length=50,
                )
            if current_step >= total_steps:
                break
    post_val_ave /= num_el
    if post_fn in [identifiability]:
        post_val_ave = 1.0 / post_val_ave
    return post_val_ave
