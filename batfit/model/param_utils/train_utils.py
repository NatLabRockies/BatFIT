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

from .noise_utils import apply_noise


def create_model_from_log(
    model_obj_file: str,
    model_state_dict_file: str | None,
    verbose: bool = True,
) -> torch.nn.Module:
    """Reconstruct a model from a pickled object and optional weights.

    Loads the pickled model architecture from ``model_obj_file`` and, when
    ``model_state_dict_file`` is not None, loads that state dict

    Parameters
    ----------
    model_obj_file: str
        Path to the pickled model object (``model.pkl``)
    model_state_dict_file: str | None
        Path to the ``.pt`` weights to load, or None to return the
        freshly-unpickled model
    verbose: bool
        Log the loaded files and parameter count

    Returns
    -------
    torch.nn.Module
        The reconstructed model
    """
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


def _noisy_input(
    model: torch.nn.Module,
    signal: torch.Tensor,
    noise_levels: torch.Tensor,
    a_min: torch.Tensor,
    a_max: torch.Tensor,
    bias_tensor: torch.Tensor | None,
    target_mode: str | None,
    device: torch.device,
) -> torch.Tensor:
    """Noise a scaled signal batch in physical space and encode it.

    Noise is skipped when ``target_mode == "encoded"`` (pre-encoded data).
    """
    if target_mode != "encoded":
        signal = apply_noise(
            batch_in=signal,
            scaler_X=model.scaler_X,
            noise_levels=noise_levels,
            a_min=a_min,
            a_max=a_max,
            bias=bias_tensor,
        )
    return model._encode(signal.to(device))


def unpack_npe_batch(
    model: torch.nn.Module,
    batch: list[torch.Tensor],
    device: torch.device,
) -> tuple[
    torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor
]:
    """Split an NPE batch into signal, protocol, end time and labels.

    Batches are ``(X, Y)``, ``(X, P, Y)``, ``(X, T, Y)`` or ``(X, P, T, Y)``
    (see :func:`batfit.utils.torch_dataset_builder.make_npe_dataset_from_np`):
    P is present for protocol models, T with
    ``signal_scaling="time_dependent_zscore"``, labels are always last.

    Parameters
    ----------
    model: torch.nn.Module
        NPE model, providing ``scaler_P`` and ``with_end_time``
    batch: list[torch.Tensor]
        Batch from an NPE DataLoader
    device: torch.device
        Device to move the protocol parameters, end time and labels to

    Returns
    -------
    tuple
        ``(x, p, t_end, y)``: the signal (left on its device, to be noised),
        the protocol parameters and end time (None when unused) and the
        labels
    """
    p = batch[1].to(device) if model.scaler_P is not None else None
    t_end = batch[-2].to(device) if model.with_end_time else None
    return batch[0], p, t_end, batch[-1].to(device)


def _gauss_npe_batch_loss(
    model: torch.nn.Module,
    batch: list[torch.Tensor],
    batch_in: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Compute the loss of one batch in the scaled parameter space.

    ``batch_in`` is the noised signal; the other inputs are read from
    ``batch`` with :func:`unpack_npe_batch`.
    """
    _, p, t_end, y = unpack_npe_batch(model, batch, device)
    inputs = [batch_in] if p is None else [batch_in, p]
    mu, gamma = model(*inputs, t_end=t_end)
    return model.loss_fn(mu, gamma, y)


def _reshape_noise_args(
    noise_levels: torch.Tensor,
    a_min: torch.Tensor,
    a_max: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reshape 1D per-channel noise levels and clip bounds to ``(1, C, 1)``."""
    assert a_min is not None and a_max is not None, (
        "a_min/a_max are required: build them with make_noise_levels and "
        "the vmin/vmax of the experiment config"
    )
    reshaped = []
    for tens in (noise_levels, a_min, a_max):
        if len(tens.shape) == 1:
            tens = torch.reshape(tens, (1, tens.shape[0], 1))
        reshaped.append(tens)
    return tuple(reshaped)


def learning_rate_schedule(
    epoch: int, epoch_end: int, lr_beg: float, lr_end: float
) -> float:
    """Piecewise learning-rate schedule.

    Use ``lr_beg`` for the first ``epoch_end // 10`` epochs
    Decays geometrically from ``lr_beg`` toward ``lr_end``

    Parameters
    ----------
    epoch: int
        Current epoch
    epoch_end: int
        Epoch at which the decay reaches ``lr_end``
    lr_beg: float
        Initial learning rate
    lr_end: float
        Final learning rate

    Returns
    -------
    float
        Learning rate for ``epoch``
    """
    # a run shorter than 2 epochs gives epoch_end = 0 (num_epochs * 3 // 4)
    epoch_end = max(epoch_end, 1)
    epoch_delay = epoch_end // 10
    if epoch < epoch_delay:
        return lr_beg
    else:
        return lr_beg * (lr_end / lr_beg) ** (
            min((epoch - epoch_delay) / epoch_end, 1.0)
        )


def temp_schedule(
    epoch: int,
    epoch_beg: int,
    epoch_end: int,
    val_beg: float,
    val_end: float,
) -> float:
    """Schedule of tempering value (for vae)

    The ramp runs between ``epoch_beg`` and ``epoch_end`` and is clamped to
    ``val_end`` afterwards.

    Parameters
    ----------
    epoch: int
        Current epoch
    epoch_beg: int
        Epoch at which the ramp starts
    epoch_end: int
        Epoch at which the ramp reaches ``val_end``
    val_beg: float
        Initial value
    val_end: float
        Final value

    Returns
    -------
    float
        Interpolated value for ``epoch``
    """
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
    save_freq: int = 1_000_000,
    restart_from: str | None = None,
    enable_cuda: bool = True,
    enable_mps: bool = True,
    trial: optuna.trial.Trial | None = None,
    noise_levels: torch.Tensor | None = torch.tensor([0, 0.010, 0.04, 1]),
    bias_tensor: torch.Tensor | None = None,
    a_min: torch.Tensor | None = None,
    a_max: torch.Tensor | None = None,
    target_mode: None | str = None,
) -> tuple[torch.nn.Module, np.ndarray]:
    """Train NPE with noise augmentation.

    Parameters
    ----------
    model: torch.nn.Module
        Probabilistic model (e.g. :class:`ProbParamCNN`,
        :class:`ProbProtParamCNN`) trained in place
    train_data_loader: torch.utils.data.DataLoader
        Training batches
    learning_rate: float
        Initial learning rate
    num_epochs: int | None
        Number of epochs; ignored when ``num_steps`` is set
    learning_rate_end: float | None
        Final learning rate (defaults to ``learning_rate / 100``)
    test_data_loader: torch.utils.data.DataLoader | None
        Optional loader for per-epoch test-loss evaluation
    num_steps: int | None
        Total training steps; overrides ``num_epochs`` when set
    num_steps_test: int | None
        Steps used when evaluating the test loss
    log_folder: str
        Directory for loss CSVs and checkpoints
    log_freq: int
        Step interval for logging the training loss
    save_freq: int
        Step interval for checkpointing the model
    restart_from: str | None
        Path to a model state dict to restart training from (weights only).
        When set, epoch/step counters resume from the existing loss CSVs.
    enable_cuda: bool
        Allow training on CUDA when available
    enable_mps: bool
        Allow training on MPS when available
    trial: optuna.trial.Trial | None
        Optuna trial enabling hyperparameter-tuning pruning
    noise_levels: torch.Tensor | None
        Per-channel noise levels for :func:`apply_noise`
    bias_tensor: torch.Tensor | None
        Optional per-channel bias added during augmentation
    a_min: torch.Tensor | None
        Per-channel lower clip applied after adding noise (physical units,
        from :func:`make_noise_levels`); required unless the data is encoded
    a_max: torch.Tensor | None
        Per-channel upper clip applied after adding noise
    target_mode: str | None
        When ``"encoded"``, skip signal-space noise augmentation

    Returns
    -------
    tuple
        ``(model, loss_hist)`` — the trained model and the training-loss
        history array
    """

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
        noise_levels, a_min, a_max = _reshape_noise_args(
            noise_levels, a_min, a_max
        )

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
                noise_levels=noise_levels,
                a_min=a_min,
                a_max=a_max,
                target_mode=target_mode,
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
            batch_in = _noisy_input(
                model,
                batch[0],
                noise_levels,
                a_min,
                a_max,
                bias_tensor,
                target_mode,
                device,
            )

            # Compute loss in the scaled parameter space
            try:
                loss = _gauss_npe_batch_loss(model, batch, batch_in, device)
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
                noise_levels=noise_levels,
                a_min=a_min,
                a_max=a_max,
                target_mode=target_mode,
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
    model: torch.nn.Module,
    test_data_loader: torch.utils.data.DataLoader,
    num_steps: int | None = None,
    enable_cuda: bool = True,
    enable_mps: bool = True,
    verbose: bool = True,
    noise_levels: torch.Tensor | None = torch.tensor([0, 0.010, 0.04, 1]),
    bias_tensor: torch.Tensor | None = None,
    a_min: torch.Tensor | None = None,
    a_max: torch.Tensor | None = None,
    target_mode: None | str = None,
) -> float:
    """Compute mean test loss.
    Use same input-noise augmentation as training

    Parameters
    ----------
    model: torch.nn.Module
        Probabilistic model to evaluate
    test_data_loader: torch.utils.data.DataLoader
        Test batches
    num_steps: int | None
        Cap on the number of batches evaluated; all batches when None
    enable_cuda: bool
        Allow evaluating on CUDA when available
    enable_mps: bool
        Allow evaluating on MPS when available
    verbose: bool
        Display a progress bar
    noise_levels: torch.Tensor | None
        Per-channel noise levels for :func:`apply_noise`
    bias_tensor: torch.Tensor | None
        Optional per-channel bias added during augmentation
    a_min: torch.Tensor | None
        Per-channel lower clip applied after adding noise (physical units,
        from :func:`make_noise_levels`); required unless the data is encoded
    a_max: torch.Tensor | None
        Per-channel upper clip applied after adding noise
    target_mode: str | None
        When ``"encoded"``, skip signal-space noise augmentation

    Returns
    -------
    float
        Sample-weighted average loss over the evaluated batches
    """
    # Device set up
    device_type = get_device_type(
        enable_cuda=enable_cuda, enable_mps=enable_mps
    )
    device = torch.device(device_type)
    if verbose:
        print("Device = ", device)

    if target_mode != "encoded":
        noise_levels, a_min, a_max = _reshape_noise_args(
            noise_levels, a_min, a_max
        )

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
            batch_in = _noisy_input(
                model,
                batch[0],
                noise_levels,
                a_min,
                a_max,
                bias_tensor,
                target_mode,
                device,
            )
            loss = _gauss_npe_batch_loss(model, batch, batch_in, device)
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
