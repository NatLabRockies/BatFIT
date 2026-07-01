"""Training utilities for conditional flow matching models.

Covers :class:`~batfit.model.paramNN.ProbParamFM` and
:class:`~batfit.model.paramNN.ProbProtParamFM`.  
Main differences with train_utils.py:

- We predict a velocity, not (mu, gamma)
- The loss is flow_matching_loss not a log-likelihood.
- Sampling is depends on whether we do prior matching or no
"""

import numpy as np
import optuna
import torch
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from prettyPlot.progressBar import print_progress_bar

from batfit import logger
from batfit.model.param_utils.model_utils import _ProbParamFMBase
from batfit.model.paramNN import ProbParamFM, ProbProtParamFM
from batfit.utils.torch_utils import (
    get_device_type,
    log_training,
    prepare_log,
    save_model,
)

from .losses import flow_matching_loss
from .metrics import accuracy, rel_accuracy
from .noise_utils import apply_noise
from .train_utils import learning_rate_schedule


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_labels_idx(model: torch.nn.Module) -> int:
    """Return the batch index that holds the degradation parameter labels.

    For protocol models the batch is ``(X, P, Y)`` so labels are at index 2.
    For plain models the batch is ``(X, Y)`` so labels are at index 1.
    """
    if isinstance(model, ProbProtParamFM):
        return 2
    elif isinstance(model, ProbParamFM):
        return 1
    else:
        raise TypeError("_get_labels_idx should only be used for FM models")


def _forward_fm(
    model: torch.nn.Module,
    x_signal: torch.Tensor,
    batch: list[torch.Tensor],
    x_t: torch.Tensor,
    t: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Call the correct FM forward signature depending on model type.

    :param model: a :class:`_ProbParamFMBase` instance
    :param x_signal: pre-processed signal tensor, shape (batch, channels, time)
    :param batch: full DataLoader batch (used to extract prot_params if needed)
    :param x_t: interpolated particle positions, shape (batch, n_params)
    :param t: flow times in [0, 1], shape (batch,)
    :param device: computation device
    :return: predicted velocity, shape (batch, n_params)
    """
    if isinstance(model, ProbProtParamFM):
        return model(x_signal, batch[1].to(device), x_t, t)
    elif isinstance(model, ProbParamFM):
        return model(x_signal, x_t, t)
    else:
        raise TypeError("_forward_fm should only be used for FM models")


def _sample_fm(
    model: torch.nn.Module,
    x_signal: torch.Tensor,
    batch: list[torch.Tensor],
    n_samples: int,
    n_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Draw posterior samples from an FM model.

    :return: samples, shape (batch, n_samples, n_params)
    """
    if isinstance(model, ProbProtParamFM):
        return model.sample(x_signal, batch[1].to(device), n_samples, n_steps)
    elif isinstance(model, ProbParamFM):
        return model.sample(x_signal, n_samples, n_steps)
    else:
        raise TypeError("_sample_fm should only be used for FM models")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_fm_model(
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
) -> tuple[torch.nn.Module, np.ndarray]:
    """Train a flow matching model (``ProbParamFM`` or ``ProbProtParamFM``).

    For each mini-batch the training step is:

    1. Apply measurement noise to the signal (same as for CNN models).
    2. Sample the base point ``x_0`` from N(0, I) or, when
       ``model.use_prior_matching=True``, from the prior U(min_par, max_par).
    3. Draw flow time ``t ~ Uniform(0, 1)``.
    4. Compute the interpolated position ``x_t`` and target velocity ``u_t``
       via ``AffineProbPath(CondOTScheduler())``.
    5. Call ``model.forward`` to predict the velocity and regress against
       ``u_t`` with ``flow_matching_loss``.

    DataLoader batch format
    -----------------------
    - ``ProbParamFM``: ``(X, Y)``
    - ``ProbProtParamFM``: ``(X, P, Y)``

    :param model: FM model to train
    :param train_data_loader: yields mini-batches
    :param learning_rate: initial learning rate for Adamax
    :param num_epochs: number of training epochs (mutually exclusive with
                       ``num_steps``)
    :param learning_rate_end: final LR; defaults to ``learning_rate / 100``
    :param test_data_loader: optional held-out set for per-epoch test loss
    :param num_steps: total gradient steps (overrides ``num_epochs``)
    :param num_steps_test: maximum test steps per epoch evaluation
    :param log_folder: directory for loss CSV and model checkpoints
    :param log_freq: log loss every N steps
    :param save_freq: save checkpoint every N steps
    :param optimizer_state_dict_filename: path to resume optimizer state
    :param enable_cuda: allow CUDA device
    :param enable_mps: allow Apple MPS device
    :param trial: Optuna trial for hyperparameter search pruning
    :param noise_levels: measurement noise levels passed to ``apply_noise``
    :param bias_tensor: measurement bias passed to ``apply_noise``
    :param scaler_X: signal scaler (needed by ``apply_noise``)
    :param a_min: lower clip for signal channels after noise injection
    :param a_max: upper clip for signal channels after noise injection
    :return: ``(trained_model, loss_history)``
    """
    device_type = get_device_type(enable_cuda=enable_cuda, enable_mps=enable_mps)
    device = torch.device(device_type)

    save_model(step=0, model=model, log_folder=log_folder,
               save_model_obj=True, save_model_weights=False, save_model_opt=False)

    if noise_levels is not None and len(noise_levels.shape) == 1:
        noise_levels = noise_levels.reshape(1, noise_levels.shape[0], 1)
    if a_min is not None and len(a_min.shape) == 1:
        a_min = a_min.reshape(1, a_min.shape[0], 1)
    if a_max is not None and len(a_max.shape) == 1:
        a_max = a_max.reshape(1, a_max.shape[0], 1)

    if learning_rate_end is None:
        learning_rate_end = learning_rate / 100.0

    logger.info(f"Device = {device}")
    model = model.to(device)

    prob_path = AffineProbPath(scheduler=CondOTScheduler())
    labels_idx = _get_labels_idx(model)

    loss_hist = np.array([])
    optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    if optimizer_state_dict_filename is not None:
        optimizer.load_state_dict(
            torch.load(optimizer_state_dict_filename, weights_only=True)
        )

    num_batch = len(train_data_loader)
    if num_steps is not None:
        total_steps = num_steps
        num_epochs = num_steps // num_batch + 1
    else:
        total_steps = num_batch * num_epochs

    prepare_log(log_folder)
    model.train()
    print_progress_bar(0, total_steps,
                       prefix=f"Loss = ? Step 0 / {total_steps} ",
                       suffix="Complete", length=50)

    current_step = 0
    for epoch in range(num_epochs):
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate_schedule(
                epoch, num_epochs * 3 // 4, learning_rate, learning_rate_end
            )

        for step, batch in enumerate(train_data_loader):
            current_step = epoch * num_batch + (step + 1)
            optimizer.zero_grad()

            batch_in = apply_noise(
                batch_in=batch[0],
                scaler_X=scaler_X,
                noise_levels=noise_levels,
                a_min=a_min,
                a_max=a_max,
                bias=bias_tensor,
            )
            x_signal = batch_in.to(device)

            # x_1 = degradation parameter labels (target for the flow)
            x_1 = batch[labels_idx].to(device)
            batch_size = x_1.shape[0]

            # Base point: prior or standard Gaussian
            if model.use_prior_matching:
                x_0 = model.sample_prior(batch_size, device)
            else:
                x_0 = torch.randn_like(x_1)

            t = torch.rand(batch_size, device=device)
            path_sample = prob_path.sample(x_0=x_0, x_1=x_1, t=t)

            try:
                pred_u = _forward_fm(model, x_signal, batch, path_sample.x_t, t, device)
                loss = flow_matching_loss(pred_u, path_sample.dx_t)

                if ~(torch.isnan(loss) | torch.isinf(loss)):
                    loss.backward()
                    optimizer.step()
            except (torch.OutOfMemoryError, RuntimeError) as err:
                if trial is not None:
                    raise optuna.exceptions.TrialPruned()
                raise err

            loss_hist = np.append(loss_hist, loss.detach().to("cpu").numpy())
            logged = False
            if current_step % save_freq == 0:
                logged = True
                log_training(current_step, loss, log_folder, filename="train_loss.csv")
                save_model(step=current_step, model=model, optimizer=optimizer,
                           device_type=device_type, log_folder=log_folder)
            elif current_step % log_freq == 0 and not logged:
                log_training(current_step, loss, log_folder, filename="train_loss.csv")

            print_progress_bar(
                current_step, total_steps,
                prefix=f"Loss = {loss.item():.4g} Step {current_step} / {total_steps} ",
                suffix="Complete", length=50,
            )
            if current_step >= total_steps:
                break
            if trial is not None and (
                trial.should_prune() or np.isnan(loss.item()) or np.isinf(loss.item())
            ):
                raise optuna.exceptions.TrialPruned()

        if test_data_loader is not None:
            test_loss = compute_test_loss_fm(
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
            )
            log_training(current_step, test_loss, log_folder, filename="test_loss.csv")
            model.train()
        else:
            test_loss = None

        if trial is not None:
            if test_loss is not None:
                trial.report(test_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    save_model(step=total_steps, model=model, optimizer=optimizer,
               device_type=device_type, log_folder=log_folder, bypass="final")
    return model, loss_hist


# ---------------------------------------------------------------------------
# Test loss (velocity MSE)
# ---------------------------------------------------------------------------

def compute_test_loss_fm(
    model: torch.nn.Module,
    test_data_loader: torch.utils.data.DataLoader,
    num_steps: int | None = None,
    enable_cuda: bool = True,
    enable_mps: bool = True,
    verbose: bool = True,
    noise_levels: torch.Tensor | None = torch.tensor([0, 0.010, 0.04, 1]),
    bias_tensor: torch.Tensor | None = None,
    scaler_X=None,
    a_min: torch.Tensor | None = torch.tensor(
        [-torch.inf, 3, -torch.inf, -torch.inf]
    ),
    a_max: torch.Tensor | None = torch.tensor([torch.inf, 4.1, 0, 0]),
) -> float:
    """Evaluate the velocity-MSE loss on a held-out set.

    Uses the same probability path sampling as ``train_fm_model``, so the
    returned value is directly comparable to the training loss.

    :param model: trained FM model
    :param test_data_loader: held-out DataLoader
    :param num_steps: cap the number of evaluated batches (None = all)
    :param enable_cuda: allow CUDA
    :param enable_mps: allow MPS
    :param verbose: print a progress bar
    :param noise_levels: measurement noise levels
    :param bias_tensor: measurement bias
    :param scaler_X: signal scaler
    :param a_min: lower clip for noisy signal
    :param a_max: upper clip for noisy signal
    :return: mean velocity MSE over the evaluated batches
    """
    device_type = get_device_type(enable_cuda=enable_cuda, enable_mps=enable_mps)
    device = torch.device(device_type)

    if noise_levels is not None and len(noise_levels.shape) == 1:
        noise_levels = noise_levels.reshape(1, noise_levels.shape[0], 1)
    if a_min is not None and len(a_min.shape) == 1:
        a_min = a_min.reshape(1, a_min.shape[0], 1)
    if a_max is not None and len(a_max.shape) == 1:
        a_max = a_max.reshape(1, a_max.shape[0], 1)

    model = model.to(device)
    prob_path = AffineProbPath(scheduler=CondOTScheduler())
    labels_idx = _get_labels_idx(model)

    total_steps = num_steps if num_steps is not None else len(test_data_loader)
    loss_sum = 0.0
    n_el = 0

    model.eval()
    if verbose:
        print_progress_bar(0, total_steps,
                           prefix=f"Test Loss = ? Step 0 / {total_steps} ",
                           suffix="Complete", length=50)

    with torch.no_grad():
        for step, batch in enumerate(test_data_loader):
            batch_in = apply_noise(
                batch_in=batch[0], scaler_X=scaler_X, noise_levels=noise_levels,
                a_min=a_min, a_max=a_max, bias=bias_tensor,
            )
            x_signal = batch_in.to(device)
            x_1 = batch[labels_idx].to(device)
            batch_size = x_1.shape[0]

            if model.use_prior_matching:
                x_0 = model.sample_prior(batch_size, device)
            else:
                x_0 = torch.randn_like(x_1)

            t = torch.rand(batch_size, device=device)
            path_sample = prob_path.sample(x_0=x_0, x_1=x_1, t=t)
            pred_u = _forward_fm(model, x_signal, batch, path_sample.x_t, t, device)
            loss = flow_matching_loss(pred_u, path_sample.dx_t)

            loss_sum += loss.item() * batch_size
            n_el += batch_size

            if verbose:
                print_progress_bar(
                    step + 1, total_steps,
                    prefix=f"Test Loss = {loss_sum / n_el:.4g} Step {step + 1} / {total_steps} ",
                    suffix="Complete", length=50,
                )
            if step + 1 >= total_steps:
                break

    return loss_sum / n_el


# ---------------------------------------------------------------------------
# Sample-based accuracy — works for both CNN and FM
# ---------------------------------------------------------------------------

def compute_post_sample_based(
    model: torch.nn.Module,
    test_data_loader: torch.utils.data.DataLoader,
    n_samples: int,
    n_ode_steps: int = 100,
    num_steps: int | None = None,
    enable_cuda: bool = True,
    enable_mps: bool = True,
    verbose: bool = True,
    noise_levels: torch.Tensor | None = torch.tensor([0, 0.010, 0.04, 1]),
    bias_tensor: torch.Tensor | None = None,
    scaler_X=None,
    a_min: torch.Tensor | None = torch.tensor(
        [-torch.inf, 3, -torch.inf, -torch.inf]
    ),
    a_max: torch.Tensor | None = torch.tensor([torch.inf, 4.1, 0, 0]),
    post_fn=rel_accuracy,
) -> torch.Tensor:
    """Sample-based accuracy metric 

    Estimates the posterior mean from samples and computes ``post_fn`` against
    ground-truth labels. 

    :param model: CNN or FM model
    :param test_data_loader: held-out DataLoader
    :param n_samples: posterior samples per observation
    :param n_ode_steps: ODE integration steps (FM only)
    :param num_steps: cap the number of evaluated batches (None = all)
    :param enable_cuda: allow CUDA
    :param enable_mps: allow MPS
    :param verbose: print a progress bar
    :param noise_levels: measurement noise levels
    :param bias_tensor: measurement bias
    :param scaler_X: signal scaler
    :param a_min: lower clip for noisy signal
    :param a_max: upper clip for noisy signal
    :param post_fn: metric function, one of ``accuracy`` or ``rel_accuracy``
    :return: per-parameter metric, shape (n_params,)
    """
    device_type = get_device_type(enable_cuda=enable_cuda, enable_mps=enable_mps)
    device = torch.device(device_type)

    if noise_levels is not None and len(noise_levels.shape) == 1:
        noise_levels = noise_levels.reshape(1, noise_levels.shape[0], 1)
    if a_min is not None and len(a_min.shape) == 1:
        a_min = a_min.reshape(1, a_min.shape[0], 1)
    if a_max is not None and len(a_max.shape) == 1:
        a_max = a_max.reshape(1, a_max.shape[0], 1)

    if not isinstance(model, _ProbParamFMBase):
        raise TypeError(
            f"compute_post_sample_based is for FM models only. "
            f"For CNN models use compute_post from train_utils. "
            f"Got {type(model).__name__}."
        )

    model = model.to(device)
    labels_idx = _get_labels_idx(model)
    total_steps = num_steps if num_steps is not None else len(test_data_loader)

    post_sum = None
    n_el = 0

    model.eval()
    if verbose:
        print_progress_bar(0, total_steps,
                           prefix=f"Sample post = ? Step 0 / {total_steps} ",
                           suffix="Complete", length=50)

    with torch.no_grad():
        for step, batch in enumerate(test_data_loader):
            batch_in = apply_noise(
                batch_in=batch[0], scaler_X=scaler_X, noise_levels=noise_levels,
                a_min=a_min, a_max=a_max, bias=bias_tensor,
            )
            x_signal = batch_in.to(device)
            y_true = batch[labels_idx].to(device)

            # Integrate n_samples ODE trajectories and average to estimate the
            # posterior mean
            samples = _sample_fm(model, x_signal, batch, n_samples, n_ode_steps, device)
            mu_est = samples.mean(dim=1)  # (batch, n_params)

            batch_post = post_fn(mu_est, y_true)  # (n_params,)
            post_sum = batch_post * y_true.shape[0] if post_sum is None else post_sum + batch_post * y_true.shape[0]
            n_el += y_true.shape[0]

            if verbose:
                print_progress_bar(
                    step + 1, total_steps,
                    prefix=f"Sample post Step {step + 1} / {total_steps} ",
                    suffix="Complete", length=50,
                )
            if step + 1 >= total_steps:
                break

    return post_sum / n_el
