import argparse
import os
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn
from prettyPlot.progressBar import print_progress_bar
from torch.utils.data import DataLoader, TensorDataset

from batfit import logger
from batfit.preprocess.sim_setup import make_params
from batfit.utils.data_utils import (
    scale_dataset_from_scaler,
    scale_input_from_scaler,
    scale_output_from_scaler,
    unscale_dataset_from_scaler,
    unscale_input_from_scaler,
    unscale_output_from_scaler,
    unscale_pred_from_scaler,
)
from batfit.utils.text_utils import shuffle_substrings
from batfit.utils.torch_utils import (
    get_device_type,
    get_num_parameters,
    load_model,
    log_training,
    make_dataset_from_np,
    prepare_log,
    save_model,
)


def make_noise_levels(
    target_mode: str,
    noise_levels: list,
    cyc_mode: str,
    vmin: float = 3.0,
    vmax: float = 4.1,
):
    noise_levels_single = torch.tensor(noise_levels).view(1, 4, 1)
    noise_levels_dis = torch.tensor(noise_levels).view(1, 4, 1)
    noise_levels_chcc = torch.tensor(noise_levels).view(1, 4, 1)
    a_min_single = torch.tensor(
        [-torch.inf, vmin, -torch.inf, -torch.inf]
    ).view(1, 4, 1)
    a_max_single = torch.tensor([torch.inf, vmax, torch.inf, torch.inf]).view(
        1, 4, 1
    )
    a_max_dis = torch.tensor([torch.inf, torch.inf, 0, 0]).view(1, 4, 1)
    a_min_dis = torch.tensor([-torch.inf, vmin, -torch.inf, -torch.inf]).view(
        1, 4, 1
    )
    a_max_dis = torch.tensor([torch.inf, torch.inf, 0, 0]).view(1, 4, 1)
    a_min_chcc = torch.tensor([-torch.inf, vmin, 0, 0]).view(1, 4, 1)
    a_max_chcc = torch.tensor([torch.inf, vmax, torch.inf, torch.inf]).view(
        1, 4, 1
    )

    if target_mode.lower() == "phionly":
        inds = [0]
    elif target_mode.lower() == "phi":
        inds = [0, 1]
    elif target_mode.lower() == "dvdq":
        inds = [0, 2]
    elif target_mode.lower() == "dqdv":
        inds = [0, 3]
    elif target_mode.lower() in shuffle_substrings("phi-dvdq"):
        inds = [0, 1, 2]
    elif target_mode.lower() in shuffle_substrings("phi-dqdv"):
        inds = [0, 1, 3]
    elif target_mode.lower() in shuffle_substrings("dvdq-dqdv"):
        inds = [0, 2, 3]
    elif target_mode.lower() in shuffle_substrings("phi-dvdq-dqdv"):
        inds = [0, 1, 2, 3]
    else:
        raise NotImplementedError

    if cyc_mode.lower() == "discharge":
        noise_levels = noise_levels_dis[:, inds, :]
        a_min = a_min_dis[:, inds, :]
        a_max = a_max_dis[:, inds, :]

    if cyc_mode.lower() == "chargecc":
        noise_levels = noise_levels_chcc[:, inds, :]
        a_min = a_min_chcc[:, inds, :]
        a_max = a_max_chcc[:, inds, :]

    if cyc_mode.lower() == "discharge-chargecc":
        noise_levels = torch.cat(
            (noise_levels_dis[:, inds, :], noise_levels_chcc[:, inds, :]),
            dim=1,
        )
        a_min = torch.cat(
            (a_min_dis[:, inds, :], a_min_chcc[:, inds, :]), dim=1
        )
        a_max = torch.cat(
            (a_max_dis[:, inds, :], a_max_chcc[:, inds, :]), dim=1
        )

    if cyc_mode.lower() in ["rh", "lh"]:
        noise_levels = noise_levels_single[:, inds, :]
        a_min = a_min_single[:, inds, :]
        a_max = a_max_single[:, inds, :]

    return noise_levels, a_min, a_max


def apply_noise(
    batch_in: torch.Tensor,
    scaler_X,
    noise_levels: torch.Tensor,
    a_min: torch.Tensor,
    a_max: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    batch_in = scaler_X.inverse_transform(batch_in)
    noise = (torch.rand(batch_in.shape) - 0.5) * torch.reshape(
        noise_levels, (1, -1, 1)
    )
    batch_in += noise
    if bias is not None:
        batch_in += bias * torch.rand(1)
    batch_in = torch.clamp(batch_in, min=a_min, max=a_max)
    batch_in = scaler_X.transform(batch_in)
    return batch_in


# -----------------------
# Encoder
# -----------------------
class ConvEncoder1D(nn.Module):
    def __init__(
        self,
        input_channels=1,
        num_points=16384,
        latent_dim=32,
        num_filt=[32, 64],
        leaky_relu_slope=0.2,
    ):
        super().__init__()
        layer_list = []
        for ifilt, n_filt in enumerate(num_filt):
            if ifilt == 0:
                layer_list.append(
                    nn.Conv1d(
                        input_channels,
                        n_filt,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    )
                )
                layer_list.append(nn.LeakyReLU(leaky_relu_slope))
            else:
                layer_list.append(
                    nn.Conv1d(
                        num_filt[ifilt - 1],
                        n_filt,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    )
                )
                layer_list.append(nn.LeakyReLU(leaky_relu_slope))
        layer_list.append(nn.Flatten())
        self.conv_net = nn.Sequential(*layer_list)
        self.fc_mu = nn.Linear(
            num_filt[-1] * num_points // 2 ** len(num_filt), latent_dim
        )
        self.fc_logvar = nn.Linear(
            num_filt[-1] * num_points // 2 ** len(num_filt), latent_dim
        )

    def forward(self, x):
        """
        x: (batch, channels=1, length)
        returns: mu (batch, latent_dim), logvar (batch, latent_dim)
        """
        h = self.conv_net(x)  # (batch, channels', L')
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def encode(self, x):
        return self.forward(x)


# -----------------------
# Decoder
# -----------------------
class ConvDecoder1D(nn.Module):
    def __init__(
        self,
        output_channels=1,
        num_points=16384,
        latent_dim=32,
        num_filt=[32, 64],
        leaky_relu_slope=0.2,
    ):
        super().__init__()
        layer_list = []
        layer_list.append(
            nn.Linear(
                latent_dim, num_filt[-1] * num_points // 2 ** len(num_filt)
            )
        )
        layer_list.append(
            nn.Unflatten(1, (num_filt[-1], num_points // 2 ** len(num_filt)))
        )
        for ifilt, n_filt in enumerate(num_filt):
            if ifilt == len(num_filt) - 1:
                layer_list.append(
                    nn.ConvTranspose1d(
                        num_filt[-ifilt - 1],
                        output_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    )
                )
            else:
                layer_list.append(
                    nn.ConvTranspose1d(
                        num_filt[-ifilt - 1],
                        num_filt[-ifilt - 2],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    )
                )
                layer_list.append(nn.LeakyReLU(leaky_relu_slope))

        self.decoder = nn.Sequential(*layer_list)

    def forward(self, z):
        out = self.decoder(z)
        return out

    def decode(self, z):
        return self.forward(z)


# -----------------------
# Loss
# -----------------------
def vae_loss(recon_x, x, mu, logvar, kl_weight=1.0):
    assert recon_x.shape == x.shape
    recon_criterion = nn.MSELoss(reduction="mean")
    recon_loss = recon_criterion(recon_x, x)
    # temp_grad = -torch.roll(recon_x[:,0,:],-1, dims=1) + recon_x[:,0,:]
    # temp_grad2 = -torch.roll(recon_x[:,0,:],-2, dims=1) + recon_x[:,0,:]
    # temp_grad3 = -torch.roll(recon_x[:,0,:],-3, dims=1) + recon_x[:,0,:]
    # temp_grad = 10*(1.0/3.0)*torch.mean(torch.relu(temp_grad[:,:-1])) #+ 10*(1.0/3.0)*torch.mean(torch.relu(temp_grad2[:,:-2]))  + 10*(1.0/3.0)*torch.mean(torch.relu(temp_grad3[:,:-3]))
    temp_grad = 10.0 * torch.mean(torch.relu(-torch.diff(recon_x[:, 0, :])))
    # KL divergence between q(z|x) and N(0,1)
    # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kl + temp_grad, recon_loss + temp_grad, kl


# -----------------------
# Schedule
# -----------------------
def learning_rate_schedule(epoch, epoch_end, lr_beg, lr_end):
    epoch_delay = epoch_end // 10
    if epoch < epoch_delay:
        return lr_beg
    else:
        return lr_beg * (lr_end / lr_beg) ** (
            min((epoch - epoch_delay) / epoch_end, 1.0)
        )


def kl_anneal_weight(epoch, epoch_end, kl_weight_beg=0.01, kl_weight_end=1.0):
    if epoch >= epoch_end:
        return kl_weight_end
    else:
        # return kl_weight_beg + (kl_weight_end-kl_weight_beg) * min(epoch / epoch_end, 1.0)
        return kl_weight_beg * (kl_weight_end / kl_weight_beg) ** min(
            epoch / epoch_end, 1.0
        )


# -----------------------
# VAE wrapper
# -----------------------
class VAECNN(nn.Module):
    def __init__(
        self,
        input_shape,
        chan_list,
        loss_fn=vae_loss,
        leaky_relu_slope=0.2,
        cyc_mode="discharge",
        latent_dim=32,
        sim_config=None,
        denoise=False,
    ):
        logger.info(
            f"Creating VAE CNN model with {latent_dim} latent dimension"
        )
        super(VAECNN, self).__init__()

        assert len(input_shape) == 2
        try:
            assert input_shape[0] < input_shape[1]
        except AssertionError:
            raise AssertionError(
                "Expect input_shape (n_chan, n_points), but (n_chan {input_shape[0]} > n_points {input_shape[1]} ) which is suspicious"
            )
        input_channels = input_shape[0]
        num_points = input_shape[1]
        self.chan_list = chan_list
        self.latent_dim = latent_dim
        assert self.latent_dim < num_points
        self.leaky_relu_slope = leaky_relu_slope
        self.chan_list = chan_list
        self.loss_fn = loss_fn
        self.denoise = denoise
        assert self.loss_fn in [
            vae_loss,
        ]
        self.sim_config = sim_config
        if self.sim_config is not None:
            self.sim_params = make_params(self.sim_config)

        assert len(chan_list) < int(np.log(num_points) / np.log(2))

        self.encoder = ConvEncoder1D(
            input_channels=input_channels,
            num_points=num_points,
            latent_dim=self.latent_dim,
            num_filt=self.chan_list,
            leaky_relu_slope=self.leaky_relu_slope,
        )
        self.decoder = ConvDecoder1D(
            output_channels=input_channels,
            num_points=num_points,
            latent_dim=self.latent_dim,
            num_filt=self.chan_list,
            leaky_relu_slope=self.leaky_relu_slope,
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        x: (batch, channels, length)
        returns: recon (batch, channels, length), mu, logvar
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def encode(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z):
        return self.decoder(z)


def compute_test_loss(
    model: torch.nn.Module,
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
    kl_weight=1.0,
):
    target_mode = "phi"
    # Device set up
    device_type = get_device_type(
        enable_cuda=enable_cuda, enable_mps=enable_mps
    )
    device = torch.device(device_type)
    logger.debug(f"Device = {device}")

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
    recon_loss_ave = 0
    kl_loss_ave = 0
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
                )
            else:
                batch_in = batch[0]

            # Compute loss
            # batch_in = batch_in[:,1,:][:,torch.newaxis,:]
            batch_in = batch_in.to(device)
            recon, mu, logvar = model(batch_in)
            if model.denoise:
                loss, recon_l, kl = model.loss_fn(
                    recon, batch[0].to(device), mu, logvar, kl_weight=kl_weight
                )
            else:
                loss, recon_l, kl = model.loss_fn(
                    recon, batch_in, mu, logvar, kl_weight=kl_weight
                )
            kl = kl * kl_weight

            loss_ave += loss.item() * batch_in.shape[0]
            recon_loss_ave += recon_l.item() * batch_in.shape[0]
            kl_loss_ave += kl.item() * batch_in.shape[0]
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
        recon_loss_ave /= num_el
        kl_loss_ave /= num_el
    return loss_ave, recon_loss_ave, kl_loss_ave


# -----------------------
# Training loop
# -----------------------
def train_model(
    model: nn.Module,
    train_data_loader: torch.utils.data.DataLoader,
    learning_rate: float,
    num_epochs: int | None,
    kl_weight_beg: float = 0.01,
    kl_weight_end: float = 1.0,
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
    scaler_X=None,
    a_min: torch.Tensor | None = torch.tensor(
        [-torch.inf, 3, -torch.inf, -torch.inf]
    ),
    a_max: torch.Tensor | None = torch.tensor([torch.inf, 4.1, 0, 0]),
):

    target_mode = "phi"
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
        autoencoder=True,
    )

    if learning_rate_end is None:
        learning_rate_end = learning_rate / 100.0

    logger.info(f"Device = {device}")
    model = model.to(device)

    loss_hist = np.array([])
    optimizer = torch.optim.Adamax(
        model.parameters(), lr=learning_rate, weight_decay=1e-5
    )
    if optimizer_state_dict_filename is not None:
        optimizer.load_state_dict(
            torch.load(optimizer_state_dict_filename, weights_only=True)
        )

    num_batch = len(train_data_loader)
    model.train()

    prepare_log(log_folder)
    if num_steps is not None:
        total_steps = num_steps
        num_epochs = num_steps // num_batch + 1
    else:
        total_steps = num_batch * num_epochs

    print_progress_bar(
        0,
        total_steps,
        prefix=f"Loss = ? Step 0 / {total_steps} ",
        suffix="Complete",
        length=50,
    )

    for epoch in range(num_epochs):
        model.train()
        # Set LR and KL weight for this epoch
        kl_weight = kl_anneal_weight(
            epoch,
            num_epochs * 3 // 4,
            kl_weight_beg=kl_weight_beg,
            kl_weight_end=kl_weight_end,
        )
        # print(kl_weight)
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
                )
            else:
                batch_in = batch[0]

            # Compute loss
            try:
                # batch_in = batch_in[:,1,:][:,torch.newaxis,:]
                batch_in = batch_in.to(device)
                recon, mu, logvar = model(batch_in)
                if model.denoise:
                    loss, recon_l, kl = model.loss_fn(
                        recon,
                        batch[0].to(device),
                        mu,
                        logvar,
                        kl_weight=kl_weight,
                    )
                else:
                    loss, recon_l, kl = model.loss_fn(
                        recon, batch_in, mu, logvar, kl_weight=kl_weight
                    )
                kl = kl * kl_weight

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
                    current_step,
                    [loss, recon_l, kl],
                    log_folder,
                    filename="train_loss.csv",
                )
                save_model(
                    step=current_step,
                    model=model,
                    optimizer=optimizer,
                    device_type=device_type,
                    log_folder=log_folder,
                    autoencoder=True,
                )
            elif current_step % log_freq == 0 and not logged:
                log_training(
                    current_step,
                    [loss, recon_l, kl],
                    log_folder,
                    filename="train_loss.csv",
                )

            logged = False

            print_progress_bar(
                current_step,
                total_steps,
                prefix=f"Loss = {loss.item():.4g} (Recon: {recon_l.item()/loss.item():.2f} KL: {kl.item()/loss.item():.2f}) Step {current_step} / {total_steps} ",
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
            test_loss, test_recon_loss, test_kl_loss = compute_test_loss(
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
                kl_weight=kl_weight_end,
            )

            log_training(
                current_step,
                [test_loss, test_recon_loss, test_kl_loss],
                log_folder,
                filename="test_loss.csv",
            )

        else:
            test_loss = None
        if trial is not None:
            if test_loss is not None:
                trial.report(test_loss, epoch)
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return model
