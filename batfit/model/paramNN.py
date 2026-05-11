import pickle

import numpy as np
import optuna
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from prettyPlot.progressBar import print_progress_bar
from torch.distributions import kl_divergence

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

from .ae import AECNN
from .metrics import accuracy, identifiability, rel_accuracy
from .vae import VAECNN


def mse_loss(output, target):
    """
    Custom mean squared error loss function.
    output: Predicted values.
    target: Ground truth labels.
    """
    loss = torch.mean((output - target) ** 2)
    return loss


def independent_gumbel_loss(mu, sigma, target):
    """
    Custom gumbel loss function.
    output: Predicted values.
    target: Ground truth labels.
    """
    epsilon = 1e-6
    sigma = torch.clamp(sigma, min=epsilon)
    beta = np.sqrt(6) * sigma / torch.pi
    z = (target - mu) / beta
    loss = torch.mean(torch.log(beta) + z + torch.exp(-z))
    return loss


def pinball_loss(mu, sigma, target):
    epsilon = 1e-6
    sigma = torch.clamp(sigma, min=epsilon)
    y5 = mu - 1.6448536269514729 * sigma
    y95 = mu + 1.6448536269514729 * sigma
    loss = torch.mean(
        torch.maximum(0.05 * (target - y5), (0.05 - 1) * (target - y5))
    ) + torch.mean(
        torch.maximum(0.95 * (target - y95), (0.95 - 1) * (target - y95))
    )
    return loss


def independent_normal_loss(mu, sigma, target):
    """
    Custom neg log like loss function.
    output: Predicted values.
    target: Ground truth labels.
    """
    # epsilon = 1e-6  # To prevent log(0) or division by zero
    # sigma = torch.clamp(sigma, min=epsilon)  # Ensure sigma is positive
    # nll = torch.sum(torch.log(sigma), dim=1) + 0.5 * torch.sum(((target - mu) ** 2) / (sigma**2), dim=1)
    # return torch.mean(nll)  # Average over the batch
    epsilon = 1e-6
    sigma = torch.clamp(sigma, min=epsilon)
    mvn = dist.MultivariateNormal(
        mu, covariance_matrix=torch.diag_embed(sigma**2)
    )
    nll = -mvn.log_prob(target)
    return nll.mean()


def elbo_independent_normal_loss(mu, sigma, prior, target, temp):
    """
    Custom neg log like loss function.
    output: Predicted values.
    target: Ground truth labels.
    """
    sigma = torch.clamp(sigma, min=1e-4)
    # post
    posterior = dist.MultivariateNormal(
        mu, covariance_matrix=torch.diag_embed(sigma**2)
    )
    if not posterior.mean.shape[0] == prior.mean.shape[0]:
        prior_mean = (
            prior.mean[0].reshape(1, -1).repeat(posterior.mean.shape[0], 1)
        )
        prior_cov = (
            prior.covariance_matrix[0]
            .reshape(1, sigma.shape[1], sigma.shape[1])
            .repeat(posterior.mean.shape[0], 1, 1)
        )
        prior = dist.MultivariateNormal(
            prior_mean, covariance_matrix=prior_cov
        )
    elbo = -posterior.log_prob(target) + temp * kl_divergence(posterior, prior)

    return elbo.mean()


def nll_loss(mu, sigma, target):
    return independent_normal_loss(mu, sigma, target)


def gumbel_loss(mu, sigma, target):
    return independent_gumbel_loss(mu, sigma, target)


def correlated_normal_loss(mu, sigma, target):
    mvn = dist.MultivariateNormal(mu, covariance_matrix=sigma)
    nll = -mvn.log_prob(target)

    return nll.mean()  # Average over the batch


def create_model_from_log(model_obj_file, model_state_dict_file, verbose=True):
    if verbose:
        logger.info(
            f"loading model from \n\t{model_obj_file} and {model_state_dict_file}"
        )
    with open(model_obj_file, "rb") as f:
        model = pickle.load(f)
    if not hasattr(model, "dependent_outputs"):
        model.dependent_outputs = False
    if not hasattr(model, "enforce_licons"):
        model.enforce_licons = False
    num_parameters = get_num_parameters(model)
    if verbose:
        print(f"\tNo. Trainable Parameters: {num_parameters}")
    if model_state_dict_file is not None:
        model = load_model(
            model, model_state_dict_file, enable_cuda=False, enable_mps=False
        )
    return model


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

    if cyc_mode.lower() in ["rh", "lh", "diffcap", "hppc", "posthppc", "chirp"]:
        noise_levels = noise_levels_single[:, inds, :]
        a_min = a_min_single[:, inds, :]
        a_max = a_max_single[:, inds, :]

    return noise_levels, a_min, a_max


def make_bias_tensor(
    target_mode: str,
    cyc_mode: str,
    bias: np.ndarray | None,
):
    if bias is None:
        return None
    if not isinstance(bias, np.ndarray):
        raise NotImplementedError
    if len(bias.shape) > 1:
        raise NotImplementedError
    bias = bias[np.newaxis, np.newaxis, :]
    bias = np.repeat(bias, 4, axis=1)
    bias = bias.astype("float32")
    bias[:, 0, :] = 0
    bias[:, 2, :] = 0
    bias[:, 3, :] = 0

    if target_mode.lower() == "phionly":
        inds = [1]
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
        bias = bias[:, inds, :]

    if cyc_mode.lower() == "chargecc":
        bias = bias[:, inds, :]

    if cyc_mode.lower() == "discharge-chargecc":
        raise NotImplementedError

    if cyc_mode.lower() in ["rh", "lh"]:
        bias = bias[:, inds, :]

    return torch.tensor(bias)


def sample_var(tens_m, tens_std, min_val, max_val, n=100):
    samp = torch.Tensor.repeat(tens_m, (n, 1)) + torch.Tensor.repeat(
        tens_std, (n, 1)
    ) * torch.randn(n, tens_m.shape[0])
    samp = torch.clamp(samp, min=min_val, max=max_val)
    return samp


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
        elif isinstance(model, ParamCNN):
            pred_scaled = model(torch.from_numpy(X_scaled))
            if model.constrain_output:
                pred_unscaled = model.inv_transform_output(
                    pred_scaled,
                    model.min_par.to("cpu"),
                    model.amp_par.to("cpu"),
                )
            elif not scale_y:
                pred_unscaled = pred_scaled
            elif scale_y:
                raise NotImplementedError
            else:
                raise NotImplementedError
            pred_unscaled = pred_unscaled.numpy()
            inp_unscaled, _ = unscale_dataset_from_scaler(
                X_scaled, pred_scaled, scaler_X_file, scaler_Y_file
            )
            probabilistic = False

        else:
            raise NotImplementedError

    if probabilistic:
        return (pred_unscaled, gamma_unscaled)
    else:
        return pred_unscaled


class ProbParamCNN(nn.Module):
    def __init__(
        self,
        input_shape,
        chan_list,
        fc_list,
        fc_mu_list,
        fc_gamma_list,
        loss_fn,
        leaky_relu_slope=0.2,
        cyc_mode="discharge",
        n_param_pred=6,
        dependent_outputs=False,
        enforce_licons=False,
        constrain_output=False,
        encoder_model=None,
        sim_config=None,
        prior=None,
    ):
        logger.info("Creating probabilistic CNN model")
        super(ProbParamCNN, self).__init__()
        self.leaky_relu_slope = leaky_relu_slope
        self.chan_list = chan_list
        self.fc_list = fc_list
        self.loss_fn = loss_fn
        self.cyc_mode = cyc_mode
        self.n_param_pred = n_param_pred
        self.constrain_output = constrain_output
        self.sim_config = sim_config
        self.dependent_outputs = dependent_outputs
        self.enforce_licons = enforce_licons
        self.encoder_model = encoder_model
        self.prior = prior
        if not self.cyc_mode.lower() == "discharge-chargecc":
            self.enforce_licons = False
        if self.enforce_licons:
            self.output_dim = self.n_param_pred - 1
        else:
            self.output_dim = self.n_param_pred
        if self.dependent_outputs:
            assert self.loss_fn == correlated_normal_loss
        else:
            assert self.loss_fn in [
                mse_loss,
                gumbel_loss,
                nll_loss,
                independent_normal_loss,
                independent_gumbel_loss,
                elbo_independent_normal_loss,
            ]

        if self.sim_config is not None:
            self.sim_params = make_params(self.sim_config)
            self.max_par = torch.from_numpy(
                np.array(
                    [
                        self.sim_params["deg_" + var_name + "_max"]
                        for var_name in self.sim_params["deg_param_names"]
                    ]
                ).astype("float32")
            )
            self.min_par = torch.from_numpy(
                np.array(
                    [
                        self.sim_params["deg_" + var_name + "_min"]
                        for var_name in self.sim_params["deg_param_names"]
                    ]
                ).astype("float32")
            )
            if self.enforce_licons:
                self.max_par = self.max_par[:-1]
                self.min_par = self.min_par[:-1]

            self.amp_par = self.max_par - self.min_par

        assert len(chan_list) < int(np.log(input_shape[1]) / np.log(2))

        if cyc_mode.lower() == "discharge-chargecc":
            input_shape_0 = input_shape[0] // 2
            input_shape_1 = input_shape[1]
        else:
            input_shape_0 = input_shape[0]
            input_shape_1 = input_shape[1]

        self.conv = []
        self.pool = []
        for ichan, chan in enumerate(chan_list):
            if ichan == 0:
                self.conv.append(
                    nn.Conv1d(
                        in_channels=input_shape_0,
                        out_channels=chan,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                self.pool.append(nn.MaxPool1d(kernel_size=2, stride=2))
            else:
                self.conv.append(
                    nn.Conv1d(
                        in_channels=chan_list[ichan - 1],
                        out_channels=chan,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
            self.pool.append(nn.MaxPool1d(kernel_size=2, stride=2))

        self.fc = []
        for ifc, fc in enumerate(fc_list):
            if ifc == 0:
                self.fc.append(
                    nn.Linear(
                        chan_list[-1] * input_shape_1 // (2 ** len(chan_list)),
                        fc,
                    )
                )
            else:
                self.fc.append(nn.Linear(fc_list[ifc - 1], fc))

        if cyc_mode.lower() == "discharge-chargecc":
            self.conv_aux = []
            self.pool_aux = []
            for ichan, chan in enumerate(chan_list):
                if ichan == 0:
                    self.conv_aux.append(
                        nn.Conv1d(
                            in_channels=input_shape_0,
                            out_channels=chan,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        )
                    )
                    self.pool_aux.append(nn.MaxPool1d(kernel_size=2, stride=2))
                else:
                    self.conv_aux.append(
                        nn.Conv1d(
                            in_channels=chan_list[ichan - 1],
                            out_channels=chan,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        )
                    )
                self.pool_aux.append(nn.MaxPool1d(kernel_size=2, stride=2))

            self.fc_aux = []
            for ifc, fc in enumerate(fc_list):
                if ifc == 0:
                    self.fc_aux.append(
                        nn.Linear(
                            chan_list[-1]
                            * input_shape_1
                            // (2 ** len(chan_list)),
                            fc,
                        )
                    )
                else:
                    self.fc_aux.append(nn.Linear(fc_list[ifc - 1], fc))

            fc_list_end = 2 * fc_list[-1]

        else:
            fc_list_end = fc_list[-1]

        self.fc_mu = []
        for ifc, fc in enumerate(fc_mu_list):
            if ifc == 0:
                self.fc_mu.append(nn.Linear(fc_list_end, fc))
            else:
                self.fc_mu.append(nn.Linear(fc_mu_list[ifc - 1], fc))
        self.fc_gamma = []
        for ifc, fc in enumerate(fc_gamma_list):
            if ifc == 0:
                self.fc_gamma.append(nn.Linear(fc_list_end, fc))
            else:
                self.fc_gamma.append(nn.Linear(fc_gamma_list[ifc - 1], fc))

        self.fc_otpt_mu = nn.Linear(fc_mu_list[-1], self.output_dim)
        if not self.dependent_outputs:
            self.fc_otpt_gamma = nn.Linear(fc_gamma_list[-1], self.output_dim)
        else:
            self.fc_otpt_gamma = nn.Linear(
                fc_gamma_list[-1], self.output_dim * (self.output_dim + 1) // 2
            )

        # self.elu_act = nn.ELU(alpha=1.0)
        self.softplus_act = nn.Softplus(beta=1.0, threshold=20.0)
        self.softplus_smooth = nn.Softplus(beta=0.1, threshold=20.0)
        self.softplus_sharp = nn.Softplus(beta=10.0, threshold=20.0)
        self.leaky_act = nn.LeakyReLU(self.leaky_relu_slope)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        # Now create a layer list
        self.layers = []
        for ichan, chan in enumerate(self.conv):
            self.layers.append(self.conv[ichan])
            self.layers.append(self.pool[ichan])
            self.layers.append(nn.LeakyReLU(self.leaky_relu_slope))
        self.layers.append(nn.Flatten())
        for ifc, fc in enumerate(self.fc):
            self.layers.append(self.fc[ifc])
            self.layers.append(nn.Tanh())

        if cyc_mode.lower() == "discharge-chargecc":
            self.layers_aux = []
            for ichan, chan in enumerate(self.conv_aux):
                self.layers_aux.append(self.conv_aux[ichan])
                self.layers_aux.append(self.pool_aux[ichan])
                self.layers_aux.append(nn.LeakyReLU(self.leaky_relu_slope))
            self.layers_aux.append(nn.Flatten())
            for ifc, fc in enumerate(self.fc_aux):
                self.layers_aux.append(self.fc_aux[ifc])
                self.layers_aux.append(nn.Tanh())

        self.mu_layers = []
        for ifc, fc in enumerate(self.fc_mu):
            self.mu_layers.append(self.fc_mu[ifc])
            self.mu_layers.append(nn.Tanh())
        self.mu_layers.append(self.fc_otpt_mu)
        if self.constrain_output:
            self.mu_layers.append(self.sigmoid)

        self.gamma_layers = []
        for ifc, fc in enumerate(self.fc_gamma):
            self.gamma_layers.append(self.fc_gamma[ifc])
            self.gamma_layers.append(nn.Tanh())
        self.gamma_layers.append(self.fc_otpt_gamma)
        if self.constrain_output and not self.dependent_outputs:
            self.gamma_layers.append(self.sigmoid)
        elif not self.constrain_output and not self.dependent_outputs:
            self.gamma_layers.append(self.softplus_act)
        elif self.dependent_outputs:
            pass

        self.cnn_layers = nn.Sequential(*self.layers)
        if cyc_mode.lower() == "discharge-chargecc":
            self.cnn_layers_aux = nn.Sequential(*self.layers_aux)
        self.model_mu_layers = nn.Sequential(*self.mu_layers)
        self.model_gamma_layers = nn.Sequential(*self.gamma_layers)

    def inv_transform_mu(self, mu_unscaled, min_par, amp_par):
        mu = mu_unscaled * amp_par + min_par
        return mu

    def inv_transform_gamma(self, gamma_unscaled, amp_par):
        gamma = gamma_unscaled * amp_par
        return gamma

    def transform_mu(self, mu_scaled, min_par, amp_par):
        mu = (mu_scaled - min_par) / amp_par
        return mu

    def transform_gamma(self, gamma_scaled, amp_par):
        gamma = gamma_scaled / amp_par
        return gamma

    def transform_output(self, mu_scaled, gamma_scaled, min_par, amp_par):
        return self.transform_mu(
            mu_scaled, min_par, amp_par
        ), self.transform_gamma(gamma_scaled, amp_par)

    def inv_transform_output(
        self, mu_unscaled, gamma_unscaled, min_par, amp_par
    ):
        return self.inv_transform_mu(
            mu_unscaled, min_par, amp_par
        ), self.inv_transform_gamma(gamma_unscaled, amp_par)

    def forward(self, x):
        if self.cyc_mode.lower() == "discharge-chargecc":
            nchans = x.shape[1]
            x_dis, x_chcc = torch.split(x, nchans // 2, dim=1)

            x_dis = self.cnn_layers(x_dis)
            x_chcc = self.cnn_layers_aux(x_chcc)

            x_conc = torch.cat((x_dis, x_chcc), dim=1)

            mu = self.model_mu_layers(x_conc)
            gamma = self.model_gamma_layers(x_conc)
            last_x = x_conc
        else:
            x = self.cnn_layers(x)

            mu = self.model_mu_layers(x)
            gamma = self.model_gamma_layers(x)
            last_x = x

        if self.dependent_outputs:
            # Create covariance matrix
            L = torch.zeros(
                last_x.size(0),
                self.output_dim,
                self.output_dim,
                device=last_x.device,
            )
            # Indices of the lower triangular matrix
            # first row is row coordinates
            # second row is col coordinates
            tril_indices = torch.tril_indices(
                row=self.output_dim, col=self.output_dim, offset=0
            )

            # Fill lower triangular
            L[:, tril_indices[0], tril_indices[1]] = gamma

            # Apply softplus to diagonal for positive definiteness
            diagonal_indices = torch.arange(self.output_dim)
            L[:, diagonal_indices, diagonal_indices] = self.softplus_act(
                L[:, diagonal_indices, diagonal_indices]
            )

            # Covariance matrix
            cov = L @ L.transpose(-1, -2)
            gamma = cov

        return mu, gamma


class ProbParamFCNN(nn.Module):
    def __init__(
        self,
        input_shape,
        hidden_list,
        fc_mu_list,
        fc_gamma_list,
        loss_fn,
        cyc_mode="discharge",
        n_param_pred=6,
        dependent_outputs=False,
        enforce_licons=False,
        constrain_output=False,
        encoder_model=None,
        sim_config=None,
    ):
        logger.info("Creating probabilistic FCNN model")
        super(ProbParamFCNN, self).__init__()
        self.hidden_list = hidden_list
        self.loss_fn = loss_fn
        self.cyc_mode = cyc_mode
        self.n_param_pred = n_param_pred
        self.constrain_output = constrain_output
        self.sim_config = sim_config
        self.dependent_outputs = dependent_outputs
        self.enforce_licons = enforce_licons
        self.encoder_model = encoder_model
        if not self.cyc_mode.lower() == "discharge-chargecc":
            self.enforce_licons = False
        if self.enforce_licons:
            self.output_dim = self.n_param_pred - 1
        else:
            self.output_dim = self.n_param_pred
        if self.dependent_outputs:
            assert self.loss_fn == correlated_normal_loss
        else:
            assert self.loss_fn in [
                mse_loss,
                gumbel_loss,
                nll_loss,
                independent_normal_loss,
                independent_gumbel_loss,
            ]

        if self.sim_config is not None:
            self.sim_params = make_params(self.sim_config)
            self.max_par = torch.from_numpy(
                np.array(
                    [
                        self.sim_params["deg_" + var_name + "_max"]
                        for var_name in self.sim_params["deg_param_names"]
                    ]
                ).astype("float32")
            )
            self.min_par = torch.from_numpy(
                np.array(
                    [
                        self.sim_params["deg_" + var_name + "_min"]
                        for var_name in self.sim_params["deg_param_names"]
                    ]
                ).astype("float32")
            )
            if self.enforce_licons:
                self.max_par = self.max_par[:-1]
                self.min_par = self.min_par[:-1]

            self.amp_par = self.max_par - self.min_par

        self.fcnn = []
        self.pool = []
        for ihidden, hidden in enumerate(hidden_list):
            if ihidden == 0:
                self.fcnn.append(
                    nn.Linear(
                        in_features=input_shape[0],
                        out_features=hidden,
                    )
                )
                self.fcnn.append(nn.Tanh())
            else:
                self.fcnn.append(
                    nn.Linear(
                        in_features=hidden_list[ihidden - 1],
                        out_features=hidden,
                    )
                )
                self.fcnn.append(nn.Tanh())

        fc_list_end = hidden_list[-1]

        self.fc_mu = []
        for ifc, fc in enumerate(fc_mu_list):
            if ifc == 0:
                self.fc_mu.append(nn.Linear(fc_list_end, fc))
            else:
                self.fc_mu.append(nn.Linear(fc_mu_list[ifc - 1], fc))
        self.fc_gamma = []
        for ifc, fc in enumerate(fc_gamma_list):
            if ifc == 0:
                self.fc_gamma.append(nn.Linear(fc_list_end, fc))
            else:
                self.fc_gamma.append(nn.Linear(fc_gamma_list[ifc - 1], fc))

        self.fc_otpt_mu = nn.Linear(fc_mu_list[-1], self.output_dim)
        if not self.dependent_outputs:
            self.fc_otpt_gamma = nn.Linear(fc_gamma_list[-1], self.output_dim)
        else:
            self.fc_otpt_gamma = nn.Linear(
                fc_gamma_list[-1], self.output_dim * (self.output_dim + 1) // 2
            )

        # self.elu_act = nn.ELU(alpha=1.0)
        self.softplus_act = nn.Softplus(beta=1.0, threshold=20.0)
        self.softplus_smooth = nn.Softplus(beta=0.1, threshold=20.0)
        self.softplus_sharp = nn.Softplus(beta=10.0, threshold=20.0)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.mu_layers = []
        for ifc, fc in enumerate(self.fc_mu):
            self.mu_layers.append(self.fc_mu[ifc])
            self.mu_layers.append(nn.Tanh())
        self.mu_layers.append(self.fc_otpt_mu)
        if self.constrain_output:
            self.mu_layers.append(self.sigmoid)

        self.gamma_layers = []
        for ifc, fc in enumerate(self.fc_gamma):
            self.gamma_layers.append(self.fc_gamma[ifc])
            self.gamma_layers.append(nn.Tanh())
        self.gamma_layers.append(self.fc_otpt_gamma)
        if self.constrain_output and not self.dependent_outputs:
            self.gamma_layers.append(self.sigmoid)
        elif not self.constrain_output and not self.dependent_outputs:
            self.gamma_layers.append(self.softplus_act)
        elif self.dependent_outputs:
            pass

        self.fcnn_layers = nn.Sequential(*self.fcnn)
        self.model_mu_layers = nn.Sequential(*self.mu_layers)
        self.model_gamma_layers = nn.Sequential(*self.gamma_layers)

    def inv_transform_mu(self, mu_unscaled, min_par, amp_par):
        mu = mu_unscaled * amp_par + min_par
        return mu

    def inv_transform_gamma(self, gamma_unscaled, amp_par):
        gamma = gamma_unscaled * amp_par
        return gamma

    def transform_mu(self, mu_scaled, min_par, amp_par):
        mu = (mu_scaled - min_par) / amp_par
        return mu

    def transform_gamma(self, gamma_scaled, amp_par):
        gamma = gamma_scaled / amp_par
        return gamma

    def transform_output(self, mu_scaled, gamma_scaled, min_par, amp_par):
        return self.transform_mu(
            mu_scaled, min_par, amp_par
        ), self.transform_gamma(gamma_scaled, amp_par)

    def inv_transform_output(
        self, mu_unscaled, gamma_unscaled, min_par, amp_par
    ):
        return self.inv_transform_mu(
            mu_unscaled, min_par, amp_par
        ), self.inv_transform_gamma(gamma_unscaled, amp_par)

    def forward(self, x):
        if self.cyc_mode.lower() == "discharge-chargecc":
            nchans = x.shape[1]
            x_dis, x_chcc = torch.split(x, nchans // 2, dim=1)

            x_dis = self.fcnn_layers(x_dis)
            x_chcc = self.fcnn_layers_aux(x_chcc)

            x_conc = torch.cat((x_dis, x_chcc), dim=1)

            mu = self.model_mu_layers(x_conc)
            gamma = self.model_gamma_layers(x_conc)
            last_x = x_conc
        else:
            x = self.fcnn_layers(x)

            mu = self.model_mu_layers(x)
            gamma = self.model_gamma_layers(x)
            last_x = x

        if self.dependent_outputs:
            # Create covariance matrix
            L = torch.zeros(
                last_x.size(0),
                self.output_dim,
                self.output_dim,
                device=last_x.device,
            )
            # Indices of the lower triangular matrix
            # first row is row coordinates
            # second row is col coordinates
            tril_indices = torch.tril_indices(
                row=self.output_dim, col=self.output_dim, offset=0
            )

            # Fill lower triangular
            L[:, tril_indices[0], tril_indices[1]] = gamma

            # Apply softplus to diagonal for positive definiteness
            diagonal_indices = torch.arange(self.output_dim)
            L[:, diagonal_indices, diagonal_indices] = self.softplus_act(
                L[:, diagonal_indices, diagonal_indices]
            )

            # Covariance matrix
            cov = L @ L.transpose(-1, -2)
            gamma = cov

        return mu, gamma


class ParamCNN(nn.Module):
    def __init__(
        self,
        input_shape,
        chan_list,
        fc_list,
        fc_mu_list,
        loss_fn,
        n_param_pred=6,
        leaky_relu_slope=0.2,
        cyc_mode="discharge",
        constrain_output=False,
        sim_config=None,
    ):
        logger.info("Creating deterministic CNN model")
        super(ParamCNN, self).__init__()
        self.leaky_relu_slope = leaky_relu_slope
        self.chan_list = chan_list
        self.fc_list = fc_list
        self.loss_fn = loss_fn
        self.cyc_mode = cyc_mode
        self.n_param_pred = n_param_pred
        self.constrain_output = constrain_output
        self.sim_config = sim_config
        if self.sim_config is not None:
            self.sim_params = make_params(self.sim_config)
            self.max_par = torch.from_numpy(
                np.array(
                    [
                        self.sim_params["deg_" + var_name + "_max"]
                        for var_name in self.sim_params["deg_param_names"]
                    ]
                ).astype("float32")
            )
            self.min_par = torch.from_numpy(
                np.array(
                    [
                        self.sim_params["deg_" + var_name + "_min"]
                        for var_name in self.sim_params["deg_param_names"]
                    ]
                ).astype("float32")
            )
            self.amp_par = self.max_par - self.min_par

        assert len(chan_list) < int(np.log(input_shape[1]) / np.log(2))

        if cyc_mode.lower() == "discharge-chargecc":
            input_shape_0 = input_shape[0] // 2
            input_shape_1 = input_shape[1]
        else:
            input_shape_0 = input_shape[0]
            input_shape_1 = input_shape[1]

        self.conv = []
        self.pool = []
        for ichan, chan in enumerate(chan_list):
            if ichan == 0:
                self.conv.append(
                    nn.Conv1d(
                        in_channels=input_shape_0,
                        out_channels=chan,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                self.pool.append(nn.MaxPool1d(kernel_size=2, stride=2))
            else:
                self.conv.append(
                    nn.Conv1d(
                        in_channels=chan_list[ichan - 1],
                        out_channels=chan,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
            self.pool.append(nn.MaxPool1d(kernel_size=2, stride=2))

        self.fc = []
        for ifc, fc in enumerate(fc_list):
            if ifc == 0:
                self.fc.append(
                    nn.Linear(
                        chan_list[-1] * input_shape_1 // (2 ** len(chan_list)),
                        fc,
                    )
                )
            else:
                self.fc.append(nn.Linear(fc_list[ifc - 1], fc))

        if cyc_mode.lower() == "discharge-chargecc":

            self.conv_aux = []
            self.pool_aux = []
            for ichan, chan in enumerate(chan_list):
                if ichan == 0:
                    self.conv_aux.append(
                        nn.Conv1d(
                            in_channels=input_shape_0,
                            out_channels=chan,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        )
                    )
                    self.pool_aux.append(nn.MaxPool1d(kernel_size=2, stride=2))
                else:
                    self.conv_aux.append(
                        nn.Conv1d(
                            in_channels=chan_list[ichan - 1],
                            out_channels=chan,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        )
                    )
                self.pool_aux.append(nn.MaxPool1d(kernel_size=2, stride=2))

            self.fc_aux = []
            for ifc, fc in enumerate(fc_list):
                if ifc == 0:
                    self.fc_aux.append(
                        nn.Linear(
                            chan_list[-1]
                            * input_shape_1
                            // (2 ** len(chan_list)),
                            fc,
                        )
                    )
                else:
                    self.fc_aux.append(nn.Linear(fc_list[ifc - 1], fc))

            fc_list_end = 2 * fc_list[-1]

        else:
            fc_list_end = fc_list[-1]

        self.fc_mu = []
        for ifc, fc in enumerate(fc_mu_list):
            if ifc == 0:
                self.fc_mu.append(nn.Linear(fc_list_end, fc))
            else:
                self.fc_mu.append(nn.Linear(fc_mu_list[ifc - 1], fc))

        self.fc_otpt_mu = nn.Linear(fc_mu_list[-1], self.n_param_pred)
        # self.elu_act = nn.ELU(alpha=1.0)
        self.softplus_act = nn.Softplus(beta=1.0, threshold=20.0)
        self.softplus_smooth = nn.Softplus(beta=0.1, threshold=20.0)
        self.softplus_sharp = nn.Softplus(beta=10.0, threshold=20.0)
        self.leaky_act = nn.LeakyReLU(self.leaky_relu_slope)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        # Now create a layer list
        self.layers = []
        for ichan, chan in enumerate(self.conv):
            self.layers.append(self.conv[ichan])
            self.layers.append(self.pool[ichan])
            self.layers.append(nn.LeakyReLU(self.leaky_relu_slope))
        self.layers.append(nn.Flatten())
        for ifc, fc in enumerate(self.fc):
            self.layers.append(self.fc[ifc])
            self.layers.append(nn.Tanh())

        if cyc_mode.lower() == "discharge-chargecc":
            self.layers_aux = []
            for ichan, chan in enumerate(self.conv_aux):
                self.layers_aux.append(self.conv_aux[ichan])
                self.layers_aux.append(self.pool_aux[ichan])
                self.layers_aux.append(nn.LeakyReLU(self.leaky_relu_slope))
            self.layers_aux.append(nn.Flatten())
            for ifc, fc in enumerate(self.fc_aux):
                self.layers_aux.append(self.fc_aux[ifc])
                self.layers_aux.append(nn.Tanh())

        self.mu_layers = []
        for ifc, fc in enumerate(self.fc_mu):
            self.mu_layers.append(self.fc_mu[ifc])
            self.mu_layers.append(nn.Tanh())
        self.mu_layers.append(self.fc_otpt_mu)
        if self.constrain_output:
            self.mu_layers.append(self.sigmoid)

        self.cnn_layers = nn.Sequential(*self.layers)
        if cyc_mode.lower() == "discharge-chargecc":
            self.cnn_layers_aux = nn.Sequential(*self.layers_aux)
        self.model_mu_layers = nn.Sequential(*self.mu_layers)

    def inv_transform_mu(self, mu_unscaled, min_par, amp_par):
        mu = mu_unscaled * amp_par + min_par
        return mu

    def transform_mu(self, mu_scaled, min_par, amp_par):
        mu = (mu_scaled - min_par) / amp_par
        return mu

    def transform_output(self, mu_scaled, min_par, amp_par):
        return self.transform_mu(mu_scaled, min_par, amp_par)

    def inv_transform_output(self, mu_unscaled, min_par, amp_par):
        return self.inv_transform_mu(mu_unscaled, min_par, amp_par)

    def forward(self, x):
        if self.cyc_mode.lower() == "discharge-chargecc":
            nchans = x.shape[1]
            x_dis, x_chcc = torch.split(x, nchans // 2, dim=1)

            x_dis = self.cnn_layers(x_dis)
            x_chcc = self.cnn_layers_aux(x_chcc)

            x_conc = torch.cat((x_dis, x_chcc), dim=1)

            mu = self.model_mu_layers(x_conc)

        else:
            x = self.cnn_layers(x)

            mu = self.model_mu_layers(x)

        return mu


def cluster_rand(input_tensor):
    output_tensor = torch.empty_like(input_tensor)
    # Region 1: [0, 0.5)
    mask1 = input_tensor < 0.5
    output_tensor[mask1] = (
        0.4 * input_tensor[mask1]
    )  # linear map from [0,0.5] -> [0,0.2]
    # Region 2: [0.5, 1]
    mask2 = ~mask1
    output_tensor[mask2] = (
        0.4 * (input_tensor[mask2] - 0.5) + 0.8
    )  # linear map from [0.5,1] -> [0.8,1]
    return output_tensor


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
        # batch_in += bias.repeat(batch_in.shape[0], 1, 1) * cluster_rand(torch.rand((batch_in.shape[0], 1, 1)))
        batch_in += bias.repeat(batch_in.shape[0], 1, 1) * torch.rand(
            (batch_in.shape[0], 1, 1)
        )
    batch_in = torch.clamp(batch_in, min=a_min, max=a_max)
    batch_in = scaler_X.transform(batch_in)
    return batch_in


def apply_noise_unscaled(
    batch_in: torch.Tensor,
    noise_levels: torch.Tensor,
    a_min: torch.Tensor,
    a_max: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    noise = (torch.rand(batch_in.shape) - 0.5) * torch.reshape(
        noise_levels, (1, -1, 1)
    )
    batch_in += noise
    if bias is not None:
        # batch_in += bias.repeat(batch_in.shape[0], 1, 1) * cluster_rand(torch.rand((batch_in.shape[0], 1, 1)))
        batch_in += bias.repeat(batch_in.shape[0], 1, 1) * torch.rand(
            (batch_in.shape[0], 1, 1)
        )
    batch_in = torch.clamp(batch_in, min=a_min, max=a_max)
    return batch_in


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
    model: nn.Module,
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
    if model.loss_fn in [elbo_independent_normal_loss]:
        assert prior is not None
        if not prior.mean.device == device:
            new_prior = dist.MultivariateNormal(
                prior.mean.to(device), prior.covariance_matrix.to(device)
            )
            prior = new_prior

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
                if isinstance(model, ParamCNN):
                    pred = model(batch_in.to(device))
                    if model.constrain_output:
                        pred = model.inv_transform_output(
                            mu,
                            model.min_par.to(device),
                            model.amp_par.to(device),
                        )
                    loss = mse_loss(pred, batch[1].to(device))
                elif isinstance(model, ProbParamCNN) or isinstance(
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
                    if model.loss_fn in [elbo_independent_normal_loss]:
                        loss = model.loss_fn(
                            mu, gamma, prior, batch[1].to(device), temp
                        )
                    else:
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
    model: ParamCNN,
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
            if isinstance(model, ParamCNN):
                pred = model(batch_in.to(device))
                if model.constrain_output:
                    pred = model.inv_transform_output(
                        pred,
                        model.min_par.to(device),
                        model.amp_par.to(device),
                    )
                loss = mse_loss(pred, batch[1].to(device))
            elif isinstance(model, ProbParamCNN) or isinstance(
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

                if model.loss_fn in [elbo_independent_normal_loss]:
                    loss = model.loss_fn(
                        mu, gamma, prior, batch[1].to(device), 1.0
                    )
                else:
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
    model: ParamCNN,
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
            if isinstance(model, ParamCNN):
                pred = model(batch_in.to(device))
                if model.constrain_output:
                    pred = model.inv_transform_output(
                        pred,
                        model.min_par.to(device),
                        model.amp_par.to(device),
                    )
                if post_fn in [accuracy, rel_accuracy]:
                    post_val = post_fn(pred, batch[1].to(device))
                elif post_fn in [identifiability]:
                    logger.error(
                        "Identifiability can only be computed with probabilistic model"
                    )
                    sys.exit()
            elif isinstance(model, ProbParamCNN) or isinstance(
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
