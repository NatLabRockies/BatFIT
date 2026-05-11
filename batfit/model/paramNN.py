from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn

from batfit import logger
from batfit.preprocess.sim_setup import make_params
from batfit.utils.torch_utils import (
    get_device_type,
    get_num_parameters,
    load_model,
    log_training,
    make_dataset_from_np,
    prepare_log,
    save_model,
)

from .losses import (                                                                                                        
    correlated_normal_loss,                                                                                                  
    gumbel_loss,                                                                                                             
    independent_gumbel_loss,                                                                                                 
    independent_normal_loss,                                                                                                 
    mse_loss,                                                                                                                
    nll_loss,                                                                                                                
    pinball_loss,                                                                                                            
) 

def _build_conv_layers(
    input_shape_0:int , chan_list: list[int]
  ) -> tuple[list[nn.Module], list[nn.Module]]:
    """
    Build convolution and maxpooling from channel list to reduce dimension
    """
    conv_l = []
    pool_l = []
   
    # Conv layers
    for ichan, chan in enumerate(chan_list):
        if ichan == 0:
            conv_l.append(
                nn.Conv1d(
                    in_channels=input_shape_0,
                    out_channels=chan,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            pool_l.append(nn.MaxPool1d(kernel_size=2, stride=2))
        else:
            conv_l.append(
                nn.Conv1d(
                    in_channels=chan_list[ichan - 1],
                    out_channels=chan,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
        pool_l.append(nn.MaxPool1d(kernel_size=2, stride=2))
    return conv_l, pool_l

def _build_conv_fc_layers(
    input_shape_1:int , chan_list: list[int], fc_list: list[int]
  ) -> list[nn.Module]:
    """
    Build fully connected layer after conv
    """
    fc_l = []
    for ifc, fc in enumerate(fc_list):
        if ifc == 0:
            fc_l.append(
                nn.Linear(
                    chan_list[-1] * input_shape_1 // (2 ** len(chan_list)),
                    fc,
                )
            )
        else:
            fc_l.append(nn.Linear(fc_list[ifc - 1], fc))
    return fc_l


def _build_cnn_encoder(
      input_shape_0, input_shape_1, chan_list, fc_list, leaky_relu_slope, cyc_mode
  ):
    conv, pool = _build_conv_layers(input_shape_0, chan_list)
    fc = _build_conv_fc_layers(input_shape_1, chan_list, fc_list)

    # CNN layers + activation
    _cnn_layers = []
    for ichan, chan in enumerate(conv):
        _cnn_layers.append(conv[ichan])
        _cnn_layers.append(pool[ichan])
        _cnn_layers.append(nn.LeakyReLU(leaky_relu_slope))
    _cnn_layers.append(nn.Flatten())
    for ifc, fc in enumerate(fc):
        _cnn_layers.append(fc[ifc])
        _cnn_layers.append(nn.Tanh())
    cnn_layers = nn.Sequential(*_cnn_layers)
    
    # If we process two separate protocols
    if cyc_mode.lower() == "discharge-chargecc":
        conv_aux, pool_aux = _build_conv_layers(input_shape_0, chan_list)
        fc_aux = _build_conv_fc_layers(input_shape_1, chan_list, fc_list)
        _cnn_layers_aux = []
        for ichan, chan in enumerate(conv_aux):
            _cnn_layers_aux.append(conv_aux[ichan])
            _cnn_layers_aux.append(pool_aux[ichan])
            _cnn_layers_aux.append(nn.LeakyReLU(leaky_relu_slope))
        _cnn_layers_aux.append(nn.Flatten())
        for ifc, fc in enumerate(fc_aux):
            _cnn_layers_aux.append(fc_aux[ifc])
            _cnn_layers_aux.append(nn.Tanh())
        cnn_layers_aux = nn.Sequential(*_cnn_layers_aux)
        fc_list_end = 2 * fc_list[-1]
    else:
        cnn_layers_aux = None
        fc_list_end = fc_list[-1]

    return cnn_layers, cnn_layers_aux, fc_list_end

class _ProbParamBase(nn.Module, ABC):
    def __init__(
        self,
        loss_fn,
        cyc_mode="discharge",
        n_param_pred=6,
        dependent_outputs=False,
        constrain_output=False,
        encoder_model=None,
        sim_config=None,
    ):
        super(_ProbParamBase, self).__init__()
        self.loss_fn = loss_fn
        self.cyc_mode = cyc_mode
        self.n_param_pred = n_param_pred
        self.constrain_output = constrain_output
        self.sim_config = sim_config
        self.dependent_outputs = dependent_outputs
        self.encoder_model = encoder_model
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
            self.amp_par = self.max_par - self.min_par

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

    @abstractmethod
    def forward(self, x):
        pass


class ProbParamCNN(_ProbParamBase):
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
        constrain_output=False,
        encoder_model=None,
        sim_config=None,
    ):
        logger.info("Creating probabilistic CNN model")
        super(ProbParamCNN, self).__init__(
            loss_fn=loss_fn,
            cyc_mode=cyc_mode,
            n_param_pred=n_param_pred,
            dependent_outputs=dependent_outputs,
            constrain_output=constrain_output,
            encoder_model=encoder_model,
            sim_config=sim_config,
        )
        self.leaky_relu_slope = leaky_relu_slope
        self.chan_list = chan_list
        self.fc_list = fc_list

        assert len(chan_list) < int(np.log(input_shape[1]) / np.log(2))

        if cyc_mode.lower() == "discharge-chargecc":
            input_shape_0 = input_shape[0] // 2
            input_shape_1 = input_shape[1]
        else:
            input_shape_0 = input_shape[0]
            input_shape_1 = input_shape[1]

        self.cnn_layers, self.cnn_layers_aux, fc_list_end = _build_cnn_encoder(input_shape_0, input_shape_1, chan_list, fc_list, leaky_relu_slope, cyc_mode)

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

        self.model_mu_layers = nn.Sequential(*self.mu_layers)
        self.model_gamma_layers = nn.Sequential(*self.gamma_layers)

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


class ProbParamFCNN(_ProbParamBase):
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
        constrain_output=False,
        encoder_model=None,
        sim_config=None,
    ):
        logger.info("Creating probabilistic FCNN model")
        super(ProbParamFCNN, self).__init__(
            loss_fn=loss_fn,
            cyc_mode=cyc_mode,
            n_param_pred=n_param_pred,                                                                           
            dependent_outputs=dependent_outputs,
            constrain_output=constrain_output,
            encoder_model=encoder_model,
            sim_config=sim_config,
        )
        self.hidden_list = hidden_list
        self.fcnn = []
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



