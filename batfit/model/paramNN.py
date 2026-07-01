import numpy as np
import torch
import torch.nn as nn

from batfit import logger
from batfit.utils.torch_utils import (
    get_device_type,
    get_num_parameters,
    load_model,
    log_training,
    make_dataset_from_np,
    prepare_log,
    save_model,
)

from .param_utils.model_utils import (
    _ParamScalingMixin,
    _ProbParamBase,
    _ProbParamFMBase,
    _VFWrapper,
    _build_cnn_encoder,
    _build_hidden_fcnn_layers,
    _build_output_heads,
)


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

        self.cnn_layers, self.cnn_layers_aux, fc_list_end = _build_cnn_encoder(
            input_shape_0,
            input_shape_1,
            chan_list,
            fc_list,
            leaky_relu_slope,
            cyc_mode,
        )

        self.model_mu_layers, self.model_gamma_layers = _build_output_heads(
            fc_list_end,
            fc_mu_list,
            fc_gamma_list,
            self.output_dim,
            self.dependent_outputs,
            self.constrain_output,
        )

    def forward(self, x):
        if self.cyc_mode.lower() == "discharge-chargecc":
            nchans = x.shape[1]
            x_dis, x_chcc = torch.split(x, nchans // 2, dim=1)

            x_dis = self.cnn_layers(x_dis)
            x_chcc = self.cnn_layers_aux(x_chcc)

            x_conc = torch.cat((x_dis, x_chcc), dim=1)

            mu = self.model_mu_layers(x_conc)
            gamma = self.model_gamma_layers(x_conc)
        else:
            x = self.cnn_layers(x)

            mu = self.model_mu_layers(x)
            gamma = self.model_gamma_layers(x)

        if self.dependent_outputs:
            gamma = self._cholesky_cov(gamma)

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
        elementary_fcnn = _build_hidden_fcnn_layers(
            input_shape[0], hidden_list
        )
        self.fcnn = []
        for ihidden, hidden in enumerate(elementary_fcnn):
            self.fcnn.append(elementary_fcnn[ihidden])
            self.fcnn.append(nn.Tanh())
        if self.cyc_mode.lower() == "discharge-chargecc":
            elementary_fcnn_aux = _build_hidden_fcnn_layers(
                input_shape[0], hidden_list
            )
            self.fcnn_aux = []
            for ihidden, hidden in enumerate(elementary_fcnn_aux):
                self.fcnn_aux.append(elementary_fcnn_aux[ihidden])
                self.fcnn_aux.append(nn.Tanh())
            fc_list_end = 2 * hidden_list[-1]
        else:
            fc_list_end = hidden_list[-1]

        self.model_mu_layers, self.model_gamma_layers = _build_output_heads(
            fc_list_end,
            fc_mu_list,
            fc_gamma_list,
            self.output_dim,
            self.dependent_outputs,
            self.constrain_output,
        )

        self.fcnn_layers = nn.Sequential(*self.fcnn)
        if self.cyc_mode.lower() == "discharge-chargecc":
            self.fcnn_layers_aux = nn.Sequential(*self.fcnn_aux)

    def forward(self, x):
        if self.cyc_mode.lower() == "discharge-chargecc":
            nchans = x.shape[1]
            x_dis, x_chcc = torch.split(x, nchans // 2, dim=1)

            x_dis = self.fcnn_layers(x_dis)
            x_chcc = self.fcnn_layers_aux(x_chcc)

            x_conc = torch.cat((x_dis, x_chcc), dim=1)

            mu = self.model_mu_layers(x_conc)
            gamma = self.model_gamma_layers(x_conc)
        else:
            x = self.fcnn_layers(x)

            mu = self.model_mu_layers(x)
            gamma = self.model_gamma_layers(x)

        if self.dependent_outputs:
            gamma = self._cholesky_cov(gamma)

        return mu, gamma


class ProbProtParamCNN(_ProbParamBase):
    """CNN encoder for electrochemical signal with protocol parameter fusion.

    The CNN encodes the input signal (time, voltage, etc.), then the flattened
    CNN output is concatenated with the protocol parameters before passing through
    optional additional FC layers. The combined representation is then split into
    independent mu and gamma prediction heads.
    """

    def __init__(
        self,
        input_shape: tuple[int, int],
        chan_list: list[int],
        fc_list: list[int],
        fc_prot_list: list[int],
        fc_mu_list: list[int],
        fc_gamma_list: list[int],
        loss_fn,
        n_prot_params: int,
        leaky_relu_slope: float = 0.2,
        cyc_mode: str = "chirp",
        n_param_pred: int = 6,
        dependent_outputs: bool = False,
        constrain_output: bool = False,
        encoder_model=None,
        sim_config=None,
    ):
        logger.info(
            "Creating probabilistic CNN model with protocol parameters"
        )
        assert cyc_mode.lower() != "discharge-chargecc"
        if cyc_mode.lower() in ["discharge-chargecc"]:
            raise NotImplementedError(
                "We do a fusing after CNN encoding, we need to make it work for dual conv encoders"
            )
        super(ProbProtParamCNN, self).__init__(
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
        self.fc_prot_list = fc_prot_list
        self.n_prot_params = n_prot_params

        assert len(chan_list) < int(np.log(input_shape[1]) / np.log(2))

        # Conv encoder that process electrochem signal
        self.cnn_layers, _, _ = _build_cnn_encoder(
            input_shape[0],
            input_shape[1],
            chan_list,
            fc_list,
            leaky_relu_slope,
            cyc_mode,
        )

        # After CNN output + prot_params concatenation
        prot_input_size = fc_list[-1] + n_prot_params
        _prot_layers = []
        if fc_prot_list:
            prot_fc = _build_hidden_fcnn_layers(prot_input_size, fc_prot_list)
            for ifc in range(len(prot_fc)):
                _prot_layers.append(prot_fc[ifc])
                _prot_layers.append(nn.Tanh())
            fc_list_end = fc_prot_list[-1]
        else:
            fc_list_end = prot_input_size
        self.prot_layers = nn.Sequential(*_prot_layers)

        self.model_mu_layers, self.model_gamma_layers = _build_output_heads(
            fc_list_end,
            fc_mu_list,
            fc_gamma_list,
            self.output_dim,
            self.dependent_outputs,
            self.constrain_output,
        )

    def forward(
        self, x: torch.Tensor, prot_params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass combining electrochemical signal and protocol parameters.

        :param x: electrochemical signal of shape (batch, channels, time)
        :param prot_params: protocol parameters of shape (batch, n_prot_params)
        :return: (mu, gamma) — predicted parameter means and variances/covariance
        """
        x = self.cnn_layers(x)
        x = torch.cat((x, prot_params), dim=1)
        x = self.prot_layers(x)

        mu = self.model_mu_layers(x)
        gamma = self.model_gamma_layers(x)

        if self.dependent_outputs:
            gamma = self._cholesky_cov(gamma)

        return mu, gamma


