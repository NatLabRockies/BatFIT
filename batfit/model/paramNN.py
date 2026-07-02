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
        num_attn_heads: int = 0,
        attn_dropout: float = 0.0,
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
            num_attn_heads=num_attn_heads,
            attn_dropout=attn_dropout,
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
        num_attn_heads: int = 0,
        attn_dropout: float = 0.0,
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
            num_attn_heads=num_attn_heads,
            attn_dropout=attn_dropout,
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


class ProbParamFM(_ProbParamFMBase):
    """Flow matching model for battery parameter estimation.

    Two encoder modes are supported:

    **CNN mode** (default): a 1-D CNN is trained jointly with the velocity
    field MLP end-to-end.  Requires ``input_shape``, ``chan_list``,
    ``fc_list``.  ``cyc_mode="discharge-chargecc"`` is supported and uses
    two independent CNN encoders whose embeddings are concatenated.

    **External encoder mode**: a pre-trained encoder (e.g. the
    ``ConvEncoder1D`` sub-module of a ``VAECNN``) is passed via
    ``encoder_model``.  Its parameters are **frozen** and only the velocity
    field MLP is trained.  The encoder must expose a ``latent_dim: int``
    attribute and its ``forward(x)`` must return ``(mu, ...)`` where the
    first element is the deterministic embedding (e.g. the VAE posterior
    mean).  Pass ``vae_model.encoder``, not the full ``VAECNN``.
    ``cyc_mode="discharge-chargecc"`` is not supported in this mode.

    Training
    --------
    Call ``forward(x, z_t, t)`` to obtain the predicted velocity and regress
    it against the target velocity from ``AffineProbPath`` using
    ``flow_matching_loss``.

    Inference
    ---------
    Call ``sample(x, n_samples)`` to draw posterior samples by integrating the
    learned ODE from N(0, I) to t=1 with the midpoint method.
    """

    def __init__(
        self,
        vf_hidden_list: list[int],
        input_shape: tuple[int, int] | None = None,
        chan_list: list[int] | None = None,
        fc_list: list[int] | None = None,
        encoder_model: nn.Module | None = None,
        leaky_relu_slope: float = 0.2,
        cyc_mode: str = "discharge",
        n_param_pred: int = 6,
        sim_config: str | None = None,
        use_prior_matching: bool = False,
        num_attn_heads: int = 0,
        attn_dropout: float = 0.0,
    ):
        """
        :param vf_hidden_list: hidden dims of the velocity field MLP
        :param input_shape: (n_channels, n_time_points); required in CNN mode
        :param chan_list: Conv1d output channels per layer; required in CNN mode
        :param fc_list: FC hidden dims after the CNN; required in CNN mode
        :param encoder_model: pre-trained encoder (e.g. ``vae.encoder``);
                              when given, CNN params are ignored and the
                              encoder weights are frozen
        :param leaky_relu_slope: negative slope for LeakyReLU in CNN mode
        :param cyc_mode: cycling mode; "discharge-chargecc" uses dual CNN
                         encoders (CNN mode only)
        :param n_param_pred: number of degradation parameters to estimate
        :param sim_config: path to sim config YAML for physical scaling;
                           None skips scaling init
        :param use_prior_matching: if True, use the empirical training-data
                                   distribution as the base (requires calling
                                   :meth:`set_prior_data` before training);
                                   when False, N(0, I) is used
        :param num_attn_heads: if > 0, insert a :class:`_SelfAttentionBlock`
                               after the last CNN conv layer; must divide
                               ``chan_list[-1]``; ignored when
                               ``encoder_model`` is provided
        :param attn_dropout: dropout inside MultiheadAttention
        """
        _cnn_mode = encoder_model is None
        if _cnn_mode and (
            input_shape is None or chan_list is None or fc_list is None
        ):
            raise ValueError(
                "CNN mode requires input_shape, chan_list, and fc_list. "
                "To use a pre-trained encoder pass encoder_model instead."
            )
        if not _cnn_mode and cyc_mode.lower() == "discharge-chargecc":
            raise NotImplementedError(
                "discharge-chargecc is not supported with an external encoder_model."
            )

        logger.info("Creating flow matching CNN model (ProbParamFM)")
        super().__init__(
            cyc_mode=cyc_mode,
            n_param_pred=n_param_pred,
            sim_config=sim_config,
            use_prior_matching=use_prior_matching,
        )
        self.vf_hidden_list = vf_hidden_list
        self.leaky_relu_slope = leaky_relu_slope

        if _cnn_mode:
            self.chan_list = chan_list
            self.fc_list = fc_list
            assert len(chan_list) < int(np.log(input_shape[1]) / np.log(2))
            input_shape_0 = (
                input_shape[0] // 2
                if cyc_mode.lower() == "discharge-chargecc"
                else input_shape[0]
            )
            self.cnn_layers, self.cnn_layers_aux, emb_dim = _build_cnn_encoder(
                input_shape_0,
                input_shape[1],
                chan_list,
                fc_list,
                leaky_relu_slope,
                cyc_mode,
                num_attn_heads=num_attn_heads,
                attn_dropout=attn_dropout,
            )
            self.encoder_model = None
        else:
            if not hasattr(encoder_model, "latent_dim"):
                raise ValueError(
                    "encoder_model must expose a latent_dim: int attribute. "
                    "Pass vae_model.encoder (a ConvEncoder1D), not the full VAECNN."
                )
            # Freeze pre-trained encoder weights; only the velocity field MLP trains
            for param in encoder_model.parameters():
                param.requires_grad_(False)
            self.encoder_model = encoder_model
            emb_dim = encoder_model.latent_dim

        # Velocity field MLP
        # Input: [z_t (n_param_pred) | t (1) | embedding (emb_dim)]
        vf_input_dim = n_param_pred + 1 + emb_dim
        vf_fc = _build_hidden_fcnn_layers(vf_input_dim, vf_hidden_list)
        _vf = []
        for layer in vf_fc:
            _vf.append(layer)
            _vf.append(nn.Tanh())
        _vf.append(nn.Linear(vf_hidden_list[-1], n_param_pred))
        self.vf_layers = nn.Sequential(*_vf)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the signal into an embedding vector.

        In CNN mode uses the jointly trained convolutional encoder. In external
        encoder mode calls ``encoder_model.forward(x)`` and takes the first
        returned tensor as the deterministic embedding (the VAE posterior mean
        when using a ``ConvEncoder1D``).

        :param x: input signal, shape (batch, channels, time)
        :return: embedding, shape (batch, emb_dim)
        """
        if self.encoder_model is not None:
            out = self.encoder_model(x)
            return out[0] if isinstance(out, (tuple, list)) else out
        else:
            if self.cyc_mode.lower() == "discharge-chargecc":
                nchans = x.shape[1]
                x_dis, x_chcc = torch.split(x, nchans // 2, dim=1)
                return torch.cat(
                    (self.cnn_layers(x_dis), self.cnn_layers_aux(x_chcc)), dim=1
                )
            else:
                return self.cnn_layers(x)

    def forward(
        self,
        x: torch.Tensor,
        z_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the velocity u(z_t, t | x) of the conditional flow.

        The velocity field u(z_t, t | x) is the quantity learned by flow
        matching: it is the time-derivative of the probability path that
        interpolates between a base sample x_0 ~ N(0, I) and a parameter
        sample x_1 ~ p(params | x). At each training step, z_t and t are
        sampled from the path (via AffineProbPath from flow_matching), and u
        is regressed against the straight-line target velocity (x_1 - x_0)
        using flow_matching_loss.

        :param x: electrochemical signal, shape (batch, channels, time)
        :param z_t: particle positions in parameter space at time t,
                    shape (batch, n_param_pred)
        :param t: flow time in [0, 1], shape (batch,)
        :return: predicted velocity, shape (batch, n_param_pred)
        """
        context = self._encode(x)
        return self._velocity_forward(z_t, t, context)

    def sample(
        self,
        x: torch.Tensor,
        n_samples: int,
        n_steps: int = 100,
    ) -> torch.Tensor:
        """Sample from the approximate posterior p(params | x).

        Encodes x once, then integrates the learned ODE from z_0 ~ N(0, I)
        at t=0 to t=1 using the midpoint method. Each observation in the
        batch independently produces n_samples posterior draws.

        :param x: electrochemical signal, shape (batch, channels, time)
        :param n_samples: number of posterior samples per observation
        :param n_steps: number of ODE integration steps
        :return: posterior samples, shape (batch, n_samples, n_param_pred)
        """
        context = self._encode(x)
        return self._sample_from_context(
            context, x.shape[0], n_samples, n_steps, x.device
        )


class ProbProtParamFM(_ProbParamFMBase):
    """Flow matching model conditioned on both signal and protocol parameters.

    The electrochemical signal is encoded by a jointly trained 1-D CNN.  Its
    embedding is concatenated with the protocol parameters and optionally
    passed through fusion FC layers before serving as the context for the
    velocity field MLP.

    Training
    --------
    Call ``forward(x, prot_params, z_t, t)`` to obtain the predicted velocity
    and regress it against the target velocity from ``AffineProbPath`` using
    ``flow_matching_loss``.

    Inference
    ---------
    Call ``sample(x, prot_params, n_samples)`` to draw posterior samples by
    integrating the learned ODE from N(0, I) to t=1 with the midpoint method.
    """

    def __init__(
        self,
        input_shape: tuple[int, int],
        chan_list: list[int],
        fc_list: list[int],
        fc_prot_list: list[int],
        vf_hidden_list: list[int],
        n_prot_params: int,
        leaky_relu_slope: float = 0.2,
        cyc_mode: str = "chirp",
        n_param_pred: int = 6,
        sim_config: str | None = None,
        use_prior_matching: bool = False,
        num_attn_heads: int = 0,
        attn_dropout: float = 0.0,
    ):
        """
        :param input_shape: (n_channels, n_time_points) of the input signal
        :param chan_list: Conv1d output channels per layer
        :param fc_list: FC hidden dims after the CNN
        :param fc_prot_list: hidden dims for the protocol fusion FC layers;
                             empty list skips fusion (prot_params concatenated
                             directly to the CNN embedding)
        :param vf_hidden_list: hidden dims of the velocity field MLP
        :param n_prot_params: number of protocol parameters
        :param leaky_relu_slope: negative slope for LeakyReLU in the CNN
        :param cyc_mode: cycling mode; "discharge-chargecc" is not supported
        :param n_param_pred: number of degradation parameters to estimate
        :param sim_config: path to sim config YAML for physical scaling;
                           None skips scaling init
        :param use_prior_matching: if True, use the empirical training-data
                                   distribution as the base (requires calling
                                   :meth:`set_prior_data` before training);
                                   when False, N(0, I) is used
        :param num_attn_heads: if > 0, insert a :class:`_SelfAttentionBlock`
                               after the last CNN conv layer; must divide
                               ``chan_list[-1]``
        :param attn_dropout: dropout inside MultiheadAttention
        """
        if cyc_mode.lower() == "discharge-chargecc":
            raise NotImplementedError(
                "discharge-chargecc is not supported in ProbProtParamFM."
            )

        logger.info(
            "Creating flow matching CNN model with protocol parameters "
            "(ProbProtParamFM)"
        )
        super().__init__(
            cyc_mode=cyc_mode,
            n_param_pred=n_param_pred,
            sim_config=sim_config,
            use_prior_matching=use_prior_matching,
        )
        self.leaky_relu_slope = leaky_relu_slope
        self.chan_list = chan_list
        self.fc_list = fc_list
        self.fc_prot_list = fc_prot_list
        self.vf_hidden_list = vf_hidden_list
        self.n_prot_params = n_prot_params

        assert len(chan_list) < int(np.log(input_shape[1]) / np.log(2))

        # CNN encoder for the electrochemical signal
        self.cnn_layers, _, _ = _build_cnn_encoder(
            input_shape[0],
            input_shape[1],
            chan_list,
            fc_list,
            leaky_relu_slope,
            cyc_mode,
            num_attn_heads=num_attn_heads,
            attn_dropout=attn_dropout,
        )

        # Protocol fusion: [cnn_emb | prot_params] -> optional FC -> context
        prot_input_size = fc_list[-1] + n_prot_params
        _prot_layers = []
        if fc_prot_list:
            prot_fc = _build_hidden_fcnn_layers(prot_input_size, fc_prot_list)
            for layer in prot_fc:
                _prot_layers.append(layer)
                _prot_layers.append(nn.Tanh())
            context_dim = fc_prot_list[-1]
        else:
            context_dim = prot_input_size
        self.prot_layers = nn.Sequential(*_prot_layers)

        # Velocity field MLP
        # Input: [z_t (n_param_pred) | t (1) | context (context_dim)]
        vf_input_dim = n_param_pred + 1 + context_dim
        vf_fc = _build_hidden_fcnn_layers(vf_input_dim, vf_hidden_list)
        _vf = []
        for layer in vf_fc:
            _vf.append(layer)
            _vf.append(nn.Tanh())
        _vf.append(nn.Linear(vf_hidden_list[-1], n_param_pred))
        self.vf_layers = nn.Sequential(*_vf)

    def _encode_context(
        self, x: torch.Tensor, prot_params: torch.Tensor
    ) -> torch.Tensor:
        """Encode signal and protocol parameters into a context vector.

        Runs the signal through the CNN encoder, concatenates the protocol
        parameters, then applies the optional fusion layers.

        :param x: electrochemical signal, shape (batch, channels, time)
        :param prot_params: protocol parameters, shape (batch, n_prot_params)
        :return: context embedding, shape (batch, context_dim)
        """
        cnn_emb = self.cnn_layers(x)
        fused = torch.cat((cnn_emb, prot_params), dim=1)
        return self.prot_layers(fused)

    def forward(
        self,
        x: torch.Tensor,
        prot_params: torch.Tensor,
        z_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the velocity u(z_t, t | x, prot_params) of the conditional flow.

        The velocity field is conditioned jointly on the electrochemical signal
        and the protocol parameters.  At each training step, z_t and t are
        sampled from the probability path (via AffineProbPath) between a noise
        sample x_0 ~ N(0, I) and the ground-truth parameter sample x_1, and
        the predicted velocity is regressed against the straight-line target
        (x_1 - x_0) using ``flow_matching_loss``.

        :param x: electrochemical signal, shape (batch, channels, time)
        :param prot_params: protocol parameters, shape (batch, n_prot_params)
        :param z_t: particle positions in parameter space at time t,
                    shape (batch, n_param_pred)
        :param t: flow time in [0, 1], shape (batch,)
        :return: predicted velocity, shape (batch, n_param_pred)
        """
        context = self._encode_context(x, prot_params)
        return self._velocity_forward(z_t, t, context)

    def sample(
        self,
        x: torch.Tensor,
        prot_params: torch.Tensor,
        n_samples: int,
        n_steps: int = 100,
    ) -> torch.Tensor:
        """Sample from the approximate posterior p(params | x, prot_params).

        Encodes x and prot_params once, then integrates the learned ODE from
        z_0 ~ N(0, I) at t=0 to t=1 using the midpoint method.  Each
        observation in the batch independently produces n_samples posterior
        draws.

        :param x: electrochemical signal, shape (batch, channels, time)
        :param prot_params: protocol parameters, shape (batch, n_prot_params)
        :param n_samples: number of posterior samples per observation
        :param n_steps: number of ODE integration steps
        :return: posterior samples, shape (batch, n_samples, n_param_pred)
        """
        context = self._encode_context(x, prot_params)
        return self._sample_from_context(
            context, x.shape[0], n_samples, n_steps, x.device
        )

