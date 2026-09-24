import numpy as np
import torch
import torch.nn as nn

from batfit import logger
from batfit.utils.scalers import ZScoreScaler

from .param_utils.model_utils import (
    _build_cnn_encoder,
    _build_hidden_fcnn_layers,
    _build_output_heads,
    _ProbParamBase,
    _ProbParamFMBase,
    encoder_channels,
    signal_scaler_shape,
)


class ProbParamCNN(_ProbParamBase):
    """CNN encoder for electrochemical signal."""

    def __init__(
        self,
        input_shape,
        chan_list,
        fc_list,
        fc_mu_list,
        fc_gamma_list,
        loss_fn,
        sim_config: str,
        leaky_relu_slope=0.2,
        cyc_mode="discharge",
        encoder_model=None,
        scaler_X: ZScoreScaler | None = None,
        param_margin: float = 0.05,
        num_attn_heads: int = 0,
        attn_dropout: float = 0.0,
        signal_scaling: str = "zscore",
        scaler_T: ZScoreScaler | None = None,
    ):
        logger.info("Creating probabilistic CNN model")
        super(ProbParamCNN, self).__init__(
            loss_fn=loss_fn,
            sim_config=sim_config,
            scaler_X_shape=signal_scaler_shape(input_shape, signal_scaling),
            cyc_mode=cyc_mode,
            encoder_model=encoder_model,
            scaler_X=scaler_X,
            param_margin=param_margin,
            signal_scaling=signal_scaling,
            scaler_T=scaler_T,
        )
        self.leaky_relu_slope = leaky_relu_slope
        self.chan_list = chan_list
        self.fc_list = fc_list

        assert len(chan_list) < int(np.log(input_shape[1]) / np.log(2))

        if cyc_mode.lower() == "discharge-chargecc":
            input_shape_0 = input_shape[0] // 2
            input_shape_1 = input_shape[1]
        else:
            input_shape_0 = encoder_channels(input_shape, signal_scaling)
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

        # the heads also see the end time with time_dependent_zscore
        self.model_mu_layers, self.model_gamma_layers = _build_output_heads(
            fc_list_end + self.n_end_time,
            fc_mu_list,
            fc_gamma_list,
            self.output_dim,
            self.param_margin,
        )

    def forward(
        self, x: torch.Tensor, t_end: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the scaled posterior mean and std.

        Parameters
        ----------
        x: torch.Tensor
            Scaled signal, shape ``(batch, channels, time)``
        t_end: torch.Tensor | None
            Scaled end time, shape ``(batch, 1)``; required with
            ``signal_scaling="time_dependent_zscore"``

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Scaled posterior means and standard deviations ``(mu, gamma)``
        """
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
            x = self._append_end_time(x, t_end)

            mu = self.model_mu_layers(x)
            gamma = self.model_gamma_layers(x)

        return mu, gamma


class ProbParamFCNN(_ProbParamBase):
    """FCNN encoder for electrochemical signal.

    With ``signal_scaling="zscore"`` the input is a flat feature vector of
    shape ``(batch, input_shape[0])`` (e.g. an encoded signal). With
    ``"time_dependent_zscore"``, ``input_shape`` is the physical signal shape
    ``(2, n_points)``: the voltage ``(batch, 1, n_points)`` is flattened
    through the hidden layers and the end time is fused before the heads.
    """

    def __init__(
        self,
        input_shape,
        hidden_list,
        fc_mu_list,
        fc_gamma_list,
        loss_fn,
        sim_config: str,
        cyc_mode="discharge",
        encoder_model=None,
        scaler_X: ZScoreScaler | None = None,
        param_margin: float = 0.05,
        signal_scaling: str = "zscore",
        scaler_T: ZScoreScaler | None = None,
    ):
        logger.info("Creating probabilistic FCNN model")
        if signal_scaling == "time_dependent_zscore":
            # physical (2, n_points) signal: flattened voltage as features
            assert len(input_shape) == 2, "input_shape must be (2, n_points)"
            scaler_X_shape = signal_scaler_shape(input_shape, signal_scaling)
            n_features = input_shape[1]
        else:
            scaler_X_shape = (1, input_shape[0])
            n_features = input_shape[0]
        super(ProbParamFCNN, self).__init__(
            loss_fn=loss_fn,
            sim_config=sim_config,
            scaler_X_shape=scaler_X_shape,
            cyc_mode=cyc_mode,
            encoder_model=encoder_model,
            scaler_X=scaler_X,
            param_margin=param_margin,
            signal_scaling=signal_scaling,
            scaler_T=scaler_T,
        )
        self.hidden_list = hidden_list
        elementary_fcnn = _build_hidden_fcnn_layers(n_features, hidden_list)
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

        # the heads also see the end time with time_dependent_zscore
        self.model_mu_layers, self.model_gamma_layers = _build_output_heads(
            fc_list_end + self.n_end_time,
            fc_mu_list,
            fc_gamma_list,
            self.output_dim,
            self.param_margin,
        )

        self.fcnn_layers = nn.Sequential(*self.fcnn)
        if self.cyc_mode.lower() == "discharge-chargecc":
            self.fcnn_layers_aux = nn.Sequential(*self.fcnn_aux)

    def forward(
        self, x: torch.Tensor, t_end: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the scaled posterior mean and std.

        Parameters
        ----------
        x: torch.Tensor
            Scaled input: ``(batch, features)``, or the voltage
            ``(batch, 1, n_points)`` with ``"time_dependent_zscore"``
        t_end: torch.Tensor | None
            Scaled end time, shape ``(batch, 1)``; required with
            ``signal_scaling="time_dependent_zscore"``

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Scaled posterior means and standard deviations ``(mu, gamma)``
        """
        if self.cyc_mode.lower() == "discharge-chargecc":
            nchans = x.shape[1]
            x_dis, x_chcc = torch.split(x, nchans // 2, dim=1)

            x_dis = self.fcnn_layers(x_dis)
            x_chcc = self.fcnn_layers_aux(x_chcc)

            x_conc = torch.cat((x_dis, x_chcc), dim=1)

            mu = self.model_mu_layers(x_conc)
            gamma = self.model_gamma_layers(x_conc)
        else:
            if self.with_end_time:
                x = torch.flatten(x, start_dim=1)
            x = self.fcnn_layers(x)
            x = self._append_end_time(x, t_end)

            mu = self.model_mu_layers(x)
            gamma = self.model_gamma_layers(x)

        return mu, gamma


class ProbProtParamCNN(_ProbParamBase):
    """CNN encoder for electrochemical signal with protocol parameter fusion."""

    def __init__(
        self,
        input_shape: tuple[int, int],
        chan_list: list[int],
        fc_list: list[int],
        fc_prot_list: list[int],
        fc_mu_list: list[int],
        fc_gamma_list: list[int],
        loss_fn,
        sim_config: str,
        leaky_relu_slope: float = 0.2,
        cyc_mode: str = "chirp",
        encoder_model=None,
        scaler_X: ZScoreScaler | None = None,
        param_margin: float = 0.05,
        num_attn_heads: int = 0,
        attn_dropout: float = 0.0,
        signal_scaling: str = "zscore",
        scaler_T: ZScoreScaler | None = None,
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
            sim_config=sim_config,
            scaler_X_shape=signal_scaler_shape(input_shape, signal_scaling),
            cyc_mode=cyc_mode,
            encoder_model=encoder_model,
            scaler_X=scaler_X,
            param_margin=param_margin,
            with_prot=True,
            signal_scaling=signal_scaling,
            scaler_T=scaler_T,
        )
        self.leaky_relu_slope = leaky_relu_slope
        self.chan_list = chan_list
        self.fc_list = fc_list
        self.fc_prot_list = fc_prot_list

        assert len(chan_list) < int(np.log(input_shape[1]) / np.log(2))

        # Conv encoder that process electrochem signal
        self.cnn_layers, _, _ = _build_cnn_encoder(
            encoder_channels(input_shape, signal_scaling),
            input_shape[1],
            chan_list,
            fc_list,
            leaky_relu_slope,
            cyc_mode,
            num_attn_heads=num_attn_heads,
            attn_dropout=attn_dropout,
        )

        # After CNN output + prot_params (+ end time) concatenation
        prot_input_size = fc_list[-1] + self.n_prot_params + self.n_end_time
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
            self.param_margin,
        )

    def forward(
        self,
        x: torch.Tensor,
        prot_params: torch.Tensor,
        t_end: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass combining electrochemical signal and protocol parameters.

        Parameters
        ----------
        x: torch.Tensor
            Electrochemical signal, shape ``(batch, channels, time)``
        prot_params: torch.Tensor
            Protocol parameters, shape ``(batch, n_prot_params)``
        t_end: torch.Tensor | None
            Scaled end time, shape ``(batch, 1)``; required with
            ``signal_scaling="time_dependent_zscore"``

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Scaled posterior means and standard deviations ``(mu, gamma)``
        """
        x = self.cnn_layers(x)
        x = torch.cat((x, prot_params), dim=1)
        x = self._append_end_time(x, t_end)
        x = self.prot_layers(x)

        mu = self.model_mu_layers(x)
        gamma = self.model_gamma_layers(x)

        return mu, gamma


class ProbParamFM(_ProbParamFMBase):
    """Flow matching model for battery parameter inference

    Two encoder modes are supported:

    **CNN mode** (default): a 1-D CNN is trained jointly with the velocity
    field MLP end-to-end.
    Requires ``input_shape``, ``chan_list``, ``fc_list``
    ``cyc_mode="discharge-chargecc"`` is supported and uses
    two independent CNN encoders whose embeddings are concatenated.

    **External encoder mode**: a pre-trained encoder
    passed via ``encoder_model``, parameters are **frozen**
    The encoder must expose a ``latent_dim: int``
    Pass ``vae_model.encoder``, not the full ``VAECNN``.
    ``cyc_mode="discharge-chargecc"`` is not supported in this mode.

    Training
    --------
    Call ``forward(x, z_t, t)`` to obtain the predicted velocity and regress
    it against the target velocity from ``AffineProbPath`` using
    ``flow_matching_loss``.

    Inference
    ---------
    Call ``sample(x, n_samples)`` to draw posterior samples by integrating the
    learned ODE from the base distribution to t=1 with the midpoint method,
    then :meth:`to_physical` to map them to physical units (or
    :meth:`sample_physical` from a physical signal).
    """

    def __init__(
        self,
        vf_hidden_list: list[int],
        sim_config: str,
        input_shape: tuple[int, int] | None = None,
        chan_list: list[int] | None = None,
        fc_list: list[int] | None = None,
        encoder_model: nn.Module | None = None,
        leaky_relu_slope: float = 0.2,
        cyc_mode: str = "discharge",
        scaler_X: ZScoreScaler | None = None,
        use_prior_matching: bool = False,
        num_attn_heads: int = 0,
        attn_dropout: float = 0.0,
        signal_scaling: str = "zscore",
        scaler_T: ZScoreScaler | None = None,
    ):
        """
        Parameters
        ----------
        vf_hidden_list: list[int]
            Hidden dims of the velocity field MLP
        sim_config: str
            Experiment configuration; provides the degradation parameters
            (and so ``n_param_pred``) and their bounds
        input_shape: tuple[int, int], optional
            ``(n_channels, n_time_points)``; required in CNN mode
        chan_list: list[int], optional
            Conv1d output channels per layer; required in CNN mode
        fc_list: list[int], optional
            FC hidden dims after the CNN; required in CNN mode
        encoder_model: nn.Module, optional
            Pre-trained encoder (e.g. ``vae.encoder``); when given, CNN
            params are ignored and the encoder weights are frozen
        leaky_relu_slope: float
            Negative slope for LeakyReLU in CNN mode
        cyc_mode: str
            Cycling mode; ``"discharge-chargecc"`` uses dual CNN encoders
            (CNN mode only)
        scaler_X: ZScoreScaler, optional
            Fitted signal scaler; None creates an identity placeholder
            (filled by ``load_state_dict``), which requires ``input_shape``
        use_prior_matching: bool
            If True, use the empirical training-data distribution as the
            base (requires calling :meth:`set_prior_data` before training);
            when False, N(0, I) is used
        num_attn_heads: int
            If > 0, insert a :class:`_SelfAttentionBlock` after the last
            CNN conv layer; must divide ``chan_list[-1]``; ignored when
            ``encoder_model`` is provided
        attn_dropout: float
            Dropout inside MultiheadAttention
        signal_scaling: str
            ``"zscore"`` or ``"time_dependent_zscore"``; the latter feeds the
            voltage to the CNN and the end time to the context (CNN mode
            only)
        scaler_T: ZScoreScaler, optional
            Fitted end-time scaler of ``"time_dependent_zscore"``; None
            creates an identity placeholder
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
            sim_config=sim_config,
            scaler_X_shape=(
                None
                if input_shape is None
                else signal_scaler_shape(input_shape, signal_scaling)
            ),
            cyc_mode=cyc_mode,
            scaler_X=scaler_X,
            use_prior_matching=use_prior_matching,
            signal_scaling=signal_scaling,
            scaler_T=scaler_T,
        )
        if self.with_end_time and not _cnn_mode:
            raise NotImplementedError(
                "time_dependent_zscore is not supported with an external "
                "encoder_model"
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
                else encoder_channels(input_shape, signal_scaling)
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
        # Input: [z_t (n_param_pred) | t (1) | embedding (emb_dim)
        #         | end time (n_end_time)]
        vf_input_dim = self.n_param_pred + 1 + emb_dim + self.n_end_time
        vf_fc = _build_hidden_fcnn_layers(vf_input_dim, vf_hidden_list)
        _vf = []
        for layer in vf_fc:
            _vf.append(layer)
            _vf.append(nn.Tanh())
        _vf.append(nn.Linear(vf_hidden_list[-1], self.n_param_pred))
        self.vf_layers = nn.Sequential(*_vf)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode signal into an embedding vector."""
        if self.encoder_model is not None:
            out = self.encoder_model(x)
            return out[0] if isinstance(out, (tuple, list)) else out
        else:
            if self.cyc_mode.lower() == "discharge-chargecc":
                nchans = x.shape[1]
                x_dis, x_chcc = torch.split(x, nchans // 2, dim=1)
                return torch.cat(
                    (self.cnn_layers(x_dis), self.cnn_layers_aux(x_chcc)),
                    dim=1,
                )
            else:
                return self.cnn_layers(x)

    def forward(
        self,
        x: torch.Tensor,
        z_t: torch.Tensor,
        t: torch.Tensor,
        t_end: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the velocity u(z_t, t | x) of the conditional flow.

        Parameters
        ----------
        x: torch.Tensor
            Electrochemical signal, shape ``(batch, channels, time)``
        z_t: torch.Tensor
            Particle positions in parameter space at time t, shape
            ``(batch, n_param_pred)``
        t: torch.Tensor
            Flow time in [0, 1], shape ``(batch,)``
        t_end: torch.Tensor | None
            Scaled end time, shape ``(batch, 1)``; required with
            ``signal_scaling="time_dependent_zscore"``

        Returns
        -------
        torch.Tensor
            Predicted velocity, shape ``(batch, n_param_pred)``
        """
        context = self._append_end_time(self._encode(x), t_end)
        return self._velocity_forward(z_t, t, context)

    def sample(
        self,
        x: torch.Tensor,
        n_samples: int,
        n_steps: int = 100,
        t_end: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample posterior p(params | x).

        Parameters
        ----------
        x: torch.Tensor
            Electrochemical signal, shape ``(batch, channels, time)``
        n_samples: int
            Number of posterior samples per observation
        n_steps: int
            Number of ODE integration steps
        t_end: torch.Tensor | None
            Scaled end time, shape ``(batch, 1)``; required with
            ``signal_scaling="time_dependent_zscore"``

        Returns
        -------
        torch.Tensor
            Posterior samples, shape ``(batch, n_samples, n_param_pred)``
        """
        context = self._append_end_time(self._encode(x), t_end)
        return self._sample_from_context(
            context, x.shape[0], n_samples, n_steps, x.device
        )


class ProbProtParamFM(_ProbParamFMBase):
    """Flow matching model conditioned on both signal and protocol parameters.

    Training
    --------
    Call ``forward(x, prot_params, z_t, t)`` to obtain the predicted velocity
    and regress it against the target velocity from ``AffineProbPath`` using
    ``flow_matching_loss``.

    Inference
    ---------
    Call ``sample(x, prot_params, n_samples)`` to draw posterior samples by
    integrating the learned ODE from the base distribution to t=1 with the
    midpoint method, then :meth:`to_physical` to map them to physical units
    (or :meth:`sample_physical` from physical inputs).
    """

    def __init__(
        self,
        input_shape: tuple[int, int],
        chan_list: list[int],
        fc_list: list[int],
        fc_prot_list: list[int],
        vf_hidden_list: list[int],
        sim_config: str,
        leaky_relu_slope: float = 0.2,
        cyc_mode: str = "chirp",
        scaler_X: ZScoreScaler | None = None,
        use_prior_matching: bool = False,
        num_attn_heads: int = 0,
        attn_dropout: float = 0.0,
        signal_scaling: str = "zscore",
        scaler_T: ZScoreScaler | None = None,
    ):
        """
        Parameters
        ----------
        input_shape: tuple[int, int]
            ``(n_channels, n_time_points)`` of the input signal
        chan_list: list[int]
            Conv1d output channels per layer
        fc_list: list[int]
            FC hidden dims after the CNN
        fc_prot_list: list[int]
            Hidden dims for the protocol fusion FC layers; empty list skips
            fusion (prot_params concatenated directly to the CNN embedding)
        vf_hidden_list: list[int]
            Hidden dims of the velocity field MLP
        sim_config: str
            Experiment configuration; provides the degradation and protocol
            parameters (and so their numbers) and their bounds
        leaky_relu_slope: float
            Negative slope for LeakyReLU in the CNN
        cyc_mode: str
            Cycling mode; ``"discharge-chargecc"`` is not supported
        scaler_X: ZScoreScaler, optional
            Fitted signal scaler; None creates an identity placeholder
            (filled by ``load_state_dict``)
        use_prior_matching: bool
            If True, use the empirical training-data distribution as the
            base (requires calling :meth:`set_prior_data` before training);
            when False, N(0, I) is used
        num_attn_heads: int
            If > 0, insert a :class:`_SelfAttentionBlock` after the last
            CNN conv layer; must divide ``chan_list[-1]``
        attn_dropout: float
            Dropout inside MultiheadAttention
        signal_scaling: str
            ``"zscore"`` or ``"time_dependent_zscore"``; the latter feeds the
            voltage to the CNN and fuses the end time with the protocol
            parameters
        scaler_T: ZScoreScaler, optional
            Fitted end-time scaler of ``"time_dependent_zscore"``; None
            creates an identity placeholder
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
            sim_config=sim_config,
            scaler_X_shape=signal_scaler_shape(input_shape, signal_scaling),
            cyc_mode=cyc_mode,
            scaler_X=scaler_X,
            use_prior_matching=use_prior_matching,
            with_prot=True,
            signal_scaling=signal_scaling,
            scaler_T=scaler_T,
        )
        self.leaky_relu_slope = leaky_relu_slope
        self.chan_list = chan_list
        self.fc_list = fc_list
        self.fc_prot_list = fc_prot_list
        self.vf_hidden_list = vf_hidden_list

        assert len(chan_list) < int(np.log(input_shape[1]) / np.log(2))

        # CNN encoder for the electrochemical signal
        self.cnn_layers, _, _ = _build_cnn_encoder(
            encoder_channels(input_shape, signal_scaling),
            input_shape[1],
            chan_list,
            fc_list,
            leaky_relu_slope,
            cyc_mode,
            num_attn_heads=num_attn_heads,
            attn_dropout=attn_dropout,
        )

        # Protocol fusion: [cnn_emb | prot_params | end time] -> optional FC
        # -> context
        prot_input_size = fc_list[-1] + self.n_prot_params + self.n_end_time
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
        vf_input_dim = self.n_param_pred + 1 + context_dim
        vf_fc = _build_hidden_fcnn_layers(vf_input_dim, vf_hidden_list)
        _vf = []
        for layer in vf_fc:
            _vf.append(layer)
            _vf.append(nn.Tanh())
        _vf.append(nn.Linear(vf_hidden_list[-1], self.n_param_pred))
        self.vf_layers = nn.Sequential(*_vf)

    def _encode_context(
        self,
        x: torch.Tensor,
        prot_params: torch.Tensor,
        t_end: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode signal, protocol parameters (and end time) into a context
        vector."""
        cnn_emb = self.cnn_layers(x)
        fused = torch.cat((cnn_emb, prot_params), dim=1)
        fused = self._append_end_time(fused, t_end)
        return self.prot_layers(fused)

    def forward(
        self,
        x: torch.Tensor,
        prot_params: torch.Tensor,
        z_t: torch.Tensor,
        t: torch.Tensor,
        t_end: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the velocity u(z_t, t | x, prot_params) of the conditional flow.

        Parameters
        ----------
        x: torch.Tensor
            Electrochemical signal, shape ``(batch, channels, time)``
        prot_params: torch.Tensor
            Protocol parameters, shape ``(batch, n_prot_params)``
        z_t: torch.Tensor
            Particle positions in parameter space at time t, shape
            ``(batch, n_param_pred)``
        t: torch.Tensor
            Flow time in [0, 1], shape ``(batch,)``
        t_end: torch.Tensor | None
            Scaled end time, shape ``(batch, 1)``; required with
            ``signal_scaling="time_dependent_zscore"``

        Returns
        -------
        torch.Tensor
            Predicted velocity, shape ``(batch, n_param_pred)``
        """
        context = self._encode_context(x, prot_params, t_end)
        return self._velocity_forward(z_t, t, context)

    def sample(
        self,
        x: torch.Tensor,
        prot_params: torch.Tensor,
        n_samples: int,
        n_steps: int = 100,
        t_end: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample from the approximate posterior p(params | x, prot_params).

        Parameters
        ----------
        x: torch.Tensor
            Electrochemical signal, shape ``(batch, channels, time)``
        prot_params: torch.Tensor
            Protocol parameters, shape ``(batch, n_prot_params)``
        n_samples: int
            Number of posterior samples per observation
        n_steps: int
            Number of ODE integration steps
        t_end: torch.Tensor | None
            Scaled end time, shape ``(batch, 1)``; required with
            ``signal_scaling="time_dependent_zscore"``

        Returns
        -------
        torch.Tensor
            Posterior samples, shape ``(batch, n_samples, n_param_pred)``
        """
        context = self._encode_context(x, prot_params, t_end)
        return self._sample_from_context(
            context, x.shape[0], n_samples, n_steps, x.device
        )
