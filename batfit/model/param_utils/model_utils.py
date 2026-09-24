from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

from batfit.preprocess.sim_setup import make_params
from batfit.utils.scalers import BoundedScaler, MarginSigmoid, ZScoreScaler
from batfit.utils.signal_encoding import SIGNAL_SCALINGS, split_end_time

from .losses import (
    gumbel_loss,
    independent_gumbel_loss,
    independent_normal_loss,
    nll_loss,
)

# ---------------------------------------------------------------------------
# Layer builder helpers
# ---------------------------------------------------------------------------


def _build_conv_layers(
    input_shape_0: int, chan_list: list[int]
) -> tuple[list[nn.Module], list[nn.Module]]:
    """Build Conv1d and MaxPool1d layer lists from a channel list."""
    conv_l = []
    pool_l = []

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


def _build_hidden_fcnn_layers(
    input_shape: int, fc_list: list[int]
) -> list[nn.Module]:
    """Build a list of Linear layers from an input size and hidden dim list."""
    fc_l = []
    for ifc, fc in enumerate(fc_list):
        if ifc == 0:
            fc_l.append(nn.Linear(input_shape, fc))
        else:
            fc_l.append(nn.Linear(fc_list[ifc - 1], fc))
    return fc_l


def _build_conv_fc_layers(
    input_shape_1: int, chan_list: list[int], fc_list: list[int]
) -> list[nn.Module]:
    """Build FC layers whose input size is the flattened output of the conv stack."""
    return _build_hidden_fcnn_layers(
        input_shape=chan_list[-1] * input_shape_1 // (2 ** len(chan_list)),
        fc_list=fc_list,
    )


class _SelfAttentionBlock(nn.Module):
    """Multi-head self-attention block inserted after the CNN conv stack."""

    def __init__(
        self, embed_dim: int, num_heads: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply pre-norm self-attention with residual connection.

        Parameters
        ----------
        x: torch.Tensor
            CNN feature map, shape ``(batch, channels, time)``

        Returns
        -------
        torch.Tensor
            Attended feature map, shape ``(batch, channels, time)``
        """
        x_t = x.transpose(1, 2)  # (batch, time, channels)
        normed = self.norm(x_t)
        attn_out, _ = self.attn(normed, normed, normed)
        return (x_t + attn_out).transpose(1, 2)  # (batch, channels, time)


def _build_cnn_encoder(
    input_shape_0: int,
    input_shape_1: int,
    chan_list: list[int],
    fc_list: list[int],
    leaky_relu_slope: float,
    cyc_mode: str,
    num_attn_heads: int = 0,
    attn_dropout: float = 0.0,
) -> tuple[nn.Sequential, nn.Sequential | None, int]:
    """Build a 1-D CNN encoder (Conv + Pool + LeakyReLU + optional attention + FC + Tanh).

    For ``cyc_mode="discharge-chargecc"`` a second independent encoder is
    built for the charge-CC half and their embeddings are concatenated.

    When ``num_attn_heads > 0``, a :class:`_SelfAttentionBlock` is inserted
    between the last convolutional block and the Flatten layer of each encoder.
    The block operates on the ``(batch, chan_list[-1], time_reduced)`` feature
    map, so ``chan_list[-1]`` must be divisible by ``num_attn_heads``.

    Parameters
    ----------
    num_attn_heads: int
        Number of attention heads; 0 disables attention
    attn_dropout: float
        Dropout inside MultiheadAttention (only used when
        ``num_attn_heads > 0``)

    Returns
    -------
    tuple
        ``(cnn_layers, cnn_layers_aux, embedding_dim)``.
        ``cnn_layers_aux`` is ``None`` unless ``cyc_mode`` is
        ``"discharge-chargecc"``.
    """
    conv, pool = _build_conv_layers(input_shape_0, chan_list)
    fc = _build_conv_fc_layers(input_shape_1, chan_list, fc_list)

    _cnn_layers = []
    for ichan in range(len(conv)):
        _cnn_layers.append(conv[ichan])
        _cnn_layers.append(pool[ichan])
        _cnn_layers.append(nn.LeakyReLU(leaky_relu_slope))
    if num_attn_heads > 0:
        _cnn_layers.append(
            _SelfAttentionBlock(chan_list[-1], num_attn_heads, attn_dropout)
        )
    _cnn_layers.append(nn.Flatten())
    for ifc in range(len(fc)):
        _cnn_layers.append(fc[ifc])
        _cnn_layers.append(nn.Tanh())
    cnn_layers = nn.Sequential(*_cnn_layers)

    if cyc_mode.lower() == "discharge-chargecc":
        conv_aux, pool_aux = _build_conv_layers(input_shape_0, chan_list)
        fc_aux = _build_conv_fc_layers(input_shape_1, chan_list, fc_list)
        _cnn_layers_aux = []
        for ichan in range(len(conv_aux)):
            _cnn_layers_aux.append(conv_aux[ichan])
            _cnn_layers_aux.append(pool_aux[ichan])
            _cnn_layers_aux.append(nn.LeakyReLU(leaky_relu_slope))
        if num_attn_heads > 0:
            _cnn_layers_aux.append(
                _SelfAttentionBlock(
                    chan_list[-1], num_attn_heads, attn_dropout
                )
            )
        _cnn_layers_aux.append(nn.Flatten())
        for ifc in range(len(fc_aux)):
            _cnn_layers_aux.append(fc_aux[ifc])
            _cnn_layers_aux.append(nn.Tanh())
        cnn_layers_aux = nn.Sequential(*_cnn_layers_aux)
        fc_list_end = 2 * fc_list[-1]
    else:
        cnn_layers_aux = None
        fc_list_end = fc_list[-1]

    return cnn_layers, cnn_layers_aux, fc_list_end


def _build_output_heads(
    fc_list_end: int,
    fc_mu_list: list[int],
    fc_gamma_list: list[int],
    output_dim: int,
    param_margin: float,
) -> tuple[nn.Sequential, nn.Sequential]:
    """Build the mu and sigma output heads of the Gaussian parameter models.

    Both heads predict in the ``[0, 1]`` space of the degradation parameters:
    mu ends with a :class:`MarginSigmoid` spanning
    ``[-param_margin, 1 + param_margin]`` and sigma with a ``Sigmoid``.

    Returns
    -------
    tuple
        ``(model_mu_layers, model_gamma_layers)``
    """
    fc_mu = _build_hidden_fcnn_layers(fc_list_end, fc_mu_list)
    _mu_layers = []
    for layer in fc_mu:
        _mu_layers.append(layer)
        _mu_layers.append(nn.Tanh())
    _mu_layers.append(nn.Linear(fc_mu_list[-1], output_dim))
    _mu_layers.append(MarginSigmoid(param_margin))

    fc_gamma = _build_hidden_fcnn_layers(fc_list_end, fc_gamma_list)
    _gamma_layers = []
    for layer in fc_gamma:
        _gamma_layers.append(layer)
        _gamma_layers.append(nn.Tanh())
    _gamma_layers.append(nn.Linear(fc_gamma_list[-1], output_dim))
    _gamma_layers.append(nn.Sigmoid())

    return nn.Sequential(*_mu_layers), nn.Sequential(*_gamma_layers)


def signal_scaler_shape(
    input_shape: tuple[int, int], signal_scaling: str
) -> tuple[int, int, int]:
    """Shape of the placeholder signal-scaler statistics of a CNN model.

    Parameters
    ----------
    input_shape : tuple[int, int]
        ``(n_channels, n_time_points)`` of the physical signal.
    signal_scaling : str
        ``"zscore"`` (one statistic per channel) or
        ``"time_dependent_zscore"`` (one statistic per time point of the
        voltage).

    Returns
    -------
    tuple[int, int, int]
        ``(1, n_channels, 1)`` or ``(1, 1, n_time_points)``.
    """
    if signal_scaling == "time_dependent_zscore":
        return (1, 1, input_shape[1])
    return (1, input_shape[0], 1)


def encoder_channels(input_shape: tuple[int, int], signal_scaling: str) -> int:
    """Number of signal channels seen by the CNN encoder.

    Parameters
    ----------
    input_shape : tuple[int, int]
        ``(n_channels, n_time_points)`` of the physical signal.
    signal_scaling : str
        With ``"time_dependent_zscore"`` the time channel is replaced by the
        end time, fused after the encoder.

    Returns
    -------
    int
        ``n_channels``, or ``n_channels - 1`` without the time channel.
    """
    if signal_scaling == "time_dependent_zscore":
        return input_shape[0] - 1
    return input_shape[0]


class _NPEBase(nn.Module):
    """Base class of all parameter-inference (NPE) models.

    Shared by the Gaussian and FM models: builds the scalers and
    parameter counts from the experiment configuration. Sets ``sim_config``,
    ``sim_params``, ``scaler_Y`` (degradation parameters, ``[0, 1]``),
    ``n_param_pred``, ``scaler_X``, ``scaler_T`` (end time, only with
    ``"time_dependent_zscore"``) and, for protocol models, ``scaler_P`` and
    ``n_prot_params``.

    Parameters
    ----------
    sim_config: str
        Experiment configuration providing the parameter bounds
    scaler_X: ZScoreScaler | None
        Fitted signal scaler; None creates an identity placeholder, to be
        filled by ``load_state_dict``
    scaler_X_shape: tuple[int, ...] | None
        Shape of the placeholder signal-scaler statistics, e.g.
        ``(1, channels, 1)``; required when ``scaler_X`` is None
    with_prot: bool
        Build the protocol-parameter scaler
    signal_scaling: str
        ``"zscore"``: the (time, voltage) signal is z-scored per channel.
        ``"time_dependent_zscore"``: the voltage is z-scored per time point
        and the end time is a separate input, fused after the encoder
    scaler_T: ZScoreScaler | None
        Fitted end-time scaler (``"time_dependent_zscore"`` only); None
        creates an identity placeholder
    """

    def __init__(
        self,
        sim_config: str,
        scaler_X: ZScoreScaler | None,
        scaler_X_shape: tuple[int, ...] | None,
        with_prot: bool,
        signal_scaling: str = "zscore",
        scaler_T: ZScoreScaler | None = None,
    ) -> None:
        super().__init__()
        assert signal_scaling in SIGNAL_SCALINGS, (
            f"Unknown signal_scaling {signal_scaling}, "
            f"use one of {SIGNAL_SCALINGS}"
        )
        self.sim_config = sim_config
        self.sim_params = make_params(sim_config)
        # one output per degradation parameter of the config
        self.scaler_Y = BoundedScaler.from_sim_params(self.sim_params, "deg")
        self.n_param_pred = len(self.sim_params["deg_param_names"])
        if with_prot:
            assert (
                "prot_param_names" in self.sim_params
            ), f"{sim_config} declares no protocol parameters"
            self.scaler_P = BoundedScaler.from_sim_params(
                self.sim_params, "prot"
            )
            self.n_prot_params = len(self.sim_params["prot_param_names"])
        else:
            self.scaler_P = None
        if scaler_X is None:
            assert scaler_X_shape is not None, (
                "scaler_X is required when the signal shape is unknown "
                "(e.g. with an external encoder)"
            )
            # identity placeholder, overwritten when loading a checkpoint
            scaler_X = ZScoreScaler(
                np.zeros(scaler_X_shape), np.ones(scaler_X_shape)
            )
        self.scaler_X = scaler_X
        self.signal_scaling = signal_scaling
        self.with_end_time = signal_scaling == "time_dependent_zscore"
        # number of end-time inputs fused after the encoder
        self.n_end_time = 1 if self.with_end_time else 0
        if self.with_end_time and scaler_T is None:
            # identity placeholder, overwritten when loading a checkpoint
            scaler_T = ZScoreScaler(np.zeros((1, 1)), np.ones((1, 1)))
        self.scaler_T = scaler_T

    def scale_signal(
        self, x: np.ndarray | torch.Tensor
    ) -> tuple[np.ndarray | torch.Tensor, ...]:
        """Scale a physical ``(time, voltage)`` signal into network inputs.

        Parameters
        ----------
        x: numpy.ndarray or torch.Tensor
            Physical signal of shape ``(batch, channels, time)``

        Returns
        -------
        tuple
            ``(x_scaled,)`` for ``"zscore"``; ``(v_scaled, t_end_scaled)``
            for ``"time_dependent_zscore"``, with the voltage of shape
            ``(batch, 1, time)`` and the end time of shape ``(batch, 1)``
        """
        if not self.with_end_time:
            return (self.scaler_X.transform(x),)
        voltage, t_end = split_end_time(x)
        return (
            self.scaler_X.transform(voltage),
            self.scaler_T.transform(t_end),
        )

    def _append_end_time(
        self, h: torch.Tensor, t_end: torch.Tensor | None
    ) -> torch.Tensor:
        """Concatenate the scaled end time to an embedding when it is used.

        Parameters
        ----------
        h: torch.Tensor
            Embedding of shape ``(batch, features)``
        t_end: torch.Tensor | None
            Scaled end time of shape ``(batch, 1)``; required with
            ``"time_dependent_zscore"``, must be None otherwise

        Returns
        -------
        torch.Tensor
            ``h``, or ``cat(h, t_end)`` of shape ``(batch, features + 1)``
        """
        if not self.with_end_time:
            assert t_end is None, "t_end is only used by time_dependent_zscore"
            return h
        assert t_end is not None, "t_end is required by time_dependent_zscore"
        return torch.cat((h, t_end), dim=1)


class _ProbParamBase(_NPEBase, ABC):
    """Base class of the Gaussian parameter-inference (NPE) models.

    Parameters
    ----------
    loss_fn: Callable
        Loss ``loss_fn(mu, sigma, target)`` computed in scaled space
    sim_config: str
        Experiment configuration providing the parameter bounds
    scaler_X_shape: tuple[int, ...]
        Shape of the signal-scaler statistics, e.g. ``(1, channels, 1)``
    cyc_mode: str
        Cycling mode of the signal
    encoder_model: torch.nn.Module | None
        Optional frozen encoder applied to the scaled signal
    scaler_X: ZScoreScaler | None
        Fitted signal scaler; None creates an identity placeholder, to be
        filled by ``load_state_dict``
    param_margin: float
        Margin of the mu head beyond the ``[0, 1]`` parameter bounds
    with_prot: bool
        Build the protocol-parameter scaler
    signal_scaling: str
        ``"zscore"`` or ``"time_dependent_zscore"`` (see :class:`_NPEBase`)
    scaler_T: ZScoreScaler | None
        Fitted end-time scaler of ``"time_dependent_zscore"``; None creates
        an identity placeholder

    The numbers of predicted degradation parameters (``n_param_pred``) and of
    protocol parameters (``n_prot_params``) are read from ``sim_config``.
    """

    def __init__(
        self,
        loss_fn,
        sim_config: str,
        scaler_X_shape: tuple[int, ...],
        cyc_mode: str = "discharge",
        encoder_model: nn.Module | None = None,
        scaler_X: ZScoreScaler | None = None,
        param_margin: float = 0.05,
        with_prot: bool = False,
        signal_scaling: str = "zscore",
        scaler_T: ZScoreScaler | None = None,
    ):
        super().__init__(
            sim_config,
            scaler_X,
            scaler_X_shape,
            with_prot,
            signal_scaling=signal_scaling,
            scaler_T=scaler_T,
        )
        if self.with_end_time and cyc_mode.lower() == "discharge-chargecc":
            raise NotImplementedError(
                "time_dependent_zscore needs a single (time, voltage) signal"
            )
        if self.with_end_time and encoder_model is not None:
            raise NotImplementedError(
                "time_dependent_zscore is not supported with an external "
                "encoder_model"
            )
        assert loss_fn in [
            gumbel_loss,
            nll_loss,
            independent_normal_loss,
            independent_gumbel_loss,
        ]
        self.loss_fn = loss_fn
        self.cyc_mode = cyc_mode
        self.encoder_model = encoder_model
        self.param_margin = param_margin
        self.output_dim = self.n_param_pred

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the optional encoder to a scaled signal."""
        if self.encoder_model is None:
            return x
        encoded = self.encoder_model.encode(x)
        # a VAE encoder returns (z, mu, logvar)
        return encoded[0] if isinstance(encoded, tuple) else encoded

    def to_physical(
        self, mu: torch.Tensor, sigma: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map scaled network outputs to physical parameter space.

        Parameters
        ----------
        mu: torch.Tensor
            Posterior mean in scaled space, shape ``(batch, n_param_pred)``
        sigma: torch.Tensor
            Posterior standard deviation in scaled space

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Physical ``(mu, sigma)``; mu is clipped to the parameter bounds
        """
        mu_phys = self.scaler_Y.clip_physical(
            self.scaler_Y.inverse_transform(mu)
        )
        sigma_phys = self.scaler_Y.inverse_transform_std(sigma)
        return mu_phys, sigma_phys

    def predict_physical(
        self,
        x: torch.Tensor,
        prot_params: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the physical posterior mean and std from a physical signal.

        Parameters
        ----------
        x: torch.Tensor
            Unscaled signal, on the model's device
        prot_params: torch.Tensor | None
            Unscaled protocol parameters, required for protocol models

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Physical ``(mu, sigma)``, shape ``(batch, n_param_pred)`` each
        """
        signal = self.scale_signal(x)
        inputs = [self._encode(signal[0])]
        if self.scaler_P is not None:
            assert prot_params is not None, "prot_params is required"
            inputs.append(self.scaler_P.transform(prot_params))
        # the end time is only passed to models that use it
        kwargs = {"t_end": signal[1]} if self.with_end_time else {}
        mu, sigma = self(*inputs, **kwargs)
        return self.to_physical(mu, sigma)

    @abstractmethod
    def forward(self, x):
        pass


# Scale of the flow-matching space.
# The flow transports samples of a source
# (base) distribution, N(0, I) or the empirical prior, to the posterior target.
# Parameters scaled to [0, 1] have mean 0.5 and std 1/sqrt(12) under a uniform
# prior, so ``(u - 0.5) * sqrt(12)`` gives the target zero mean and unit
# variance: the target then sits close to the N(0, I) source, which keeps the
# transport paths short and the learned velocities of order one.
_FLOW_SCALE = float(np.sqrt(12.0))


class _ProbParamFMBase(_NPEBase, ABC):
    """Abstract base class for flow matching parameter estimation models.

    The data loaders provide degradation parameters ``u`` scaled to
    ``[0, 1]`` by ``scaler_Y``. The flow operates in the space
    ``(u - 0.5) * sqrt(12)``; the conversions between ``u`` and the flow
    space are internal to this class. :meth:`to_physical` maps posterior
    samples back to physical units, clamped to the parameter bounds.

    Subclasses must assign self.vf_layers (an nn.Sequential) in their
    __init__. Its input dimension must be (n_param_pred + 1 + context_dim)
    and its output dimension must be n_param_pred.

    Parameters
    ----------
    sim_config: str
        Experiment configuration providing the parameter bounds
    scaler_X_shape: tuple[int, ...] | None
        Shape of the placeholder signal-scaler statistics; None requires
        ``scaler_X``
    cyc_mode: str
        Cycling mode of the signal
    scaler_X: ZScoreScaler | None
        Fitted signal scaler; None creates an identity placeholder, to be
        filled by ``load_state_dict``
    use_prior_matching: bool
        Start the flow from the empirical training labels (registered with
        :meth:`set_prior_data`) instead of N(0, I)
    with_prot: bool
        Build the protocol-parameter scaler
    signal_scaling: str
        ``"zscore"`` or ``"time_dependent_zscore"`` (see :class:`_NPEBase`)
    scaler_T: ZScoreScaler | None
        Fitted end-time scaler of ``"time_dependent_zscore"``; None creates
        an identity placeholder
    """

    def __init__(
        self,
        sim_config: str,
        scaler_X_shape: tuple[int, ...] | None,
        cyc_mode: str = "discharge",
        scaler_X: ZScoreScaler | None = None,
        use_prior_matching: bool = False,
        with_prot: bool = False,
        signal_scaling: str = "zscore",
        scaler_T: ZScoreScaler | None = None,
    ):
        super().__init__(
            sim_config,
            scaler_X,
            scaler_X_shape,
            with_prot,
            signal_scaling=signal_scaling,
            scaler_T=scaler_T,
        )
        if self.with_end_time and cyc_mode.lower() == "discharge-chargecc":
            raise NotImplementedError(
                "time_dependent_zscore needs a single (time, voltage) signal"
            )
        self.cyc_mode = cyc_mode
        self.use_prior_matching = use_prior_matching

    def _u_to_flow(self, u: torch.Tensor) -> torch.Tensor:
        """Map ``[0, 1]``-scaled parameters to the flow space."""
        # centre and rescale so the posterior target is close to the source
        # distribution of the flow (see _FLOW_SCALE)
        return (u - 0.5) * _FLOW_SCALE

    def _flow_to_u(self, z: torch.Tensor) -> torch.Tensor:
        """Map flow-space points back to ``[0, 1]``-scaled parameters."""
        return z / _FLOW_SCALE + 0.5

    def to_physical(self, samples: torch.Tensor) -> torch.Tensor:
        """Map flow-space posterior samples to physical parameter space.

        Parameters
        ----------
        samples: torch.Tensor
            Samples returned by :meth:`sample`, shape
            ``(batch, n_samples, n_param_pred)``

        Returns
        -------
        torch.Tensor
            Physical samples, same shape, clamped to the parameter bounds
        """
        u = torch.clamp(self._flow_to_u(samples), min=0.0, max=1.0)
        return self.scaler_Y.inverse_transform(u)

    def sample_physical(
        self,
        x: torch.Tensor,
        prot_params: torch.Tensor | None = None,
        n_samples: int = 1000,
        n_steps: int = 100,
    ) -> torch.Tensor:
        """Draw physical posterior samples from a physical signal.

        Parameters
        ----------
        x: torch.Tensor
            Unscaled signal, on the model's device
        prot_params: torch.Tensor | None
            Unscaled protocol parameters, required for protocol models
        n_samples: int
            Number of posterior samples per observation
        n_steps: int
            Number of ODE integration steps

        Returns
        -------
        torch.Tensor
            Physical samples, shape ``(batch, n_samples, n_param_pred)``
        """
        signal = self.scale_signal(x)
        inputs = [signal[0]]
        if self.scaler_P is not None:
            assert prot_params is not None, "prot_params is required"
            inputs.append(self.scaler_P.transform(prot_params))
        # the end time is only passed to models that use it
        kwargs = {"t_end": signal[1]} if self.with_end_time else {}
        samples = self.sample(
            *inputs, n_samples=n_samples, n_steps=n_steps, **kwargs
        )
        return self.to_physical(samples)

    @property
    def vf_layers(self) -> nn.Sequential:
        """Velocity field MLP; must be set by the subclass __init__."""
        try:
            return self._vf_layers
        except AttributeError:
            raise AttributeError(
                f"{type(self).__name__} must define self.vf_layers "
                "(an nn.Sequential) in its __init__ before calling "
                "_velocity_forward or _sample_from_context."
            )

    @vf_layers.setter
    def vf_layers(self, value: nn.Sequential) -> None:
        self._vf_layers = value

    def _velocity_forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the velocity field given a pre-computed context embedding.

        Parameters
        ----------
        z_t: torch.Tensor
            Particle positions in parameter space, shape ``(batch, n_param_pred)``
        t: torch.Tensor
            Flow time in [0, 1], shape ``(batch,)``
        context: torch.Tensor
            Conditioning embedding, shape ``(batch, context_dim)``

        Returns
        -------
        torch.Tensor
            Velocity vectors, shape ``(batch, n_param_pred)``
        """
        # torchdiffeq passes t as a 0-dim scalar; training code passes (batch,)
        if t.dim() == 0:
            t_exp = t.expand(z_t.shape[0], 1)
        else:
            t_exp = t.unsqueeze(-1)
        vf_input = torch.cat([z_t, t_exp, context], dim=-1)
        return self.vf_layers(vf_input)

    def set_prior_data(self, Y_train_scaled: torch.Tensor) -> None:
        """Register the training labels as the empirical base distribution.

        The labels are stored in the flow space; :meth:`sample_prior` then
        draws random rows from this buffer. The buffer is persisted in both
        ``model.pkl`` (full pickle) and every ``.pt`` checkpoint (state dict),
        so it is available at inference time without any extra files.
        **This might create memory issues though**

        Call this after constructing the model but before training.

        Parameters
        ----------
        Y_train_scaled: torch.Tensor
            Training labels scaled to ``[0, 1]`` (as in the DataLoader),
            shape ``(n_train, n_param_pred)``
        """
        # stored in the flow space, like the targets x_1 of the training loop,
        # so source and target distributions share the same coordinates
        self.register_buffer(
            "Y_prior", self._u_to_flow(Y_train_scaled.float())
        )

    def sample_prior(self, n: int, device: torch.device) -> torch.Tensor:
        """Sample n points from the empirical base distribution.

        Parameters
        ----------
        n: int
            Number of samples
        device: torch.device
            Target torch device

        Returns
        -------
        torch.Tensor
            Prior samples, shape ``(n, n_param_pred)``

        Raises
        ------
        RuntimeError
            If :meth:`set_prior_data` was not called first
        """
        if not (hasattr(self, "Y_prior") and self.Y_prior is not None):
            raise RuntimeError(
                "sample_prior() requires set_prior_data() to be called first "
                "with the [0, 1]-scaled Y_train tensor."
            )
        idx = torch.randint(
            0, self.Y_prior.shape[0], (n,), device=self.Y_prior.device
        )
        return self.Y_prior[idx].to(device)

    def _sample_from_context(
        self,
        context: torch.Tensor,
        batch_size: int,
        n_samples: int,
        n_steps: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Integrate the learned ODE from the base distribution to the
        posterior; samples are returned in the flow space."""
        context_rep = context.repeat_interleave(n_samples, dim=0)
        n_particles = batch_size * n_samples
        if self.use_prior_matching:
            z_0 = self.sample_prior(n_particles, device)
        else:
            z_0 = torch.randn(n_particles, self.n_param_pred, device=device)

        wrapper = _VFWrapper(self._velocity_forward)
        solver = ODESolver(velocity_model=wrapper)

        time_grid = torch.tensor([0.0, 1.0], device=device)
        sol = solver.sample(
            time_grid=time_grid,
            x_init=z_0,
            method="midpoint",
            step_size=1.0 / n_steps,
            context=context_rep,
        )

        return sol.reshape(batch_size, n_samples, self.n_param_pred)

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Predict the velocity field for a training batch."""
        pass

    @abstractmethod
    def sample(self, *args, **kwargs) -> torch.Tensor:
        """Draw posterior samples by integrating the learned ODE."""
        pass


class _VFWrapper(ModelWrapper):
    """Stateless adapter so ODESolver can call our velocity field method.

    This is because ODESolver requires a ModelWrapper subclass.
    """

    def __init__(self, velocity_fn):
        super().__init__(model=None)
        self._velocity_fn = velocity_fn

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, **extras
    ) -> torch.Tensor:
        return self._velocity_fn(x, t, extras["context"])
