from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

from batfit.preprocess.sim_setup import make_params
from batfit.utils.scalers import BoundedScaler, MarginSigmoid, ZScoreScaler

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


class _ParamScalingMixin:
    """Physical parameter space scaling/unscaling utilities.

    Both Gaussian and flow matching base classes inherit from this
    """

    def _init_scaling(self, sim_config: str | None) -> None:
        """Initialise physical parameter bounds from a sim_config

        Parameters
        ----------
        sim_config: str | None
            Path to a YAML experiment configuration file, or None
        """
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

    def inv_transform_mu(
        self,
        mu_unscaled: torch.Tensor,
        min_par: torch.Tensor,
        amp_par: torch.Tensor,
    ) -> torch.Tensor:
        return mu_unscaled * amp_par + min_par

    def inv_transform_gamma(
        self, gamma_unscaled: torch.Tensor, amp_par: torch.Tensor
    ) -> torch.Tensor:
        return gamma_unscaled * amp_par

    def transform_mu(
        self,
        mu_scaled: torch.Tensor,
        min_par: torch.Tensor,
        amp_par: torch.Tensor,
    ) -> torch.Tensor:
        return (mu_scaled - min_par) / amp_par

    def transform_gamma(
        self, gamma_scaled: torch.Tensor, amp_par: torch.Tensor
    ) -> torch.Tensor:
        return gamma_scaled / amp_par

    def transform_output(
        self,
        mu_scaled: torch.Tensor,
        gamma_scaled: torch.Tensor,
        min_par: torch.Tensor,
        amp_par: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.transform_mu(
            mu_scaled, min_par, amp_par
        ), self.transform_gamma(gamma_scaled, amp_par)

    def inv_transform_output(
        self,
        mu_unscaled: torch.Tensor,
        gamma_unscaled: torch.Tensor,
        min_par: torch.Tensor,
        amp_par: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inv_transform_mu(
            mu_unscaled, min_par, amp_par
        ), self.inv_transform_gamma(gamma_unscaled, amp_par)


class _ProbParamBase(nn.Module, ABC):
    """Base class of the Gaussian parameter-inference (NPE) models.

    The network maps a z-scored signal (and ``[0, 1]``-scaled protocol
    parameters) to the posterior mean and standard deviation of the
    degradation parameters, both in the ``[0, 1]`` space defined by the
    parameter bounds of ``sim_config``. The model holds its scalers, so
    :meth:`predict_physical` goes from physical inputs to physical outputs.

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
    ):
        super(_ProbParamBase, self).__init__()
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

        self.sim_config = sim_config
        self.sim_params = make_params(sim_config)
        # one output per degradation parameter of the config
        self.scaler_Y = BoundedScaler.from_sim_params(self.sim_params, "deg")
        self.n_param_pred = len(self.sim_params["deg_param_names"])
        self.output_dim = self.n_param_pred
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
            # identity placeholder, overwritten when loading a checkpoint
            scaler_X = ZScoreScaler(
                np.zeros(scaler_X_shape), np.ones(scaler_X_shape)
            )
        self.scaler_X = scaler_X

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
        x_scaled = self._encode(self.scaler_X.transform(x))
        if self.scaler_P is None:
            mu, sigma = self(x_scaled)
        else:
            assert prot_params is not None, "prot_params is required"
            mu, sigma = self(x_scaled, self.scaler_P.transform(prot_params))
        return self.to_physical(mu, sigma)

    @abstractmethod
    def forward(self, x):
        pass


class _ProbParamFMBase(nn.Module, ABC, _ParamScalingMixin):
    """Abstract base class for flow matching parameter estimation models.

    Subclasses must assign self.vf_layers (an nn.Sequential) in their
    __init__. Its input dimension must be (n_param_pred + 1 + context_dim)
    and its output dimension must be n_param_pred.
    """

    def __init__(
        self,
        cyc_mode: str = "discharge",
        n_param_pred: int = 6,
        sim_config: str | None = None,
        use_prior_matching: bool = False,
    ):
        super().__init__()
        self.cyc_mode = cyc_mode
        self.n_param_pred = n_param_pred
        self.use_prior_matching = use_prior_matching
        self._init_scaling(sim_config)
        if use_prior_matching and sim_config is None:
            raise ValueError(
                "use_prior_matching=True requires sim_config so that "
                "min_par and amp_par are available for prior sampling."
            )

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

    def set_prior_data(self, Y_train: torch.Tensor) -> None:
        """Register scaled training labels as the empirical base distribution.

        Once set, :meth:`sample_prior` draws random rows from this buffer
        instead of the parametric U(min_par, max_par) prior.
        The buffer is persisted in both ``model.pkl`` (full pickle) and every ``.pt``
        checkpoint (state dict), so it is automatically available at inference
        time without any extra files. **This might create memory issues though**

        Call this after constructing the model but before training, passing the
        **scaled** Y_train that matches the DataLoader label space (e.g.
        z-scored when ``scale_y=True``).

        Parameters
        ----------
        Y_train: torch.Tensor
            Scaled training labels, shape ``(n_train, n_param_pred)``
        """
        self.register_buffer("Y_prior", Y_train.float())

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
                "with the scaled Y_train tensor."
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
        """Integrate the learned ODE from N(0, I) to the posterior."""
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
