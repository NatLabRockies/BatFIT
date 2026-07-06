from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

from batfit.preprocess.sim_setup import make_params

from .losses import (
    correlated_normal_loss,
    gumbel_loss,
    independent_gumbel_loss,
    independent_normal_loss,
    mse_loss,
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
    """Multi-head self-attention block inserted after the CNN conv stack.

    Applies pre-norm self-attention along the time dimension of a
    ``(batch, channels, time)`` feature map, treating each time step as a
    token of dimension ``channels``.  The residual connection preserves the
    input statistics so the block can be toggled off (``num_attn_heads=0``)
    to recover the plain-CNN behaviour without any weight surgery.

    :param embed_dim: token dimension (= output channels of the last Conv1d);
                      must be divisible by ``num_heads``
    :param num_heads: number of attention heads
    :param dropout: attention weight dropout probability
    :raises ValueError: if ``embed_dim`` is not divisible by ``num_heads``
    """

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

        :param x: CNN feature map, shape ``(batch, channels, time)``
        :return: attended feature map, shape ``(batch, channels, time)``
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

    :param num_attn_heads: number of attention heads; 0 disables attention
    :param attn_dropout: dropout inside MultiheadAttention (only used when
                         ``num_attn_heads > 0``)
    :return: ``(cnn_layers, cnn_layers_aux, embedding_dim)``
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
    dependent_outputs: bool,
    constrain_output: bool,
) -> tuple[nn.Sequential, nn.Sequential]:
    """Build the mu and gamma output heads for Gaussian parameter models.

    :return: (model_mu_layers, model_gamma_layers)
    """
    fc_mu = _build_hidden_fcnn_layers(fc_list_end, fc_mu_list)
    fc_otpt_mu = nn.Linear(fc_mu_list[-1], output_dim)

    _mu_layers = []
    for layer in fc_mu:
        _mu_layers.append(layer)
        _mu_layers.append(nn.Tanh())
    _mu_layers.append(fc_otpt_mu)
    if constrain_output:
        _mu_layers.append(nn.Sigmoid())

    fc_gamma = _build_hidden_fcnn_layers(fc_list_end, fc_gamma_list)
    if not dependent_outputs:
        fc_otpt_gamma = nn.Linear(fc_gamma_list[-1], output_dim)
    else:
        fc_otpt_gamma = nn.Linear(
            fc_gamma_list[-1], output_dim * (output_dim + 1) // 2
        )

    _gamma_layers = []
    for layer in fc_gamma:
        _gamma_layers.append(layer)
        _gamma_layers.append(nn.Tanh())
    _gamma_layers.append(fc_otpt_gamma)

    if constrain_output and not dependent_outputs:
        _gamma_layers.append(nn.Sigmoid())
    elif not constrain_output and not dependent_outputs:
        _gamma_layers.append(nn.Softplus(beta=1.0, threshold=20.0))

    return nn.Sequential(*_mu_layers), nn.Sequential(*_gamma_layers)


class _ParamScalingMixin:
    """Mixin providing physical parameter space scaling/unscaling utilities.

    Both Gaussian and flow matching base classes inherit from this mixin to
    share the sim_config initialisation logic and transform methods.
    """

    def _init_scaling(self, sim_config: str | None) -> None:
        """Initialise physical parameter bounds from a sim_config YAML path.

        Sets self.sim_config, self.sim_params, self.max_par, self.min_par,
        and self.amp_par when sim_config is provided.

        :param sim_config: path to a YAML experiment configuration file, or None
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


class _ProbParamBase(nn.Module, ABC, _ParamScalingMixin):
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
        self.dependent_outputs = dependent_outputs
        self.encoder_model = encoder_model
        self.output_dim = self.n_param_pred
        self._init_scaling(sim_config)

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

    def _cholesky_cov(self, gamma: torch.Tensor) -> torch.Tensor:
        """Build a positive-definite covariance matrix via Cholesky decomposition.

        :param gamma: flattened lower-triangular entries, shape (batch, n*(n+1)//2)
        :return: covariance matrices, shape (batch, n, n)
        """
        # Create covariance matrix
        L = torch.zeros(
            gamma.size(0),
            self.output_dim,
            self.output_dim,
            device=gamma.device,
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
        L[:, diagonal_indices, diagonal_indices] = (
            torch.nn.functional.softplus(
                L[:, diagonal_indices, diagonal_indices],
                beta=1.0,
                threshold=20.0,
            )
        )
        return L @ L.transpose(-1, -2)

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

        :param z_t: particle positions in parameter space, shape (batch, n_param_pred)
        :param t: flow time in [0, 1], shape (batch,)
        :param context: conditioning embedding, shape (batch, context_dim)
        :return: velocity vectors, shape (batch, n_param_pred)
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
        instead of the parametric U(min_par, max_par) prior.  The buffer is
        persisted in both ``model.pkl`` (full pickle) and every ``.pt``
        checkpoint (state dict), so it is automatically available at inference
        time without any extra files.

        Call this after constructing the model but before training, passing the
        **scaled** Y_train that matches the DataLoader label space (e.g.
        z-scored when ``scale_y=True``).

        :param Y_train: scaled training labels, shape (n_train, n_param_pred)
        """
        self.register_buffer("Y_prior", Y_train.float())

    def sample_prior(self, n: int, device: torch.device) -> torch.Tensor:
        """Sample n points from the empirical base distribution.

        Draws n rows uniformly at random from the training labels registered
        via :meth:`set_prior_data`.  Raises :exc:`RuntimeError` if
        :meth:`set_prior_data` has not been called — the physical-space
        parametric fallback was removed because it is inconsistent with
        z-scored training labels.

        :param n: number of samples
        :param device: target torch device
        :return: prior samples, shape (n, n_param_pred)
        :raises RuntimeError: if :meth:`set_prior_data` was not called first
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
        """Integrate the learned ODE from N(0, I) to the posterior.

        Shared sampling logic used by all FM subclasses. Repeats the context
        for each sample, draws initial noise, and runs the midpoint ODE solver.
        The context is forwarded to the velocity model at every ODE step via
        ODESolver's model_extras mechanism.

        :param context: conditioning embedding, shape (batch, context_dim)
        :param batch_size: number of observations in the batch
        :param n_samples: number of posterior samples per observation
        :param n_steps: number of ODE integration steps
        :param device: target torch device
        :return: posterior samples of shape (batch, n_samples, n_param_pred)
        """
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

    The ODESolver requires a ModelWrapper subclass. This adapter delegates to
    _velocity_forward, extracting the conditioning context from the keyword
    arguments that ODESolver passes at each integration step via model_extras.
    """

    def __init__(self, velocity_fn):
        super().__init__(model=None)
        self._velocity_fn = velocity_fn

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, **extras
    ) -> torch.Tensor:
        return self._velocity_fn(x, t, extras["context"])
