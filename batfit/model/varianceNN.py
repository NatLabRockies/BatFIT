import numpy as np
import torch
import torch.nn as nn

from batfit import logger
from batfit.preprocess.sim_setup import make_params
from batfit.utils.scalers import BoundedScaler, ZScoreScaler


class _VariancePredBase(nn.Module):
    """Base class of the amortized NPE-sigma estimators.

    The network predicts the NPE posterior standard deviation of every
    degradation parameter as a z-scored ``log(sigma)`` (``scaler_logsigma``,
    fitted on the training sigmas)
    The inputs are scaled to ``[0, 1]`` from the bounds of
    ``sim_config``: the degradation-parameter mean with ``scaler_Y`` and, for
    protocol models, the protocol parameters with ``scaler_P``.

    Parameters
    ----------
    hidden_list: list[int]
        Widths of the hidden Tanh layers
    sim_config: str
        Experiment configuration; provides the parameters (and so their
        numbers ``n_deg``, ``n_prot``) and their bounds
    scaler_logsigma: ZScoreScaler | None
        Z-score of ``log(sigma)`` fitted on the training set; None creates an
        identity placeholder, to be filled by ``load_state_dict``
    with_prot: bool
        Condition on the protocol parameters
    """

    def __init__(
        self,
        hidden_list: list[int],
        sim_config: str,
        scaler_logsigma: ZScoreScaler | None,
        with_prot: bool,
    ) -> None:
        super().__init__()
        self.hidden_list = hidden_list
        self.sim_config = sim_config
        self.sim_params = make_params(sim_config)

        self.scaler_Y = BoundedScaler.from_sim_params(self.sim_params, "deg")
        self.n_deg = len(self.sim_params["deg_param_names"])
        if with_prot:
            self.scaler_P = BoundedScaler.from_sim_params(
                self.sim_params, "prot"
            )
            self.n_prot = len(self.sim_params["prot_param_names"])
        else:
            self.scaler_P = None
            self.n_prot = 0
        if scaler_logsigma is None:
            # identity placeholder, overwritten when loading a checkpoint
            shape = (1, self.n_deg)
            scaler_logsigma = ZScoreScaler(np.zeros(shape), np.ones(shape))
        self.scaler_logsigma = scaler_logsigma

        # linear head: the target (z-scored log sigma) is unbounded
        layers: list[nn.Module] = []
        in_dim = self.n_prot + self.n_deg
        for hidden_dim in hidden_list:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self.n_deg))
        self.layers = nn.Sequential(*layers)

    def to_physical(self, sigma_scaled: torch.Tensor) -> torch.Tensor:
        """Map the network output (z-scored log sigma) to physical sigma.

        Differentiable, so it can be used inside a gradient-based protocol
        optimisation.

        Parameters
        ----------
        sigma_scaled: torch.Tensor
            Output of ``forward``, shape ``(batch, n_deg)``

        Returns
        -------
        torch.Tensor
            Physical sigma, shape ``(batch, n_deg)``
        """
        return torch.exp(self.scaler_logsigma.inverse_transform(sigma_scaled))


class VariancePredFCNN(_VariancePredBase):
    """MLP predicting NPE sigma from protocol parameters and the NPE mean.

    Parameters
    ----------
    hidden_list: list[int]
        Widths of the hidden Tanh layers
    sim_config: str
        Experiment configuration; must declare protocol parameters
    scaler_logsigma: ZScoreScaler | None
        Z-score of ``log(sigma)`` fitted on the training set; None creates an
        identity placeholder, to be filled by ``load_state_dict``
    """

    def __init__(
        self,
        hidden_list: list[int],
        sim_config: str,
        scaler_logsigma: ZScoreScaler | None = None,
    ) -> None:
        logger.info("Creating variance predictor MLP")
        super().__init__(
            hidden_list, sim_config, scaler_logsigma, with_prot=True
        )

    def forward(
        self,
        prot_params: torch.Tensor,
        mu: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the z-scored log sigma from scaled inputs.

        Parameters
        ----------
        prot_params: torch.Tensor
            Protocol parameters scaled with ``scaler_P``, shape
            ``(batch, n_prot)``
        mu: torch.Tensor
            Degradation-parameter mean scaled with ``scaler_Y``, shape
            ``(batch, n_deg)``

        Returns
        -------
        torch.Tensor
            Z-scored log sigma, shape ``(batch, n_deg)``; see
            :meth:`to_physical`
        """
        x = torch.cat([prot_params, mu], dim=-1)
        return self.layers(x)

    def predict_physical(
        self, prot_params: torch.Tensor, mu: torch.Tensor
    ) -> torch.Tensor:
        """Predict physical sigma from physical protocol parameters and mean.

        Parameters
        ----------
        prot_params: torch.Tensor
            Protocol parameters, shape ``(batch, n_prot)``
        mu: torch.Tensor
            Degradation-parameter mean, shape ``(batch, n_deg)``

        Returns
        -------
        torch.Tensor
            Physical sigma, shape ``(batch, n_deg)``
        """
        prot_params_scaled = self.scaler_P.transform(prot_params)
        mu_scaled = self.scaler_Y.transform(mu)
        sigma_scaled = self(prot_params_scaled, mu_scaled)
        return self.to_physical(sigma_scaled)


class VariancePredNoProtFCNN(_VariancePredBase):
    """MLP predicting NPE sigma from the NPE mean only (no protocol).

    Parameters
    ----------
    hidden_list: list[int]
        Widths of the hidden Tanh layers
    sim_config: str
        Experiment configuration providing the parameter bounds
    scaler_logsigma: ZScoreScaler | None
        Z-score of ``log(sigma)`` fitted on the training set; None creates an
        identity placeholder, to be filled by ``load_state_dict``
    """

    def __init__(
        self,
        hidden_list: list[int],
        sim_config: str,
        scaler_logsigma: ZScoreScaler | None = None,
    ) -> None:
        logger.info("Creating variance predictor MLP (no protocol)")
        super().__init__(
            hidden_list, sim_config, scaler_logsigma, with_prot=False
        )

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        """Predict the z-scored log sigma from the scaled mean.

        Parameters
        ----------
        mu: torch.Tensor
            Degradation-parameter mean scaled with ``scaler_Y``, shape
            ``(batch, n_deg)``

        Returns
        -------
        torch.Tensor
            Z-scored log sigma, shape ``(batch, n_deg)``; see
            :meth:`to_physical`
        """
        return self.layers(mu)

    def predict_physical(self, mu: torch.Tensor) -> torch.Tensor:
        """Predict physical sigma from the physical mean.

        Parameters
        ----------
        mu: torch.Tensor
            Degradation-parameter mean, shape ``(batch, n_deg)``

        Returns
        -------
        torch.Tensor
            Physical sigma, shape ``(batch, n_deg)``
        """
        sigma_scaled = self(self.scaler_Y.transform(mu))
        return self.to_physical(sigma_scaled)
