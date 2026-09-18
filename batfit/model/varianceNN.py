import numpy as np
import torch
import torch.nn as nn

from batfit import logger
from batfit.preprocess.sim_setup import make_params


class VariancePredFCNN(nn.Module):
    """Deterministic MLP predicting NPE sigma given scaled (prot_params, deg_mean).

    Parameters
    ----------
    n_prot: int
        Number of protocol parameters
    n_deg: int
        Number of degradation parameters
    hidden_list: list[int]
        Widths of the hidden FC layers
    sim_config: str
        Path to the simulation YAML config; required to build min_par and
        amp_par for sigma rescaling
    output_activation: str
        ``"sigmoid"`` (default, historical behaviour) or ``"linear"``
        (unbounded output for z-scored log-sigma targets)
    """

    def __init__(
        self,
        n_prot: int,
        n_deg: int,
        hidden_list: list[int],
        sim_config: str,
        output_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        logger.info("Creating variance predictor MLP")
        assert output_activation in ("sigmoid", "linear"), (
            f"output_activation must be 'sigmoid' or 'linear', "
            f"got {output_activation}"
        )
        self.n_prot = n_prot
        self.n_deg = n_deg
        self.hidden_list = hidden_list
        self.sim_config = sim_config
        self.output_activation = output_activation

        sim_params = make_params(sim_config)
        self.max_par = torch.from_numpy(
            np.array(
                [
                    sim_params["deg_" + name + "_max"]
                    for name in sim_params["deg_param_names"]
                ]
            ).astype("float32")
        )
        self.min_par = torch.from_numpy(
            np.array(
                [
                    sim_params["deg_" + name + "_min"]
                    for name in sim_params["deg_param_names"]
                ]
            ).astype("float32")
        )
        self.amp_par = self.max_par - self.min_par

        layers: list[nn.Module] = []
        in_dim = n_prot + n_deg
        for hidden_dim in hidden_list:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, n_deg))
        if output_activation == "sigmoid":
            # Sigmoid mirrors constrain_output=True in ProbProtParamCNN;
            # physical sigma is recovered by inv_transform_gamma.
            layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(
        self,
        prot_params: torch.Tensor,
        mu: torch.Tensor,
    ) -> torch.Tensor:
        """Return the sigma prediction in the network's trained target space.

        Parameters
        ----------
        prot_params: torch.Tensor
            MinMax-scaled protocol params, shape ``(batch, n_prot)``
        mu: torch.Tensor
            MinMax-scaled degradation param mean, shape ``(batch, n_deg)``

        Returns
        -------
        torch.Tensor
            Shape ``(batch, n_deg)``. With ``output_activation="sigmoid"``,
            values in (0, 1); multiply by amp_par via inv_transform_gamma
            (or invert the sigma MinMaxScaler when trained with scale_sigma)
            to obtain physical sigma. With ``output_activation="linear"``,
            z-scored log sigma; invert with scaler_logsigma and exponentiate.
        """
        x = torch.cat([prot_params, mu], dim=-1)
        return self.layers(x)

    def inv_transform_gamma(
        self,
        gamma_sigmoid: torch.Tensor,
        amp_par: torch.Tensor,
    ) -> torch.Tensor:
        """Convert sigmoid output to physical sigma (mirrors _ProbParamBase).

        Parameters
        ----------
        gamma_sigmoid: torch.Tensor
            Sigmoid output of ``forward()``, shape ``(batch, n_deg)``
        amp_par: torch.Tensor
            Parameter amplitude tensor, shape ``(n_deg,)``

        Returns
        -------
        torch.Tensor
            Physical sigma, shape ``(batch, n_deg)``
        """
        return gamma_sigmoid * amp_par

    def transform_gamma(
        self,
        gamma_physical: torch.Tensor,
        amp_par: torch.Tensor,
    ) -> torch.Tensor:
        """Convert physical sigma to sigmoid-space target for loss computation.

        Parameters
        ----------
        gamma_physical: torch.Tensor
            Physical sigma values, shape ``(batch, n_deg)``
        amp_par: torch.Tensor
            Parameter amplitude tensor, shape ``(n_deg,)``

        Returns
        -------
        torch.Tensor
            Normalised sigma in (0, 1), shape ``(batch, n_deg)``
        """
        return gamma_physical / amp_par


class VariancePredNoProtFCNN(nn.Module):
    """Deterministic MLP predicting NPE sigma given scaled (deg_mean).

    Parameters
    ----------
    n_deg: int
        Number of degradation parameters
    hidden_list: list[int]
        Widths of the hidden FC layers
    sim_config: str
        Path to the simulation YAML config; required to build min_par and
        amp_par for sigma rescaling
    """

    def __init__(
        self,
        n_deg: int,
        hidden_list: list[int],
        sim_config: str,
    ) -> None:
        super().__init__()
        logger.info("Creating variance predictor MLP")
        self.n_deg = n_deg
        self.hidden_list = hidden_list
        self.sim_config = sim_config

        sim_params = make_params(sim_config)
        self.max_par = torch.from_numpy(
            np.array(
                [
                    sim_params["deg_" + name + "_max"]
                    for name in sim_params["deg_param_names"]
                ]
            ).astype("float32")
        )
        self.min_par = torch.from_numpy(
            np.array(
                [
                    sim_params["deg_" + name + "_min"]
                    for name in sim_params["deg_param_names"]
                ]
            ).astype("float32")
        )
        self.amp_par = self.max_par - self.min_par

        layers: list[nn.Module] = []
        in_dim = n_deg
        for hidden_dim in hidden_list:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, n_deg))
        # Sigmoid mirrors constrain_output=True in ProbProtParamCNN;
        # physical sigma is recovered by inv_transform_gamma.
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(
        self,
        mu: torch.Tensor,
    ) -> torch.Tensor:
        """Return sigmoid-scaled sigma given scaled protocol and deg-param mean.

        Parameters
        ----------
        mu: torch.Tensor
            MinMax-scaled degradation param mean, shape ``(batch, n_deg)``

        Returns
        -------
        torch.Tensor
            Sigmoid output in (0, 1), shape ``(batch, n_deg)``; multiply by
            amp_par via inv_transform_gamma to obtain physical sigma
        """
        return self.layers(mu)

    def inv_transform_gamma(
        self,
        gamma_sigmoid: torch.Tensor,
        amp_par: torch.Tensor,
    ) -> torch.Tensor:
        """Convert sigmoid output to physical sigma (mirrors _ProbParamBase).

        Parameters
        ----------
        gamma_sigmoid: torch.Tensor
            Sigmoid output of ``forward()``, shape ``(batch, n_deg)``
        amp_par: torch.Tensor
            Parameter amplitude tensor, shape ``(n_deg,)``

        Returns
        -------
        torch.Tensor
            Physical sigma, shape ``(batch, n_deg)``
        """
        return gamma_sigmoid * amp_par

    def transform_gamma(
        self,
        gamma_physical: torch.Tensor,
        amp_par: torch.Tensor,
    ) -> torch.Tensor:
        """Convert physical sigma to sigmoid-space target for loss computation.

        Parameters
        ----------
        gamma_physical: torch.Tensor
            Physical sigma values, shape ``(batch, n_deg)``
        amp_par: torch.Tensor
            Parameter amplitude tensor, shape ``(n_deg,)``

        Returns
        -------
        torch.Tensor
            Normalised sigma in (0, 1), shape ``(batch, n_deg)``
        """
        return gamma_physical / amp_par
