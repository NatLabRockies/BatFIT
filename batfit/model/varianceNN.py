import numpy as np
import torch
import torch.nn as nn

from batfit import logger
from batfit.preprocess.sim_setup import make_params


class VariancePredFCNN(nn.Module):
    """Deterministic MLP predicting NPE sigma given scaled (prot_params, deg_mean).

    The output uses a Sigmoid activation and is rescaled by the degradation
    parameter amplitude (amp_par = max_par - min_par from sim_config), mirroring
    the constrained-output approach of ProbProtParamCNN. The sigma in physical
    space is recovered via inv_transform_gamma(sigmoid_out, amp_par).

    Inputs are expected to be pre-scaled (MinMax) before being passed to forward.
    Use the scalers saved by gen_var_dataset.py for this.

    :param n_prot: number of protocol parameters
    :param n_deg: number of degradation parameters
    :param hidden_list: widths of the hidden FC layers
    :param sim_config: path to the simulation YAML config; required to build
        min_par and amp_par for sigma rescaling
    """

    def __init__(
        self,
        n_prot: int,
        n_deg: int,
        hidden_list: list[int],
        sim_config: str,
    ) -> None:
        super().__init__()
        logger.info("Creating variance predictor MLP")
        self.n_prot = n_prot
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
        in_dim = n_prot + n_deg
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
        prot_params: torch.Tensor,
        mu: torch.Tensor,
    ) -> torch.Tensor:
        """Return sigmoid-scaled sigma given scaled protocol and deg-param mean.

        :param prot_params: MinMax-scaled protocol params, shape (batch, n_prot)
        :param mu: MinMax-scaled degradation param mean, shape (batch, n_deg)
        :return: sigmoid output in (0, 1), shape (batch, n_deg); multiply by
            amp_par via inv_transform_gamma to obtain physical sigma
        """
        x = torch.cat([prot_params, mu], dim=-1)
        return self.layers(x)

    def inv_transform_gamma(
        self,
        gamma_sigmoid: torch.Tensor,
        amp_par: torch.Tensor,
    ) -> torch.Tensor:
        """Convert sigmoid output to physical sigma (mirrors _ProbParamBase).

        :param gamma_sigmoid: Sigmoid output of forward(), shape (batch, n_deg)
        :param amp_par: parameter amplitude tensor, shape (n_deg,)
        :return: physical sigma, shape (batch, n_deg)
        """
        return gamma_sigmoid * amp_par

    def transform_gamma(
        self,
        gamma_physical: torch.Tensor,
        amp_par: torch.Tensor,
    ) -> torch.Tensor:
        """Convert physical sigma to sigmoid-space target for loss computation.

        :param gamma_physical: physical sigma values, shape (batch, n_deg)
        :param amp_par: parameter amplitude tensor, shape (n_deg,)
        :return: normalised sigma in (0, 1), shape (batch, n_deg)
        """
        return gamma_physical / amp_par


class VariancePredNoProtFCNN(nn.Module):
    """Deterministic MLP predicting NPE sigma given scaled (deg_mean).

    The output uses a Sigmoid activation and is rescaled by the degradation
    parameter amplitude (amp_par = max_par - min_par from sim_config), mirroring
    the constrained-output approach of ProbParamCNN. The sigma in physical
    space is recovered via inv_transform_gamma(sigmoid_out, amp_par).

    Inputs are expected to be pre-scaled (MinMax) before being passed to forward.
    Use the scalers saved by gen_var_dataset.py for this.

    :param n_prot: number of protocol parameters
    :param n_deg: number of degradation parameters
    :param hidden_list: widths of the hidden FC layers
    :param sim_config: path to the simulation YAML config; required to build
        min_par and amp_par for sigma rescaling
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

        :param mu: MinMax-scaled degradation param mean, shape (batch, n_deg)
        :return: sigmoid output in (0, 1), shape (batch, n_deg); multiply by
            amp_par via inv_transform_gamma to obtain physical sigma
        """
        return self.layers(mu)

    def inv_transform_gamma(
        self,
        gamma_sigmoid: torch.Tensor,
        amp_par: torch.Tensor,
    ) -> torch.Tensor:
        """Convert sigmoid output to physical sigma (mirrors _ProbParamBase).

        :param gamma_sigmoid: Sigmoid output of forward(), shape (batch, n_deg)
        :param amp_par: parameter amplitude tensor, shape (n_deg,)
        :return: physical sigma, shape (batch, n_deg)
        """
        return gamma_sigmoid * amp_par

    def transform_gamma(
        self,
        gamma_physical: torch.Tensor,
        amp_par: torch.Tensor,
    ) -> torch.Tensor:
        """Convert physical sigma to sigmoid-space target for loss computation.

        :param gamma_physical: physical sigma values, shape (batch, n_deg)
        :param amp_par: parameter amplitude tensor, shape (n_deg,)
        :return: normalised sigma in (0, 1), shape (batch, n_deg)
        """
        return gamma_physical / amp_par
