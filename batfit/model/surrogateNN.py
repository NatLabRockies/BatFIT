import numpy as np
import torch
import torch.nn as nn

from batfit import logger
from batfit.preprocess.sim_setup import make_params
from batfit.utils.scalers import BoundedScaler, MarginSigmoid, ZScoreScaler

from .surrogate_utils.losses import mae_loss, mse_loss


class SurrogateFCNN(nn.Module):
    """Fully-connected surrogate mapping ``(time, deg_params...)`` to voltage.

    The network works in scaled space: time is z-scored (``scaler_t``), the
    degradation parameters are scaled to ``[0, 1]`` from the bounds of
    ``sim_config`` (``scaler_Y``) and the voltage is predicted in the
    ``[0, 1]`` space of ``[vmin, vmax]`` (``scaler_V``), through a sigmoid
    head widened by ``voltage_margin``. The model holds its scalers, so
    :meth:`predict_physical` goes from physical inputs to volts.

    Parameters
    ----------
    fc_list: list[int]
        Widths of the hidden Tanh layers
    sim_config: str
        Experiment configuration; provides the degradation parameters (and so
        ``n_param_pred``), their bounds and ``vmin``/``vmax``
    loss_fn: Callable
        Loss ``loss_fn(pred, target)`` computed in scaled space
    cyc_mode: str
        Cycling mode of the data
    scaler_t: ZScoreScaler | None
        Fitted time scaler; None creates an identity placeholder, to be
        filled by ``load_state_dict``
    voltage_margin: float
        Margin of the output head beyond ``[vmin, vmax]``, in volts
    """

    def __init__(
        self,
        fc_list: list[int],
        sim_config: str,
        loss_fn=mae_loss,
        cyc_mode: str = "discharge",
        scaler_t: ZScoreScaler | None = None,
        voltage_margin: float = 0.5,
    ):
        logger.info("Creating Surrogate model")
        super(SurrogateFCNN, self).__init__()
        assert loss_fn in [mae_loss, mse_loss]
        self.fc_list = fc_list
        self.loss_fn = loss_fn
        self.cyc_mode = cyc_mode
        self.output_dim = 1
        self.voltage_margin = voltage_margin

        self.sim_config = sim_config
        self.sim_params = make_params(sim_config)
        self.scaler_Y = BoundedScaler.from_sim_params(self.sim_params, "deg")
        self.scaler_V = BoundedScaler.from_sim_params(
            self.sim_params, "voltage"
        )
        self.n_param_pred = len(self.sim_params["deg_param_names"])
        if scaler_t is None:
            # identity placeholder, overwritten when loading a checkpoint
            scaler_t = ZScoreScaler(np.zeros((1, 1)), np.ones((1, 1)))
        self.scaler_t = scaler_t

        # the margin is given in volts; the head works on [0, 1] voltages
        voltage_range = self.sim_params["vmax"] - self.sim_params["vmin"]
        relative_margin = voltage_margin / voltage_range

        input_dim = self.n_param_pred + 1
        fcnn = []
        for ihidden, hidden in enumerate(fc_list):
            in_features = input_dim if ihidden == 0 else fc_list[ihidden - 1]
            fcnn.append(
                nn.Linear(in_features=in_features, out_features=hidden)
            )
            fcnn.append(nn.Tanh())
        fcnn.append(nn.Linear(fc_list[-1], self.output_dim))
        fcnn.append(MarginSigmoid(relative_margin))
        self.fcnn_layers = nn.Sequential(*fcnn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the scaled voltage from scaled ``(time, deg_params...)``.

        Parameters
        ----------
        x: torch.Tensor
            Scaled inputs, shape ``(batch, n_param_pred + 1)``; column 0 is
            time

        Returns
        -------
        torch.Tensor
            Scaled voltage, shape ``(batch, 1)``
        """
        return self.fcnn_layers(x)

    def scale_input(
        self, time: torch.Tensor, deg_params: torch.Tensor
    ) -> torch.Tensor:
        """Scale physical time and degradation parameters into network input.

        Parameters
        ----------
        time: torch.Tensor
            Time in seconds, shape ``(batch, 1)``
        deg_params: torch.Tensor
            Degradation parameters, shape ``(batch, n_param_pred)``

        Returns
        -------
        torch.Tensor
            Scaled inputs, shape ``(batch, n_param_pred + 1)``
        """
        time_scaled = self.scaler_t.transform(time)
        deg_params_scaled = self.scaler_Y.transform(deg_params)
        return torch.cat((time_scaled, deg_params_scaled), dim=1)

    def to_physical(self, voltage: torch.Tensor) -> torch.Tensor:
        """Map a scaled voltage prediction to volts.

        The voltage is not clamped to ``[vmin, vmax]``: a clamp would create
        flat regions with zero gradient for gradient-based MCMC samplers.
        """
        return self.scaler_V.inverse_transform(voltage)

    def predict_physical(
        self, time: torch.Tensor, deg_params: torch.Tensor
    ) -> torch.Tensor:
        """Predict the voltage in volts from physical inputs.

        Parameters
        ----------
        time: torch.Tensor
            Time in seconds, shape ``(batch, 1)``, on the model's device
        deg_params: torch.Tensor
            Degradation parameters, shape ``(batch, n_param_pred)``

        Returns
        -------
        torch.Tensor
            Voltage in volts, shape ``(batch, 1)``
        """
        voltage_scaled = self(self.scale_input(time, deg_params))
        return self.to_physical(voltage_scaled)
