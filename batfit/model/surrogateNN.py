import pickle

import numpy as np
import optuna
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from prettyPlot.progressBar import print_progress_bar

from batfit import logger
from batfit.preprocess.sim_setup import make_params
from batfit.utils.data_utils import (
    scale_dataset_from_scaler,
    scale_input_from_scaler,
    scale_output_from_scaler,
    unscale_dataset_from_scaler,
    unscale_input_from_scaler,
    unscale_output_from_scaler,
    unscale_pred_from_scaler,
)
from batfit.utils.text_utils import shuffle_substrings
from batfit.utils.torch_utils import (
    get_device_type,
    get_num_parameters,
    load_model,
    log_training,
    make_dataset_from_np,
    prepare_log,
    save_model,
)
from .surrogate_utils.losses import mae_loss, mse_loss


class SurrogateFCNN(nn.Module):
    def __init__(
        self,
        fc_list,
        n_param_pred=6,
        loss_fn=mae_loss,
        constrain_output=False,
        cyc_mode="discharge",
        sim_config=None,
    ):
        logger.info("Creating Surrogate model")
        super(SurrogateFCNN, self).__init__()
        input_shape = (n_param_pred + 1,)
        self.fc_list = fc_list
        self.n_param_pred = n_param_pred
        self.constrain_output = constrain_output
        self.sim_config = sim_config
        self.output_dim = 1
        self.cyc_mode = (cyc_mode,)
        self.loss_fn = loss_fn
        assert self.loss_fn in [
            mae_loss,
            mse_loss,
        ]

        if self.sim_config is not None:
            self.sim_params = make_params(self.sim_config)
            self.max_v = np.float32(self.sim_params["vmax"]) + 0.5
            self.min_v = np.float32(self.sim_params["vmin"]) - 0.5
            self.amp_v = self.max_v - self.min_v

        self.fcnn = []
        for ihidden, hidden in enumerate(fc_list):
            if ihidden == 0:
                self.fcnn.append(
                    nn.Linear(
                        in_features=input_shape[0],
                        out_features=hidden,
                    )
                )
                self.fcnn.append(nn.Tanh())
            else:
                self.fcnn.append(
                    nn.Linear(
                        in_features=fc_list[ihidden - 1],
                        out_features=hidden,
                    )
                )
                self.fcnn.append(nn.Tanh())

        self.fcnn.append(nn.Linear(fc_list[-1], self.output_dim))
        if self.constrain_output:
            self.fcnn.append(nn.Sigmoid())
        else:
            self.fcnn.append(nn.ReLU())

        self.fcnn_layers = nn.Sequential(*self.fcnn)

    def inv_transform_output(self, x_unscaled, min_v, amp_v):
        x = x_unscaled * amp_v + min_v
        return x

    def transform_output(self, x_scaled, min_v, amp_v):
        x = (x_scaled - min_v) / amp_v
        return x

    def forward(self, x):
        # for layer in self.fcnn_layers:
        #    try:
        #        x = layer(x)
        #    except RuntimeError:
        #        breakpoint()

        x = self.fcnn_layers(x)
        return x


