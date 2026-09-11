from typing import Tuple

import torch
import torch.nn as nn

from batfit import logger
from batfit.utils.torch_utils import get_device_type, get_num_parameters


class BaseMLP(nn.Module):
    """A standard flexible MLP for binary classification."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        layers = []
        current_dim = input_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim

        # Output layer to single logit
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class JointBoundaryClassifier(nn.Module):
    """
    Wraps two separate MLPs. One discriminates p1 from p3, the other p2 from p3.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int = 64, num_layers: int = 3
    ):
        super().__init__()
        self.net1 = BaseMLP(input_dim, hidden_dim, num_layers)
        self.net2 = BaseMLP(input_dim, hidden_dim, num_layers)

        # Log parameter count
        num_params = get_num_parameters(self)
        logger.info(
            f"Initialized JointBoundaryClassifier with {num_params} trainable parameters."
        )

        self.device = get_device_type()
        self.to(self.device)
        logger.info(f"Model moved to device: {self.device}")

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training. Takes distinct inputs for each network.
        Returns unnormalized logits (use BCEWithLogitsLoss).
        """
        logits1 = self.net1(x1)
        logits2 = self.net2(x2)
        return logits1, logits2

    @torch.no_grad()
    def filter_samples(
        self, x: torch.Tensor, threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Evaluates a batch of candidate samples from the joint distribution.
        Returns a boolean tensor where True means the sample is accepted (in both stable regions).

        :param x: Candidate samples tensor of shape (batch_size, input_dim)
        :param threshold: Probability threshold for acceptance
        :return: Boolean mask of shape (batch_size, 1)
        """
        x = x.to(self.device)

        # Pass the same samples through both boundary detectors
        logits1 = self.net1(x)
        logits2 = self.net2(x)

        prob1 = torch.sigmoid(logits1)
        prob2 = torch.sigmoid(logits2)

        # Sample must be confidently predicted as valid by BOTH networks
        accepted = (prob1 >= threshold) & (prob2 >= threshold)
        return accepted
