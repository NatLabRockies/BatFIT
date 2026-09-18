import numpy as np
import torch
import torch.distributions as dist
from torch.distributions import kl_divergence


def mse_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error between predictions and targets.

    Parameters
    ----------
    output: torch.Tensor
        Predicted values, shape (batch, n_params)
    target: torch.Tensor
        Ground-truth values, shape (batch, n_params)

    Returns
    -------
    torch.Tensor
        Scalar mean squared error
    """
    loss = torch.mean((output - target) ** 2)
    return loss


def independent_gumbel_loss(
    mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    r"""Negative log-likelihood under an independent Gumbel distribution.

    The Gumbel scale ``beta = sqrt(6) * sigma / pi`` is chosen so that the
    distribution's standard deviation equals ``sigma``.

    Implemented following Getter et al. "Statistical Treatment of Convolutional Neural Network Superresolution of Inland
Surface Wind for Subgrid-Scale Variability Quantification", AIES, 2024.

    Parameters
    ----------
    mu: torch.Tensor
        Predicted location per parameter, shape (batch, n_params)
    sigma: torch.Tensor
        Predicted standard deviation per parameter, shape (batch, n_params)
    target: torch.Tensor
        Ground-truth values, shape (batch, n_params)

    Returns
    -------
    torch.Tensor
        Scalar mean negative log-likelihood
    """
    epsilon = 1e-6
    sigma = torch.clamp(sigma, min=epsilon)
    beta = np.sqrt(6) * sigma / torch.pi
    z = (target - mu) / beta
    loss = torch.mean(torch.log(beta) + z + torch.exp(-z))
    return loss


def pinball_loss(
    mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Pinball (quantile) loss at the 5% and 95% quantiles.

    The quantiles are derived from a Gaussian assumption as
    ``mu +/- 1.6449 * sigma`` (the 90% central interval), penalising
    miscoverage of the predicted interval.

    Parameters
    ----------
    mu: torch.Tensor
        Predicted mean per parameter, shape (batch, n_params)
    sigma: torch.Tensor
        Predicted standard deviation per parameter, shape (batch, n_params)
    target: torch.Tensor
        Ground-truth values, shape (batch, n_params)

    Returns
    -------
    torch.Tensor
        Scalar pinball loss summed over the two quantiles
    """
    epsilon = 1e-6
    sigma = torch.clamp(sigma, min=epsilon)
    y5 = mu - 1.6448536269514729 * sigma
    y95 = mu + 1.6448536269514729 * sigma
    loss = torch.mean(
        torch.maximum(0.05 * (target - y5), (0.05 - 1) * (target - y5))
    ) + torch.mean(
        torch.maximum(0.95 * (target - y95), (0.95 - 1) * (target - y95))
    )
    return loss


def independent_normal_loss(
    mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Negative log-likelihood under a diagonal (independent) Gaussian.

    Models each parameter with an independent normal whose covariance is the
    diagonal matrix ``diag(sigma ** 2)``.

    Parameters
    ----------
    mu: torch.Tensor
        Predicted mean per parameter, shape (batch, n_params)
    sigma: torch.Tensor
        Predicted standard deviation per parameter, shape (batch, n_params)
    target: torch.Tensor
        Ground-truth values, shape (batch, n_params)

    Returns
    -------
    torch.Tensor
        Scalar mean negative log-likelihood
    """
    # epsilon = 1e-6  # To prevent log(0) or division by zero
    # sigma = torch.clamp(sigma, min=epsilon)  # Ensure sigma is positive
    # nll = torch.sum(torch.log(sigma), dim=1) + 0.5 * torch.sum(((target - mu) ** 2) / (sigma**2), dim=1)
    # return torch.mean(nll)  # Average over the batch
    epsilon = 1e-6
    sigma = torch.clamp(sigma, min=epsilon)
    mvn = dist.MultivariateNormal(
        mu, covariance_matrix=torch.diag_embed(sigma**2)
    )
    nll = -mvn.log_prob(target)
    return nll.mean()


def nll_loss(
    mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Alias for :func:`independent_normal_loss`.

    Parameters
    ----------
    mu: torch.Tensor
        Predicted mean per parameter, shape (batch, n_params)
    sigma: torch.Tensor
        Predicted standard deviation per parameter, shape (batch, n_params)
    target: torch.Tensor
        Ground-truth values, shape (batch, n_params)

    Returns
    -------
    torch.Tensor
        Scalar mean negative log-likelihood
    """
    return independent_normal_loss(mu, sigma, target)


def gumbel_loss(
    mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Alias for :func:`independent_gumbel_loss`.

    Parameters
    ----------
    mu: torch.Tensor
        Predicted location per parameter, shape (batch, n_params)
    sigma: torch.Tensor
        Predicted standard deviation per parameter, shape (batch, n_params)
    target: torch.Tensor
        Ground-truth values, shape (batch, n_params)

    Returns
    -------
    torch.Tensor
        Scalar mean negative log-likelihood
    """
    return independent_gumbel_loss(mu, sigma, target)


def correlated_normal_loss(
    mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Negative log-likelihood under a full-covariance Gaussian.

    Unlike :func:`independent_normal_loss`, ``sigma`` is the full covariance
    matrix rather than per-parameter standard deviations.

    Parameters
    ----------
    mu: torch.Tensor
        Predicted mean per parameter, shape (batch, n_params)
    sigma: torch.Tensor
        Predicted covariance matrix, shape (batch, n_params, n_params)
    target: torch.Tensor
        Ground-truth values, shape (batch, n_params)

    Returns
    -------
    torch.Tensor
        Scalar mean negative log-likelihood
    """
    mvn = dist.MultivariateNormal(mu, covariance_matrix=sigma)
    nll = -mvn.log_prob(target)

    return nll.mean()  # Average over the batch


def flow_matching_loss(
    predicted_velocity: torch.Tensor, target_velocity: torch.Tensor
) -> torch.Tensor:
    """L2 loss between the predicted and target velocity vectors.

    Used to train a conditional flow matching model. The target velocity is
    provided by sampling a probability path (e.g. via AffineProbPath) between
    a noise sample x_0 and the ground-truth parameters x_1.

    Parameters
    ----------
    predicted_velocity: torch.Tensor
        Velocity predicted by the model, shape (batch, n_params)
    target_velocity: torch.Tensor
        Target velocity from the path sample, shape (batch, n_params)

    Returns
    -------
    torch.Tensor
        Scalar mean squared error
    """
    return torch.pow(predicted_velocity - target_velocity, 2).mean()
