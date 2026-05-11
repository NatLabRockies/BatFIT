import numpy as np
import torch
import torch.distributions as dist
from torch.distributions import kl_divergence


def mse_loss(output, target):
    """
    Custom mean squared error loss function.
    output: Predicted values.
    target: Ground truth labels.
    """
    loss = torch.mean((output - target) ** 2)
    return loss


def independent_gumbel_loss(mu, sigma, target):
    """
    Custom gumbel loss function.
    output: Predicted values.
    target: Ground truth labels.
    """
    epsilon = 1e-6
    sigma = torch.clamp(sigma, min=epsilon)
    beta = np.sqrt(6) * sigma / torch.pi
    z = (target - mu) / beta
    loss = torch.mean(torch.log(beta) + z + torch.exp(-z))
    return loss


def pinball_loss(mu, sigma, target):
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


def independent_normal_loss(mu, sigma, target):
    """
    Custom neg log like loss function.
    output: Predicted values.
    target: Ground truth labels.
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


def elbo_independent_normal_loss(mu, sigma, prior, target, temp):
    """
    Custom neg log like loss function.
    output: Predicted values.
    target: Ground truth labels.
    """
    sigma = torch.clamp(sigma, min=1e-4)
    # post
    posterior = dist.MultivariateNormal(
        mu, covariance_matrix=torch.diag_embed(sigma**2)
    )
    if not posterior.mean.shape[0] == prior.mean.shape[0]:
        prior_mean = (
            prior.mean[0].reshape(1, -1).repeat(posterior.mean.shape[0], 1)
        )
        prior_cov = (
            prior.covariance_matrix[0]
            .reshape(1, sigma.shape[1], sigma.shape[1])
            .repeat(posterior.mean.shape[0], 1, 1)
        )
        prior = dist.MultivariateNormal(
            prior_mean, covariance_matrix=prior_cov
        )
    elbo = -posterior.log_prob(target) + temp * kl_divergence(posterior, prior)

    return elbo.mean()


def nll_loss(mu, sigma, target):
    return independent_normal_loss(mu, sigma, target)


def gumbel_loss(mu, sigma, target):
    return independent_gumbel_loss(mu, sigma, target)


def correlated_normal_loss(mu, sigma, target):
    mvn = dist.MultivariateNormal(mu, covariance_matrix=sigma)
    nll = -mvn.log_prob(target)

    return nll.mean()  # Average over the batch
