import torch

from batfit.model.param_utils.losses import (
    correlated_normal_loss,
    independent_normal_loss,
)
from batfit.model.param_utils.losses import mse_loss as mse_loss_param
from batfit.model.param_utils.losses import (
    pinball_loss,
)
from batfit.model.surrogate_utils.losses import mae_loss as mae_loss_surr
from batfit.model.surrogate_utils.losses import mse_loss as mse_loss_surr


def test_mse_surr():
    # zero if equal inputs:
    x = torch.tensor([1.0, 2.0, 3.0])
    assert mse_loss_surr(x, x).item() == 0.0

    # known val
    output = torch.tensor([0.0, 0.0])
    target = torch.tensor([1.0, 1.0])
    assert abs(mse_loss_surr(output, target).item() - 1.0) < 1e-6


def test_mae_surr():
    # zero if equal inputs:
    x = torch.tensor([1.0, 2.0, 3.0])
    assert mae_loss_surr(x, x).item() == 0.0

    # known val
    output = torch.tensor([0.0, 2.0])
    target = torch.tensor([1.0, 4.0])
    # mean(|0-1|, |2-4|) = mean(1, 2) = 1.5
    assert abs(mae_loss_surr(output, target).item() - 1.5) < 1e-6


def test_mse_loss_param():
    """Both mse_loss implementations produce the same value."""
    output = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.5, 2.5, 3.5])
    assert (
        abs(
            mse_loss_param(output, target).item()
            - mse_loss_surr(output, target).item()
        )
        < 1e-6
    )


def test_independent_normal_loss_param():
    # returns scalar
    batch = 8
    n_params = 4
    mu = torch.zeros(batch, n_params)
    sigma = torch.ones(batch, n_params)
    target = torch.zeros(batch, n_params)
    loss = independent_normal_loss(mu, sigma, target)
    assert loss.shape == torch.Size([])


def test_correlated_normal_loss_param():
    # returns scalar
    batch = 8
    n_params = 3
    mu = torch.zeros(batch, n_params)
    # Build a valid positive definite covariance matrix
    cov = torch.eye(n_params).unsqueeze(0).expand(batch, -1, -1)
    target = torch.zeros(batch, n_params)
    loss = correlated_normal_loss(mu, cov, target)
    assert loss.shape == torch.Size([])


def test_pinball_loss_param():
    # returns scalar
    batch = 16
    n_params = 4
    mu = torch.zeros(batch, n_params)
    sigma = torch.ones(batch, n_params)
    target = torch.zeros(batch, n_params)
    loss = pinball_loss(mu, sigma, target)
    assert loss.shape == torch.Size([])
