import torch

from batfit.model.param_utils.losses import (
    correlated_normal_loss,
    independent_normal_loss,
)
from batfit.model.paramNN import ProbParamCNN, ProbParamFCNN


def test_ProbParamCNN():
    batch = 4
    n_points = 64
    n_param_pred = 3

    model = ProbParamCNN(
        input_shape=(2, n_points),
        chan_list=[8],
        fc_list=[16],
        fc_mu_list=[8],
        fc_gamma_list=[8],
        loss_fn=independent_normal_loss,
        cyc_mode="discharge",
        n_param_pred=n_param_pred,
        constrain_output=True,
    )
    x = torch.rand(batch, 2, n_points)
    mu, gamma = model(x)
    assert mu.shape == (batch, n_param_pred)
    assert gamma.shape == (batch, n_param_pred)
    # constrain_output applies Sigmoid to mu -> values in (0, 1)
    assert mu.min().item() >= 0.0
    assert mu.max().item() <= 1.0


def test_ProbParamCNN_dependent_outputs():
    batch = 4
    n_points = 64
    n_param_pred = 3
    # dependent_outputs produces a full covariance matrix (batch, n, n)
    model = ProbParamCNN(
        input_shape=(2, n_points),
        chan_list=[8],
        fc_list=[16],
        fc_mu_list=[8],
        fc_gamma_list=[8],
        loss_fn=correlated_normal_loss,
        cyc_mode="discharge",
        n_param_pred=n_param_pred,
        dependent_outputs=True,
        constrain_output=False,
    )
    x = torch.rand(batch, 2, n_points)
    mu, cov = model(x)
    assert mu.shape == (batch, n_param_pred)
    assert cov.shape == (batch, n_param_pred, n_param_pred)


def test_ProbParamCNN_discharge_chargecc():
    batch = 4
    n_points = 64
    n_param_pred = 3
    # discharge-chargecc splits the 4-channel input into two 2-channel halves
    model = ProbParamCNN(
        input_shape=(4, n_points),
        chan_list=[8],
        fc_list=[16],
        fc_mu_list=[8],
        fc_gamma_list=[8],
        loss_fn=independent_normal_loss,
        cyc_mode="discharge-chargecc",
        n_param_pred=n_param_pred,
        constrain_output=False,
    )
    x = torch.rand(batch, 4, n_points)
    mu, gamma = model(x)
    assert mu.shape == (batch, n_param_pred)
    assert gamma.shape == (batch, n_param_pred)


def test_ProbParamFCNN():
    batch = 4
    input_dim = 32
    n_param_pred = 3

    model = ProbParamFCNN(
        input_shape=(input_dim,),
        hidden_list=[16],
        fc_mu_list=[8],
        fc_gamma_list=[8],
        loss_fn=independent_normal_loss,
        cyc_mode="discharge",
        n_param_pred=n_param_pred,
        constrain_output=True,
    )
    x = torch.rand(batch, input_dim)
    mu, gamma = model(x)
    assert mu.shape == (batch, n_param_pred)
    assert gamma.shape == (batch, n_param_pred)

    # discharge-chargecc: input is 2*input_dim wide, split into two halves
    model_dc = ProbParamFCNN(
        input_shape=(input_dim,),
        hidden_list=[16],
        fc_mu_list=[8],
        fc_gamma_list=[8],
        loss_fn=independent_normal_loss,
        cyc_mode="discharge-chargecc",
        n_param_pred=n_param_pred,
        constrain_output=False,
    )
    x_dc = torch.rand(batch, 2 * input_dim)
    mu_dc, gamma_dc = model_dc(x_dc)
    assert mu_dc.shape == (batch, n_param_pred)
    assert gamma_dc.shape == (batch, n_param_pred)


def test_transform_output():
    n_param_pred = 3
    model = ProbParamCNN(
        input_shape=(2, 64),
        chan_list=[8],
        fc_list=[16],
        fc_mu_list=[8],
        fc_gamma_list=[8],
        loss_fn=independent_normal_loss,
        cyc_mode="discharge",
        n_param_pred=n_param_pred,
        constrain_output=False,
    )
    min_par = torch.tensor([0.5, 0.6, 0.7])
    amp_par = torch.tensor([0.4, 0.3, 0.2])
    mu = torch.rand(4, n_param_pred)
    gamma = torch.rand(4, n_param_pred)

    mu_s, gamma_s = model.transform_output(mu, gamma, min_par, amp_par)
    mu_r, gamma_r = model.inv_transform_output(mu_s, gamma_s, min_par, amp_par)
    assert torch.allclose(mu, mu_r, atol=1e-5)
    assert torch.allclose(gamma, gamma_r, atol=1e-5)
