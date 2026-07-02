import pytest
import torch
import torch.nn as nn

from batfit.model.param_utils.losses import (
    correlated_normal_loss,
    independent_normal_loss,
)
from batfit.model.param_utils.model_utils import _SelfAttentionBlock
from batfit.model.paramNN import (
    ProbParamCNN,
    ProbParamFCNN,
    ProbParamFM,
    ProbProtParamCNN,
    ProbProtParamFM,
)


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


def test_ProbProtParamCNN():
    batch = 4
    n_points = 64
    n_param_pred = 3
    n_prot_params = 3

    # With fc_prot_list: CNN out + prot_params -> fc_prot_list -> mu/gamma heads
    model = ProbProtParamCNN(
        input_shape=(2, n_points),
        chan_list=[8],
        fc_list=[16],
        fc_prot_list=[32],
        fc_mu_list=[8],
        fc_gamma_list=[8],
        loss_fn=independent_normal_loss,
        n_prot_params=n_prot_params,
        cyc_mode="chirp",
        n_param_pred=n_param_pred,
        constrain_output=False,
    )
    x = torch.rand(batch, 2, n_points)
    prot_params = torch.rand(batch, n_prot_params)
    mu, gamma = model(x, prot_params)
    assert mu.shape == (batch, n_param_pred)
    assert gamma.shape == (batch, n_param_pred)
    # gamma should be positive (Softplus output)
    assert gamma.min().item() > 0.0

    # Without fc_prot_list: CNN out + prot_params fed directly to mu/gamma heads
    model_noprot = ProbProtParamCNN(
        input_shape=(2, n_points),
        chan_list=[8],
        fc_list=[16],
        fc_prot_list=[],
        fc_mu_list=[8],
        fc_gamma_list=[8],
        loss_fn=independent_normal_loss,
        n_prot_params=n_prot_params,
        cyc_mode="chirp",
        n_param_pred=n_param_pred,
        constrain_output=False,
    )
    mu2, gamma2 = model_noprot(x, prot_params)
    assert mu2.shape == (batch, n_param_pred)
    assert gamma2.shape == (batch, n_param_pred)

    # discharge-chargecc mode must raise
    with pytest.raises(AssertionError):
        ProbProtParamCNN(
            input_shape=(2, n_points),
            chan_list=[8],
            fc_list=[16],
            fc_prot_list=[],
            fc_mu_list=[8],
            fc_gamma_list=[8],
            loss_fn=independent_normal_loss,
            n_prot_params=n_prot_params,
            cyc_mode="discharge-chargecc",
            n_param_pred=n_param_pred,
        )


def test_ProbParamFM():
    batch = 4
    n_points = 64
    n_channels = 2
    n_param_pred = 3
    n_samples = 5

    # --- CNN mode ---
    model = ProbParamFM(
        input_shape=(n_channels, n_points),
        chan_list=[8],
        fc_list=[16],
        vf_hidden_list=[32],
        cyc_mode="discharge",
        n_param_pred=n_param_pred,
    )

    x = torch.rand(batch, n_channels, n_points)
    z_t = torch.rand(batch, n_param_pred)
    t = torch.rand(batch)

    velocity = model(x, z_t, t)
    assert velocity.shape == (batch, n_param_pred)

    samples = model.sample(x, n_samples=n_samples, n_steps=10)
    assert samples.shape == (batch, n_samples, n_param_pred)

    # discharge-chargecc: dual CNN encoder
    model_dc = ProbParamFM(
        input_shape=(2 * n_channels, n_points),
        chan_list=[8],
        fc_list=[16],
        vf_hidden_list=[32],
        cyc_mode="discharge-chargecc",
        n_param_pred=n_param_pred,
    )
    x_dc = torch.rand(batch, 2 * n_channels, n_points)
    velocity_dc = model_dc(x_dc, torch.rand(batch, n_param_pred), torch.rand(batch))
    assert velocity_dc.shape == (batch, n_param_pred)
    samples_dc = model_dc.sample(x_dc, n_samples=n_samples, n_steps=10)
    assert samples_dc.shape == (batch, n_samples, n_param_pred)

    # missing CNN args must raise
    with pytest.raises(ValueError):
        ProbParamFM(vf_hidden_list=[32], n_param_pred=n_param_pred)

    # --- External encoder mode ---
    latent_dim = 8

    class _DummyEncoder(nn.Module):
        """Minimal stand-in for ConvEncoder1D."""

        latent_dim = 8

        def forward(self, x):
            mu = torch.zeros(x.shape[0], self.latent_dim)
            logvar = torch.zeros(x.shape[0], self.latent_dim)
            return mu, logvar

    enc = _DummyEncoder()
    model_vae = ProbParamFM(
        vf_hidden_list=[32],
        encoder_model=enc,
        n_param_pred=n_param_pred,
    )

    # encoder weights must be frozen
    assert all(not p.requires_grad for p in enc.parameters())

    velocity_vae = model_vae(x, torch.rand(batch, n_param_pred), torch.rand(batch))
    assert velocity_vae.shape == (batch, n_param_pred)

    samples_vae = model_vae.sample(x, n_samples=n_samples, n_steps=10)
    assert samples_vae.shape == (batch, n_samples, n_param_pred)

    # discharge-chargecc with external encoder must raise
    with pytest.raises(NotImplementedError):
        ProbParamFM(
            vf_hidden_list=[32],
            encoder_model=enc,
            cyc_mode="discharge-chargecc",
            n_param_pred=n_param_pred,
        )

    # encoder_model without latent_dim must raise
    with pytest.raises(ValueError):
        ProbParamFM(
            vf_hidden_list=[32],
            encoder_model=nn.Linear(16, 8),
            n_param_pred=n_param_pred,
        )

    # --- Prior matching ---
    n_param_pred_pm = 6  # must match the YAML
    sim_config = "batfit/default_exps/spm_discharge.yaml"
    model_pm = ProbParamFM(
        input_shape=(n_channels, n_points),
        chan_list=[8],
        fc_list=[16],
        vf_hidden_list=[32],
        n_param_pred=n_param_pred_pm,
        sim_config=sim_config,
        use_prior_matching=True,
    )
    # sample_prior requires set_prior_data to have been called
    with pytest.raises(RuntimeError):
        model_pm.sample_prior(8, device=torch.device("cpu"))

    Y_prior = torch.rand(50, n_param_pred_pm)  # simulates scaled Y_train
    model_pm.set_prior_data(Y_prior)
    prior_samples = model_pm.sample_prior(8, device=torch.device("cpu"))
    assert prior_samples.shape == (8, n_param_pred_pm)
    # every row must be one of the registered training samples
    for row in prior_samples:
        assert any(torch.allclose(row, Y_prior[i]) for i in range(len(Y_prior)))

    x_pm = torch.rand(batch, n_channels, n_points)
    samples_pm = model_pm.sample(x_pm, n_samples=n_samples, n_steps=10)
    assert samples_pm.shape == (batch, n_samples, n_param_pred_pm)

    # use_prior_matching=True without sim_config must raise
    with pytest.raises(ValueError):
        ProbParamFM(
            input_shape=(n_channels, n_points),
            chan_list=[8],
            fc_list=[16],
            vf_hidden_list=[32],
            n_param_pred=n_param_pred,
            use_prior_matching=True,
        )


def test_ProbProtParamFM():
    batch = 4
    n_points = 64
    n_channels = 2
    n_param_pred = 3
    n_prot_params = 3
    n_samples = 5

    x = torch.rand(batch, n_channels, n_points)
    prot_params = torch.rand(batch, n_prot_params)
    z_t = torch.rand(batch, n_param_pred)
    t = torch.rand(batch)

    # With fc_prot_list: fusion layers between CNN emb + prot_params and VF
    model = ProbProtParamFM(
        input_shape=(n_channels, n_points),
        chan_list=[8],
        fc_list=[16],
        fc_prot_list=[32],
        vf_hidden_list=[32],
        n_prot_params=n_prot_params,
        cyc_mode="chirp",
        n_param_pred=n_param_pred,
    )
    velocity = model(x, prot_params, z_t, t)
    assert velocity.shape == (batch, n_param_pred)

    samples = model.sample(x, prot_params, n_samples=n_samples, n_steps=10)
    assert samples.shape == (batch, n_samples, n_param_pred)

    # Without fc_prot_list: CNN emb + prot_params fed directly to VF
    model_noprot = ProbProtParamFM(
        input_shape=(n_channels, n_points),
        chan_list=[8],
        fc_list=[16],
        fc_prot_list=[],
        vf_hidden_list=[32],
        n_prot_params=n_prot_params,
        cyc_mode="chirp",
        n_param_pred=n_param_pred,
    )
    velocity2 = model_noprot(x, prot_params, z_t, t)
    assert velocity2.shape == (batch, n_param_pred)

    samples2 = model_noprot.sample(x, prot_params, n_samples=n_samples, n_steps=10)
    assert samples2.shape == (batch, n_samples, n_param_pred)

    # discharge-chargecc must raise
    with pytest.raises(NotImplementedError):
        ProbProtParamFM(
            input_shape=(n_channels, n_points),
            chan_list=[8],
            fc_list=[16],
            fc_prot_list=[],
            vf_hidden_list=[32],
            n_prot_params=n_prot_params,
            cyc_mode="discharge-chargecc",
            n_param_pred=n_param_pred,
        )

    # --- Prior matching ---
    n_param_pred_pm = 6  # must match the YAML
    sim_config = "batfit/default_exps/spm_discharge.yaml"
    model_pm = ProbProtParamFM(
        input_shape=(n_channels, n_points),
        chan_list=[8],
        fc_list=[16],
        fc_prot_list=[32],
        vf_hidden_list=[32],
        n_prot_params=n_prot_params,
        cyc_mode="discharge",
        n_param_pred=n_param_pred_pm,
        sim_config=sim_config,
        use_prior_matching=True,
    )
    with pytest.raises(RuntimeError):
        model_pm.sample_prior(8, device=torch.device("cpu"))

    Y_prior_prot = torch.rand(50, n_param_pred_pm)
    model_pm.set_prior_data(Y_prior_prot)
    prior_samples = model_pm.sample_prior(8, device=torch.device("cpu"))
    assert prior_samples.shape == (8, n_param_pred_pm)

    x_pm = torch.rand(batch, n_channels, n_points)
    prot_pm = torch.rand(batch, n_prot_params)
    samples_pm = model_pm.sample(x_pm, prot_pm, n_samples=n_samples, n_steps=10)
    assert samples_pm.shape == (batch, n_samples, n_param_pred_pm)


def test__SelfAttentionBlock():
    """Shape preservation, gradient flow, and invalid num_heads validation."""
    batch, channels, time = 3, 8, 16
    block = _SelfAttentionBlock(embed_dim=channels, num_heads=4)

    x = torch.rand(batch, channels, time)
    out = block(x)

    # Output shape must exactly match the input
    assert out.shape == (batch, channels, time)

    # Gradients must flow through the residual path
    x_grad = torch.rand(batch, channels, time, requires_grad=True)
    block(x_grad).sum().backward()
    assert x_grad.grad is not None
    assert x_grad.grad.shape == x_grad.shape

    # embed_dim not divisible by num_heads must raise
    with pytest.raises(ValueError):
        _SelfAttentionBlock(embed_dim=8, num_heads=3)

    # Single head must also work (trivial case)
    block_single = _SelfAttentionBlock(embed_dim=channels, num_heads=1)
    assert block_single(x).shape == (batch, channels, time)


def test_ProbParamCNN_attention():
    """CNN NPE with a self-attention block produces correct output shapes."""
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
        constrain_output=False,
        num_attn_heads=4,
        attn_dropout=0.0,
    )
    x = torch.rand(batch, 2, n_points)
    mu, gamma = model(x)
    assert mu.shape == (batch, n_param_pred)
    assert gamma.shape == (batch, n_param_pred)
    # Softplus output head must produce positive sigmas
    assert gamma.min().item() > 0.0

    # num_attn_heads not dividing chan_list[-1] must raise during construction
    with pytest.raises(ValueError):
        ProbParamCNN(
            input_shape=(2, n_points),
            chan_list=[8],
            fc_list=[16],
            fc_mu_list=[8],
            fc_gamma_list=[8],
            loss_fn=independent_normal_loss,
            n_param_pred=n_param_pred,
            num_attn_heads=3,
        )


def test_ProbParamFM_attention():
    """FM model with a self-attention block produces correct output shapes."""
    batch = 4
    n_points = 64
    n_channels = 2
    n_param_pred = 3
    n_samples = 5

    model = ProbParamFM(
        input_shape=(n_channels, n_points),
        chan_list=[8],
        fc_list=[16],
        vf_hidden_list=[32],
        cyc_mode="discharge",
        n_param_pred=n_param_pred,
        num_attn_heads=4,
        attn_dropout=0.0,
    )
    x = torch.rand(batch, n_channels, n_points)
    z_t = torch.rand(batch, n_param_pred)
    t = torch.rand(batch)

    velocity = model(x, z_t, t)
    assert velocity.shape == (batch, n_param_pred)

    samples = model.sample(x, n_samples=n_samples, n_steps=10)
    assert samples.shape == (batch, n_samples, n_param_pred)
