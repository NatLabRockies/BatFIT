import numpy as np
import pytest
import torch
import torch.nn as nn

from batfit.model.param_utils.losses import independent_normal_loss
from batfit.model.param_utils.model_utils import (
    _SelfAttentionBlock,
    encoder_channels,
    signal_scaler_shape,
)
from batfit.model.paramNN import (
    ProbParamCNN,
    ProbParamFCNN,
    ProbParamFM,
    ProbProtParamCNN,
    ProbProtParamFM,
)
from batfit.utils.scalers import ZScoreScaler


def test_ProbParamCNN():
    batch = 4
    n_points = 64
    n_param_pred = 6  # parameters declared in the YAML
    sim_config = "batfit/default_exps/spm_discharge.yaml"

    model = ProbParamCNN(
        input_shape=(2, n_points),
        chan_list=[8],
        fc_list=[16],
        fc_mu_list=[8],
        fc_gamma_list=[8],
        loss_fn=independent_normal_loss,
        sim_config=sim_config,
        cyc_mode="discharge",
        param_margin=0.1,
    )
    x = torch.rand(batch, 2, n_points)
    mu, gamma = model(x)
    assert mu.shape == (batch, n_param_pred)
    assert gamma.shape == (batch, n_param_pred)
    # scaled outputs: mu within [-margin, 1 + margin], sigma within (0, 1)
    assert mu.min().item() >= -0.1
    assert mu.max().item() <= 1.1
    assert gamma.min().item() > 0.0
    assert gamma.max().item() < 1.0
    # scalers are submodules, saved in the state dict
    keys = model.state_dict().keys()
    assert {"scaler_X.means", "scaler_Y.low", "scaler_Y.high"} <= set(keys)
    # without a fitted scaler_X, an identity placeholder is created
    assert torch.all(model.scaler_X.means == 0.0)
    assert torch.all(model.scaler_X.stds == 1.0)

    # discharge-chargecc splits the 4-channel input into two 2-channel halves
    model_dc = ProbParamCNN(
        input_shape=(4, n_points),
        chan_list=[8],
        fc_list=[16],
        fc_mu_list=[8],
        fc_gamma_list=[8],
        loss_fn=independent_normal_loss,
        sim_config=sim_config,
        cyc_mode="discharge-chargecc",
    )
    mu_dc, gamma_dc = model_dc(torch.rand(batch, 4, n_points))
    assert mu_dc.shape == (batch, n_param_pred)
    assert gamma_dc.shape == (batch, n_param_pred)

    # the number of predicted parameters is read from the config
    assert model.n_param_pred == n_param_pred

    # time_dependent_zscore: voltage-only CNN, end time fused before the heads
    model_td = ProbParamCNN(
        input_shape=(2, n_points),
        chan_list=[8],
        fc_list=[16],
        fc_mu_list=[8],
        fc_gamma_list=[8],
        loss_fn=independent_normal_loss,
        sim_config=sim_config,
        signal_scaling="time_dependent_zscore",
    )
    mu_td, gamma_td = model_td(
        torch.rand(batch, 1, n_points), t_end=torch.rand(batch, 1)
    )
    assert mu_td.shape == (batch, n_param_pred)
    assert gamma_td.shape == (batch, n_param_pred)
    # placeholder scalers: one voltage statistic per time point, one for T
    assert tuple(model_td.scaler_X.means.shape) == (1, 1, n_points)
    assert {"scaler_T.means", "scaler_T.stds"} <= set(model_td.state_dict())
    assert model.scaler_T is None
    # the end time is required in this mode and rejected in the other
    with pytest.raises(AssertionError):
        model_td(torch.rand(batch, 1, n_points))
    with pytest.raises(AssertionError):
        model(x, t_end=torch.rand(batch, 1))
    # dual encoders need two end times: not supported
    with pytest.raises(NotImplementedError):
        ProbParamCNN(
            input_shape=(4, n_points),
            chan_list=[8],
            fc_list=[16],
            fc_mu_list=[8],
            fc_gamma_list=[8],
            loss_fn=independent_normal_loss,
            sim_config=sim_config,
            cyc_mode="discharge-chargecc",
            signal_scaling="time_dependent_zscore",
        )
    # unknown scaling modes are rejected
    with pytest.raises(AssertionError):
        ProbParamCNN(
            input_shape=(2, n_points),
            chan_list=[8],
            fc_list=[16],
            fc_mu_list=[8],
            fc_gamma_list=[8],
            loss_fn=independent_normal_loss,
            sim_config=sim_config,
            signal_scaling="minmax",
        )


def test_ProbParamFCNN():
    batch = 4
    input_dim = 32
    n_param_pred = 6
    sim_config = "batfit/default_exps/spm_discharge.yaml"

    model = ProbParamFCNN(
        input_shape=(input_dim,),
        hidden_list=[16],
        fc_mu_list=[8],
        fc_gamma_list=[8],
        loss_fn=independent_normal_loss,
        sim_config=sim_config,
        cyc_mode="discharge",
    )
    x = torch.rand(batch, input_dim)
    mu, gamma = model(x)
    assert mu.shape == (batch, n_param_pred)
    assert gamma.shape == (batch, n_param_pred)
    # flat input: the signal scaler has 2D statistics
    assert tuple(model.scaler_X.means.shape) == (1, input_dim)

    # discharge-chargecc: input is 2*input_dim wide, split into two halves
    model_dc = ProbParamFCNN(
        input_shape=(input_dim,),
        hidden_list=[16],
        fc_mu_list=[8],
        fc_gamma_list=[8],
        loss_fn=independent_normal_loss,
        sim_config=sim_config,
        cyc_mode="discharge-chargecc",
    )
    x_dc = torch.rand(batch, 2 * input_dim)
    mu_dc, gamma_dc = model_dc(x_dc)
    assert mu_dc.shape == (batch, n_param_pred)
    assert gamma_dc.shape == (batch, n_param_pred)

    # time_dependent_zscore: flattened voltage, end time fused before heads
    n_points = 64
    model_td = ProbParamFCNN(
        input_shape=(2, n_points),
        hidden_list=[16],
        fc_mu_list=[8],
        fc_gamma_list=[8],
        loss_fn=independent_normal_loss,
        sim_config=sim_config,
        signal_scaling="time_dependent_zscore",
    )
    assert tuple(model_td.scaler_X.means.shape) == (1, 1, n_points)
    mu_td, gamma_td = model_td(
        torch.rand(batch, 1, n_points), t_end=torch.rand(batch, 1)
    )
    assert mu_td.shape == (batch, n_param_pred)
    assert gamma_td.shape == (batch, n_param_pred)
    with pytest.raises(AssertionError):
        model_td(torch.rand(batch, 1, n_points))
    # the physical API splits the (time, voltage) signal
    t_end = torch.rand(batch, 1) * 4000.0 + 1000.0
    grid = torch.linspace(0.0, 1.0, n_points)
    x_phys = torch.stack(
        (t_end * grid, torch.rand(batch, n_points) + 3.0), dim=1
    )
    model_td.eval()
    with torch.no_grad():
        mu_phys, sigma_phys = model_td.predict_physical(x_phys)
    assert mu_phys.shape == (batch, n_param_pred)


def test_to_physical():
    model = ProbParamCNN(
        input_shape=(2, 64),
        chan_list=[8],
        fc_list=[16],
        fc_mu_list=[8],
        fc_gamma_list=[8],
        loss_fn=independent_normal_loss,
        sim_config="batfit/default_exps/spm_discharge.yaml",
    )
    low, high = model.scaler_Y.low, model.scaler_Y.high
    # 0 -> lower bound, 1 -> upper bound, overshoot clipped to the bounds
    mu = torch.stack([torch.zeros(6), torch.ones(6), torch.full((6,), 1.05)])
    sigma = torch.full((3, 6), 0.1)
    mu_phys, sigma_phys = model.to_physical(mu, sigma)
    assert torch.allclose(mu_phys[0], low)
    assert torch.allclose(mu_phys[1], high)
    assert torch.allclose(mu_phys[2], high)
    # sigma only picks up the range of each parameter
    assert torch.allclose(sigma_phys, 0.1 * (high - low).expand(3, -1))


def test_predict_physical():
    torch.manual_seed(0)
    batch = 4
    n_points = 64
    X = np.random.rand(10, 2, n_points).astype("float32") * 0.5 + 3.5
    scaler_X = ZScoreScaler.fit(X, axis=(0, 2))
    model = ProbProtParamCNN(
        input_shape=(2, n_points),
        chan_list=[8],
        fc_list=[16],
        fc_prot_list=[8],
        fc_mu_list=[8],
        fc_gamma_list=[8],
        loss_fn=independent_normal_loss,
        sim_config="batfit/default_exps/spm_chirp.yaml",
        scaler_X=scaler_X,
    )
    model.eval()
    x = torch.from_numpy(X[:batch])
    # physical protocol parameters inside the configured bounds
    p = model.scaler_P.inverse_transform(torch.rand(batch, 3))

    with torch.no_grad():
        mu, sigma = model.predict_physical(x, p)
        # same result as scaling by hand then calling to_physical
        mu_scaled, sigma_scaled = model(
            scaler_X.transform(x), model.scaler_P.transform(p)
        )
        mu_ref, sigma_ref = model.to_physical(mu_scaled, sigma_scaled)
    assert torch.allclose(mu, mu_ref)
    assert torch.allclose(sigma, sigma_ref)
    assert torch.all(mu >= model.scaler_Y.low)
    assert torch.all(mu <= model.scaler_Y.high)

    # protocol models require protocol parameters
    with pytest.raises(AssertionError):
        model.predict_physical(x)

    # time_dependent_zscore: same physical API, (time, voltage) signal in
    t_end = np.random.uniform(1000.0, 5000.0, (10, 1)).astype("float32")
    grid = np.linspace(0.0, 1.0, n_points, dtype="float32")
    X_td = np.stack((t_end * grid, X[:, 1, :]), axis=1)
    scaler_V = ZScoreScaler.fit(X_td[:, 1:, :], axis=0, min_std=1e-3)
    scaler_T = ZScoreScaler.fit(t_end, axis=0)
    for model_td in (
        ProbParamCNN(
            input_shape=(2, n_points),
            chan_list=[8],
            fc_list=[16],
            fc_mu_list=[8],
            fc_gamma_list=[8],
            loss_fn=independent_normal_loss,
            sim_config="batfit/default_exps/spm_discharge.yaml",
            scaler_X=scaler_V,
            signal_scaling="time_dependent_zscore",
            scaler_T=scaler_T,
        ),
        ProbProtParamCNN(
            input_shape=(2, n_points),
            chan_list=[8],
            fc_list=[16],
            fc_prot_list=[8],
            fc_mu_list=[8],
            fc_gamma_list=[8],
            loss_fn=independent_normal_loss,
            sim_config="batfit/default_exps/spm_chirp.yaml",
            scaler_X=scaler_V,
            signal_scaling="time_dependent_zscore",
            scaler_T=scaler_T,
        ),
    ):
        model_td.eval()
        x_td = torch.from_numpy(X_td[:batch])
        inputs = [scaler_V.transform(x_td[:, 1:, :])]
        p_td = None
        if model_td.scaler_P is not None:
            p_td = model_td.scaler_P.inverse_transform(torch.rand(batch, 3))
            inputs.append(model_td.scaler_P.transform(p_td))
        with torch.no_grad():
            mu, sigma = model_td.predict_physical(x_td, p_td)
            mu_scaled, sigma_scaled = model_td(
                *inputs, t_end=scaler_T.transform(x_td[:, 0, -1:])
            )
            mu_ref, sigma_ref = model_td.to_physical(mu_scaled, sigma_scaled)
        assert torch.allclose(mu, mu_ref)
        assert torch.allclose(sigma, sigma_ref)


def test_ProbProtParamCNN():
    batch = 4
    n_points = 64
    n_param_pred = 6
    n_prot_params = 3
    sim_config = "batfit/default_exps/spm_chirp.yaml"

    # With fc_prot_list: CNN out + prot_params -> fc_prot_list -> mu/gamma heads
    model = ProbProtParamCNN(
        input_shape=(2, n_points),
        chan_list=[8],
        fc_list=[16],
        fc_prot_list=[32],
        fc_mu_list=[8],
        fc_gamma_list=[8],
        loss_fn=independent_normal_loss,
        sim_config=sim_config,
        cyc_mode="chirp",
    )
    x = torch.rand(batch, 2, n_points)
    prot_params = torch.rand(batch, n_prot_params)
    mu, gamma = model(x, prot_params)
    assert mu.shape == (batch, n_param_pred)
    assert gamma.shape == (batch, n_param_pred)
    # gamma should be positive (Sigmoid output)
    assert gamma.min().item() > 0.0
    # protocol scaler and parameter counts built from the config
    assert model.scaler_P.low.shape == (n_prot_params,)
    assert model.n_prot_params == n_prot_params
    assert model.n_param_pred == n_param_pred

    # a config without protocol parameters is rejected
    with pytest.raises(AssertionError):
        ProbProtParamCNN(
            input_shape=(2, n_points),
            chan_list=[8],
            fc_list=[16],
            fc_prot_list=[],
            fc_mu_list=[8],
            fc_gamma_list=[8],
            loss_fn=independent_normal_loss,
            sim_config="batfit/default_exps/spm_discharge.yaml",
        )

    # Without fc_prot_list: CNN out + prot_params fed directly to mu/gamma heads
    model_noprot = ProbProtParamCNN(
        input_shape=(2, n_points),
        chan_list=[8],
        fc_list=[16],
        fc_prot_list=[],
        fc_mu_list=[8],
        fc_gamma_list=[8],
        loss_fn=independent_normal_loss,
        sim_config=sim_config,
        cyc_mode="chirp",
    )
    mu2, gamma2 = model_noprot(x, prot_params)
    assert mu2.shape == (batch, n_param_pred)
    assert gamma2.shape == (batch, n_param_pred)

    # time_dependent_zscore: end time fused with the protocol parameters
    for fc_prot_list in ([32], []):
        model_td = ProbProtParamCNN(
            input_shape=(2, n_points),
            chan_list=[8],
            fc_list=[16],
            fc_prot_list=fc_prot_list,
            fc_mu_list=[8],
            fc_gamma_list=[8],
            loss_fn=independent_normal_loss,
            sim_config=sim_config,
            signal_scaling="time_dependent_zscore",
        )
        mu_td, gamma_td = model_td(
            torch.rand(batch, 1, n_points),
            prot_params,
            t_end=torch.rand(batch, 1),
        )
        assert mu_td.shape == (batch, n_param_pred)
        assert gamma_td.shape == (batch, n_param_pred)

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
            sim_config=sim_config,
            cyc_mode="discharge-chargecc",
        )


def test_ProbParamFM():
    batch = 4
    n_points = 64
    n_channels = 2
    n_param_pred = 6  # parameters declared in the YAML
    n_samples = 5
    sim_config = "batfit/default_exps/spm_discharge.yaml"

    # --- CNN mode ---
    model = ProbParamFM(
        input_shape=(n_channels, n_points),
        chan_list=[8],
        fc_list=[16],
        vf_hidden_list=[32],
        sim_config=sim_config,
        cyc_mode="discharge",
    )
    assert model.n_param_pred == n_param_pred
    assert {"scaler_X.means", "scaler_Y.low"} <= set(model.state_dict())

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
        sim_config=sim_config,
        cyc_mode="discharge-chargecc",
    )
    x_dc = torch.rand(batch, 2 * n_channels, n_points)
    velocity_dc = model_dc(
        x_dc, torch.rand(batch, n_param_pred), torch.rand(batch)
    )
    assert velocity_dc.shape == (batch, n_param_pred)
    samples_dc = model_dc.sample(x_dc, n_samples=n_samples, n_steps=10)
    assert samples_dc.shape == (batch, n_samples, n_param_pred)

    # time_dependent_zscore: end time appended to the CNN context
    model_td = ProbParamFM(
        input_shape=(n_channels, n_points),
        chan_list=[8],
        fc_list=[16],
        vf_hidden_list=[32],
        sim_config=sim_config,
        signal_scaling="time_dependent_zscore",
    )
    v_td = torch.rand(batch, 1, n_points)
    t_end = torch.rand(batch, 1)
    assert model_td(v_td, z_t, t, t_end=t_end).shape == (batch, n_param_pred)
    samples_td = model_td.sample(v_td, n_samples, n_steps=10, t_end=t_end)
    assert samples_td.shape == (batch, n_samples, n_param_pred)
    with pytest.raises(AssertionError):
        model_td(v_td, z_t, t)

    # missing CNN args must raise
    with pytest.raises(ValueError):
        ProbParamFM(vf_hidden_list=[32], sim_config=sim_config)

    # --- External encoder mode ---
    class _DummyEncoder(nn.Module):
        """Minimal stand-in for ConvEncoder1D."""

        latent_dim = 8

        def forward(self, x):
            mu = torch.zeros(x.shape[0], self.latent_dim)
            logvar = torch.zeros(x.shape[0], self.latent_dim)
            return mu, logvar

    # the signal shape is unknown with an external encoder: pass scaler_X
    scaler_X = ZScoreScaler(
        np.zeros((1, n_channels, 1)), np.ones((1, n_channels, 1))
    )
    enc = _DummyEncoder()
    model_vae = ProbParamFM(
        vf_hidden_list=[32],
        sim_config=sim_config,
        encoder_model=enc,
        scaler_X=scaler_X,
    )

    # encoder weights must be frozen
    assert all(not p.requires_grad for p in enc.parameters())

    velocity_vae = model_vae(
        x, torch.rand(batch, n_param_pred), torch.rand(batch)
    )
    assert velocity_vae.shape == (batch, n_param_pred)

    samples_vae = model_vae.sample(x, n_samples=n_samples, n_steps=10)
    assert samples_vae.shape == (batch, n_samples, n_param_pred)

    # external encoder without scaler_X must raise
    with pytest.raises(AssertionError):
        ProbParamFM(
            vf_hidden_list=[32], sim_config=sim_config, encoder_model=enc
        )

    # discharge-chargecc with external encoder must raise
    with pytest.raises(NotImplementedError):
        ProbParamFM(
            vf_hidden_list=[32],
            sim_config=sim_config,
            encoder_model=enc,
            scaler_X=scaler_X,
            cyc_mode="discharge-chargecc",
        )

    # time_dependent_zscore with external encoder must raise
    with pytest.raises(NotImplementedError):
        ProbParamFM(
            vf_hidden_list=[32],
            sim_config=sim_config,
            encoder_model=enc,
            scaler_X=scaler_X,
            signal_scaling="time_dependent_zscore",
        )

    # encoder_model without latent_dim must raise
    with pytest.raises(ValueError):
        ProbParamFM(
            vf_hidden_list=[32],
            sim_config=sim_config,
            encoder_model=nn.Linear(16, 8),
            scaler_X=scaler_X,
        )

    # --- Prior matching ---
    model_pm = ProbParamFM(
        input_shape=(n_channels, n_points),
        chan_list=[8],
        fc_list=[16],
        vf_hidden_list=[32],
        sim_config=sim_config,
        use_prior_matching=True,
    )
    # sample_prior requires set_prior_data to have been called
    with pytest.raises(RuntimeError):
        model_pm.sample_prior(8, device=torch.device("cpu"))

    Y_prior_u = torch.rand(50, n_param_pred)  # labels scaled to [0, 1]
    model_pm.set_prior_data(Y_prior_u)
    # the prior is stored in the flow space: (u - 0.5) * sqrt(12)
    Y_prior_flow = (Y_prior_u - 0.5) * np.sqrt(12.0)
    assert torch.allclose(model_pm.Y_prior, Y_prior_flow)
    prior_samples = model_pm.sample_prior(8, device=torch.device("cpu"))
    assert prior_samples.shape == (8, n_param_pred)
    # every row must be one of the registered training samples
    for row in prior_samples:
        assert any(
            torch.allclose(row, Y_prior_flow[i])
            for i in range(len(Y_prior_flow))
        )

    x_pm = torch.rand(batch, n_channels, n_points)
    samples_pm = model_pm.sample(x_pm, n_samples=n_samples, n_steps=10)
    assert samples_pm.shape == (batch, n_samples, n_param_pred)


def test_ProbProtParamFM():
    batch = 4
    n_points = 64
    n_channels = 2
    n_param_pred = 6
    n_prot_params = 3
    n_samples = 5
    sim_config = "batfit/default_exps/spm_chirp.yaml"

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
        sim_config=sim_config,
        cyc_mode="chirp",
    )
    # parameter counts read from the config
    assert model.n_param_pred == n_param_pred
    assert model.n_prot_params == n_prot_params
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
        sim_config=sim_config,
        cyc_mode="chirp",
    )
    velocity2 = model_noprot(x, prot_params, z_t, t)
    assert velocity2.shape == (batch, n_param_pred)

    samples2 = model_noprot.sample(
        x, prot_params, n_samples=n_samples, n_steps=10
    )
    assert samples2.shape == (batch, n_samples, n_param_pred)

    # time_dependent_zscore: end time fused with the protocol parameters
    v_td = torch.rand(batch, 1, n_points)
    t_end = torch.rand(batch, 1)
    for fc_prot_list in ([32], []):
        model_td = ProbProtParamFM(
            input_shape=(n_channels, n_points),
            chan_list=[8],
            fc_list=[16],
            fc_prot_list=fc_prot_list,
            vf_hidden_list=[32],
            sim_config=sim_config,
            signal_scaling="time_dependent_zscore",
        )
        velocity_td = model_td(v_td, prot_params, z_t, t, t_end=t_end)
        assert velocity_td.shape == (batch, n_param_pred)
        samples_td = model_td.sample(
            v_td, prot_params, n_samples=n_samples, n_steps=10, t_end=t_end
        )
        assert samples_td.shape == (batch, n_samples, n_param_pred)

    # discharge-chargecc must raise
    with pytest.raises(NotImplementedError):
        ProbProtParamFM(
            input_shape=(n_channels, n_points),
            chan_list=[8],
            fc_list=[16],
            fc_prot_list=[],
            vf_hidden_list=[32],
            sim_config=sim_config,
            cyc_mode="discharge-chargecc",
        )

    # a config without protocol parameters is rejected
    with pytest.raises(AssertionError):
        ProbProtParamFM(
            input_shape=(n_channels, n_points),
            chan_list=[8],
            fc_list=[16],
            fc_prot_list=[],
            vf_hidden_list=[32],
            sim_config="batfit/default_exps/spm_discharge.yaml",
        )

    # --- Prior matching ---
    model_pm = ProbProtParamFM(
        input_shape=(n_channels, n_points),
        chan_list=[8],
        fc_list=[16],
        fc_prot_list=[32],
        vf_hidden_list=[32],
        sim_config=sim_config,
        use_prior_matching=True,
    )
    with pytest.raises(RuntimeError):
        model_pm.sample_prior(8, device=torch.device("cpu"))

    model_pm.set_prior_data(torch.rand(50, n_param_pred))
    prior_samples = model_pm.sample_prior(8, device=torch.device("cpu"))
    assert prior_samples.shape == (8, n_param_pred)

    samples_pm = model_pm.sample(
        x, prot_params, n_samples=n_samples, n_steps=10
    )
    assert samples_pm.shape == (batch, n_samples, n_param_pred)


def test_to_physical_fm():
    model = ProbParamFM(
        input_shape=(2, 64),
        chan_list=[8],
        fc_list=[16],
        vf_hidden_list=[32],
        sim_config="batfit/default_exps/spm_discharge.yaml",
    )
    low, high = model.scaler_Y.low, model.scaler_Y.high
    half_width = np.sqrt(3.0)  # flow-space image of the bounds
    # flow space: -sqrt(3) -> lower bound, 0 -> middle, +sqrt(3) -> upper
    # bound, beyond +sqrt(3) -> clamped to the upper bound
    z = torch.stack(
        [
            torch.full((6,), -half_width),
            torch.zeros(6),
            torch.full((6,), half_width),
            torch.full((6,), 5.0),
        ]
    ).unsqueeze(
        0
    )  # (batch=1, n_samples=4, n_param_pred=6)
    samples = model.to_physical(z)
    assert samples.shape == (1, 4, 6)
    assert torch.allclose(samples[0, 0], low, atol=1e-5)
    assert torch.allclose(samples[0, 1], 0.5 * (low + high), atol=1e-5)
    assert torch.allclose(samples[0, 2], high, atol=1e-5)
    assert torch.allclose(samples[0, 3], high)


def test_sample_physical():
    torch.manual_seed(0)
    batch = 3
    n_points = 64
    X = np.random.rand(10, 2, n_points).astype("float32") * 0.5 + 3.5
    scaler_X = ZScoreScaler.fit(X, axis=(0, 2))
    model = ProbProtParamFM(
        input_shape=(2, n_points),
        chan_list=[8],
        fc_list=[16],
        fc_prot_list=[8],
        vf_hidden_list=[16],
        sim_config="batfit/default_exps/spm_chirp.yaml",
        scaler_X=scaler_X,
    )
    model.eval()
    x = torch.from_numpy(X[:batch])
    # physical protocol parameters inside the configured bounds
    p = model.scaler_P.inverse_transform(torch.rand(batch, 3))

    with torch.no_grad():
        torch.manual_seed(1)
        samples = model.sample_physical(x, p, n_samples=7, n_steps=5)
        # same result as scaling by hand, sampling, then to_physical
        torch.manual_seed(1)
        samples_flow = model.sample(
            scaler_X.transform(x),
            model.scaler_P.transform(p),
            n_samples=7,
            n_steps=5,
        )
        samples_ref = model.to_physical(samples_flow)
    assert samples.shape == (batch, 7, 6)
    assert torch.allclose(samples, samples_ref)
    assert torch.all(samples >= model.scaler_Y.low)
    assert torch.all(samples <= model.scaler_Y.high)

    # protocol models require protocol parameters
    with pytest.raises(AssertionError):
        model.sample_physical(x)

    # time_dependent_zscore: same physical API, (time, voltage) signal in
    t_end = np.random.uniform(1000.0, 5000.0, (10, 1)).astype("float32")
    grid = np.linspace(0.0, 1.0, n_points, dtype="float32")
    X_td = np.stack((t_end * grid, X[:, 1, :]), axis=1)
    scaler_V = ZScoreScaler.fit(X_td[:, 1:, :], axis=0, min_std=1e-3)
    scaler_T = ZScoreScaler.fit(t_end, axis=0)
    for model_td in (
        ProbParamFM(
            input_shape=(2, n_points),
            chan_list=[8],
            fc_list=[16],
            vf_hidden_list=[16],
            sim_config="batfit/default_exps/spm_discharge.yaml",
            scaler_X=scaler_V,
            signal_scaling="time_dependent_zscore",
            scaler_T=scaler_T,
        ),
        ProbProtParamFM(
            input_shape=(2, n_points),
            chan_list=[8],
            fc_list=[16],
            fc_prot_list=[8],
            vf_hidden_list=[16],
            sim_config="batfit/default_exps/spm_chirp.yaml",
            scaler_X=scaler_V,
            signal_scaling="time_dependent_zscore",
            scaler_T=scaler_T,
        ),
    ):
        model_td.eval()
        x_td = torch.from_numpy(X_td[:batch])
        inputs = [scaler_V.transform(x_td[:, 1:, :])]
        p_td = None
        if model_td.scaler_P is not None:
            p_td = model_td.scaler_P.inverse_transform(torch.rand(batch, 3))
            inputs.append(model_td.scaler_P.transform(p_td))
        with torch.no_grad():
            torch.manual_seed(1)
            samples = model_td.sample_physical(
                x_td, p_td, n_samples=7, n_steps=5
            )
            torch.manual_seed(1)
            samples_flow = model_td.sample(
                *inputs,
                n_samples=7,
                n_steps=5,
                t_end=scaler_T.transform(x_td[:, 0, -1:]),
            )
        assert torch.allclose(samples, model_td.to_physical(samples_flow))


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
    n_param_pred = 6
    sim_config = "batfit/default_exps/spm_discharge.yaml"

    model = ProbParamCNN(
        input_shape=(2, n_points),
        chan_list=[8],
        fc_list=[16],
        fc_mu_list=[8],
        fc_gamma_list=[8],
        loss_fn=independent_normal_loss,
        sim_config=sim_config,
        cyc_mode="discharge",
        num_attn_heads=4,
        attn_dropout=0.0,
    )
    x = torch.rand(batch, 2, n_points)
    mu, gamma = model(x)
    assert mu.shape == (batch, n_param_pred)
    assert gamma.shape == (batch, n_param_pred)
    # Sigmoid output head must produce positive sigmas
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
            sim_config=sim_config,
            num_attn_heads=3,
        )


def test_ProbProtParamCNN_attention():
    """Protocol CNN NPE with attention layer produces correct output shapes."""
    batch = 4
    n_points = 64
    n_param_pred = 6
    n_prot_params = 3
    sim_config = "batfit/default_exps/spm_chirp.yaml"

    model = ProbProtParamCNN(
        input_shape=(2, n_points),
        chan_list=[8],
        fc_list=[16],
        fc_prot_list=[32],
        fc_mu_list=[8],
        fc_gamma_list=[8],
        loss_fn=independent_normal_loss,
        sim_config=sim_config,
        cyc_mode="chirp",
        num_attn_heads=4,
        attn_dropout=0.0,
    )
    x = torch.rand(batch, 2, n_points)
    prot_params = torch.rand(batch, n_prot_params)
    mu, gamma = model(x, prot_params)
    assert mu.shape == (batch, n_param_pred)
    assert gamma.shape == (batch, n_param_pred)
    assert gamma.min().item() > 0.0

    # num_attn_heads not dividing chan_list[-1] must raise
    with pytest.raises(ValueError):
        ProbProtParamCNN(
            input_shape=(2, n_points),
            chan_list=[8],
            fc_list=[16],
            fc_prot_list=[],
            fc_mu_list=[8],
            fc_gamma_list=[8],
            loss_fn=independent_normal_loss,
            sim_config=sim_config,
            num_attn_heads=3,
        )


def test_ProbParamFM_attention():
    """FM model with a self-attention block produces correct output shapes."""
    batch = 4
    n_points = 64
    n_channels = 2
    n_param_pred = 6
    n_samples = 5

    model = ProbParamFM(
        input_shape=(n_channels, n_points),
        chan_list=[8],
        fc_list=[16],
        vf_hidden_list=[32],
        sim_config="batfit/default_exps/spm_discharge.yaml",
        cyc_mode="discharge",
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


def test_ProbProtParamFM_attention():
    """Protocol FM model with attention and prior matching produces correct shapes."""
    batch = 4
    n_points = 64
    n_channels = 2
    n_param_pred = 6
    n_prot_params = 3
    n_samples = 5
    sim_config = "batfit/default_exps/spm_chirp.yaml"

    model = ProbProtParamFM(
        input_shape=(n_channels, n_points),
        chan_list=[8],
        fc_list=[16],
        fc_prot_list=[32],
        vf_hidden_list=[32],
        sim_config=sim_config,
        cyc_mode="chirp",
        num_attn_heads=4,
        attn_dropout=0.0,
    )
    x = torch.rand(batch, n_channels, n_points)
    prot_params = torch.rand(batch, n_prot_params)
    z_t = torch.rand(batch, n_param_pred)
    t = torch.rand(batch)

    velocity = model(x, prot_params, z_t, t)
    assert velocity.shape == (batch, n_param_pred)

    samples = model.sample(x, prot_params, n_samples=n_samples, n_steps=10)
    assert samples.shape == (batch, n_samples, n_param_pred)

    # Prior matching via set_prior_data/sample_prior (inherited from base)
    model.set_prior_data(torch.rand(50, n_param_pred))
    prior_samples = model.sample_prior(8, device=torch.device("cpu"))
    assert prior_samples.shape == (8, n_param_pred)

    # num_attn_heads not dividing chan_list[-1] must raise
    with pytest.raises(ValueError):
        ProbProtParamFM(
            input_shape=(n_channels, n_points),
            chan_list=[8],
            fc_list=[16],
            fc_prot_list=[],
            vf_hidden_list=[32],
            sim_config=sim_config,
            num_attn_heads=3,
        )


def test_signal_scaler_shape():
    assert signal_scaler_shape((2, 64), "zscore") == (1, 2, 1)
    assert signal_scaler_shape((2, 64), "time_dependent_zscore") == (1, 1, 64)


def test_encoder_channels():
    assert encoder_channels((2, 64), "zscore") == 2
    # the time channel is replaced by the end time
    assert encoder_channels((2, 64), "time_dependent_zscore") == 1


def test_scale_signal():
    n_points = 32
    t_end = np.random.uniform(1000.0, 5000.0, (6, 1)).astype("float32")
    grid = np.linspace(0.0, 1.0, n_points, dtype="float32")
    voltage = np.random.uniform(3.0, 4.1, (6, n_points)).astype("float32")
    X = np.stack((t_end * grid, voltage), axis=1)

    # zscore: one scaled (time, voltage) input
    scaler_X = ZScoreScaler.fit(X, axis=(0, 2))
    model = ProbParamFM(
        input_shape=(2, n_points),
        chan_list=[8],
        fc_list=[16],
        vf_hidden_list=[16],
        sim_config="batfit/default_exps/spm_discharge.yaml",
        scaler_X=scaler_X,
    )
    (x_scaled,) = model.scale_signal(X)
    assert np.allclose(x_scaled, scaler_X.transform(X))

    # time_dependent_zscore: scaled voltage and scaled end time
    scaler_V = ZScoreScaler.fit(X[:, 1:, :], axis=0, min_std=1e-3)
    scaler_T = ZScoreScaler.fit(t_end, axis=0)
    model_td = ProbParamFM(
        input_shape=(2, n_points),
        chan_list=[8],
        fc_list=[16],
        vf_hidden_list=[16],
        sim_config="batfit/default_exps/spm_discharge.yaml",
        scaler_X=scaler_V,
        signal_scaling="time_dependent_zscore",
        scaler_T=scaler_T,
    )
    v_scaled, t_scaled = model_td.scale_signal(X)
    assert v_scaled.shape == (6, 1, n_points)
    assert np.allclose(v_scaled.mean(axis=0), 0.0, atol=1e-4)
    assert t_scaled.shape == (6, 1)
    assert np.allclose(t_scaled, scaler_T.transform(t_end))
    # works on torch tensors too
    v_t, t_t = model_td.scale_signal(torch.from_numpy(X))
    assert isinstance(v_t, torch.Tensor) and isinstance(t_t, torch.Tensor)


def test__append_end_time():
    h = torch.rand(4, 10)
    t_end = torch.rand(4, 1)
    model = ProbParamFM(
        input_shape=(2, 32),
        chan_list=[8],
        fc_list=[16],
        vf_hidden_list=[16],
        sim_config="batfit/default_exps/spm_discharge.yaml",
    )
    # zscore: embedding unchanged, no end time accepted
    assert model._append_end_time(h, None) is h
    with pytest.raises(AssertionError):
        model._append_end_time(h, t_end)

    model_td = ProbParamFM(
        input_shape=(2, 32),
        chan_list=[8],
        fc_list=[16],
        vf_hidden_list=[16],
        sim_config="batfit/default_exps/spm_discharge.yaml",
        signal_scaling="time_dependent_zscore",
    )
    out = model_td._append_end_time(h, t_end)
    assert out.shape == (4, 11)
    assert torch.equal(out[:, -1:], t_end)
    with pytest.raises(AssertionError):
        model_td._append_end_time(h, None)
