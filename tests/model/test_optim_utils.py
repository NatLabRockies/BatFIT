import numpy as np
import pytest
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from batfit.model.param_utils.losses import independent_normal_loss
from batfit.model.param_utils.noise_utils import make_noise_levels
from batfit.model.param_utils.optim_utils import (
    evaluate_sigma,
    optimize_protocol,
    predict_mu_sigma,
    sigma_physical,
)
from batfit.model.paramNN import ProbParamCNN, ProbParamFM, ProbProtParamCNN
from batfit.model.varianceNN import VariancePredFCNN
from batfit.utils.scalers import CustomScaler


def _tiny_var_model():
    return VariancePredFCNN(
        n_prot=3,
        n_deg=6,
        hidden_list=[8],
        sim_config="batfit/default_exps/spm_chirp.yaml",
    )


def test_predict_mu_sigma():
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cpu")
    n_curves = 5
    n_points = 32
    n_noise = 2

    X = np.random.rand(n_curves, 2, n_points).astype("float32") + 3.0
    scaler_x = CustomScaler.fit(X, axis=(0, 2))
    X_scaled = scaler_x.transform(X).astype("float32")
    noise_levels, a_min, a_max = make_noise_levels(
        target_mode="phi",
        noise_levels=[0, 0.001, 0.001, 2.0],
        cyc_mode="chargecc",
    )
    shared = dict(
        scaler_x=scaler_x,
        noise_levels=noise_levels,
        a_min=a_min,
        a_max=a_max,
        n_noise=n_noise,
        device=device,
    )

    # --- CNN NPE without protocol conditioning (constrain_output path) ---
    n_deg = 6  # must match the sim_config YAML
    cnn = ProbParamCNN(
        input_shape=(2, n_points),
        chan_list=[4],
        fc_list=[8],
        fc_mu_list=[8],
        fc_gamma_list=[8],
        loss_fn=independent_normal_loss,
        cyc_mode="chargecc",
        n_param_pred=n_deg,
        constrain_output=True,
        sim_config="batfit/default_exps/spm_nochirp.yaml",
    )
    cnn.eval()
    mu, sigma = predict_mu_sigma(X_scaled, cnn, batch_size=2, **shared)
    assert mu.shape == (n_curves, n_deg)
    assert sigma.shape == (n_curves, n_deg)
    assert mu.dtype == np.float32
    assert np.all(sigma > 0)
    # constrain_output unscales mu into the physical prior range
    min_par = cnn.min_par.numpy()
    max_par = (cnn.min_par + cnn.amp_par).numpy()
    assert np.all(mu >= min_par - 1e-5)
    assert np.all(mu <= max_par + 1e-5)

    # --- CNN NPE with protocol conditioning ---
    n_prot = 3
    P_scaled = np.random.rand(n_curves, n_prot).astype("float32")
    prot_cnn = ProbProtParamCNN(
        input_shape=(2, n_points),
        chan_list=[4],
        fc_list=[8],
        fc_prot_list=[8],
        fc_mu_list=[8],
        fc_gamma_list=[8],
        loss_fn=independent_normal_loss,
        n_prot_params=n_prot,
        cyc_mode="chirp",
        n_param_pred=3,
        constrain_output=False,
    )
    prot_cnn.eval()
    mu_p, sigma_p = predict_mu_sigma(
        X_scaled, prot_cnn, P_scaled=P_scaled, **shared
    )
    assert mu_p.shape == (n_curves, 3)
    assert sigma_p.shape == (n_curves, 3)

    # --- FM NPE (posterior samples -> mean/std in physical space) ---
    fm = ProbParamFM(
        input_shape=(2, n_points),
        chan_list=[4],
        fc_list=[8],
        vf_hidden_list=[8],
        cyc_mode="chargecc",
        n_param_pred=3,
    )
    fm.eval()
    scaler_y = MinMaxScaler()
    scaler_y.fit(np.random.rand(20, 3).astype("float32"))
    mu_fm, sigma_fm = predict_mu_sigma(
        X_scaled,
        fm,
        scaler_Y=scaler_y,
        n_samples=5,
        n_ode_steps=5,
        **shared,
    )
    assert mu_fm.shape == (n_curves, 3)
    assert sigma_fm.shape == (n_curves, 3)
    assert np.all(np.isfinite(mu_fm))
    assert np.all(sigma_fm >= 0)

    # FM NPE without scaler_Y must raise
    with pytest.raises(AssertionError):
        predict_mu_sigma(X_scaled, fm, **shared)


def test_sigma_physical():
    torch.manual_seed(0)
    device = torch.device("cpu")
    batch = 4
    var_model = _tiny_var_model()
    sigma_out = torch.rand(batch, 6)

    # scaler_sigma=None -> amp_par unscaling via inv_transform_gamma
    out = sigma_physical(sigma_out, var_model, None, device)
    ref = var_model.inv_transform_gamma(sigma_out, var_model.amp_par)
    assert torch.allclose(out, ref, atol=1e-6)

    # scaler_sigma provided -> differentiable inverse MinMax transform
    scaler_sigma = MinMaxScaler()
    scaler_sigma.fit(np.random.rand(20, 6).astype("float32") * 0.1)
    sigma_in = torch.rand(batch, 6, requires_grad=True)
    out_sc = sigma_physical(sigma_in, var_model, scaler_sigma, device)
    ref_sc = scaler_sigma.inverse_transform(sigma_in.detach().numpy())
    assert np.allclose(out_sc.detach().numpy(), ref_sc, atol=1e-5)
    # gradients must flow through the inverse transform
    out_sc.sum().backward()
    assert sigma_in.grad is not None

    # StandardScaler (log_sigma mode) -> differentiable exp(z * scale + mean)
    scaler_logsigma = StandardScaler()
    scaler_logsigma.fit(
        np.log(np.random.rand(20, 6).astype("float32") * 0.1 + 1e-3)
    )
    z_in = torch.randn(batch, 6, requires_grad=True)
    out_log = sigma_physical(z_in, var_model, scaler_logsigma, device)
    ref_log = np.exp(
        scaler_logsigma.inverse_transform(z_in.detach().numpy())
    )
    assert np.allclose(out_log.detach().numpy(), ref_log, rtol=1e-5)
    # physical sigma is strictly positive by construction
    assert out_log.min().item() > 0.0
    # gradients must flow through the exp inverse transform
    out_log.sum().backward()
    assert z_in.grad is not None


def test_evaluate_sigma():
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cpu")
    var_model = _tiny_var_model()
    P_scaled = np.random.rand(3).astype("float32")
    mu_scaled = np.random.rand(6).astype("float32")

    sigma = evaluate_sigma(P_scaled, mu_scaled, var_model, None, device)
    assert sigma.shape == (6,)
    assert np.all(np.isfinite(sigma))
    assert np.all(sigma > 0)


def test_optimize_protocol():
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cpu")
    var_model = _tiny_var_model()
    mu_scaled = np.random.rand(1, 6).astype("float32")
    param_idx = 0

    bounds = [(0.0, 1.0)] * 3
    p_opt, sigma_opt = optimize_protocol(
        mu_scaled, var_model, param_idx, bounds, 2, None, device
    )
    assert p_opt.shape == (3,)
    assert np.all(p_opt >= 0.0) and np.all(p_opt <= 1.0)
    # reported optimum must match a direct evaluation at p_opt
    sigma_eval = evaluate_sigma(
        p_opt, mu_scaled.flatten(), var_model, None, device
    )
    assert np.isclose(sigma_eval[param_idx], sigma_opt, atol=1e-5)

    # clamped dimension (e.g. amplitude fixed to 0) must be respected
    bounds_clamped = [(0.0, 1.0), (0.0, 0.0), (0.0, 1.0)]
    p_clamped, sigma_clamped = optimize_protocol(
        mu_scaled, var_model, param_idx, bounds_clamped, 2, None, device
    )
    assert p_clamped[1] == 0.0
    # the constrained optimum cannot beat the unconstrained one
    assert sigma_clamped >= sigma_opt - 1e-6
