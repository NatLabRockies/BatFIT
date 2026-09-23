import numpy as np
import torch

from batfit.model.param_utils.losses import independent_normal_loss
from batfit.model.param_utils.noise_utils import make_noise_levels
from batfit.model.param_utils.optim_utils import (
    evaluate_sigma,
    optimize_protocol,
    predict_mu_sigma,
)
from batfit.model.paramNN import ProbParamCNN, ProbParamFM, ProbProtParamCNN
from batfit.model.varianceNN import VariancePredFCNN
from batfit.utils.scalers import ZScoreScaler


def _tiny_var_model():
    return VariancePredFCNN(
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
    # every NPE below carries this signal scaler, used to apply the noise
    scaler_x = ZScoreScaler.fit(X, axis=(0, 2))
    X_scaled = scaler_x.transform(X)
    noise_levels, a_min, a_max = make_noise_levels(
        target_mode="phi",
        noise_levels=[0, 0.001, 0.001, 2.0],
        cyc_mode="chargecc",
        vmin=3.0,
        vmax=4.2,
    )
    shared = dict(
        noise_levels=noise_levels,
        a_min=a_min,
        a_max=a_max,
        n_noise=n_noise,
        device=device,
    )

    # --- CNN NPE without protocol conditioning ---
    n_deg = 6  # must match the sim_config YAML
    cnn = ProbParamCNN(
        input_shape=(2, n_points),
        chan_list=[4],
        fc_list=[8],
        fc_mu_list=[8],
        fc_gamma_list=[8],
        loss_fn=independent_normal_loss,
        sim_config="batfit/default_exps/spm_nochirp.yaml",
        cyc_mode="chargecc",
        scaler_X=scaler_x,
    )
    cnn.eval()
    mu, sigma = predict_mu_sigma(X_scaled, cnn, batch_size=2, **shared)
    assert mu.shape == (n_curves, n_deg)
    assert sigma.shape == (n_curves, n_deg)
    assert mu.dtype == np.float32
    assert np.all(sigma > 0)
    # to_physical maps mu into the physical prior range
    assert np.all(mu >= cnn.scaler_Y.low.numpy() - 1e-5)
    assert np.all(mu <= cnn.scaler_Y.high.numpy() + 1e-5)

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
        sim_config="batfit/default_exps/spm_chirp.yaml",
        cyc_mode="chirp",
        scaler_X=scaler_x,
    )
    prot_cnn.eval()
    mu_p, sigma_p = predict_mu_sigma(
        X_scaled, prot_cnn, P_scaled=P_scaled, **shared
    )
    assert mu_p.shape == (n_curves, n_deg)
    assert sigma_p.shape == (n_curves, n_deg)

    # --- FM NPE (posterior samples -> mean/std in physical space) ---
    fm = ProbParamFM(
        input_shape=(2, n_points),
        chan_list=[4],
        fc_list=[8],
        vf_hidden_list=[8],
        sim_config="batfit/default_exps/spm_nochirp.yaml",
        cyc_mode="chargecc",
        scaler_X=scaler_x,
    )
    fm.eval()
    mu_fm, sigma_fm = predict_mu_sigma(
        X_scaled,
        fm,
        n_samples=5,
        n_ode_steps=5,
        **shared,
    )
    assert mu_fm.shape == (n_curves, n_deg)
    assert sigma_fm.shape == (n_curves, n_deg)
    assert np.all(np.isfinite(mu_fm))
    assert np.all(sigma_fm >= 0)
    # posterior samples are clamped to the prior, so the mean is inside it
    assert np.all(mu_fm >= fm.scaler_Y.low.numpy() - 1e-5)
    assert np.all(mu_fm <= fm.scaler_Y.high.numpy() + 1e-5)


def test_evaluate_sigma():
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cpu")
    var_model = _tiny_var_model()
    P_scaled = np.random.rand(3).astype("float32")
    mu_scaled = np.random.rand(6).astype("float32")

    sigma = evaluate_sigma(P_scaled, mu_scaled, var_model, device)
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
        mu_scaled, var_model, param_idx, bounds, 2, device
    )
    assert p_opt.shape == (3,)
    assert np.all(p_opt >= 0.0) and np.all(p_opt <= 1.0)
    # reported optimum must match a direct evaluation at p_opt
    sigma_eval = evaluate_sigma(p_opt, mu_scaled.flatten(), var_model, device)
    assert np.isclose(sigma_eval[param_idx], sigma_opt, atol=1e-5)

    # clamped dimension (e.g. amplitude fixed to 0) must be respected
    bounds_clamped = [(0.0, 1.0), (0.0, 0.0), (0.0, 1.0)]
    p_clamped, sigma_clamped = optimize_protocol(
        mu_scaled, var_model, param_idx, bounds_clamped, 2, device
    )
    assert p_clamped[1] == 0.0
    # the constrained optimum cannot beat the unconstrained one
    assert sigma_clamped >= sigma_opt - 1e-6
