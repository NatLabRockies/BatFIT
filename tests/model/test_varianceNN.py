import numpy as np
import torch

from batfit.model.varianceNN import VariancePredFCNN, VariancePredNoProtFCNN
from batfit.utils.scalers import ZScoreScaler


def test_VariancePredFCNN():
    batch = 8
    n_prot = 3
    n_deg = 6
    sim_config = "batfit/default_exps/spm_chirp.yaml"

    # log-sigma z-score fitted on training sigmas
    sigma_train = np.random.rand(40, n_deg).astype("float32") * 0.1 + 1e-3
    scaler_logsigma = ZScoreScaler.fit(np.log(sigma_train), axis=0)
    model = VariancePredFCNN(
        hidden_list=[32, 16],
        sim_config=sim_config,
        scaler_logsigma=scaler_logsigma,
    )
    # numbers of parameters read from the config
    assert model.n_prot == n_prot
    assert model.n_deg == n_deg
    # scalers are submodules, saved in the state dict
    keys = set(model.state_dict())
    assert {"scaler_P.low", "scaler_Y.low", "scaler_logsigma.means"} <= keys

    prot_params = torch.rand(batch, n_prot)
    mu = torch.rand(batch, n_deg)
    sigma_scaled = model(prot_params, mu)
    # linear head: unbounded z-scored log sigma
    assert sigma_scaled.shape == (batch, n_deg)
    assert isinstance(model.layers[-1], torch.nn.Linear)

    # predict_physical: physical inputs -> physical sigma
    p_phys = model.scaler_P.inverse_transform(prot_params)
    mu_phys = model.scaler_Y.inverse_transform(mu)
    sigma_phys = model.predict_physical(p_phys, mu_phys)
    assert torch.allclose(sigma_phys, model.to_physical(sigma_scaled))
    assert sigma_phys.min().item() > 0.0

    # gradients flow from physical sigma back to the protocol parameters
    p_req = prot_params.clone().requires_grad_(True)
    model.to_physical(model(p_req, mu)).sum().backward()
    assert p_req.grad is not None


def test_VariancePredNoProtFCNN():
    batch = 5
    n_deg = 6
    model = VariancePredNoProtFCNN(
        hidden_list=[16],
        sim_config="batfit/default_exps/spm_discharge.yaml",
    )
    assert model.n_prot == 0
    assert model.scaler_P is None
    mu = torch.rand(batch, n_deg)
    sigma_scaled = model(mu)
    assert sigma_scaled.shape == (batch, n_deg)
    mu_phys = model.scaler_Y.inverse_transform(mu)
    sigma_phys = model.predict_physical(mu_phys)
    assert torch.allclose(sigma_phys, model.to_physical(sigma_scaled))


def test_to_physical_variance():
    n_deg = 6
    log_sigma = np.log(np.random.rand(30, n_deg) * 0.1 + 1e-3)
    scaler_logsigma = ZScoreScaler.fit(log_sigma.astype("float32"), axis=0)
    model = VariancePredFCNN(
        hidden_list=[8],
        sim_config="batfit/default_exps/spm_chirp.yaml",
        scaler_logsigma=scaler_logsigma,
    )
    # exact inverse of z-scoring log sigma
    sigma = torch.tensor(np.exp(log_sigma[:4]), dtype=torch.float32)
    z = scaler_logsigma.transform(torch.log(sigma))
    assert torch.allclose(model.to_physical(z), sigma, rtol=1e-5)
    # strictly positive by construction
    assert model.to_physical(torch.randn(3, n_deg)).min().item() > 0.0
