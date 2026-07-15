import torch

from batfit.model.varianceNN import VariancePredFCNN


def test_VariancePredFCNN():
    batch = 8
    n_prot = 3
    n_deg = 6
    hidden_list = [32, 16]
    sim_config = "batfit/default_exps/spm_chirp.yaml"

    model = VariancePredFCNN(
        n_prot=n_prot,
        n_deg=n_deg,
        hidden_list=hidden_list,
        sim_config=sim_config,
    )

    prot_params = torch.rand(batch, n_prot)
    mu = torch.rand(batch, n_deg)

    sigma_sigmoid = model(prot_params, mu)

    # Output shape
    assert sigma_sigmoid.shape == (batch, n_deg)
    # Sigmoid output strictly in (0, 1)
    assert sigma_sigmoid.min().item() > 0.0
    assert sigma_sigmoid.max().item() < 1.0

    # inv_transform_gamma recovers physical sigma (positive)
    sigma_physical = model.inv_transform_gamma(sigma_sigmoid, model.amp_par)
    assert sigma_physical.shape == (batch, n_deg)
    assert sigma_physical.min().item() > 0.0

    # transform_gamma is the inverse of inv_transform_gamma
    sigma_roundtrip = model.inv_transform_gamma(
        model.transform_gamma(sigma_physical, model.amp_par), model.amp_par
    )
    assert torch.allclose(sigma_physical, sigma_roundtrip, atol=1e-5)

    # Gradients flow through the full forward + rescale path
    loss = sigma_physical.sum()
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None

    # output_activation="linear": no final Sigmoid, unbounded output
    model_lin = VariancePredFCNN(
        n_prot=n_prot,
        n_deg=n_deg,
        hidden_list=hidden_list,
        sim_config=sim_config,
        output_activation="linear",
    )
    assert not isinstance(model_lin.layers[-1], torch.nn.Sigmoid)
    out_lin = model_lin(prot_params, mu)
    assert out_lin.shape == (batch, n_deg)
    out_lin.sum().backward()
    for p in model_lin.parameters():
        assert p.grad is not None
