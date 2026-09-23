import numpy as np
import pytest
import torch

from batfit.model.surrogate_utils.losses import mae_loss
from batfit.model.surrogateNN import SurrogateFCNN
from batfit.utils.scalers import MarginSigmoid, ZScoreScaler


def test_SurrogateFCNN():
    batch = 8
    sim_config = "batfit/default_exps/spm_discharge.yaml"
    model = SurrogateFCNN(fc_list=[32, 32], sim_config=sim_config)
    # parameter count read from the config: 6 deg params + time
    assert model.n_param_pred == 6
    x = torch.rand(batch, model.n_param_pred + 1)
    out = model(x)
    assert out.shape == (batch, 1)

    # cyc_mode must be stored as a plain string, not an accidental tuple
    assert model.cyc_mode == "discharge"

    # bounded output head: 0.5 V margin relative to the [vmin, vmax] window
    head = model.fcnn_layers[-1]
    assert isinstance(head, MarginSigmoid)
    v_range = model.sim_params["vmax"] - model.sim_params["vmin"]
    assert np.isclose(float(head.margin), 0.5 / v_range)

    # scalers are submodules, saved in the state dict
    keys = set(model.state_dict())
    assert {"scaler_t.means", "scaler_Y.low", "scaler_V.low"} <= keys

    # only MAE/MSE losses are supported
    with pytest.raises(AssertionError):
        SurrogateFCNN(fc_list=[8], sim_config=sim_config, loss_fn=torch.abs)


def test_scale_input():
    time = np.linspace(0.0, 3600.0, 20).reshape(-1, 1).astype("float32")
    scaler_t = ZScoreScaler.fit(time, axis=0)
    model = SurrogateFCNN(
        fc_list=[8],
        sim_config="batfit/default_exps/spm_discharge.yaml",
        loss_fn=mae_loss,
        scaler_t=scaler_t,
    )
    low, high = model.scaler_Y.low, model.scaler_Y.high
    deg_params = torch.stack([low, high])
    x = model.scale_input(torch.from_numpy(time[:2]), deg_params)
    assert x.shape == (2, 7)
    # column 0 is the z-scored time, then the [0, 1]-scaled parameters
    assert torch.allclose(
        x[:, :1], scaler_t.transform(torch.from_numpy(time[:2]))
    )
    assert torch.allclose(x[0, 1:], torch.zeros(6))
    assert torch.allclose(x[1, 1:], torch.ones(6))


def test_to_physical_surrogate():
    model = SurrogateFCNN(
        fc_list=[8], sim_config="batfit/default_exps/spm_discharge.yaml"
    )
    vmin = model.sim_params["vmin"]
    vmax = model.sim_params["vmax"]
    voltage = model.to_physical(torch.tensor([[0.0], [1.0], [1.1]]))
    # 0 -> vmin, 1 -> vmax; values past the bounds are NOT clamped
    assert torch.allclose(voltage[:2, 0], torch.tensor([vmin, vmax]))
    assert voltage[2, 0] > vmax


def test_predict_physical_surrogate():
    torch.manual_seed(0)
    model = SurrogateFCNN(
        fc_list=[8], sim_config="batfit/default_exps/spm_discharge.yaml"
    )
    model.eval()
    time = torch.linspace(0.0, 3600.0, 5).reshape(-1, 1)
    deg_params = model.scaler_Y.inverse_transform(torch.rand(5, 6))
    with torch.no_grad():
        voltage = model.predict_physical(time, deg_params)
        # same result as scaling by hand then calling to_physical
        voltage_scaled = model(model.scale_input(time, deg_params))
        voltage_ref = model.to_physical(voltage_scaled)
    assert voltage.shape == (5, 1)
    assert torch.allclose(voltage, voltage_ref)
    # within the widened voltage window of the output head
    vmin = model.sim_params["vmin"]
    vmax = model.sim_params["vmax"]
    assert torch.all(voltage >= vmin - 0.5)
    assert torch.all(voltage <= vmax + 0.5)
