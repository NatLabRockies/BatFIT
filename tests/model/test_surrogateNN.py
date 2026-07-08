import pytest
import torch
import torch.nn as nn

from batfit.model.surrogateNN import SurrogateFCNN


def test_surrogate_forward():
    n_param_pred = 4
    n_points = 10
    batch = 8
    model = SurrogateFCNN(fc_list=[32, 32], n_param_pred=n_param_pred)
    x = torch.rand(batch, n_param_pred + 1)
    out = model(x)
    assert out.shape == (batch, 1)

    # cyc_mode must be stored as a plain string, not an accidental tuple
    assert model.cyc_mode == "discharge"

    # default (raw voltage targets): final activation is ReLU
    assert isinstance(model.fcnn_layers[-1], nn.ReLU)

    # scale_y=True (standardized targets can be negative): no final
    # activation, output head is the bare Linear layer
    model_scaled = SurrogateFCNN(
        fc_list=[32, 32], n_param_pred=n_param_pred, scale_y=True
    )
    assert isinstance(model_scaled.fcnn_layers[-1], nn.Linear)
    out_scaled = model_scaled(x)
    assert out_scaled.shape == (batch, 1)

    # constrain_output and scale_y are mutually exclusive
    with pytest.raises(ValueError):
        SurrogateFCNN(
            fc_list=[32, 32],
            n_param_pred=n_param_pred,
            constrain_output=True,
            scale_y=True,
        )


def test_surrogate_transform_output():
    n_param_pred = 3
    model = SurrogateFCNN(fc_list=[16], n_param_pred=n_param_pred)
    x = torch.tensor([3.5, 4.0, 2.8])
    min_v, amp_v = 2.5, 2.0
    scaled = model.transform_output(x, min_v, amp_v)
    recovered = model.inv_transform_output(scaled, min_v, amp_v)
    assert torch.allclose(x, recovered, atol=1e-5)
