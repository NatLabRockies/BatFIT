import torch

from batfit.model.surrogateNN import SurrogateFCNN


def test_surrogate_forward():
    n_param_pred = 4
    n_points = 10
    batch = 8
    model = SurrogateFCNN(fc_list=[32, 32], n_param_pred=n_param_pred)
    x = torch.rand(batch, n_param_pred + 1)
    out = model(x)
    assert out.shape == (batch, 1)


def test_surrogate_transform_output():
    n_param_pred = 3
    model = SurrogateFCNN(fc_list=[16], n_param_pred=n_param_pred)
    x = torch.tensor([3.5, 4.0, 2.8])
    min_v, amp_v = 2.5, 2.0
    scaled = model.transform_output(x, min_v, amp_v)
    recovered = model.inv_transform_output(scaled, min_v, amp_v)
    assert torch.allclose(x, recovered, atol=1e-5)
