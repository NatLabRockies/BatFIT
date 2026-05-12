import torch

from batfit.model.param_utils.metrics import accuracy, identifiability, rel_accuracy


def test_identifiability():
    # shape
    # (batch=8, n_params=4) -> shape (n_params,)
    std = torch.ones(8, 4)
    result = identifiability(std)
    assert result.shape == torch.Size([4])

    # known value
    # 1/std = 1/0.5 = 2.0 for all elements
    std = torch.full((8, 3), 0.5)
    result = identifiability(std)
    assert torch.allclose(result, torch.full((3,), 2.0))


def test_accuracy():
    # zero if same input
    x = torch.ones(8, 4)
    result = accuracy(x, x)
    assert torch.all(result == 0.0)
    
    # shape
    # (batch=8, n_params=4) -> shape (n_params,)
    output_mean = torch.zeros(8, 4)
    target = torch.ones(8, 4)
    result = accuracy(output_mean, target)
    assert result.shape == torch.Size([4])

    # known value
    # mean(|0 - 1|) = 1.0 per parameter
    output_mean = torch.zeros(8, 3)
    target = torch.ones(8, 3)
    result = accuracy(output_mean, target)
    assert torch.allclose(result, torch.ones(3))


def test_rel_accuracy():
    # zero if same input
    x = torch.ones(8, 4)
    result = rel_accuracy(x, x)
    assert torch.all(result == 0.0)

    # shape
    output_mean = torch.ones(8, 4)
    target = torch.full((8, 4), 2.0)
    result = rel_accuracy(output_mean, target)
    assert result.shape == torch.Size([4])

    # known values
    # |1 - 2| / max(1, 2) = 0.5 for all elements
    output_mean = torch.ones(8, 3)
    target = torch.full((8, 3), 2.0)
    result = rel_accuracy(output_mean, target)
    assert torch.allclose(result, torch.full((3,), 0.5))
