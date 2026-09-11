import torch

from batfit.model.param_utils.noise_utils import (
    apply_noise_unscaled,
    make_noise_levels,
    sample_var,
)


def test_make_noise_levels_shape():
    noise_levels, a_min, a_max = make_noise_levels(
        target_mode="phi",
        noise_levels=[0.0, 0.01, 0.02, 0.03],
        cyc_mode="discharge",
    )
    # phi -> inds [0, 1], so 2 channels
    assert noise_levels.shape == (1, 2, 1)
    assert a_min.shape == (1, 2, 1)
    assert a_max.shape == (1, 2, 1)

    noise_levels, a_min, a_max = make_noise_levels(
        target_mode="phionly",
        noise_levels=[0.0, 0.01, 0.02, 0.03],
        cyc_mode="chirp",
    )
    assert noise_levels.shape == (1, 1, 1)
    assert a_min.shape == (1, 1, 1)
    assert a_max.shape == (1, 1, 1)

    noise_levels, a_min, a_max = make_noise_levels(
        target_mode="phi",
        noise_levels=[0.0, 0.01, 0.02, 0.03],
        cyc_mode="discharge-chargecc",
    )
    # phi -> 2 channels per mode, concatenated -> 4 total
    assert noise_levels.shape == (1, 4, 1)


def test_apply_noise_unscaled():
    # check that output are clamped
    batch, channels, length = 4, 2, 10
    x = torch.zeros(batch, channels, length)
    noise_levels = torch.tensor([5, 5]).view(1, 2, 1)
    a_min = torch.tensor([-0.1, -0.1]).view(1, 2, 1)
    a_max = torch.tensor([0.1, 0.1]).view(1, 2, 1)
    out = apply_noise_unscaled(x, noise_levels, a_min, a_max)
    assert out.shape == (batch, channels, length)
    assert out.max().item() <= 0.1 + 1e-6
    assert out.min().item() >= -0.1 - 1e-6
