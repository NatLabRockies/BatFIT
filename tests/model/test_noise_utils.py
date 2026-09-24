import pytest
import torch

from batfit.model.param_utils.noise_utils import (
    apply_noise_unscaled,
    make_noise_levels,
    make_signal_noise_levels,
)
from batfit.model.paramNN import ProbParamFM


def test_make_noise_levels_shape():
    noise_levels, a_min, a_max = make_noise_levels(
        target_mode="phi",
        noise_levels=[0.0, 0.01, 0.02, 0.03],
        cyc_mode="discharge",
        vmin=2.5,
        vmax=4.2,
    )
    # phi -> inds [0, 1], so 2 channels
    assert noise_levels.shape == (1, 2, 1)
    assert a_min.shape == (1, 2, 1)
    assert a_max.shape == (1, 2, 1)
    # the voltage channel is clipped at the given vmin
    assert abs(a_min[0, 1, 0].item() - 2.5) < 1e-6

    noise_levels, a_min, a_max = make_noise_levels(
        target_mode="phionly",
        noise_levels=[0.0, 0.01, 0.02, 0.03],
        cyc_mode="chirp",
        vmin=2.5,
        vmax=4.2,
    )
    assert noise_levels.shape == (1, 1, 1)
    assert a_min.shape == (1, 1, 1)
    assert a_max.shape == (1, 1, 1)
    # the voltage channel is clipped at the given vmax
    assert abs(a_max[0, 0, 0].item() - 4.2) < 1e-6

    noise_levels, a_min, a_max = make_noise_levels(
        target_mode="phi",
        noise_levels=[0.0, 0.01, 0.02, 0.03],
        cyc_mode="discharge-chargecc",
        vmin=2.5,
        vmax=4.2,
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


def test_make_signal_noise_levels():
    noise = [0.0, 0.01, 0.02, 0.03]
    model = ProbParamFM(
        input_shape=(2, 32),
        chan_list=[8],
        fc_list=[16],
        vf_hidden_list=[16],
        sim_config="batfit/default_exps/spm_discharge.yaml",
    )
    # zscore: (time, voltage) channels, voltage clipped at the config vmin
    levels, a_min, a_max = make_signal_noise_levels(
        model, "phi", noise, "discharge"
    )
    assert levels.shape == (1, 2, 1)
    assert abs(a_min[0, 1, 0].item() - model.sim_params["vmin"]) < 1e-6

    model_td = ProbParamFM(
        input_shape=(2, 32),
        chan_list=[8],
        fc_list=[16],
        vf_hidden_list=[16],
        sim_config="batfit/default_exps/spm_discharge.yaml",
        signal_scaling="time_dependent_zscore",
    )
    # time_dependent_zscore: voltage channel only
    levels, a_min, a_max = make_signal_noise_levels(
        model_td, "phi", noise, "discharge"
    )
    assert levels.shape == (1, 1, 1)
    assert abs(levels[0, 0, 0].item() - 0.01) < 1e-6
    assert abs(a_min[0, 0, 0].item() - model_td.sim_params["vmin"]) < 1e-6
    # only the (time, voltage) signal can drop its time channel
    with pytest.raises(AssertionError):
        make_signal_noise_levels(model_td, "dvdq", noise, "discharge")
