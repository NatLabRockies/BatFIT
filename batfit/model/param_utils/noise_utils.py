import numpy as np
import torch

from batfit.utils.text_utils import shuffle_substrings


def make_noise_levels(
    target_mode: str,
    noise_levels: list,
    cyc_mode: str,
    vmin: float = 3.0,
    vmax: float = 4.1,
):
    noise_levels_single = torch.tensor(noise_levels).view(1, 4, 1)
    noise_levels_dis = torch.tensor(noise_levels).view(1, 4, 1)
    noise_levels_chcc = torch.tensor(noise_levels).view(1, 4, 1)
    a_min_single = torch.tensor(
        [-torch.inf, vmin, -torch.inf, -torch.inf]
    ).view(1, 4, 1)
    a_max_single = torch.tensor([torch.inf, vmax, torch.inf, torch.inf]).view(
        1, 4, 1
    )
    a_max_dis = torch.tensor([torch.inf, torch.inf, 0, 0]).view(1, 4, 1)
    a_min_dis = torch.tensor([-torch.inf, vmin, -torch.inf, -torch.inf]).view(
        1, 4, 1
    )
    a_max_dis = torch.tensor([torch.inf, torch.inf, 0, 0]).view(1, 4, 1)
    a_min_chcc = torch.tensor([-torch.inf, vmin, 0, 0]).view(1, 4, 1)
    a_max_chcc = torch.tensor([torch.inf, vmax, torch.inf, torch.inf]).view(
        1, 4, 1
    )

    if target_mode.lower() == "phionly":
        inds = [0]
    elif target_mode.lower() == "phi":
        inds = [0, 1]
    elif target_mode.lower() == "dvdq":
        inds = [0, 2]
    elif target_mode.lower() == "dqdv":
        inds = [0, 3]
    elif target_mode.lower() in shuffle_substrings("phi-dvdq"):
        inds = [0, 1, 2]
    elif target_mode.lower() in shuffle_substrings("phi-dqdv"):
        inds = [0, 1, 3]
    elif target_mode.lower() in shuffle_substrings("dvdq-dqdv"):
        inds = [0, 2, 3]
    elif target_mode.lower() in shuffle_substrings("phi-dvdq-dqdv"):
        inds = [0, 1, 2, 3]
    else:
        raise NotImplementedError

    if cyc_mode.lower() == "discharge":
        noise_levels = noise_levels_dis[:, inds, :]
        a_min = a_min_dis[:, inds, :]
        a_max = a_max_dis[:, inds, :]

    if cyc_mode.lower() == "chargecc":
        noise_levels = noise_levels_chcc[:, inds, :]
        a_min = a_min_chcc[:, inds, :]
        a_max = a_max_chcc[:, inds, :]

    if cyc_mode.lower() == "discharge-chargecc":
        noise_levels = torch.cat(
            (noise_levels_dis[:, inds, :], noise_levels_chcc[:, inds, :]),
            dim=1,
        )
        a_min = torch.cat(
            (a_min_dis[:, inds, :], a_min_chcc[:, inds, :]), dim=1
        )
        a_max = torch.cat(
            (a_max_dis[:, inds, :], a_max_chcc[:, inds, :]), dim=1
        )

    if cyc_mode.lower() in [
        "rh",
        "lh",
        "diffcap",
        "hppc",
        "posthppc",
        "chirp",
    ]:
        noise_levels = noise_levels_single[:, inds, :]
        a_min = a_min_single[:, inds, :]
        a_max = a_max_single[:, inds, :]

    return noise_levels, a_min, a_max


def make_bias_tensor(
    target_mode: str,
    cyc_mode: str,
    bias: np.ndarray | None,
):
    if bias is None:
        return None
    if not isinstance(bias, np.ndarray):
        raise NotImplementedError
    if len(bias.shape) > 1:
        raise NotImplementedError
    bias = bias[np.newaxis, np.newaxis, :]
    bias = np.repeat(bias, 4, axis=1)
    bias = bias.astype("float32")
    bias[:, 0, :] = 0
    bias[:, 2, :] = 0
    bias[:, 3, :] = 0

    if target_mode.lower() == "phionly":
        inds = [1]
    elif target_mode.lower() == "phi":
        inds = [0, 1]
    elif target_mode.lower() == "dvdq":
        inds = [0, 2]
    elif target_mode.lower() == "dqdv":
        inds = [0, 3]
    elif target_mode.lower() in shuffle_substrings("phi-dvdq"):
        inds = [0, 1, 2]
    elif target_mode.lower() in shuffle_substrings("phi-dqdv"):
        inds = [0, 1, 3]
    elif target_mode.lower() in shuffle_substrings("dvdq-dqdv"):
        inds = [0, 2, 3]
    elif target_mode.lower() in shuffle_substrings("phi-dvdq-dqdv"):
        inds = [0, 1, 2, 3]
    else:
        raise NotImplementedError

    if cyc_mode.lower() == "discharge":
        bias = bias[:, inds, :]

    if cyc_mode.lower() == "chargecc":
        bias = bias[:, inds, :]

    if cyc_mode.lower() == "discharge-chargecc":
        raise NotImplementedError

    if cyc_mode.lower() in ["rh", "lh"]:
        bias = bias[:, inds, :]

    return torch.tensor(bias)


def sample_var(tens_m, tens_std, min_val, max_val, n=100):
    samp = torch.Tensor.repeat(tens_m, (n, 1)) + torch.Tensor.repeat(
        tens_std, (n, 1)
    ) * torch.randn(n, tens_m.shape[0])
    samp = torch.clamp(samp, min=min_val, max=max_val)
    return samp


def cluster_rand(input_tensor):
    output_tensor = torch.empty_like(input_tensor)
    # Region 1: [0, 0.5)
    mask1 = input_tensor < 0.5
    output_tensor[mask1] = (
        0.4 * input_tensor[mask1]
    )  # linear map from [0,0.5] -> [0,0.2]
    # Region 2: [0.5, 1]
    mask2 = ~mask1
    output_tensor[mask2] = (
        0.4 * (input_tensor[mask2] - 0.5) + 0.8
    )  # linear map from [0.5,1] -> [0.8,1]
    return output_tensor


def apply_noise(
    batch_in: torch.Tensor,
    scaler_X,
    noise_levels: torch.Tensor,
    a_min: torch.Tensor,
    a_max: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    batch_in = scaler_X.inverse_transform(batch_in)
    noise = (torch.rand(batch_in.shape) - 0.5) * torch.reshape(
        noise_levels, (1, -1, 1)
    )
    batch_in += noise
    if bias is not None:
        # batch_in += bias.repeat(batch_in.shape[0], 1, 1) * cluster_rand(torch.rand((batch_in.shape[0], 1, 1)))
        batch_in += bias.repeat(batch_in.shape[0], 1, 1) * torch.rand(
            (batch_in.shape[0], 1, 1)
        )
    batch_in = torch.clamp(batch_in, min=a_min, max=a_max)
    batch_in = scaler_X.transform(batch_in)
    return batch_in


def apply_noise_unscaled(
    batch_in: torch.Tensor,
    noise_levels: torch.Tensor,
    a_min: torch.Tensor,
    a_max: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    noise = (torch.rand(batch_in.shape) - 0.5) * torch.reshape(
        noise_levels, (1, -1, 1)
    )
    batch_in += noise
    if bias is not None:
        # batch_in += bias.repeat(batch_in.shape[0], 1, 1) * cluster_rand(torch.rand((batch_in.shape[0], 1, 1)))
        batch_in += bias.repeat(batch_in.shape[0], 1, 1) * torch.rand(
            (batch_in.shape[0], 1, 1)
        )
    batch_in = torch.clamp(batch_in, min=a_min, max=a_max)
    return batch_in
