import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from classifier import JointBoundaryClassifier
from fuse_utils import load_config, prepare_datasets
from torch.utils.data import DataLoader

from batfit import BATFIT_EXP, logger
from batfit.utils.torch_utils import (
    load_model,
    log_training,
    prepare_log,
    save_model,
)

if __name__ == "__main__":
    # --- Example Usage ---
    # Dummy data generation representing your stable regions
    p1_samples = np.load("data/assembled_data_diffcap.npz")["Y_data"]
    p2_samples = np.load("data/assembled_data_hppc.npz")["Y_data"]
    config = os.path.join(BATFIT_EXP, "p2d_diffcap.yaml")
    mins, maxs, param_names = load_config(config)
    n_samples = min(p1_samples.shape[0], p2_samples.shape[0])
    input_dim = p1_samples.shape[1]
    rand_uniform = torch.rand(n_samples, input_dim, dtype=torch.float32)
    p3_samples = mins + (maxs - mins) * rand_uniform

    model = JointBoundaryClassifier(
        input_dim=p1_samples.shape[1], hidden_dim=256, num_layers=2
    )

    model = load_model(model, "train_log/model_final.pt")

    # Example of applying the filter to new samples drawn from q(theta)
    model_device = model.device
    # candidate_samples = torch.rand(100, p1_samples.shape[1]).to(model_device) # Represents samples from MVN q(theta)
    # candidate_samples = torch.Tensor(p2_samples).to(model_device)
    candidate_samples = p3_samples.to(model_device)
    model.eval()
    accepted_mask = model.filter_samples(candidate_samples, threshold=0.6)
    final_valid_samples = candidate_samples[accepted_mask.squeeze()].to("cpu")
    logger.info(
        f"Retained {len(final_valid_samples)} valid samples out of {candidate_samples.shape[0]}."
    )
