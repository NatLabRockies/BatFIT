from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import TensorDataset

from batfit import logger


def load_config(
    yaml_path: str,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Loads the YAML config and extracts the min/max boundaries and parameter names.
    """
    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)

    try:
        param_names_str = config["degradation parameter names"]
        param_names = [name.strip() for name in param_names_str.split(",")]

        min_dict = config["min degradation parameter"]
        max_dict = config["max degradation parameter"]

        # Ensure ordered extraction based on the parameter names list
        mins = [min_dict[name] for name in param_names]
        maxs = [max_dict[name] for name in param_names]

    except KeyError as e:
        logger.error(f"Missing key in YAML config: {e}")
        raise

    return (
        torch.tensor(mins, dtype=torch.float32),
        torch.tensor(maxs, dtype=torch.float32),
        param_names,
    )


def prepare_datasets(
    p1_samples: np.ndarray, p2_samples: np.ndarray, yaml_path: str
) -> Tuple[TensorDataset, TensorDataset, int]:
    """
    Balances the datasets, generates p3, and returns DataLoaders ready datasets.
    """
    mins, maxs, param_names = load_config(yaml_path)
    input_dim = len(param_names)

    # Validation
    if p1_samples.shape[1] != input_dim or p2_samples.shape[1] != input_dim:
        error_msg = f"Dimension mismatch. Expected {input_dim} dims, got p1:{p1_samples.shape[1]}, p2:{p2_samples.shape[1]}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Balance datasets by taking the minimum length
    n_samples = min(p1_samples.shape[0], p2_samples.shape[0])
    logger.info(f"Balancing classes: Using {n_samples} samples per class.")

    # Convert to tensors and slice to exact matching size
    t_p1 = torch.tensor(p1_samples[:n_samples], dtype=torch.float32)
    t_p2 = torch.tensor(p2_samples[:n_samples], dtype=torch.float32)

    assert p1_samples.shape[1] == p2_samples.shape[1]

    # Generate p3 samples from uniform distribution within the hypercube
    # Using uniform distribution: U(min, max) = min + (max - min) * U(0, 1)
    rand_uniform = torch.rand(n_samples, input_dim, dtype=torch.float32)
    t_p3 = mins + (maxs - mins) * rand_uniform

    # Create Labels (1 for valid simulator samples, 0 for p3 hypercube samples)
    labels_valid = torch.ones(n_samples, 1, dtype=torch.float32)
    labels_invalid = torch.zeros(n_samples, 1, dtype=torch.float32)

    # Combine p1 and p3 for Classifier 1
    x1 = torch.cat([t_p1, t_p3], dim=0)
    y1 = torch.cat([labels_valid, labels_invalid], dim=0)
    dataset_1 = TensorDataset(x1, y1)

    # Combine p2 and p3 for Classifier 2
    x2 = torch.cat([t_p2, t_p3], dim=0)
    y2 = torch.cat([labels_valid, labels_invalid], dim=0)
    dataset_2 = TensorDataset(x2, y2)

    return dataset_1, dataset_2, input_dim
