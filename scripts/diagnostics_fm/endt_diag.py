import os
import sys

import numpy as np
import yaml
from prettyPlot.plotting import *
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from batfit import logger
from batfit.basicutilityc import ReadInput as ri
from batfit.model.paramNN import *


def get_param_name(dim):
    filename = f"/projects/mlbatt/LHX_2/BatFIT/batfit/default_exps/p2d_diffcap_{dim}dim_ext.yaml"
    with open(filename, "r") as f:
        data = yaml.safe_load(f)
    params = data.get("degradation parameter names", [])
    if isinstance(params, str):
        return [p.strip() for p in params.split(",")]
    elif isinstance(params, list):
        return params
    else:
        return []


def get_sensitivities(X, Y, param_names):
    scaler = StandardScaler()
    param_scaled = scaler.fit_transform(X)
    model = LinearRegression()
    model.fit(X, Y)
    sensitivities = model.coef_

    for iname, name in enumerate(param_names):
        logger.info(f"\t{name} : {sensitivities[iname]}")

    return sensitivities


def apply_filter(X, Y, filter_vals=None):
    if filter_vals is None:
        return X, Y
    else:
        assert X.shape[0] == Y.shape[0]
        assert len(filter_vals) == 2
        y_min, y_max = filter_vals
        mask = (Y >= y_min) & (Y <= y_max)
    return X[mask], Y[mask]


def scan_1d_unsafe_supports(
    X, Y, param_names, target_val, min_prob=0.99, num_bins=50, min_samples=20
):
    """
    Scans each parameter independently to find contiguous ranges (support)
    where the probability of Y < target_val (failure/unsafe state) is >= min_prob.

    Parameters:
    - min_prob: The required probability threshold (e.g., 0.99 for 99% chance of being unsafe).
    - num_bins: The resolution of the scan. Higher = finer boundaries.
    - min_samples: Ignores regions with too few samples to be statistically reliable.
    """
    # 1. Create a binary target mask: 1 if Y is in the unsafe zone, 0 otherwise
    Y_unsafe = (Y < target_val).astype(int)

    d = X.shape[1]
    found_any = False

    for i in range(d):
        xi = X[:, i]
        name = param_names[i]

        # 2. Divide the support of the variable into bins
        bin_edges = np.linspace(xi.min(), xi.max(), num_bins + 1)
        bin_indices = np.clip(np.digitize(xi, bin_edges) - 1, 0, num_bins - 1)

        unsafe_bins = []

        # 3. Check the probability of Y < target_val in each bin
        for b in range(num_bins):
            in_bin = bin_indices == b
            count = np.sum(in_bin)

            if count >= min_samples:
                prob = np.mean(Y_unsafe[in_bin])
                if prob >= min_prob:
                    unsafe_bins.append(
                        {
                            "low": bin_edges[b],
                            "high": bin_edges[b + 1],
                            "prob": prob,
                            "count": count,
                        }
                    )

        # 4. Merge contiguous unsafe bins for cleaner reporting
        if unsafe_bins:
            found_any = True
            unsafe_regions = []
            current_region = unsafe_bins[0].copy()

            for b in unsafe_bins[1:]:
                # If this bin touches the previous one, merge them
                if np.isclose(current_region["high"], b["low"]):
                    current_region["high"] = b["high"]
                    total_count = current_region["count"] + b["count"]
                    current_region["prob"] = (
                        (current_region["prob"] * current_region["count"])
                        + (b["prob"] * b["count"])
                    ) / total_count
                    current_region["count"] = total_count
                else:
                    unsafe_regions.append(current_region)
                    current_region = b.copy()

            unsafe_regions.append(current_region)

            # 5. Log the findings
            logger.info(f"\n--- {name} ---")
            for region in unsafe_regions:
                logger.info(
                    f"  UNSAFE Support : [{region['low']:.4e}, {region['high']:.4e}]"
                )
                logger.info(
                    f"  Failure Prob   : {region['prob']*100:.2f}% (based on {region['count']} samples)"
                )

    if not found_any:
        logger.info(
            f"No isolated 1D regions found where failure probability is >= {min_prob*100}%."
        )


def endt_diag():

    # for dim in [10, 15, 19, 23]:
    for dim in [12, 31]:
        logger.info(f"Dim = {dim}")
        param_names = get_param_name(dim)
        A = np.load(
            f"/scratch/mhassana/LHX_2/data_p2d_diffcap_{dim}dim_ext_4M/data_split.npz"
        )
        endt = A["X_train"][:, 0, -1]
        param = A["Y_train"]
        # X, Y = apply_filter(param, endt, filter_vals=[100000,1e10])
        # get_sensitivities(X, Y, param_names)
        X, Y = apply_filter(param, endt)
        scan_1d_unsafe_supports(
            X,
            Y,
            param_names,
            target_val=120000,
            min_prob=0.99,
            num_bins=50,
            min_samples=100,
        )


if __name__ == "__main__":
    endt_diag()
