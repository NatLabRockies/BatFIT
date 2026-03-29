import os
import sys
from pathlib import Path

import numpy as np
import yaml

from batfit import logger
from batfit.basicutilityc import ReadInput as ri


def is_valid_model(model_folder):
    assert os.path.isabs(model_folder)
    files = os.listdir(model_folder)
    exist_recipe = False
    exist_post = False
    for filename in files:
        if filename.startswith("recipe"):
            exist_recipe = True
        if filename.startswith("post.txt"):
            exist_post = True

    if exist_recipe and exist_post:
        return True
    else:
        return False


def get_model_id(model_folder):
    if model_folder.startswith("models"):
        model_id = int(model_folder[6:])
    elif os.path.isabs(model_folder):
        folder_leaf = Path(model_folder).name
        model_id = int(folder_leaf[6:])
    else:
        logger.error(f"Cannot extract id from {model_folder}")
        sys.exit()
    return model_id


def read_perf(model_folder):
    assert os.path.isabs(model_folder)
    rmse_file = Path(os.path.join(model_folder, "post.txt"))
    if not rmse_file.is_file():
        logger.error(f"The file {rmse_file} does not exist")
        sys.exit()
    else:
        with open(rmse_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("PERF"):
                    perf = float(line.split()[1][:])
                if line.startswith("COV"):
                    cov = float(line.split()[1][:])
    return perf, cov


def read_recipe(model_folder):
    assert os.path.isabs(model_folder)
    files = os.listdir(model_folder)
    for filename in files:
        if filename.startswith("recipe"):
            break
    recipe_filename = os.path.join(model_folder, filename)
    return ri.basic_input(recipe_filename)


def find_best(root_folder):
    folder = root_folder
    best_perf = np.inf
    best_cov = np.inf
    best_perf_cov = np.inf
    worst_perf = 0
    worst_cov = 0
    worst_perf_cov = 0

    # Get list of model folder
    model_folders = []
    files = os.listdir(root_folder)
    for filename in files:
        if os.path.isdir(
            os.path.join(root_folder, filename)
        ) and filename.startswith("models"):
            if is_valid_model(os.path.join(root_folder, filename)):
                model_folders += [os.path.join(root_folder, filename)]

    logger.info(f"Found {len(model_folders)} valid model folder")

    for model_folder in model_folders:
        model_id = get_model_id(model_folder)
        recipe = read_recipe(model_folder)
        perf, cov = read_perf(model_folder)
        perf_cov = perf + cov
        if perf < best_perf:
            best_perf = perf
            best_perf_recipe = recipe
            best_perf_id = model_id
        if perf > worst_perf:
            worst_perf = perf
            worst_perf_recipe = recipe
            worst_perf_id = model_id
        if cov < best_cov:
            best_cov = cov
            best_cov_recipe = recipe
            best_cov_id = model_id
        if cov > worst_cov:
            worst_cov = cov
            worst_cov_recipe = recipe
            worst_cov_id = model_id
        if perf_cov < best_perf_cov:
            best_perf_cov = perf_cov
            best_perf_cov_recipe = recipe
            best_perf_cov_id = model_id
        if perf_cov > worst_perf_cov:
            worst_perf_cov = perf_cov
            worst_perf_cov_recipe = recipe
            worst_perf_cov_id = model_id

    results = {}
    results["best_perf"] = best_perf
    results["best_perf_recipe"] = best_perf_recipe
    results["best_perf_id"] = best_perf_id
    results["best_cov"] = best_cov
    results["best_cov_recipe"] = best_cov_recipe
    results["best_cov_id"] = best_cov_id
    results["best_perf_cov"] = best_perf_cov
    results["best_perf_cov_recipe"] = best_perf_cov_recipe
    results["best_perf_cov_id"] = best_perf_cov_id
    results["worst_perf"] = worst_perf
    results["worst_perf_recipe"] = worst_perf_recipe
    results["worst_perf_id"] = worst_perf_id
    results["worst_cov"] = worst_cov
    results["worst_cov_recipe"] = worst_cov_recipe
    results["worst_cov_id"] = worst_cov_id
    results["worst_perf_cov"] = worst_perf_cov
    results["worst_perf_cov_recipe"] = worst_perf_cov_recipe
    results["worst_perf_cov_id"] = worst_perf_cov_id
    return results


root_folder = "/scratch/mhassana/LHX/tune_diffcap_1M"
results = find_best(root_folder)
print(f"\tPERF {results['best_perf']:.2g} model {results['best_perf_id']}")
print(f"\tCOV {results['best_cov']:.2g} model {results['best_cov_id']}")
print(
    f"\tPERF+COV {results['best_perf_cov']:.2g} model {results['best_perf_cov_id']}"
)

print(f"\tPERF {results['worst_perf']:.2g} model {results['worst_perf_id']}")
print(f"\tCOV {results['worst_cov']:.2g} model {results['worst_cov_id']}")
print(
    f"\tPERF+COV {results['worst_perf_cov']:.2g} model {results['worst_perf_cov_id']}"
)
