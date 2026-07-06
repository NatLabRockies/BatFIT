"""Standalone helpers for working with individual raw ``solution*.npz`` files.

These are not used by the current ``sols.pkl``-based assembly pipeline
(:mod:`batfit.utils.assembly`) but are kept as general-purpose utilities for
inspecting or parsing raw per-simulation solution files directly.
"""

import os

import numpy as np


def get_sol_list(data_root_folder: str) -> list[str]:
    """Return the filenames in ``data_root_folder`` matching ``solution*.npz``."""
    list_files = os.listdir(data_root_folder)
    ind_remove = []
    for ifile, file in enumerate(list_files):
        if not file.startswith("solution") or not file.endswith(".npz"):
            ind_remove.append(ifile)

    for indr in reversed(ind_remove):
        list_files.pop(indr)

    return list_files


def from_name_to_params(filename: str) -> list[float]:
    """Parse the degradation parameter values encoded in a solution filename.

    Expects filenames of the form ``solution_<p1>_<p2>_..._<pn>.npz``.
    """
    if filename.startswith("solution") and filename.endswith(".npz"):
        filename_par = filename[:-4].split("_")
    params = []
    filename_par.pop(0)
    for parstr in filename_par:
        params.append(float(parstr))
    return params


def get_max_time(data_root_folder: str) -> float:
    """Return the smallest max-time across all solution files in a folder.

    Useful to find a common time horizon safe to interpolate onto across a
    whole dataset of raw solution files.
    """
    list_files = get_sol_list(data_root_folder)
    max_t = [
        np.amax(np.load(os.path.join(data_root_folder, file))["t"])
        for file in list_files
    ]
    max_t = np.array(max_t)
    max_t = np.amin(max_t)
    return max_t
