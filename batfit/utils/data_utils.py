"""Backward-compatible re-exports for the dataset assembly/scaling modules.

The actual implementations now live in focused modules under ``batfit.utils``:
:mod:`~batfit.utils.assembly`, :mod:`~batfit.utils.raw_sol_utils`,
:mod:`~batfit.utils.scalers`, :mod:`~batfit.utils.dataset_split`, and
:mod:`~batfit.utils.dataset_scaling`. This module re-exports their public
names so existing ``from batfit.utils.data_utils import ...`` call sites keep
working unchanged, and hosts the small shared I/O helper :func:`load_pickle`.

Dataset assembly workflow (in order):

1. **Assemble and filter** (:mod:`~batfit.utils.assembly`) — read raw
   physics-based model solutions from ``sols.pkl``, reject the ones that
   fail :func:`~batfit.utils.assembly.passes_quality_filters` (a
   rejection-sampling prior over buggy/out-of-range simulations), and
   convert the accepted ones into raw ``(X, [P], Y)`` numpy arrays.
2. **Split** (:mod:`~batfit.utils.dataset_split`) — train/test split those
   raw arrays.
3. **Scale** (:mod:`~batfit.utils.dataset_scaling`) — fit scalers on the
   train split and apply them to both splits. Models are only ever trained
   and evaluated on this scaled data, never on raw/unscaled arrays.
"""

import pickle

from batfit.utils.assembly import (
    assemble_all_data,
    assemble_surrogate_data,
    augment_data,
    check_assembled_data_shape,
    check_assembled_surrogate_data_shape,
    from_combined_sols_to_data,
    from_param_to_surrogate_data,
    from_sol_dict_to_xy,
    passes_quality_filters,
)
from batfit.utils.dataset_scaling import (
    scale_dataset_from_np,
    scale_protocol_dataset_from_np,
    scale_surrogate_dataset_from_np,
)
from batfit.utils.dataset_split import (
    split_dataset_from_np,
    split_protocol_dataset_from_np,
    split_surrogate_dataset_from_np,
)
from batfit.utils.raw_sol_utils import (
    from_name_to_params,
    get_max_time,
    get_sol_list,
)
from batfit.utils.scalers import (
    CustomScaler,
    scale_dataset_from_scaler,
    scale_input_from_scaler,
    scale_output_from_scaler,
    unscale_dataset_from_scaler,
    unscale_input_from_scaler,
    unscale_output_from_scaler,
    unscale_pred_from_scaler,
    unscale_pred_std_from_scaler,
)

__all__ = [
    "assemble_all_data",
    "assemble_surrogate_data",
    "augment_data",
    "check_assembled_data_shape",
    "check_assembled_surrogate_data_shape",
    "from_combined_sols_to_data",
    "from_param_to_surrogate_data",
    "from_sol_dict_to_xy",
    "passes_quality_filters",
    "scale_dataset_from_np",
    "scale_protocol_dataset_from_np",
    "scale_surrogate_dataset_from_np",
    "split_dataset_from_np",
    "split_protocol_dataset_from_np",
    "split_surrogate_dataset_from_np",
    "from_name_to_params",
    "get_max_time",
    "get_sol_list",
    "CustomScaler",
    "scale_dataset_from_scaler",
    "scale_input_from_scaler",
    "scale_output_from_scaler",
    "unscale_dataset_from_scaler",
    "unscale_input_from_scaler",
    "unscale_output_from_scaler",
    "unscale_pred_from_scaler",
    "unscale_pred_std_from_scaler",
    "load_pickle",
]


def load_pickle(path: str):
    """Load a pickled object (typically a scikit-learn scaler).

    :param path: path to the pickle file
    :return: the unpickled object
    """
    with open(path, "rb") as f:
        return pickle.load(f)
