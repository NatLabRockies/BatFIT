import pickle

from batfit.utils.assembly import (
    assemble_all_data,
    augment_data,
    check_assembled_data_shape,
    from_combined_sols_to_data,
    from_param_to_surrogate_data,
    from_sol_dict_to_xy,
    passes_quality_filters,
)
from batfit.utils.dataset_split import (
    split_dataset_from_np,
    split_protocol_dataset_from_np,
)
from batfit.utils.raw_sol_utils import (
    from_name_to_params,
    get_max_time,
    get_sol_list,
)

__all__ = [
    "assemble_all_data",
    "augment_data",
    "check_assembled_data_shape",
    "from_combined_sols_to_data",
    "from_param_to_surrogate_data",
    "from_sol_dict_to_xy",
    "passes_quality_filters",
    "split_dataset_from_np",
    "split_protocol_dataset_from_np",
    "from_name_to_params",
    "get_max_time",
    "get_sol_list",
    "load_pickle",
]


def load_pickle(path: str):
    """Load a pickled object (typically a scikit-learn scaler).

    Parameters
    ----------
    path: str
        Path to the pickle file

    Returns
    -------
    object
        The unpickled object
    """
    with open(path, "rb") as f:
        return pickle.load(f)
