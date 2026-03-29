import os
import pickle

from file_management import *
from prettyPlot.plotting import *
from read import *

from batfit import logger
from batfit.preprocess.utils import reduce_npoints_dict


def make_target_data(data_type: str, rpt_ids=[-1, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
    """
    Extract and store data by RPT, protocl and cell id
    """
    data_folder = "data_target"
    os.makedirs(data_folder, exist_ok=True)

    assert data_type.lower() in ["hppc", "diffcap"]
    if data_type.lower() == "diffcap":
        n_points = 512
    elif data_type.lower() == "hppc":
        n_points = 4000
    else:
        raise NotImplementedError

    file_names = get_all_protocol_all_rpt_data_file(
        data_type=data_type.lower()
    )
    for protocol in file_names:
        data_target = {}
        logger.info(f"Making BOL data for {protocol}")
        for rpt_id in rpt_ids:
            logger.info(f"\tRPT {rpt_id}")
            for cell_id in file_names[protocol][rpt_id]:
                logger.info(f"\t\tCell {cell_id}")
                filename = file_names[protocol][rpt_id][cell_id]
                cycle_df = read_single_csv(filename)
                time = get_elapsed_test_time(cycle_df)
                phis_c = get_voltage(cycle_df)
                sol_dict = {"t": time, "phis_c": phis_c}
                new_sol_dict = reduce_npoints_dict(sol_dict, n_points)
                data_target[cell_id] = new_sol_dict
            if rpt_id == -1:
                subfolder = "BOL"
            else:
                subfolder = f"RPT_{rpt_id}"
            os.makedirs(os.path.join(data_folder, subfolder), exist_ok=True)
            with open(
                os.path.join(
                    data_folder,
                    subfolder,
                    f"{protocol}_{data_type.lower()}.pkl",
                ),
                "wb",
            ) as f:
                pickle.dump(data_target, f)


if __name__ == "__main__":
    make_target_data(data_type="diffcap")
    make_target_data(data_type="hppc")
