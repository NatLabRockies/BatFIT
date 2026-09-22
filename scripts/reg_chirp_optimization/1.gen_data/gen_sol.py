import argparse
import os
import time

import batfit.utils.parallel as parallel_env
from batfit import BATFIT_EXP
from batfit.preprocess.sim_setup import make_params
from batfit.preprocess.sol_gen import multi_run

parser = argparse.ArgumentParser(description="Dataset generator")
parser.add_argument(
    "-sim_config",
    "--sim_config",
    type=str,
    metavar="",
    required=False,
    help="Sim config file",
    default=os.path.join(BATFIT_EXP, "spm_chirp.yaml"),
)
parser.add_argument(
    "-folder_save",
    "--folder_save",
    type=str,
    metavar="",
    required=False,
    help="Data folder",
    default=".",
)
parser.add_argument(
    "-n_points_reduce",
    "--n_points_reduce",
    type=int,
    metavar="",
    required=False,
    help="Number of time points retained per simulated signal",
    default=2048,  # HPC chirp: 2048, nochirp: 512
)
args, unknown = parser.parse_known_args()

sim_params = make_params(args.sim_config, parallel_env=parallel_env)
time_s = time.time()
multi_run(
    sim_params=sim_params,
    parallel_env=parallel_env,
    folder_save=args.folder_save,
    n_points_reduce=args.n_points_reduce,
    store_current=True,
)
time_e = time.time()
parallel_env.printRoot(f"Total time elapsed {time_e - time_s:.2f}s")
