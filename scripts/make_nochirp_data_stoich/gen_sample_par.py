import argparse
import os
import sys

from batfit import BATFIT_EXP, logger
from batfit.preprocess.param_sampling import *
from batfit.preprocess.sim_setup import make_params

parser = argparse.ArgumentParser(description="Parameter sampling")
parser.add_argument(
    "-n_int",
    "--n_int",
    type=int,
    metavar="",
    required=False,
    help="Number of interior samples",
    default=10,
)
parser.add_argument(
    "-n_bound",
    "--n_bound",
    type=int,
    metavar="",
    required=False,
    help="Number of boundary samples",
    default=0,
)
parser.add_argument(
    "-sim_config",
    "--sim_config",
    type=str,
    metavar="",
    required=False,
    help="Sim config file",
    default=os.path.join(BATFIT_EXP, "spm_discharge.yaml"),
)
parser.add_argument(
    "-folder_save",
    "--folder_save",
    type=str,
    metavar="",
    required=False,
    help="data folder",
    default=".",
)

parser.add_argument(
    "-licons",
    "--lithium_conservation",
    action="store_true",
    help="Ensure lithium conservation between the charge and the discharge",
)

args, unknown = parser.parse_known_args()
if len(unknown) > 0:
    logger.warning(f"Unknown args {unknown}")


n_int = args.n_int
n_bound = args.n_bound
li_cons = args.lithium_conservation
sim_params = make_params(args.sim_config)

deg_param_names = None
prot_param_names = None
if n_bound + n_int == 0:
    logger.error("No sample parameter requested")
    sys.exit()

if n_int > 0:
    deg_int_samples, prot_int_samples = get_samples(
        n_int=n_int,
        deg_param_names=deg_param_names,
        sim_params=sim_params,
        li_cons=li_cons,
        uniform=True,
    )
if n_bound > 0:
    deg_bound_samples, prot_bound_samples = get_bounding_samples(
        n_bound=n_bound,
        deg_param_names=deg_param_names,
        sim_params=sim_params,
        li_cons=li_cons,
    )

if n_bound > 0 and n_int > 0:
    deg_samples = np.vstack((deg_int_samples, deg_bound_samples))
if n_bound == 0:
    deg_samples = deg_int_samples
if n_int == 0:
    deg_samples = deg_bound_samples

write_exec(
    deg_samples,
    deg_param_names=deg_param_names,
    folder_save=args.folder_save,
    sim_params=sim_params,
)
