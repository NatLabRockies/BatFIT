import os
import sys

# On macOS, pip's torch and BATMODS-lite's solver (scikit-sundae) each load
# their own OpenMP runtime; importing both aborts with "OMP: Error #15". Allow
# the duplicate runtime (Intel's documented workaround) unless the user set the
# variable. Remove once torch and scikit-sundae share one runtime (e.g. torch
# from conda-forge).
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

BATFIT_DIR = os.path.dirname(os.path.realpath(__file__))
BATFIT_EXP = os.path.join(BATFIT_DIR, "default_exps")
BATFIT_REG = os.path.join(BATFIT_DIR, "..", "scripts", "reg_tests")

from batfit.logging_config import setup_logging

logger = setup_logging(level="INFO")

__version__ = "0.0.1"
