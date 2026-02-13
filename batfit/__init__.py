import os

BATFIT_DIR = os.path.dirname(os.path.realpath(__file__))
BATFIT_EXP = os.path.join(BATFIT_DIR, "default_exps")

from batfit.logging_config import setup_logging

logger = setup_logging(level="INFO")

__version__ = "0.0.1"
