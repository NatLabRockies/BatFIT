"""Plot simulated current/voltage signals from a gen_data sols.pkl."""

import argparse
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from batfit import logger

parser = argparse.ArgumentParser(description="Plot generated signals")
parser.add_argument(
    "-folder_save",
    "--folder_save",
    type=str,
    metavar="",
    required=False,
    help="gen_data output folder holding sols.pkl",
    default=".",
)
parser.add_argument(
    "-n_max",
    "--n_max",
    type=int,
    metavar="",
    required=False,
    help="Maximum number of curves to overlay",
    default=20,
)
args, unknown = parser.parse_known_args()

with open(os.path.join(args.folder_save, "sols.pkl"), "rb") as f:
    sols = pickle.load(f)

keys = list(sols)[: args.n_max]
fig, (ax_i, ax_v) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
for key in keys:
    sol = sols[key]["sol"]
    t_min = sol["t"] / 60.0
    ax_i.plot(t_min, -sol["i"] * 1000.0, linewidth=1)
    ax_v.plot(t_min, sol["phis_c"], linewidth=1)
ax_i.set_ylabel("I [mA]")
ax_v.set_xlabel("t [min]")
ax_v.set_ylabel(r"$\phi$ [V]")
ax_i.set_title(f"{len(keys)} simulated signals")
fig.tight_layout()
out_file = os.path.join(args.folder_save, "all_sol.png")
fig.savefig(out_file, dpi=150)
plt.close(fig)
logger.info(f"Saved {out_file}")
