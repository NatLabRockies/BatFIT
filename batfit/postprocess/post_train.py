import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from prettyPlot.plotting import pretty_labels


def plot_loss(loss_hist_file, figure_folder="Figures", fig_name="loss.png"):
    loss_data = np.genfromtxt(loss_hist_file, delimiter=";", skip_header=1)
    plt.figure()
    plt.plot(loss_data[:, 0], loss_data[:, 1], color="k")
    pretty_labels("# Step", "Loss", fontsize=16, fontname="Times")
    # os.makedirs(figure_folder, exist_ok=True)
    log_dir = Path(figure_folder)
    log_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(figure_folder, fig_name))
    plt.close()
