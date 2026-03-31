import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable MPS fallback
import pickle

import numpy as np
import torch
from prettyPlot.plotting import *

from batfit import BATFIT_DIR, BATFIT_EXP, logger
from batfit.basicutilityc import ReadInput as ri
from batfit.model.paramNN import *
from batfit.utils.data_utils import *
from batfit.utils.data_utils import scale_input_from_scaler
from batfit.utils.torch_utils import *
from batfit.utils.torch_utils import get_device_type


def plot_loss(model_folder:str):
    # Read
    filename = os.path.join(model_folder, "test_loss.csv")
    vals_test = np.loadtxt(filename, delimiter=";", skiprows=1)
    filename = os.path.join(model_folder, "train_loss.csv")
    vals_train = np.loadtxt(filename, delimiter=";", skiprows=1)

    # plot
    fig = plt.figure(figsize=(6, 6))
    plt.plot(vals_train[::50, 0], vals_train[::50, 1], linewidth=3, color="k", label="train")
    plt.plot(vals_test[::20, 0], vals_test[::20, 1], '--', linewidth=3, color="k", label="test")
    ax = plt.gca()
    maxe5 = int(vals_train[-1, 0]/1e5)
    ax.set_xticks([i*1e5 for i in range(maxe5+1)])
    #ax.set_yscale("log")
    pretty_labels("# Step", "Negative Log-Likelihood", fontsize=20, fontname="Times", grid=False)
    pretty_legend(fontsize=20, fontname="Times")
    plt.show()




if __name__ == "__main__":
    import shutil
    import sys

    plot_loss("/Users/mhassana/Desktop/GitHub/BatFIT_mar26/scripts/diagnostics/trained_models/diffcap/models85")
    plot_loss("/Users/mhassana/Desktop/GitHub/BatFIT_mar26/scripts/diagnostics/trained_models/hppc/models292")
    
    #test_perf(inp, mode="normal")
    #test_perf(inp, mode="model_discrepancy")
    #plot_perf(inp, mode="model_discrepancy")
