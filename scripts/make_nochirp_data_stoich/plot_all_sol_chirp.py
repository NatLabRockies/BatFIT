import pickle

from prettyPlot.plotting import *

# with open("data_spm_discharge_val/sols.pkl", "rb") as f:
with open("data_spm_chirp/sols.pkl", "rb") as f:
    sols = pickle.load(f)

fig = plt.figure()
for key in sols:
    plt.plot(
        sols[key]["sol"]["t"] / 60,
        sols[key]["sol"]["phis_c"],
        color="k",
        linewidth=1,
    )
pretty_labels("t [min]", "phi [V]")
ax = plt.gca()
ax.set_ylim([2, 5])
plt.show()
