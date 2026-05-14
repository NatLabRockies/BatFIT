import pickle

from prettyPlot.plotting import *

# with open("data_spm_discharge_val/sols.pkl", "rb") as f:
with open("data_spm_chirp/sols.pkl", "rb") as f:
    sols = pickle.load(f)

fig = plt.figure()
for i, key in enumerate(sols):
    plt.plot(
        sols[key]["sol"]["t"] / 60,
        sols[key]["sol"]["phis_c"],
        color="k",
        linewidth=1,
    )
    if i > 10:
        break
pretty_labels("t [min]", "phi [V]")
ax = plt.gca()
ax.set_ylim([2, 5])

plt.show()
fig = plt.figure()
for i, key in enumerate(sols):
    if i > 0:
        break
    plt.plot(
        sols[key]["sol"]["t"] / 60,
        -sols[key]["sol"]["i"],
        color="k",
        linewidth=1,
    )
pretty_labels("t [min]", "I [A]")
ax = plt.gca()
plt.show()

plt.show()
fig = plt.figure()
for i, key in enumerate(sols):
    if i > 0:
        break
    plt.plot(
        sols[key]["sol"]["t"] / 60,
        sols[key]["sol"]["phis_c"],
        color="k",
        linewidth=1,
    )
pretty_labels("t [min]", r"$\phi [V]$")
ax = plt.gca()
plt.show()
