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
        linewidth=3,
    )
pretty_labels("t [min]", "I [A]", grid=False, fontsize=20, fontname="Times")
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
        linewidth=3,
    )
    keysave=key
pretty_labels("t [min]", r"$\phi [V]$", grid=False, fontsize=20, fontname="Times")
ax = plt.gca()
plt.show()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
ax1.plot(sols[keysave]["sol"]["t"] / 60, 1000*(-sols[keysave]["sol"]["i"]), color='k', linewidth=2)
ax2.plot(sols[keysave]["sol"]["t"] / 60, sols[keysave]["sol"]["phis_c"], color='k', linewidth=2)

# Set y-axis labels to show they are present for both
pretty_labels("", "I [mA]", grid=False, fontsize=20, fontname="Times", ax=ax1)
pretty_labels("t [min]", r"$\phi [V]$", grid=False, fontsize=20, fontname="Times", ax=ax2)

# Explicitly hide the x-axis tick marks for the top plot
ax1.tick_params(axis='x', which='both', length=0)

# Remove the gap between the plots to make the shared axis look cleaner
plt.subplots_adjust(hspace=0.1)

# Save the figure
plt.show()

