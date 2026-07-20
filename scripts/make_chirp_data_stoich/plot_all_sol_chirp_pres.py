import pickle

from prettyPlot.plotting import *

# with open("data_spm_discharge_val/sols.pkl", "rb") as f:
with open("data_spm_chirp/sols.pkl", "rb") as f:
    sols = pickle.load(f)

for i, key in enumerate(sols):
    if i > 0:
        break
    keysave=key

minc = sols[keysave]['prot_params'][0]
maxc = sols[keysave]['prot_params'][0] + sols[keysave]['prot_params'][2]*2+1

mask_chirp = (sols[keysave]["sol"]["t"] >= minc) & (sols[keysave]["sol"]["t"] <= maxc)
mask_nochirp_min = (sols[keysave]["sol"]["t"] <= minc)
mask_nochirp_max = (sols[keysave]["sol"]["t"] > maxc)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
ax1.plot(sols[keysave]["sol"]["t"][mask_nochirp_min] / 60, 1000*(-sols[keysave]["sol"]["i"][mask_nochirp_min]), color='k', linewidth=2)
ax2.plot(sols[keysave]["sol"]["t"][mask_nochirp_min] / 60, sols[keysave]["sol"]["phis_c"][mask_nochirp_min], color='k', linewidth=2)
ax1.plot(sols[keysave]["sol"]["t"][mask_nochirp_max] / 60, 1000*(-sols[keysave]["sol"]["i"][mask_nochirp_max]), color='k', linewidth=2)
ax2.plot(sols[keysave]["sol"]["t"][mask_nochirp_max] / 60, sols[keysave]["sol"]["phis_c"][mask_nochirp_max], color='k', linewidth=2)
ax1.plot(sols[keysave]["sol"]["t"][mask_chirp] / 60, 1000*(-sols[keysave]["sol"]["i"][mask_chirp]), color='r', linewidth=2)
ax2.plot(sols[keysave]["sol"]["t"][mask_chirp] / 60, sols[keysave]["sol"]["phis_c"][mask_chirp], color='r', linewidth=2)

# Set y-axis labels to show they are present for both
pretty_labels("", "I [mA]", grid=False, fontsize=20, fontname="Times", ax=ax1)
pretty_labels("t [min]", r"$\phi [V]$", grid=False, fontsize=20, fontname="Times", ax=ax2)

# Explicitly hide the x-axis tick marks for the top plot
ax1.tick_params(axis='x', which='both', length=0)

# Remove the gap between the plots to make the shared axis look cleaner
plt.subplots_adjust(hspace=0.1)

# Save the figure
plt.show()

