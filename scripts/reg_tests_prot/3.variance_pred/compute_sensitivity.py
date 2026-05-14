import numpy as np
from prettyPlot.plotting import *
from SALib.analyze import delta

dataset = np.load("var_pred_data/var_pred_dataset.npz")

# P mu and sigma are between 0 and 1
P = dataset["P_train"]
Mu = dataset["Mu_train"]
Sigma = dataset["Sigma_train"]

X = np.hstack((Mu, P))
names = [f'Mu_{i}' for i in range(6)] + [f'P_{i}' for i in range(3)]
num_vars = len(names)
bounds = [[X[:, i].min(), X[:, i].max()] for i in range(num_vars)]
# Define the SALib problem dictionary
problem = {
    'num_vars': num_vars,
    'names': names,
    'bounds': bounds
}

var_names = [r"i$_{\rm 0,a}$", r"D$_{\rm s,c}$", r"x$_{\rm 0,a}$", r"x$_{\rm 0,c}$", r"i$_{\rm 0,c}$", r"$\varepsilon_{\rm AM,s,c}$"]
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
for i in range(6):
    Y = Sigma[:, i]
    Si = delta.analyze(problem, X, Y, num_resamples=10)
    indices = Si['delta']
    errors = Si['delta_conf']
    
    ax = axes[i]
    ax.bar(names, indices, yerr=errors, capsize=5, color='#4C72B0', edgecolor='black', alpha=0.8)
    ax.set_title(fr'$\sigma$ {var_names[i]}', fontsize=12, fontweight='bold')
    ax.set_ylabel('Delta Index', fontsize=10)
    ax.set_ylim(0, max(indices) + max(errors) + 0.1) # Add headroom for error bars
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()






