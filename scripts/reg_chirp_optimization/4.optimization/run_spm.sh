# Optimize the chirp to minimise NPE uncertainty over nochirp val observations
python run_optimization_clean.py training_recipes/recipe_spm.yml
# Conditional-average figures (variance reduction, optimal chirp, sigmas)
python plot_optimization_clean.py training_recipes/recipe_spm.yml
