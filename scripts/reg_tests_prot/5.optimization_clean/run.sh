# Evaluate the chirp benefit over nochirp observations, then plot the
# conditionally averaged variance reduction and optimal chirp parameters
python run_optimization_clean.py training_recipes/recipe_clean.yml
python plot_optimization_clean.py training_recipes/recipe_clean.yml
