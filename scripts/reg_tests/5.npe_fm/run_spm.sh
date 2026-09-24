# Train FM
python train_nn.py training_recipes/recipe_spm.yml
# Test FM
python test_nn.py training_recipes/recipe_spm.yml
# Train and test with the time-dependent z-score signal scaling
python train_nn.py training_recipes/recipe_spm_tdzscore.yml
python test_nn.py training_recipes/recipe_spm_tdzscore.yml
