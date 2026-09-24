# Train FM
python train_nn.py training_recipes/recipe_p2d.yml
# Test FM
python test_nn.py training_recipes/recipe_p2d.yml
# Train and test with the time-dependent z-score signal scaling
python train_nn.py training_recipes/recipe_p2d_tdzscore.yml
python test_nn.py training_recipes/recipe_p2d_tdzscore.yml
