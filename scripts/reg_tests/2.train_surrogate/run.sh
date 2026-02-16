# Save test and train dataset separately and consistently
python prepare_datasets.py training_recipes/recipe.yml
# Train surrogate
python train_nn.py training_recipes/recipe.yml
# Test surrogate
python test_nn.py training_recipes/recipe.yml
