# Build the variance-predictor dataset from the frozen chirp FM NPE
python gen_var_dataset.py training_recipes/recipe_gen_dataset_spm.yml
# Train the variance predictor (log_sigma -> linear head)
python train_var_pred.py training_recipes/recipe_var_pred_spm.yml
# Parity plots + per-parameter sigma error table on the test split
python test_var_pred.py training_recipes/recipe_var_pred_spm.yml
