# Build the variance-predictor dataset from the frozen chirp FM NPE
python gen_var_dataset.py training_recipes/recipe_gen_dataset_p2d.yml
# Train the variance predictor (log_sigma -> linear head)
python train_var_pred.py training_recipes/recipe_var_pred_p2d.yml
# Parity plots + per-parameter sigma error table on the val split
python test_var_pred.py training_recipes/recipe_var_pred_p2d.yml
# Global (delta) sensitivity of the NPE sigma to (mu, protocol) inputs
python compute_sensitivity.py training_recipes/recipe_var_pred_p2d.yml
