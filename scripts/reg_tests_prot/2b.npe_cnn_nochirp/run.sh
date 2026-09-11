# Train the nochirp (chargecc) NPE by reusing the discharge pipeline's
# recipe-generic training script. No test step here: reg_tests/4.npe_cnn's
# test_nn.py requires a trained surrogate, and the nochirp NPE is exercised
# downstream by 5.optimization_clean.
python ../../reg_tests/4.npe_cnn/train_nn.py training_recipes/recipe.yml
