# Chirp FM NPE (ProbProtParamFM)
python train_nn_prot.py training_recipes/recipe_spm_chirp.yml
python test_nn_prot.py training_recipes/recipe_spm_chirp.yml

# Nochirp FM NPE (ProbParamFM, surrogate-free test)
python train_nn.py training_recipes/recipe_spm_nochirp.yml
python test_nn.py training_recipes/recipe_spm_nochirp.yml

# Same FM NPEs with the time-dependent z-score signal scaling
python train_nn_prot.py training_recipes/recipe_spm_chirp_tdzscore.yml
python test_nn_prot.py training_recipes/recipe_spm_chirp_tdzscore.yml
python train_nn.py training_recipes/recipe_spm_nochirp_tdzscore.yml
python test_nn.py training_recipes/recipe_spm_nochirp_tdzscore.yml
