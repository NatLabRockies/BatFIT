rm -r 1.gen_data/data_spm_discharge
rm -r 2.train_surrogate/training_recipes/recipe.yml
rm -r 3.surrogate_mcmc/cal_recipes/recipe.yml
rm -r 4.npe_cnn/training_recipes/recipe.yml
rm -r 5.npe_fm/training_recipes/recipe.yml
cd 1.gen_data
bash run.sh
cd ../2.train_surrogate
bash run_change_recipe_local.sh
bash run.sh
cd ../3.surrogate_mcmc
bash run_change_recipe_local.sh
bash run.sh
cd ../4.npe_cnn
bash run_change_recipe_local.sh
bash run.sh
cd ../5.npe_fm
bash run_change_recipe_local.sh
bash run.sh
cd ..

