rm -r 1.gen_data/data_spm_discharge 
rm -r 2.train_surrogate/training_recipes/recipe.yml 
rm -r 3.surrogate_mcmc/cal_recipes/recipe.yml 
rm -r 4.npe/training_recipes/recipe.yml
cd 1.gen_data
bash run.sh
cd ../2.train_surrogate
bash run_change_recipe_local.sh
bash run.sh
cd ../3.surrogate_mcmc
bash run_change_recipe_local.sh
bash run.sh
cd ../4.npe
bash run_change_recipe_local.sh
bash run.sh
cd ..

