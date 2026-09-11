rm -r 1.gen_data/data_spm_chirp
rm -f 2.npe_cnn_prot/training_recipes/recipe.yml
rm -f 3.npe_fm_prot/training_recipes/recipe.yml

cd 1.gen_data
bash run.sh
cd ../2.npe_cnn_prot
bash run_change_recipe_local.sh
bash run.sh
cd ../3.npe_fm_prot
bash run_change_recipe_local.sh
bash run.sh
cd ..
