rm -r 1.gen_data/data_spm_chirp
rm -f 2.npe_prot/training_recipes/recipe.yml

cd 1.gen_data
bash run.sh
cd ../2.npe_prot
bash run_change_recipe_local.sh
bash run.sh
cd ..
