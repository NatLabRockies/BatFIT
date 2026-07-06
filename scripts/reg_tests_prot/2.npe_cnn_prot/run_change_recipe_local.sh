in_yml=training_recipes/recipe_rel.yml
out_yml=training_recipes/recipe.yml
r_path=$(pwd)/..
s_path=`python -c "from batfit import BATFIT_DIR; print(BATFIT_DIR)"`/default_exps

python write_absolute_recipe.py $in_yml $out_yml $r_path $s_path
