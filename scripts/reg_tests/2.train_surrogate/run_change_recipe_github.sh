in_yml=training_recipes/recipe_rel.yml
out_yml=training_recipes/recipe.yml
d_path=$GITHUB_WORKSPACE/scripts/reg_tests/1.gen_data
s_path=$GITHUB_WORKSPACE/batfit/default_exps

python write_absolute_recipe.py $in_yml $out_yml $d_path $s_path
