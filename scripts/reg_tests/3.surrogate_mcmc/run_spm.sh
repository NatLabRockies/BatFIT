root_dir=`python -c "from batfit import BATFIT_DIR; print(BATFIT_DIR)"`
n_procs=4
mpiexec -n $n_procs python bayesCal_synth_parallel.py cal_recipes/recipe.yml
python test_mcmc.py cal_recipes/recipe.yml
