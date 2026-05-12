root_dir=`python -c "from batfit import BATFIT_DIR; print(BATFIT_DIR)"`
n_procs=4
yaml_in=${root_dir}/default_exps/spm_chirp.yaml
folder_save=./data_spm_chirp
rm -r $folder_save
# Sample the parameter space (lithium conservation enforced)
python gen_sample_par.py -n_int 100000 -sim_config $yaml_in -folder_save $folder_save
# Generate data
mpiexec -n $n_procs python gen_sol.py -sim_config $yaml_in -folder_save $folder_save
