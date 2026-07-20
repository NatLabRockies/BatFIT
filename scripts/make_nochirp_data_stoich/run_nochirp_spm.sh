root_dir=`python -c "from batfit import BATFIT_DIR; print(BATFIT_DIR)"`
n_procs=10
yaml_in=${root_dir}/default_exps/spm_nochirp.yaml
folder_save=./data_spm_nochirp
rm -r $folder_save
# Sample the parameter space
python gen_sample_par.py -n_int 10000 -sim_config $yaml_in -folder_save $folder_save
# Generate data
#python gen_sol.py -sim_config $yaml_in -folder_save $folder_save
mpiexec -n $n_procs python gen_sol.py -sim_config $yaml_in -folder_save $folder_save 
