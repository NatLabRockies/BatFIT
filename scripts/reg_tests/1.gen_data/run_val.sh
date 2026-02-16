root_dir=`python -c "from batfit import BATFIT_DIR; print(BATFIT_DIR)"`
n_procs=4
yaml_in=${root_dir}/default_exps/spm_discharge.yaml
folder_save_val=./data_spm_discharge_val
# Generate data
mpiexec -n $n_procs python gen_sol.py -sim_config $yaml_in -folder_save $folder_save_val 

