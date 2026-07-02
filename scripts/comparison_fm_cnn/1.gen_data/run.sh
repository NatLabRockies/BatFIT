root_dir=`python -c "from batfit import BATFIT_DIR; print(BATFIT_DIR)"`
n_procs=10
yaml_in=${root_dir}/default_exps/spm_discharge.yaml
folder_save=./data_spm_discharge
rm -rf $folder_save
# Sample the parameter space (larger than reg_test to get meaningful statistics)
python gen_sample_par.py -n_int 10000 -sim_config $yaml_in -folder_save $folder_save
# Generate data
mpiexec -n $n_procs python gen_sol.py -sim_config $yaml_in -folder_save $folder_save

# If you are using slurm
# n_procs=$SLURM_NPROCS
# srun -n $n_procs python gen_sol.py -sim_config $yaml_in -folder_save $folder_save
