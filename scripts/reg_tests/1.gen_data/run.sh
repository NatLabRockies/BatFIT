root_dir=`python -c "from batfit import BATFIT_DIR; print(BATFIT_DIR)"`
n_procs=4
yaml_in=${root_dir}/default_exps/spm_discharge.yaml
folder_save=./data_spm_discharge
rm -r $folder_save
# Sample the parameter space
python gen_sample_par.py -n_int 10 -sim_config $yaml_in -folder_save $folder_save
# Generate data
mpiexec -n $n_procs python gen_sol.py -sim_config $yaml_in -folder_save $folder_save 

# If you are using slurm
# n_procs=$SLURM_NPROCS
#srun -n $n_procs python gen_sol.py -sim_config $yaml_in -folder_save $folder_save
