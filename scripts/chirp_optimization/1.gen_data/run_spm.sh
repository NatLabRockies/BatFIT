root_dir=`python -c "from batfit import BATFIT_DIR; print(BATFIT_DIR)"`
n_procs=4
n_int=40                          # HPC: 2000000
chirp_n_points=256                # HPC: 2048
nochirp_n_points=128              # HPC: 512

# ---- chirp branch (cyc_mode: chirp) ----
yaml_in=${root_dir}/default_exps/spm_chirp.yaml
folder_save=./data_spm_chirp
rm -rf $folder_save
# Sample the parameter space (degradation + protocol params)
python gen_sample_par.py -n_int $n_int -sim_config $yaml_in -folder_save $folder_save
# Generate data (fine time grid for the chirp signal)
mpiexec -n $n_procs python gen_sol.py -sim_config $yaml_in -folder_save $folder_save -n_points_reduce $chirp_n_points

# ---- nochirp branch (cyc_mode: chargecc) ----
yaml_in=${root_dir}/default_exps/spm_nochirp.yaml
folder_save=./data_spm_nochirp
rm -rf $folder_save
# Same degradation bounds, no protocol params
python gen_sample_par.py -n_int $n_int -sim_config $yaml_in -folder_save $folder_save
# Generate data (coarse time grid for the plain CC charge)
mpiexec -n $n_procs python gen_sol.py -sim_config $yaml_in -folder_save $folder_save -n_points_reduce $nochirp_n_points
