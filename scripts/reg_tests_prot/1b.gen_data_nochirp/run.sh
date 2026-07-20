root_dir=`python -c "from batfit import BATFIT_DIR; print(BATFIT_DIR)"`
n_procs=8
yaml_in=${root_dir}/default_exps/spm_nochirp.yaml
folder_save=./data_spm_nochirp
rm -r $folder_save
# The material file is resolved relative to the working directory
cp ../1.gen_data/graphite_nmc532.yaml .
# Sample the parameter space (same bounds as the chirp config, no protocol params)
python ../1.gen_data/gen_sample_par.py -n_int 100 -sim_config $yaml_in -folder_save $folder_save
# Generate data
mpiexec -n $n_procs python ../1.gen_data/gen_sol.py -sim_config $yaml_in -folder_save $folder_save
