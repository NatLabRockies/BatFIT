rm -rf 1.gen_data/data_spm_discharge
cd 1.gen_data
bash run_spm.sh
cd ../2.surrogate
bash run_spm.sh
cd ../3.surrogate_mcmc
bash run_spm.sh
cd ../4.npe_gaussian
bash run_spm.sh
cd ../5.npe_fm
bash run_spm.sh
cd ..
