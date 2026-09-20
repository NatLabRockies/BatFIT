rm -rf 1.gen_data/data_spm_discharge
cd 1.gen_data
bash run.sh
cd ../2.surrogate
bash run.sh
cd ../3.surrogate_mcmc
bash run.sh
cd ../4.npe_gaussian
bash run.sh
cd ../5.npe_fm
bash run.sh
cd ..
