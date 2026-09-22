rm -rf 1.gen_data/data_p2d_discharge
cd 1.gen_data
bash run_p2d.sh
cd ../2.surrogate
bash run_p2d.sh
cd ../3.surrogate_mcmc
bash run_p2d.sh
cd ../4.npe_gaussian
bash run_p2d.sh
cd ../5.npe_fm
bash run_p2d.sh
cd ..
