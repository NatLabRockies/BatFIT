# Full P2D chirp-optimization suite: FM NPE (chirp + nochirp baseline),
# variance predictor, and the chirp-benefit optimization.
# 1.gen_data -> 2.npe_fm -> 3.variance_pred -> 4.optimization
# (2.npe_gaussian is standalone and NOT part of this suite.)
cd 1.gen_data
bash run_p2d.sh
cd ../2.npe_fm
bash run_p2d.sh
cd ../3.variance_pred
bash run_p2d.sh
cd ../4.optimization
bash run_p2d.sh
cd ..
