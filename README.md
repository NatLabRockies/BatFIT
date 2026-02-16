# BATFIT

## Summary

This package implements several data-based techniques for parameter fitting in Li-ion battery models. It uses [BATMODS-lite](https://github.com/NatLabRockies/batmods-lite) to generate the data. 

The repository contains the code that is used for the paper "Neural posterior estimation is accurate, tractable and scalable for inverse parameter inference in Li-ion batteries", M. Hassanaly, C. R. Randall, P. J.  Weddle, P. J. Gasper, C. Kelly, T. Tanim, E. J. Dufek, K. Smith.

## Installation

We recommend using a conda environment

```
conda create -n batfit python=3.12
conda activate batfit
```

Once the files are available on your machine, use your terminal to navigate into the folder and execute one of the following depending on your installation preference.

```
pip install .             (basic installation)
pip install -e .[dev]     (editable installation with developer options)
```

## Get Started

The regression tests (run as part of the CI) show how to use the basic capabilities of the code.
1. Data generation with BATMODS-lite (``scripts/reg_tests/1.gen_data``). This demonstrates how to generate data from a single particle model in parallel. This has been tested with up to 1200 workers but uses 4 workers here. 
Once BATFIT is installed
```
bash run.sh
``` 

2. Train a surrogate of the physics-based model (``scripts/reg_tests/2.train_surrogate``). This demonstrate how to preprocess the data and train a surrogate model of the single particle model that can be used for simulation-based inference.
Once BATFIT is installed, and ``run_change_recipe_local.sh`` has been changed to used your path.
```
bash run_change_recipe_local.sh 
bash run.sh
```

3. Run MCMC to identify parameters with the trained surrogate (``scripts/reg_tests/3.surrogate_mcmc``). This demonstrates how to run MCMC with a data-based surrogate instead of a physics-based model.
Once BATFIT is installed, and ``run_change_recipe_local.sh`` has been changed to used your path.
```
bash run_change_recipe_local.sh 
bash run.sh
```

4. Use Neural Posterior Estimation to approximate the parameter posterior PDF (``scripts/reg_tests/4.npe``)
Once BATFIT is installed, and ``run_change_recipe_local.sh`` has been changed to used your path.
```
bash run_change_recipe_local.sh
bash run.sh
```

## Citing this Work

```
SWR BatFit

arxiv citation
```

## Acknowledgements
This work was authored by the National Laboratory of the Rockies (NLR) for the U.S. Department of Energy (DOE) under Contract No. DE-AC36-08GO28308. This work was supported by funding from DOE's Vehicle Technologies Office (VTO). The research was performed using computational resources sponsored by the Department of Energy's Office of Critical Minerals and Energy Innovation (CMEI) and located at the National Laboratory of the Rockies. The views expressed in the repository do not necessarily represent the views of the DOE or the U.S. Government.

