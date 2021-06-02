DOA Algorithm based on MM Algorithm
===================================

This repository contains implementation for the algorithms and experiments of the paper [Refinement of Direction of Arrival Estimators by Majorization-Minimization Optimization on the Array Manifold](https://ieeexplore.ieee.org/document/9414798) by Robin Scheibler and Masahito Togami.
Abstract
--------

We propose a generalized formulation of direction of arrival estimation that
includes many existing methods such as steered response power, subspace,
coherent and incoherent, as well as speech sparsity-based methods. Unlike most
conventional methods that rely exclusively on grid search, we introduce a
continuous optimization algorithm to refine DOA estimates beyond the resolution
of the initial grid. The algorithm is derived from the
majorization-minimization (MM) technique. We derive two surrogate functions,
one quadratic and one linear. Both lead to efficient iterative algorithms that
do not require hyperparameters, such as step size, and ensure that the DOA
estimates never leave the array manifold, without the need for a projection
step. In numerical experiments, we show that the accuracy after a few
iterations of the MM algorithm nearly removes dependency on the resolution of
the initial grid used. We find that the quadratic surrogate function leads to
very fast convergence, but the simplicity of the linear algorithm is very
attractive, and the performance gap small.

Authors
-------

* Robin Scheibler
* Masahito Togami

Experiments
-----------

We work with [anaconda](https://www.anaconda.com/products/individual) to simplify the environement setup.

    conda env create -f environment.yml
    conda activate doamm
    
For the experiment on simulated data, run the following.

    # Generate Fig. 1 and Table 2
    # - this will create the folder ./sim_results/YYYMMDD-HHmmss_experiment1_effect_grid_size
    python ./doa_experiment_para.py ./doa_experiment_para.py ./config_experiment_grid.yml
    # - plot Fig. 1
    python ./make_figure1.py ./sim_results/YYYYmmdd-HHMMSS_experiment1_effect_grid_size
    # - plot Table 2
    python ./make_table_runtime.py ./sim_results/YYYYmmdd-HHMMSS_experiment1_effect_grid_size
    
    # Generate Table 1
    # - this will create the folder ./sim_results/YYYMMDD-HHmmss_experiment1_effect_s
    python ./doa_experiment_para.py ./doa_experiment_para.py ./config_experiment_s.yml
    python ./make_table1.py ./sim_results/YYYYmmdd-HHMMSS_experiment1_effect_s

Run
---

    python ./doa_experiment.py
