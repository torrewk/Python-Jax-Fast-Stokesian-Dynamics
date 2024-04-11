# jfsd: Fast Stokesian Dynamics with JAX

> [!CAUTION]
> This software is still in active development. Changes can and will occur without warning. That said
> we are very grateful for feedback.


This repository contains a Python implementation of the Fast Stokesian Dynamics methods (A. Fiore, J. Swan, 2019, https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/fast-stokesian-dynamics/970BD1B80B43E21CD355C7BAD4644D46) which leverages the library JAX to just-in-time compile and run on GPU the simulation.


## Installation

The required softwares to install/run the jfsd are you need Python version >= 3.8. It is also recommended to run the code on a GPU machine with cuda installed:

1) CUDA (tested with 11.8) - https://developer.nvidia.com/cuda-toolkit
2) cuDNN (tested with 8.9.4) - https://developer.nvidia.com/cudnn
<!-- 3) Python >= 3.8 -->
<!-- 4) JAX - https://jax.readthedocs.io/en/latest/notebooks/quickstart.html -->
<!-- 5) JAX MD - https://github.com/jax-md/jax-md/blob/main/README.md -->


To install the latest version of jfsd

```sh
pip install "git+https://github.com/torrewk/Python-Jax-Fast-Stokesian-Dynamics[test]"
```

This will install all the requirements including JAX from PyPi.

## Test whether the code runs correctly on your machine

From the console type:

```bash
pytest tests
```

## Run your own simulations

Currently, the recommended way is to edit the script `scripts/JFSD.py` and change the simulation parameters to your liking, then run on the command line:

```bash
python scripts/JFSD.py
```

During the simulation, the particles trajectories are saved in a numpy array of shape (N_s, N_p, 3), with N_s the number of frames stored and N_p the number of particles.

