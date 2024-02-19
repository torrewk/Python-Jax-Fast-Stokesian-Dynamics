This repository contains a Python implementation of the Fast Stokesian Dynamics methods (A. Fiore, J. Swan, 2019, https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/fast-stokesian-dynamics/970BD1B80B43E21CD355C7BAD4644D46) which leverages the library JAX to just-in-time compile and run on GPU the simulation.

The required softwares to install/run the PSE plugin are:

1) CUDA (tested with 11.8) - https://developer.nvidia.com/cuda-toolkit
2) cuDNN (tested with 8.9.4) - https://developer.nvidia.com/cudnn
3) Python3 (tested 3.10.6)
4) JAX - https://jax.readthedocs.io/en/latest/notebooks/quickstart.html
5) JAX MD - https://github.com/jax-md/jax-md/blob/main/README.md

It is recommended to install both JAX and JAX MD via 'pip', for simplicity. 

To run a simulation:

1) Set the simulation parameters (number of particles, box size, number of steps, etc.) by editing the file JFSD.py 

2) Run JFSD.py

During the simulation, the particles trajectories are saved in a numpy array of shape (N_s, N_p, 3), with N_s the number of frames stored and N_p the number of particles.


