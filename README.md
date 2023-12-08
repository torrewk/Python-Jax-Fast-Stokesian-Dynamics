This repository contains a Python implementation of the Fast Stokesian Dynamics methods (A. Fiore, J. Swan, 2019, https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/fast-stokesian-dynamics/970BD1B80B43E21CD355C7BAD4644D46) which leverages the library JAX to just-in-time compile and run on GPU the simulation.




The required softwares to install/run the PSE plugin are:

1) CUDA (tested with 11.8) - https://developer.nvidia.com/cuda-toolkit
2) cuDNN (tested with 8.9.4) - https://developer.nvidia.com/cudnn
3) Python3 (tested 3.10.6)
4) JAX - https://jax.readthedocs.io/en/latest/notebooks/quickstart.html
5) JAX MD - https://github.com/jax-md/jax-md/blob/main/README.md

It is recommended to install both JAX and JAX MD via 'pip', for simplicity. 




To run a simulation use the script main.py, from which the user can choose box size, shear rate, temperature, number of particles, and all the needed input parameters. 
During the simulation, the trajectory is saved in a numpy array of shape (N_s, N_p, 3), with N_s the number of frames stored and N_p the number of particles.
To visually inspect the trajectory, use the script 'create_trajectory_xyz.py' to convert the numpy array (containing the trajectory) into a .xyz file, which can be used to render the actual particle motion (using softwares like vmd, https://www.ks.uiuc.edu/Research/vmd/, or similar). 



The script main.py is already set up to run the 'dancing sphere' test, where 3 spheres sediment vertically in a newtonian fluid. The file 'DancingSpheres_reference.npy' contains the original points that can be used as comparison. Be aware that, because of the periodicity in all directions, the trajectory need to be wrapped correctly to see physical results. In any case, the values of the coordinates should be the same as the given reference, regardless of the periodicity.




The software is still in a its 'alpha' version. Thus, expect bugs and/or large room for improvement. 


TO DO:

-Complete implementation of shear (at the moment boundary conditions are not set properly for shear).

-Optimize user interface and add/improve comments in the code.

-Implement a more efficient method to compute the matrix square root operation (first should check if the bottleneck is here, if not, this is of secondary importance).

-Avoid use of functions from JAX_MD library, so to reduce future compatibility issues (also secondary importance for now).

-Optimize memory usage in the code (at the moment might not be able to simulate more ~1000 particles because of this, the problem can be surely solved by a more efficient use/recycling of the memory during the simulation).

-Optimize the base python code to reduce the compilation time (for now, can be up to 1 minute, maybe more depending on the machine used).

-Create unit tests (the 'dancing sphere test' will be one of the tests in it).
