This repository contains a Python implementation of the Fast Stokesian Dynamics methods (A. Fiore, J. Swan, 2019, https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/fast-stokesian-dynamics/970BD1B80B43E21CD355C7BAD4644D46) which leverages the library JAX to just-in-time compile and run on GPU the simulation.

Pre-requisites:
-cuda 11.8 (https://developer.nvidia.com/cuda-11-8-0-download-archive)
-cuDNN 8.6 for cuda 11 (https://developer.nvidia.com/rdp/cudnn-archive)
-Python >= 3.10

Set up work (virtual) environment:

1) download JFSD

2) go into the directory of the project and type the command:
		python3.10 -m venv .venv 

3) activate the environment:
		source .venv/bin/activate
		
4) (from the virtual environment) install the required packages (it's important to match these versions):
		pip install scipy==1.10.1
		pip install freud-analysis
		pip install jraph==0.0.6.dev0
		pip install absl-py==2.1.0

5) Install jax via pip, with support for GPU (JFSD does not work on CPU at the moment):
		pip install jax==0.4.14
		pip install jaxlib==0.4.14+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

6) Install pytest inside the virtual environment:
		pip install pytest
		
7) Reboot the environment (needed fore pytest to work):
		deactivate && source .venv/bin/activate
	
8) Run the JFSD unit tests:
		pytest test_class.py
		
		
If no test fails, run the main code via:
		python JFSD.py


During the simulation, the particles trajectories,velocities, and stresslets are saved in a numpy array of shape (N_s, N_p, N_c), with N_s the number of frames stored, N_p the number of particles and N_c the number of d.o.f. (3 for trajectories, 6 for velocities, 5 for stresslets).


