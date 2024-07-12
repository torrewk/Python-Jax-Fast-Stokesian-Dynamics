# jfsd: a Python implementation of Fast Stokesian Dynamics with Jax

This repository contains a Python implementation of the Fast Stokesian Dynamics methods (A. Fiore, J. Swan, 2019, https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/fast-stokesian-dynamics/970BD1B80B43E21CD355C7BAD4644D46) which leverages the library JAX to just-in-time compile and run on GPU the simulation.

## Pre-requisites:
- cuda 11.8 (https://developer.nvidia.com/cuda-11-8-0-download-archive)
- cuDNN 8.6 for cuda 11 (https://developer.nvidia.com/rdp/cudnn-archive)
- Python >= 3.10

## Set up work (virtual) environment:

```bash
git clone https://github.com/UtrechtUniversity/ibridges-servers-uu.git
```

## Go into the directory of the project and type the command:
```bash
python3 -m venv .venv
```


## activate the environment:
```bash
source .venv/bin/activate
```

## Install correct version of jaxlib
```bash
pip install jaxlib==0.4.14+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Install jfsd and rest of dependencies

```bash
pip install ".[test]"
```

		
## Reboot the environment (needed fore pytest to work):
```bash
deactivate && source .venv/bin/activate
```


## Run the JFSD unit tests
```bash
pytest test_class.py
```
		
		
## If no test fails, run the main code via:
```bash
python jfsd/JFSD.py
```


During the simulation, the particles trajectories,velocities, and stresslets are saved in a numpy array of shape (N_s, N_p, N_c), with N_s the number of frames stored, N_p the number of particles and N_c the number of d.o.f. (3 for trajectories, 6 for velocities, 5 for stresslets).


