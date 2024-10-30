# jfsd: a Python implementation of Fast Stokesian Dynamics with Jax

This repository contains a Python implementation of the Fast Stokesian Dynamics methods (A. Fiore, J. Swan, 2019, https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/fast-stokesian-dynamics/970BD1B80B43E21CD355C7BAD4644D46) which leverages the library JAX to just-in-time compile and run on GPU the simulation. The accuracy of hydrodynamic interactions can be scaled down in order to obtain Rotne-Prager-Yamakawa (up to 2-body effects, and no lubrication) or Brownian Dynamics (only 1-body effects). Moreover, the average height of asperities on particles can be specified, enabling simulations of rough colloids. 

## Installation Guide

### Pre-requisites:
- cuda (tested with version 11.8, https://developer.nvidia.com/cuda-11-8-0-download-archive)
- cuDNN (tested with 8.6 for cuda 11, https://developer.nvidia.com/rdp/cudnn-archive)
- Python >= 3.9

### 1 - Set up work (virtual) environment:

```bash
git clone https://github.com/torrewk/Python-Jax-Fast-Stokesian-Dynamics.git
```

### 2 - Go into the directory of the project and type the command:
```bash
python3 -m venv .venv
```


### 3 - activate the environment:
```bash
source .venv/bin/activate
```

### 4 - Install correct version of jaxlib
```bash
pip install jaxlib==0.4.17+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### 5 - Install jfsd and rest of dependencies

```bash
pip install ".[test]"
```

		
### 6 - Reboot the environment (needed fore pytest to work):
```bash
deactivate && source .venv/bin/activate
```


### 7 - Run the JFSD unit tests
```bash
pytest tests/test_class.py
```
		
## How to Run Simulations

### 1 - Create a configuration .toml in files/ (use "files/example_configuration.toml" as reference)	
### 2 - Run the main code, from the project directory, via:
```bash
jfsd -c files/<your_config_file.toml> -o <directory_name_for_output>
```
Replace <your_config_file.toml> with the name of the file created in step (1), and <directory_name_for_output> with the name of the folder where you want to store the trajetory data.


## Analyzing Trajectory Data
During the simulation, the particles trajectories,velocities, and stresslets are saved in a numpy array of shape (N<sub>s</sub>, N<sub>p</sub>, N<sub>c</sub>), with N<sub>s</sub> the number of frames stored, N<sub>p</sub> the number of particles and N<sub>c</sub> the number of d.o.f. (3 for trajectories, 6 for velocities, 5 for stresslets). 
**Lengths in the simulation are expressed in units of the particles radius _a_, and momenta in units of _γa_, with unity mass for each particle.**  

## Common/Known Issues

### "Unable to load cuPTI"

This issue can occur if the `PATH` or `LD_LIBRARY_PATH` environment variables are not set correctly to include your CUDA installation, especially the `cuPTI` (CUDA Profiling Tools Interface) library, which JAX requires.

To solve this (temporarily), add the following line to your environment:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64
```
For a permanent solution, edit the .bashrc file in your home directory:

```bash
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64' >> ~/.bashrc
```
and apply the changes:

```bash
source ~/.bashrc
```
Note that, if cuda is installed in a different location than '/usr/local/', you need to modify the lines above accordingly.
