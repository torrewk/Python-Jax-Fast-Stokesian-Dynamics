# jfsd: a Python implementation of Fast Stokesian Dynamics with Jax

This repository contains a Python implementation of the Fast Stokesian Dynamics methods (A. Fiore, J. Swan, 2019, https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/fast-stokesian-dynamics/970BD1B80B43E21CD355C7BAD4644D46) which leverages the library JAX to just-in-time compile and run on GPU the simulation. The accuracy of hydrodynamic interactions can be scaled down in order to obtain Rotne-Prager-Yamakawa (up to 2-body effects, and no lubrication) or Brownian Dynamics (only 1-body effects). Boundary conditions can be switch from periodic to open, and the average height of asperities on particles can be specified, enabling simulations of rough colloids. 

## Installation Guide  

### Pre-requisites:  
- **CUDA** (tested with version **11.8**): [Download](https://developer.nvidia.com/cuda-11-8-0-download-archive)  
- **cuDNN** (tested with **8.6** for CUDA 11): [Download](https://developer.nvidia.com/rdp/cudnn-archive)  
- **Python** ≥ 3.9  

### Installation Options  

You can install **JFSD** in two ways:  

1. **Via pip (Stable Release)**:  
   The latest stable version of JFSD can be installed directly from PyPI:  

   ```bash
   pip install jfsd
   ```  
   **Note**: This may not include the latest developments in the main branch.  

2. **From Source (Latest Development Version)**:  

   #### 1 - Clone the repository  
   ```bash
   git clone https://github.com/torrewk/Python-Jax-Fast-Stokesian-Dynamics.git
   cd Python-Jax-Fast-Stokesian-Dynamics
   ```  

   #### 2 - Create and activate a virtual environment  
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```  

   #### 3 - Install the required JAX library  
   ```bash
   pip install jaxlib==0.4.17+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   ```  

   #### 4 - Install JFSD and dependencies  
   ```bash
   pip install ".[test]"
   ```  

   #### 5 - Reboot the environment (needed for pytest to work)  
   ```bash
   deactivate && source .venv/bin/activate
   ```  

   #### 6 - Run unit tests to verify the installation  
   ```bash
   pytest tests/test_class.py
   ```  

For the latest updates and bug fixes, using the **source installation** is recommended.
		
## How to Run Simulations

### 1 - Create a configuration .toml in files/ (use "files/example_configuration.toml" as reference)	
### 2a - Run the main code, from the project directory, via:
```bash
jfsd -c files/<your_config_file.toml> -o <directory_name_for_output>
```
Replace <your_config_file.toml> with the name of the file created in step (1), and <directory_name_for_output> with the name of the folder where you want to store the trajetory data.
### 2b - To provide initial particle positions to jfsd, run it via:
```bash
jfsd -c files/<your_config_file.toml> -s <initial_particle_positions.npy> -o <directory_name_for_output>
```
Replace <initial_particle_positions.npy> with the name of your initial configuration, and modify <your_config_file.toml> to accept initial positions from file (instead of randomly creating it, see 'example_configuration.toml'). Note that this file must be a numpy array of shape (N<sub>p</sub>, 3), with N<sub>p</sub> the number of particles.


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

## Acknowledgment

The authors acknowledge the Dutch Research Council (NWO) for funding through OCENW.KLEIN.354, as well as the International Fine Particle Research Institute (IFPRI) for funding through collaboration grant CRR-118-01.

We developed JFSD as a way to give back to the fluid dynamics community, whose contributions have shaped our own research. We hope this implementation helps keep the Fast Stokesian Dynamics framework accessible for future studies, continuing the legacy of Fiore and Swan’s excellent algorithms.

We are also grateful to Luca Leone, Athanasios Machas, and Dimos Aslanis for their valuable insights, which improved our understanding of FSD, Google JAX, and the user/developer experience, respectively.

## License

This project is licensed under the Apache License, Version 2.0. See the `LICENSE` file for the full terms and conditions. For a concise summary, refer to `LICENSE_SHORT`.

