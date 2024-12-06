Quickstart
==========

## How to Run Simulations

1 - Create a configuration .toml in files/ (use "files/example_configuration.toml" as reference)

2 - Run the main code, from the project directory, via:

.. code-block:: shell

	jfsd -c files/<your_config_file.toml> -o <directory_name_for_output>

Replace <your_config_file.toml> with the name of the file created in step (1), and <directory_name_for_output> with the name of the folder where you want to store the trajetory data.

3 - To provide initial particle positions to jfsd, run it via:

.. code-block:: shell
	jfsd -c files/<your_config_file.toml> -s <initial_particle_positions.npy> -o <directory_name_for_output>

Replace <initial_particle_positions.npy> with the name of your initial configuration, and modify <your_config_file.toml> to accept initial positions from file (instead of randomly creating it, see 'example_configuration.toml'). Note that this file must be a numpy array of shape (N<sub>p</sub>, 3), with N<sub>p</sub> the number of particles.

## Analyzing Trajectory Data

During the simulation, the particles trajectories,velocities, and stresslets are saved in a numpy array of shape (N<sub>s</sub>, N<sub>p</sub>, N<sub>c</sub>), with N<sub>s</sub> the number of frames stored, N<sub>p</sub> the number of particles and N<sub>c</sub> the number of d.o.f. (3 for trajectories, 6 for velocities, 5 for stresslets). 
**Lengths in the simulation are expressed in units of the particles radius _a_, and momenta in units of _γa_, with unity mass for each particle.**  

## Common/Known Issues

1 - "Unable to load cuPTI"

This issue can occur if the `PATH` or `LD_LIBRARY_PATH` environment variables are not set correctly to include your CUDA installation, especially the `cuPTI` (CUDA Profiling Tools Interface) library, which JAX requires.

To solve this (temporarily), add the following line to your environment:

.. code-block:: shell
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64

For a permanent solution, edit the .bashrc file in your home directory:

.. code-block:: shell
	echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64' >> ~/.bashrc

and apply the changes:

.. code-block:: shell
	source ~/.bashrc

Note that, if cuda is installed in a different location than '/usr/local/', you need to modify the lines above accordingly.
