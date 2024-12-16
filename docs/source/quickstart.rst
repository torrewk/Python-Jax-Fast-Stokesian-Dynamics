Quickstart
==========

How to Run Simulations
-----------------------

1. Create a configuration `.toml` (use `files/example_configuration.toml` as a reference).

2. Run the main code from the project directory via:

   .. code-block:: shell

       jfsd --config files/<your_config_file.toml> -output <directory_name_for_output>

   Replace `<your_config_file.toml>` with the name of the file created in step (1), and `<directory_name_for_output>` with the name of the folder where you want to store the trajectory data.

3. To provide initial particle positions to `jfsd`, run it via:

   .. code-block:: shell

       jfsd --config files/<your_config_file.toml> --start-configuration <initial_particle_positions.npy> -output <directory_name_for_output>

   Replace `<initial_particle_positions.npy>` with the name of your initial configuration, and modify `<your_config_file.toml>` to accept initial positions from a file (instead of randomly creating it, see `example_configuration.toml`). Note that this file must be a NumPy array of shape `(N_p, 3)`, where `N_p` is the number of particles.

Analyzing Trajectory Data
-------------------------

During the simulation, the particles' trajectories, velocities, and stresslets are saved in a NumPy array of shape `(N_s, N_p, N_c)`, where:

- `N_s` is the number of frames stored,
- `N_p` is the number of particles,
- `N_c` is the number of degrees of freedom (3 for trajectories, 6 for velocities, 5 for stresslets).

**Lengths in the simulation are expressed in units of the particle radius (_a_), and momenta in units of _\u03b3a_, with unity mass for each particle.**

Common/Known Issues
-------------------

1. **"Unable to load cuPTI"**

   This issue can occur if the `PATH` or `LD_LIBRARY_PATH` environment variables are not set correctly to include your CUDA installation, especially the `cuPTI` (CUDA Profiling Tools Interface) library, which JAX requires.

   To solve this temporarily, add the following line to your environment:

   .. code-block:: shell

       export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64

   For a permanent solution, edit the `.bashrc` file in your home directory:

   .. code-block:: shell

       echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64' >> ~/.bashrc

   Apply the changes:

   .. code-block:: shell

       source ~/.bashrc

   Note: If CUDA is installed in a different location than `/usr/local/`, modify the lines above accordingly.
