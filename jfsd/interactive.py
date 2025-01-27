import numpy as np

from jfsd import main, utils


def interactive_main():
    """Interactive setup for Stokesian Dynamics simulations of colloidal particles."""
    
    print("This software performs Stokesian Dynamics simulations of colloidal particles in a tricyclic periodic box.\n")

    # User input for simulation parameters
    hydrodynamics_flag = int(
        input(
            "Select hydrodynamics mode:\n"
            "0 - Brownian Dynamics\n"
            "1 - Rotne-Prager-Yamakawa\n"
            "2 - Stokesian Dynamics\n"
            "Enter choice: "
        )
    )
    boundary_flag = int(input("Enter 0 for periodic boundary conditions or 1 for open (infinite) boundaries: "))
    
    num_steps = int(input("Enter the number of simulation timesteps: "))

    writing_period = int(input("Enter the data storing period (how often to save results): "))
    if writing_period > num_steps:
        raise ValueError("Storing period cannot be larger than the total number of simulation timesteps!")

    num_particles = int(input("Enter the number of particles in the box: "))
    box_length_x = float(input("Enter the x-length of the simulation box: "))
    box_length_y = float(input("Enter the y-length of the simulation box: "))
    box_length_z = float(input("Enter the z-length of the simulation box: "))

    time_step = float(input("Enter the simulation time step: "))

    temperature = float(
        input("Enter the temperature (set to 1 for single-particle diffusion coefficient = 1): ")
    )
    interaction_strength = float(input("Enter the interaction strength (in units of thermal energy): ")) * temperature
    interaction_cutoff = float(input("Enter the interaction space cutoff (in units of particle radius): "))

    shear_rate = float(input("Enter the shear rate amplitude: "))
    shear_frequency = float(input("Enter the shear rate frequency: "))
    friction_coefficient = float(input("Enter the friction coefficient: "))
    friction_range = float(input("Enter the friction range (in units of particle radius): "))

    store_stresslet = bool(int(input("Store stresslet? (1 = Yes, 0 = No): ")))
    store_velocity = bool(int(input("Store velocity? (1 = Yes, 0 = No): ")))
    store_orientation = bool(int(input("Store orientation? (1 = Yes, 0 = No): ")))

    # Applied forces
    has_constant_force = bool(int(input("Apply constant (uniform) forces? (1 = Yes, 0 = No): ")))
    constant_forces = np.zeros((num_particles, 3))
    if has_constant_force:
        fx = float(input("Enter constant force in x-direction: "))
        fy = float(input("Enter constant force in y-direction: "))
        fz = float(input("Enter constant force in z-direction: "))
        constant_forces[:, 0] = fx
        constant_forces[:, 1] = fy
        constant_forces[:, 2] = fz

    # Applied torques
    has_constant_torque = bool(int(input("Apply constant (uniform) torques? (1 = Yes, 0 = No): ")))
    constant_torques = np.zeros((num_particles, 3))
    if has_constant_torque:
        tx = float(input("Enter constant torque in x-direction: "))
        ty = float(input("Enter constant torque in y-direction: "))
        tz = float(input("Enter constant torque in z-direction: "))
        constant_torques[:, 0] = tx
        constant_torques[:, 1] = ty
        constant_torques[:, 2] = tz

    print(f"Volume Fraction: {num_particles / (box_length_x * box_length_y * box_length_z) * (4 * np.pi / 3):.4f}")

    # Initial particle configuration
    init_positions_seed = int(
        input("Enter a seed for the initial particle configuration (0 for manual entry or file loading): ")
    )
    if init_positions_seed == 0:
        traj_file = input("Enter .npy filename (leave empty for manual input): ")
        if traj_file:
            positions = np.load(traj_file)
        else:
            positions = np.zeros((num_particles, 3))
            for i in range(num_particles):
                positions[i, 0] = float(input(f"Enter x-coordinate of particle {i + 1}: "))
                positions[i, 1] = float(input(f"Enter y-coordinate of particle {i + 1}: "))
                positions[i, 2] = float(input(f"Enter z-coordinate of particle {i + 1}: "))
    else:
        positions = utils.create_hardsphere_configuration(box_length_x, num_particles, init_positions_seed, 0.001)

    output_filename = input("Enter output filename: ")

    # Random seeds for Brownian motion
    seed_rfd, seed_ffwave, seed_ffreal, seed_nf = 0, 0, 0, 0
    if temperature > 0:
        if hydrodynamics_flag == 2:
            print("Brownian motion requires 4 seeds.")
            seed_rfd = int(input("Enter seed for random finite difference: "))
            seed_ffwave = int(input("Enter seed for wave-space far-field Brownian motion: "))
            seed_ffreal = int(input("Enter seed for real-space far-field Brownian motion: "))
            seed_nf = int(input("Enter seed for real-space near-field Brownian motion: "))
        elif hydrodynamics_flag == 1:
            print("Brownian motion requires 2 seeds.")
            seed_ffwave = int(input("Enter seed for wave-space far-field Brownian motion: "))
            seed_ffreal = int(input("Enter seed for real-space far-field Brownian motion: "))
        elif hydrodynamics_flag == 0:
            print("Brownian motion requires 1 seed.")
            seed_nf = int(input("Enter seed for Brownian forces: "))

    # Start the main simulation
    main.main(
        num_steps,
        writing_period,
        time_step,  # Simulation timestep
        box_length_x,
        box_length_y,
        box_length_z,  # Box sizes
        num_particles,  # Number of particles
        0.5,  # Max box strain
        temperature,  # Thermal energy
        1,  # Colloid radius (keep as 1)
        0.5,  # Ewald parameter (keep as 0.5)
        0.001,  # Error tolerance
        interaction_strength,  # Strength of bonds
        0,  # Buoyancy
        interaction_cutoff,  # Potential cutoff
        positions,
        seed_rfd,
        seed_ffwave,
        seed_ffreal,
        seed_nf,
        shear_rate,
        shear_frequency,
        output_filename,  # Output file name
        store_stresslet,
        store_velocity,
        store_orientation,
        constant_forces,
        constant_torques,
        hydrodynamics_flag,
        boundary_flag,
        0,
        friction_coefficient,
        friction_range,
    )

