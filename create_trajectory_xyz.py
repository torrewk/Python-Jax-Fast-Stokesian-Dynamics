import numpy as np

def save_trajectory_to_xyz(filename, trajectory):
    num_atoms = trajectory.shape[1]  # Assuming the trajectory has dimensions (num_steps, num_atoms, 3)
    num_steps = trajectory.shape[0]

    with open(filename, 'w') as file:
        for step in range(num_steps):
            file.write(f"{num_atoms}\n")
            file.write(f"Step: {step}\n")

            # Write particle positions to file for the current step
            for atom in range(num_atoms):
                file.write(f"P {trajectory[step, atom, 0]} {trajectory[step, atom, 1]} {trajectory[step, atom, 2]}\n")

# Assuming 'trajectory_data' is a NumPy array with shape (num_steps, num_atoms, 3)
trajectory_data = np.load('DancingSpheres.npy')

# Save trajectory to XYZ file
save_trajectory_to_xyz("trajectory.xyz", trajectory_data)


