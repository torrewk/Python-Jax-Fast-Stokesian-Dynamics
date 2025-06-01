import jax.numpy as jnp


# jit with static shift_fn and num_particles
from jax import jit
from functools import partial


@partial(jit, static_argnames=('shift_fn', 'num_particles'))
def update_positions(
    shear_rate,
    positions,
    net_vel,
    time_step,
    shift_fn,
    lx,
    ly,
    lz,
    num_particles,
):
    """Update particle positions and neighbor lists

    Parameters
    ----------
    shear_rate: (float)
        Shear rate at current time step
    positions: (float)
        Array (num_particles,3) of particles positions
    net_vel: (float)
        Array (6*num_particles) of linear/angular velocities relative to the background flow
    time_step: (float)
        Timestep used to advance positions
    shift_fn: (function)
        Function to apply shifts with boundary conditions 
    lx, ly, lz: (float)
        Box dimensions
    num_particles: (int)
        Number of particles in the system

    Returns
    -------
    positions (in-place update)

    """
    box_half = jnp.array([lx, ly, lz]) / 2
    
    # Define array of displacement r(t+time_step)-r(t)
    dR = jnp.zeros((num_particles, 3), float)
    # Compute actual displacement due to velocities (relative to background flow)
    dR = dR.at[:, 0].set(time_step * net_vel[(0)::6])
    dR = dR.at[:, 1].set(time_step * net_vel[(1)::6])
    dR = dR.at[:, 2].set(time_step * net_vel[(2)::6])
    # Apply displacement and compute wrapped shift (Lees Edwards boundary conditions)
    positions = (
        shift_fn(positions + box_half, dR) - box_half
    )

    # Define array of displacement r(t+time_step)-r(t) (this time for displacement given by background flow)
    dR = jnp.zeros((num_particles, 3), float)
    dR = dR.at[:, 0].set(
        time_step * shear_rate * positions[:, 1]
    )  # Assuming y:gradient direction, x:background flow direction
    positions = (
        shift_fn(positions + box_half, dR) - box_half
    )  # Apply shift
    
    return positions