from functools import partial

import jax.numpy as jnp
from jax import Array, jit

def update_box_tilt_factor(
    time_step: float, shear_rate_amplitude: float, tilt_factor: float,
    step: int, angular_frequency: float, phase: float = 0
) -> float:
    """Update tilt factors of box (in case of shear)

    Parameters
    ----------
    time_step: (float)
        Timestep value
    shear_rate_amplitude: (float)
        Shear rate amplitude
    tilt_factor: (float)
        Box tilt factor before the update
    step: (int)
        Current time step
    angular_frequency: (float)
        Angular frequency of applied oscillatory shear
    phase: (float)
        Phase of applied oscillatory shear

    Returns
    -------
    tilt_factor

    """
    current_time = step * time_step
    tilt_factor = jnp.where(
        angular_frequency == 0,
        tilt_factor + time_step * shear_rate_amplitude,
        shear_rate_amplitude * jnp.sin(angular_frequency * current_time + phase) / angular_frequency,
    )
    if tilt_factor >= 0.5:
        tilt_factor = -0.5 + (tilt_factor - 0.5)
    return tilt_factor

def update_shear_rate(
    time_step: float, step: int, shear_rate_amplitude: float, angular_frequency: float, phase: float = 0
) -> float:
    """Update shear rate

    Parameters
    ----------
    time_step: (float)
        Timestep value
    step: (int)
        Current time step
    shear_rate_amplitude: (float)
        Shear rate amplitude
    angular_frequency: (float)
        Angular frequency of applied oscillatory shear
    phase: (float)
        Phase of applied oscillatory shear

    Returns
    -------
    shear_rate

    """
    current_time = step * time_step
    shear_rate = shear_rate_amplitude * jnp.cos(angular_frequency * current_time + phase)
    return shear_rate

@partial(jit, static_argnums=[0, 1, 2])
def compute_sheared_grid(
    num_x: int,
    num_y: int,
    num_z: int,
    tilt_factor: float,
    box_length_x: float,
    box_length_y: float,
    box_length_z: float,
    gaussian_splitting: float,
    ewald_squared: float,
) -> Array:
    """Compute wave vectors on a given grid, needed for FFT.

    Parameters
    ----------
    num_x: (int)
        Number of grid points in x direction
    num_y: (int)
        Number of grid points in y direction
    num_z: (int)
        Number of grid points in z direction
    tilt_factor: (float)
        Current box tilt factor
    box_length_x: (float)
        Box size in x direction
    box_length_y: (float)
        Box size in y direction
    box_length_z: (float)
        Box size in z direction
    gaussian_splitting: (float)
        Gaussian splitting parameter
    ewald_squared: (float)
        Squared Ewald split parameter

    Returns
    -------
    grid_k

    """
    grid_k = jnp.zeros((num_x * num_y * num_z, 4), float)
    
    index_x = jnp.repeat(jnp.repeat(jnp.arange(num_z), num_y), num_x)
    index_z = jnp.resize(jnp.arange(num_x), num_x * num_y * num_z)
    index_y = jnp.resize(jnp.repeat(jnp.arange(num_z), num_y), num_x * num_y * num_z)

    grid_k_x = jnp.where(index_x < (num_x + 1) / 2, index_x, (index_x - num_x))
    grid_k_y = (
        jnp.where(index_y < (num_y + 1) / 2, index_y, (index_y - num_y))
        - tilt_factor * grid_k_x * box_length_y / box_length_x
    ) / box_length_y
    grid_k_x = grid_k_x / box_length_x
    grid_k_z = jnp.where(index_z < (num_z + 1) / 2, index_z, (index_z - num_z)) / box_length_z
    grid_k_x *= 2.0 * jnp.pi
    grid_k_y *= 2.0 * jnp.pi
    grid_k_z *= 2.0 * jnp.pi

    k_sq = grid_k_x * grid_k_x + grid_k_y * grid_k_y + grid_k_z * grid_k_z
    grid_k_w = jnp.where(
        k_sq > 0,
        6.0
        * jnp.pi
        * (1.0 + k_sq / 4.0 / ewald_squared)
        * jnp.exp(-(1 - gaussian_splitting) * k_sq / 4.0 / ewald_squared)
        / (k_sq)
        / (num_x * num_y * num_z),
        0,
    )

    grid_k = grid_k.at[:, 0].set(grid_k_x)
    grid_k = grid_k.at[:, 1].set(grid_k_y)
    grid_k = grid_k.at[:, 2].set(grid_k_z)
    grid_k = grid_k.at[:, 3].set(grid_k_w)

    grid_k = jnp.reshape(grid_k, (num_x, num_y, num_z, 4))
    grid_k = jnp.array(grid_k)

    return grid_k