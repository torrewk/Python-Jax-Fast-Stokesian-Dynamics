from functools import partial

import jax.numpy as jnp
from jax import Array, jit


def update_box_tilt_factor(
    dt: float, shear_rate_0: float, tilt_factor: float, step: int, omega: float, phase: float = 0
) -> float:
    """Update tilt factors of box (in case of shear)

    Parameters
    ----------
    dt: (float)
        Timestep value
    shear_rate_0: (float)
        Shear rate amplitude
    tilt_factor: (float)
        Box tilt factor before the update
    step: (int)
        Current time step
    omega: (float)
        Angular frequerncy of applied oscillatory shear
    phase: (float)
        Phase of applied oscillatory shear

    Returns
    -------
    tilt_factor

    """
    current_time = step * dt
    tilt_factor = jnp.where(
        omega == 0,
        tilt_factor + dt * shear_rate_0,
        shear_rate_0 * jnp.sin(omega * current_time + phase) / omega,
    )
    if tilt_factor >= 0.5:
        tilt_factor = -0.5 + (tilt_factor - 0.5)
    return tilt_factor


def update_shear_rate(
    dt: float, step: int, shear_rate_0: float, omega: float, phase: float = 0
) -> float:
    """Update shear rate

    Parameters
    ----------
    dt: (float)
        Timestep value
    step: int
        Current time step
    shear_rate_0: (float)
        Shear rate amplitude
    omega: (float)
        Angular frequerncy of applied oscillatory shear
    phase: (float)
        Phase of applied oscillatory shear

    Returns
    -------
    shear_rate

    """
    current_time = step * dt
    shear_rate = shear_rate_0 * jnp.cos(omega * current_time + phase)
    return shear_rate


@partial(jit, static_argnums=[0, 1, 2])
def compute_sheared_grid(
    Nx: int,
    Nz: int,
    Ny: int,
    tilt_factor: float,
    Lx: float,
    Ly: float,
    Lz: float,
    eta: float,
    xisq: float,
) -> Array:
    """Compute wave vectors on a given grid, needed for FFT.

    Parameters
    ----------
    Nx: (int)
        Number of grid points in x direction
    Ny: (int)
        Number of grid points in y direction
    Nz: (int)
        Number of grid points in z direction
    tilt_factor: (float)
        Current box tilt factor
    Lx: (float)
        Box size in x direction
    Ly: (float)
        Box size in y direction
    Lz: (float)
        Box size in z direction
    eta: (float)
        Gaussian splitting parameter
    xisq: (float)
        Squared Ewald split parameter

    Returns
    -------
    gridk

    """
    gridk = jnp.zeros((Nx * Ny * Nz, 4), float)
    # Here create arrays that store the indices that we would have if this function was not vectorized (using for loop instead)
    Nxx = jnp.repeat(jnp.repeat(jnp.arange(Nz), Ny), Nx)
    Nzz = jnp.resize(jnp.arange(Nx), Nx * Ny * Nz)
    Nyy = jnp.resize(jnp.repeat(jnp.arange(Nz), Ny), Nx * Ny * Nz)

    gridk_x = jnp.where(Nxx < (Nx + 1) / 2, Nxx, (Nxx - Nx))
    gridk_y = (
        jnp.where(Nyy < (Ny + 1) / 2, Nyy, (Nyy - Ny)) - tilt_factor * gridk_x * Ly / Lx
    ) / Ly
    gridk_x = gridk_x / Lx
    gridk_z = jnp.where(Nzz < (Nz + 1) / 2, Nzz, (Nzz - Nz)) / Lz
    gridk_x *= 2.0 * jnp.pi
    gridk_y *= 2.0 * jnp.pi
    gridk_z *= 2.0 * jnp.pi

    # k dot k and fourth component (contains the scaling factor of the FFT)
    k_sq = gridk_x * gridk_x + gridk_y * gridk_y + gridk_z * gridk_z
    gridk_w = jnp.where(
        k_sq > 0,
        6.0
        * jnp.pi
        * (1.0 + k_sq / 4.0 / xisq)
        * jnp.exp(-(1 - eta) * k_sq / 4.0 / xisq)
        / (k_sq)
        / (Nx * Ny * Nz),
        0,
    )

    # store the results
    gridk = gridk.at[:, 0].set(gridk_x)
    gridk = gridk.at[:, 1].set(gridk_y)
    gridk = gridk.at[:, 2].set(gridk_z)
    gridk = gridk.at[:, 3].set(gridk_w)

    # Reshape to match the gridded quantities
    gridk = jnp.reshape(gridk, (Nx, Ny, Nz, 4))
    gridk = jnp.array(gridk)

    return gridk
