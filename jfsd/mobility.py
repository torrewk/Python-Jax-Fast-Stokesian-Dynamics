from functools import partial
from jax import jit, Array
import jax.numpy as jnp
from jax.typing import ArrayLike


@partial(jit, static_argnums=[0, 1, 2, 3, 4])
def GeneralizedMobility_periodic(
    N: int,
    Nx: int,
    Ny: int,
    Nz: int,
    gaussP: int,
    gridk: ArrayLike,
    m_self: ArrayLike,
    all_indices_x: ArrayLike,
    all_indices_y: ArrayLike,
    all_indices_z: ArrayLike,
    gaussian_grid_spacing1: ArrayLike,
    gaussian_grid_spacing2: ArrayLike,
    r: ArrayLike,
    indices_i: ArrayLike,
    indices_j: ArrayLike,
    f1: ArrayLike,
    f2: ArrayLike,
    g1: ArrayLike,
    g2: ArrayLike,
    h1: ArrayLike,
    h2: ArrayLike,
    h3: ArrayLike,
    generalized_forces: ArrayLike,
) -> Array:
    """Compute the matrix-vector product of the grandmobility matrix with a generalized force vector (and stresslet), in periodic boundary conditions.

    Parameters
    ----------
    N: (int)
        Number of particles
    Nx: (int)
        Number of grid points in x direction
    Ny: (int)
        Number of grid points in y direction
    Nz: (int)
        Number of grid points in z direction
    gaussP: (int)
        Gaussian support size for wave space calculation
    gridk: (float)
        Array (Nx,Ny,Nz,4) containing wave vectors and scaling factors for far-field wavespace calculation
    m_self: (float)
        Array (,2) containing mobility self contributions
    all_indices_x: (int)
        Array (,N*gaussP*gaussP*gaussP) containing all the x-indices of wave grid points overlapping with each particle Gaussian support
    all_indices_y: (int)
        Array (,N*gaussP*gaussP*gaussP) containing all the y-indices of wave grid points overlapping with each particle Gaussian support
    all_indices_z: (int)
        Array (,N*gaussP*gaussP*gaussP) containing all the z-indices of wave grid points overlapping with each particle Gaussian support
    gaussian_grid_spacing1: (float)
        Array (,gaussP*gaussP*gaussP) containing scaled distances from support center to each gridpoint in the gaussian support (for FFT)
    gaussian_grid_spacing2: (float)
        Array (,gaussP*gaussP*gaussP) containing scaled distances from support center to each gridpoint in the gaussian support (for inverse FFT)
    r: (float)
        Array (n_pair_ff,3) containing units vectors connecting each pair of particles in the far-field neighbor list
    indices_i: (int)
        Array (,n_pair_ff) of indices of first particle in neighbor list pairs
    indices_j: (int)
        Array (,n_pair_ff) of indices of second particle in neighbor list pairs
    f1: (float)
        Array (,n_pair_ff) containing mobility scalar function evaluated for the current particle configuration
    f2: (float)
        Array (,n_pair_ff) containing mobility scalar function evaluated for the current particle configuration
    g1: (float)
        Array (,n_pair_ff) containing mobility scalar function evaluated for the current particle configuration
    g2: (float)
        Array (,n_pair_ff) containing mobility scalar function evaluated for the current particle configuration
    h1: (float)
        Array (,n_pair_ff) containing mobility scalar function evaluated for the current particle configuration
    h2: (float)
        Array (,n_pair_ff) containing mobility scalar function evaluated for the current particle configuration
    h3: (float)
        Array (,n_pair_ff) containing mobility scalar function evaluated for the current particle configuration
    generalized_forces: (float)
        Array (,11*N) containing input generalized forces (force/torque/stresslet)

    Returns
    -------
    generalized_velocities (linear/angular velocities and rateOfStrain)

    """

    # Helper function
    def swap_real_imag(cplx_arr: ArrayLike) -> Array:
        """Perform an operation on a complex number.

        Take a complex number as input and return a complex number with real part equal to (minus) the imaginary part of the input,
        and an imaginary part equal to the real part of the input.

        Parameters
        ----------
        cplx_arr: (complex)
            Array of complex values

        Returns
        -------
        -jnp.imag(cplx_arr) + 1j * jnp.real(cplx_arr)

        """
        return -jnp.imag(cplx_arr) + 1j * jnp.real(cplx_arr)

    # Get forces,torques,couplets from generalized forces (3*N vector: f1x,f1y,f1z, ... , fNx,fNy,fNz and same for torque, while couplet is 5N)
    forces = jnp.zeros(3 * N)
    forces = forces.at[0::3].set(generalized_forces.at[0 : (6 * N) : 6].get())
    forces = forces.at[1::3].set(generalized_forces.at[1 : (6 * N) : 6].get())
    forces = forces.at[2::3].set(generalized_forces.at[2 : (6 * N) : 6].get())
    torques = jnp.zeros(3 * N)
    torques = torques.at[0::3].set(generalized_forces.at[3 : (6 * N) : 6].get())
    torques = torques.at[1::3].set(generalized_forces.at[4 : (6 * N) : 6].get())
    torques = torques.at[2::3].set(generalized_forces.at[5 : (6 * N) : 6].get())
    stresslet = jnp.zeros(5 * N)
    stresslet = stresslet.at[0::5].set(generalized_forces.at[(6 * N + 0) :: 5].get())  # Sxx
    stresslet = stresslet.at[1::5].set(generalized_forces.at[(6 * N + 1) :: 5].get())  # Sxy
    stresslet = stresslet.at[2::5].set(generalized_forces.at[(6 * N + 2) :: 5].get())  # Sxz
    stresslet = stresslet.at[3::5].set(generalized_forces.at[(6 * N + 3) :: 5].get())  # Syz
    stresslet = stresslet.at[4::5].set(generalized_forces.at[(6 * N + 4) :: 5].get())  # Syy

    # Get 'couplet' from generalized forces (8*N vector)
    couplets = jnp.zeros(8 * N)
    couplets = couplets.at[::8].set(stresslet.at[::5].get())  # C[0] = S[0]
    couplets = couplets.at[1::8].set(
        stresslet.at[1::5].get() + torques.at[2::3].get() * 0.5
    )  # C[1] = S[1] + L[2]/2
    couplets = couplets.at[2::8].set(
        stresslet.at[2::5].get() - torques.at[1::3].get() * 0.5
    )  # C[2] = S[2] - L[1]/2
    couplets = couplets.at[3::8].set(
        stresslet.at[3::5].get() + torques.at[::3].get() * 0.5
    )  # C[3] = S[3] + L[0]/2
    couplets = couplets.at[4::8].add(stresslet.at[4::5].get())  # C[4] = S[4]
    couplets = couplets.at[5::8].set(
        stresslet.at[1::5].get() - torques.at[2::3].get() * 0.5
    )  # C[5] = S[1] - L[2]/2
    couplets = couplets.at[6::8].set(
        stresslet.at[2::5].get() + torques.at[1::3].get() * 0.5
    )  # C[6] = S[2] + L[1]/2
    couplets = couplets.at[7::8].set(
        stresslet.at[3::5].get() - torques.at[::3].get() * 0.5
    )  # C[7] = S[3] - L[0]/2

    ##########################################################################################################################################
    ######################################## WAVE SPACE CONTRIBUTION #########################################################################
    ##########################################################################################################################################

    # Create Grids for current iteration
    gridX = jnp.zeros((Nx, Ny, Nz))
    gridY = jnp.zeros((Nx, Ny, Nz))
    gridZ = jnp.zeros((Nx, Ny, Nz))
    gridXX = jnp.zeros((Nx, Ny, Nz))
    gridXY = jnp.zeros((Nx, Ny, Nz))
    gridXZ = jnp.zeros((Nx, Ny, Nz))
    gridYX = jnp.zeros((Nx, Ny, Nz))
    gridYY = jnp.zeros((Nx, Ny, Nz))
    gridYZ = jnp.zeros((Nx, Ny, Nz))
    gridZX = jnp.zeros((Nx, Ny, Nz))
    gridZY = jnp.zeros((Nx, Ny, Nz))

    gridX = gridX.at[all_indices_x, all_indices_y, all_indices_z].add(
        jnp.ravel(
            (
                jnp.swapaxes(
                    jnp.swapaxes(
                        jnp.swapaxes(
                            gaussian_grid_spacing1
                            * jnp.resize(forces.at[0::3].get(), (gaussP, gaussP, gaussP, N)),
                            3,
                            2,
                        ),
                        2,
                        1,
                    ),
                    1,
                    0,
                )
            )
        )
    )
    gridY = gridY.at[all_indices_x, all_indices_y, all_indices_z].add(
        jnp.ravel(
            (
                jnp.swapaxes(
                    jnp.swapaxes(
                        jnp.swapaxes(
                            gaussian_grid_spacing1
                            * jnp.resize(forces.at[1::3].get(), (gaussP, gaussP, gaussP, N)),
                            3,
                            2,
                        ),
                        2,
                        1,
                    ),
                    1,
                    0,
                )
            )
        )
    )
    gridZ = gridZ.at[all_indices_x, all_indices_y, all_indices_z].add(
        jnp.ravel(
            (
                jnp.swapaxes(
                    jnp.swapaxes(
                        jnp.swapaxes(
                            gaussian_grid_spacing1
                            * jnp.resize(forces.at[2::3].get(), (gaussP, gaussP, gaussP, N)),
                            3,
                            2,
                        ),
                        2,
                        1,
                    ),
                    1,
                    0,
                )
            )
        )
    )
    gridXX = gridXX.at[all_indices_x, all_indices_y, all_indices_z].add(
        jnp.ravel(
            (
                jnp.swapaxes(
                    jnp.swapaxes(
                        jnp.swapaxes(
                            gaussian_grid_spacing1
                            * jnp.resize(couplets.at[0::8].get(), (gaussP, gaussP, gaussP, N)),
                            3,
                            2,
                        ),
                        2,
                        1,
                    ),
                    1,
                    0,
                )
            )
        )
    )
    gridXY = gridXY.at[all_indices_x, all_indices_y, all_indices_z].add(
        jnp.ravel(
            (
                jnp.swapaxes(
                    jnp.swapaxes(
                        jnp.swapaxes(
                            gaussian_grid_spacing1
                            * jnp.resize(couplets.at[1::8].get(), (gaussP, gaussP, gaussP, N)),
                            3,
                            2,
                        ),
                        2,
                        1,
                    ),
                    1,
                    0,
                )
            )
        )
    )
    gridXZ = gridXZ.at[all_indices_x, all_indices_y, all_indices_z].add(
        jnp.ravel(
            (
                jnp.swapaxes(
                    jnp.swapaxes(
                        jnp.swapaxes(
                            gaussian_grid_spacing1
                            * jnp.resize(couplets.at[2::8].get(), (gaussP, gaussP, gaussP, N)),
                            3,
                            2,
                        ),
                        2,
                        1,
                    ),
                    1,
                    0,
                )
            )
        )
    )
    gridYZ = gridYZ.at[all_indices_x, all_indices_y, all_indices_z].add(
        jnp.ravel(
            (
                jnp.swapaxes(
                    jnp.swapaxes(
                        jnp.swapaxes(
                            gaussian_grid_spacing1
                            * jnp.resize(couplets.at[3::8].get(), (gaussP, gaussP, gaussP, N)),
                            3,
                            2,
                        ),
                        2,
                        1,
                    ),
                    1,
                    0,
                )
            )
        )
    )
    gridYY = gridYY.at[all_indices_x, all_indices_y, all_indices_z].add(
        jnp.ravel(
            (
                jnp.swapaxes(
                    jnp.swapaxes(
                        jnp.swapaxes(
                            gaussian_grid_spacing1
                            * jnp.resize(couplets.at[4::8].get(), (gaussP, gaussP, gaussP, N)),
                            3,
                            2,
                        ),
                        2,
                        1,
                    ),
                    1,
                    0,
                )
            )
        )
    )
    gridYX = gridYX.at[all_indices_x, all_indices_y, all_indices_z].add(
        jnp.ravel(
            (
                jnp.swapaxes(
                    jnp.swapaxes(
                        jnp.swapaxes(
                            gaussian_grid_spacing1
                            * jnp.resize(couplets.at[5::8].get(), (gaussP, gaussP, gaussP, N)),
                            3,
                            2,
                        ),
                        2,
                        1,
                    ),
                    1,
                    0,
                )
            )
        )
    )
    gridZX = gridZX.at[all_indices_x, all_indices_y, all_indices_z].add(
        jnp.ravel(
            (
                jnp.swapaxes(
                    jnp.swapaxes(
                        jnp.swapaxes(
                            gaussian_grid_spacing1
                            * jnp.resize(couplets.at[6::8].get(), (gaussP, gaussP, gaussP, N)),
                            3,
                            2,
                        ),
                        2,
                        1,
                    ),
                    1,
                    0,
                )
            )
        )
    )
    gridZY = gridZY.at[all_indices_x, all_indices_y, all_indices_z].add(
        jnp.ravel(
            (
                jnp.swapaxes(
                    jnp.swapaxes(
                        jnp.swapaxes(
                            gaussian_grid_spacing1
                            * jnp.resize(couplets.at[7::8].get(), (gaussP, gaussP, gaussP, N)),
                            3,
                            2,
                        ),
                        2,
                        1,
                    ),
                    1,
                    0,
                )
            )
        )
    )

    # Apply FFT
    gridX = jnp.fft.fftn(gridX)
    gridY = jnp.fft.fftn(gridY)
    gridZ = jnp.fft.fftn(gridZ)
    gridXX = jnp.fft.fftn(gridXX)
    gridXY = jnp.fft.fftn(gridXY)
    gridXZ = jnp.fft.fftn(gridXZ)
    gridYZ = jnp.fft.fftn(gridYZ)
    gridYY = jnp.fft.fftn(gridYY)
    gridYX = jnp.fft.fftn(gridYX)
    gridZX = jnp.fft.fftn(gridZX)
    gridZY = jnp.fft.fftn(gridZY)
    gridZZ = -gridXX - gridYY

    gridk_sqr = (
        gridk.at[:, :, :, 0].get() * gridk.at[:, :, :, 0].get()
        + gridk.at[:, :, :, 1].get() * gridk.at[:, :, :, 1].get()
        + gridk.at[:, :, :, 2].get() * gridk.at[:, :, :, 2].get()
    )
    gridk_mod = jnp.sqrt(gridk_sqr)

    kdF = jnp.where(
        gridk_mod > 0,
        (
            gridk.at[:, :, :, 0].get() * gridX
            + gridk.at[:, :, :, 1].get() * gridY
            + gridk.at[:, :, :, 2].get() * gridZ
        )
        / gridk_sqr,
        0,
    )

    Cdkx = (
        gridk.at[:, :, :, 0].get() * gridXX
        + gridk.at[:, :, :, 1].get() * gridXY
        + gridk.at[:, :, :, 2].get() * gridXZ
    )
    Cdky = (
        gridk.at[:, :, :, 0].get() * gridYX
        + gridk.at[:, :, :, 1].get() * gridYY
        + gridk.at[:, :, :, 2].get() * gridYZ
    )
    Cdkz = (
        gridk.at[:, :, :, 0].get() * gridZX
        + gridk.at[:, :, :, 1].get() * gridZY
        + gridk.at[:, :, :, 2].get() * gridZZ
    )

    kdcdk = jnp.where(
        gridk_mod > 0,
        (
            gridk.at[:, :, :, 0].get() * Cdkx
            + gridk.at[:, :, :, 1].get() * Cdky
            + gridk.at[:, :, :, 2].get() * Cdkz
        )
        / gridk_sqr,
        0,
    )

    Fkxx = gridk.at[:, :, :, 0].get() * gridX
    Fkxy = gridk.at[:, :, :, 1].get() * gridX
    Fkxz = gridk.at[:, :, :, 2].get() * gridX
    Fkyx = gridk.at[:, :, :, 0].get() * gridY
    Fkyy = gridk.at[:, :, :, 1].get() * gridY
    Fkyz = gridk.at[:, :, :, 2].get() * gridY
    Fkzx = gridk.at[:, :, :, 0].get() * gridZ
    Fkzy = gridk.at[:, :, :, 1].get() * gridZ
    kkxx = gridk.at[:, :, :, 0].get() * gridk.at[:, :, :, 0].get()
    kkxy = gridk.at[:, :, :, 0].get() * gridk.at[:, :, :, 1].get()
    kkxz = gridk.at[:, :, :, 0].get() * gridk.at[:, :, :, 2].get()
    kkyx = gridk.at[:, :, :, 1].get() * gridk.at[:, :, :, 0].get()
    kkyy = gridk.at[:, :, :, 1].get() * gridk.at[:, :, :, 1].get()
    kkyz = gridk.at[:, :, :, 1].get() * gridk.at[:, :, :, 2].get()
    kkzx = gridk.at[:, :, :, 2].get() * gridk.at[:, :, :, 0].get()
    kkzy = gridk.at[:, :, :, 2].get() * gridk.at[:, :, :, 1].get()
    Cdkkxx = gridk.at[:, :, :, 0].get() * Cdkx
    Cdkkxy = gridk.at[:, :, :, 1].get() * Cdkx
    Cdkkxz = gridk.at[:, :, :, 2].get() * Cdkx
    Cdkkyx = gridk.at[:, :, :, 0].get() * Cdky
    Cdkkyy = gridk.at[:, :, :, 1].get() * Cdky
    Cdkkyz = gridk.at[:, :, :, 2].get() * Cdky
    Cdkkzx = gridk.at[:, :, :, 0].get() * Cdkz
    Cdkkzy = gridk.at[:, :, :, 1].get() * Cdkz

    # UF part
    B = jnp.where(
        gridk_mod > 0,
        gridk.at[:, :, :, 3].get()
        * (jnp.sin(gridk_mod) / gridk_mod)
        * (jnp.sin(gridk_mod) / gridk_mod),
        0,
    )  # scaling factor
    gridX = B * (gridX - gridk.at[:, :, :, 0].get() * kdF)
    gridY = B * (gridY - gridk.at[:, :, :, 1].get() * kdF)
    gridZ = B * (gridZ - gridk.at[:, :, :, 2].get() * kdF)

    # UC part (here B is imaginary so we absorb the imaginary unit in the funtion 'swap_real_imag()' which returns -Im(c)+i*Re(c)
    B = jnp.where(
        gridk_mod > 0,
        gridk.at[:, :, :, 3].get()
        * (jnp.sin(gridk_mod) / gridk_mod)
        * (3 * (jnp.sin(gridk_mod) - gridk_mod * jnp.cos(gridk_mod)) / (gridk_mod * gridk_sqr)),
        0,
    )  # scaling factor

    gridX += B * swap_real_imag((Cdkx - kdcdk * gridk.at[:, :, :, 0].get()))
    gridY += B * swap_real_imag((Cdky - kdcdk * gridk.at[:, :, :, 1].get()))
    gridZ += B * swap_real_imag((Cdkz - kdcdk * gridk.at[:, :, :, 2].get()))

    # DF part
    B = jnp.where(
        gridk_mod > 0,
        gridk.at[:, :, :, 3].get()
        * (-1)
        * (jnp.sin(gridk_mod) / gridk_mod)
        * (3 * (jnp.sin(gridk_mod) - gridk_mod * jnp.cos(gridk_mod)) / (gridk_mod * gridk_sqr)),
        0,
    )  # scaling factor
    gridXX = B * swap_real_imag((Fkxx - kkxx * kdF))
    gridXY = B * swap_real_imag((Fkxy - kkxy * kdF))
    gridXZ = B * swap_real_imag((Fkxz - kkxz * kdF))
    gridYX = B * swap_real_imag((Fkyx - kkyx * kdF))
    gridYY = B * swap_real_imag((Fkyy - kkyy * kdF))
    gridYZ = B * swap_real_imag((Fkyz - kkyz * kdF))
    gridZX = B * swap_real_imag((Fkzx - kkzx * kdF))
    gridZY = B * swap_real_imag((Fkzy - kkzy * kdF))

    # DC part
    B = jnp.where(
        gridk_mod > 0,
        gridk.at[:, :, :, 3].get()
        * (9)
        * ((jnp.sin(gridk_mod) - gridk_mod * jnp.cos(gridk_mod)) / (gridk_mod * gridk_sqr))
        * ((jnp.sin(gridk_mod) - gridk_mod * jnp.cos(gridk_mod)) / (gridk_mod * gridk_sqr)),
        0,
    )  # scaling factor
    gridXX += B * (Cdkkxx - kkxx * kdcdk)
    gridXY += B * (Cdkkxy - kkxy * kdcdk)
    gridXZ += B * (Cdkkxz - kkxz * kdcdk)
    gridYX += B * (Cdkkyx - kkyx * kdcdk)
    gridYY += B * (Cdkkyy - kkyy * kdcdk)
    gridYZ += B * (Cdkkyz - kkyz * kdcdk)
    gridZX += B * (Cdkkzx - kkzx * kdcdk)
    gridZY += B * (Cdkkzy - kkzy * kdcdk)

    # Inverse FFT
    gridX = jnp.real(jnp.fft.ifftn(gridX, norm="forward"))
    gridY = jnp.real(jnp.fft.ifftn(gridY, norm="forward"))
    gridZ = jnp.real(jnp.fft.ifftn(gridZ, norm="forward"))
    gridXX = jnp.real(jnp.fft.ifftn(gridXX, norm="forward"))
    gridXY = jnp.real(jnp.fft.ifftn(gridXY, norm="forward"))
    gridXZ = jnp.real(jnp.fft.ifftn(gridXZ, norm="forward"))
    gridYX = jnp.real(jnp.fft.ifftn(gridYX, norm="forward"))
    gridYY = jnp.real(jnp.fft.ifftn(gridYY, norm="forward"))
    gridYZ = jnp.real(jnp.fft.ifftn(gridYZ, norm="forward"))
    gridZX = jnp.real(jnp.fft.ifftn(gridZX, norm="forward"))
    gridZY = jnp.real(jnp.fft.ifftn(gridZY, norm="forward"))

    # Compute Linear velocities and Velocity gradients (from which we can extract angular velocities and rate of strain)
    w_lin_velocities = jnp.zeros((N, 3), float)
    w_velocity_gradient = jnp.zeros((N, 8), float)

    w_lin_velocities = w_lin_velocities.at[:, 0].set(
        jnp.sum(
            gaussian_grid_spacing2
            * jnp.reshape(
                gridX.at[all_indices_x, all_indices_y, all_indices_z].get(),
                (N, gaussP, gaussP, gaussP),
            ),
            axis=(1, 2, 3),
        )
    )
    w_lin_velocities = w_lin_velocities.at[:, 1].set(
        jnp.sum(
            gaussian_grid_spacing2
            * jnp.reshape(
                gridY.at[all_indices_x, all_indices_y, all_indices_z].get(),
                (N, gaussP, gaussP, gaussP),
            ),
            axis=(1, 2, 3),
        )
    )
    w_lin_velocities = w_lin_velocities.at[:, 2].set(
        jnp.sum(
            gaussian_grid_spacing2
            * jnp.reshape(
                gridZ.at[all_indices_x, all_indices_y, all_indices_z].get(),
                (N, gaussP, gaussP, gaussP),
            ),
            axis=(1, 2, 3),
        )
    )

    w_velocity_gradient = w_velocity_gradient.at[:, 0].set(
        jnp.sum(
            gaussian_grid_spacing2
            * jnp.reshape(
                gridXX.at[all_indices_x, all_indices_y, all_indices_z].get(),
                (N, gaussP, gaussP, gaussP),
            ),
            axis=(1, 2, 3),
        )
    )

    w_velocity_gradient = w_velocity_gradient.at[:, 5].set(
        jnp.sum(
            gaussian_grid_spacing2
            * jnp.reshape(
                gridXY.at[all_indices_x, all_indices_y, all_indices_z].get(),
                (N, gaussP, gaussP, gaussP),
            ),
            axis=(1, 2, 3),
        )
    )

    w_velocity_gradient = w_velocity_gradient.at[:, 6].set(
        jnp.sum(
            gaussian_grid_spacing2
            * jnp.reshape(
                gridXZ.at[all_indices_x, all_indices_y, all_indices_z].get(),
                (N, gaussP, gaussP, gaussP),
            ),
            axis=(1, 2, 3),
        )
    )

    w_velocity_gradient = w_velocity_gradient.at[:, 7].set(
        jnp.sum(
            gaussian_grid_spacing2
            * jnp.reshape(
                gridYZ.at[all_indices_x, all_indices_y, all_indices_z].get(),
                (N, gaussP, gaussP, gaussP),
            ),
            axis=(1, 2, 3),
        )
    )

    w_velocity_gradient = w_velocity_gradient.at[:, 4].set(
        jnp.sum(
            gaussian_grid_spacing2
            * jnp.reshape(
                gridYY.at[all_indices_x, all_indices_y, all_indices_z].get(),
                (N, gaussP, gaussP, gaussP),
            ),
            axis=(1, 2, 3),
        )
    )

    w_velocity_gradient = w_velocity_gradient.at[:, 1].set(
        jnp.sum(
            gaussian_grid_spacing2
            * jnp.reshape(
                gridYX.at[all_indices_x, all_indices_y, all_indices_z].get(),
                (N, gaussP, gaussP, gaussP),
            ),
            axis=(1, 2, 3),
        )
    )

    w_velocity_gradient = w_velocity_gradient.at[:, 2].set(
        jnp.sum(
            gaussian_grid_spacing2
            * jnp.reshape(
                gridZX.at[all_indices_x, all_indices_y, all_indices_z].get(),
                (N, gaussP, gaussP, gaussP),
            ),
            axis=(1, 2, 3),
        )
    )

    w_velocity_gradient = w_velocity_gradient.at[:, 3].set(
        jnp.sum(
            gaussian_grid_spacing2
            * jnp.reshape(
                gridZY.at[all_indices_x, all_indices_y, all_indices_z].get(),
                (N, gaussP, gaussP, gaussP),
            ),
            axis=(1, 2, 3),
        )
    )

    ##########################################################################################################################################
    ######################################## REAL SPACE CONTRIBUTION #########################################################################
    ##########################################################################################################################################

    # Allocate arrays for Linear velocities and Velocity gradients (from which we can extract angular velocities and rate of strain)
    r_lin_velocities = jnp.zeros((N, 3), float)
    r_velocity_gradient = jnp.zeros((N, 8), float)

    # SELF CONTRIBUTIONS
    r_lin_velocities = r_lin_velocities.at[:, 0].set(m_self.at[0].get() * forces.at[0::3].get())
    r_lin_velocities = r_lin_velocities.at[:, 1].set(m_self.at[0].get() * forces.at[1::3].get())
    r_lin_velocities = r_lin_velocities.at[:, 2].set(m_self.at[0].get() * forces.at[2::3].get())

    r_velocity_gradient = r_velocity_gradient.at[:, 0].set(
        m_self.at[1].get() * (couplets.at[0::8].get() - 4 * couplets.at[0::8].get())
    )
    r_velocity_gradient = r_velocity_gradient.at[:, 5].set(
        m_self.at[1].get() * (couplets.at[1::8].get() - 4 * couplets.at[5::8].get())
    )
    r_velocity_gradient = r_velocity_gradient.at[:, 6].set(
        m_self.at[1].get() * (couplets.at[2::8].get() - 4 * couplets.at[6::8].get())
    )
    r_velocity_gradient = r_velocity_gradient.at[:, 7].set(
        m_self.at[1].get() * (couplets.at[3::8].get() - 4 * couplets.at[7::8].get())
    )

    r_velocity_gradient = r_velocity_gradient.at[:, 4].set(
        m_self.at[1].get() * (couplets.at[4::8].get() - 4 * couplets.at[4::8].get())
    )
    r_velocity_gradient = r_velocity_gradient.at[:, 1].set(
        m_self.at[1].get() * (couplets.at[5::8].get() - 4 * couplets.at[1::8].get())
    )
    r_velocity_gradient = r_velocity_gradient.at[:, 2].set(
        m_self.at[1].get() * (couplets.at[6::8].get() - 4 * couplets.at[2::8].get())
    )
    r_velocity_gradient = r_velocity_gradient.at[:, 3].set(
        m_self.at[1].get() * (couplets.at[7::8].get() - 4 * couplets.at[3::8].get())
    )

    # # Pair contributions

    # # Geometric quantities
    rdotf_j = (
        r.at[:, 0].get() * forces.at[3 * indices_j + 0].get()
        + r.at[:, 1].get() * forces.at[3 * indices_j + 1].get()
        + r.at[:, 2].get() * forces.at[3 * indices_j + 2].get()
    )
    mrdotf_i = -(
        r.at[:, 0].get() * forces.at[3 * indices_i + 0].get()
        + r.at[:, 1].get() * forces.at[3 * indices_i + 1].get()
        + r.at[:, 2].get() * forces.at[3 * indices_i + 2].get()
    )

    Cj_dotr = jnp.array(
        [
            couplets.at[8 * indices_j + 0].get() * r.at[:, 0].get()
            + couplets.at[8 * indices_j + 1].get() * r.at[:, 1].get()
            + couplets.at[8 * indices_j + 2].get() * r.at[:, 2].get(),
            couplets.at[8 * indices_j + 5].get() * r.at[:, 0].get()
            + couplets.at[8 * indices_j + 4].get() * r.at[:, 1].get()
            + couplets.at[8 * indices_j + 3].get() * r.at[:, 2].get(),
            couplets.at[8 * indices_j + 6].get() * r.at[:, 0].get()
            + couplets.at[8 * indices_j + 7].get() * r.at[:, 1].get()
            - (couplets.at[8 * indices_j + 0].get() + couplets.at[8 * indices_j + 4].get())
            * r.at[:, 2].get(),
        ]
    )

    Ci_dotmr = jnp.array(
        [
            -couplets.at[8 * indices_i + 0].get() * r.at[:, 0].get()
            - couplets.at[8 * indices_i + 1].get() * r.at[:, 1].get()
            - couplets.at[8 * indices_i + 2].get() * r.at[:, 2].get(),
            -couplets.at[8 * indices_i + 5].get() * r.at[:, 0].get()
            - couplets.at[8 * indices_i + 4].get() * r.at[:, 1].get()
            - couplets.at[8 * indices_i + 3].get() * r.at[:, 2].get(),
            -couplets.at[8 * indices_i + 6].get() * r.at[:, 0].get()
            - couplets.at[8 * indices_i + 7].get() * r.at[:, 1].get()
            + (couplets.at[8 * indices_i + 0].get() + couplets.at[8 * indices_i + 4].get())
            * r.at[:, 2].get(),
        ]
    )

    rdotC_j = jnp.array(
        [
            couplets.at[8 * indices_j + 0].get() * r.at[:, 0].get()
            + couplets.at[8 * indices_j + 5].get() * r.at[:, 1].get()
            + couplets.at[8 * indices_j + 6].get() * r.at[:, 2].get(),
            couplets.at[8 * indices_j + 1].get() * r.at[:, 0].get()
            + couplets.at[8 * indices_j + 4].get() * r.at[:, 1].get()
            + couplets.at[8 * indices_j + 7].get() * r.at[:, 2].get(),
            couplets.at[8 * indices_j + 2].get() * r.at[:, 0].get()
            + couplets.at[8 * indices_j + 3].get() * r.at[:, 1].get()
            - (couplets.at[8 * indices_j + 0].get() + couplets.at[8 * indices_j + 4].get())
            * r.at[:, 2].get(),
        ]
    )

    mrdotC_i = jnp.array(
        [
            -couplets.at[8 * indices_i + 0].get() * r.at[:, 0].get()
            - couplets.at[8 * indices_i + 5].get() * r.at[:, 1].get()
            - couplets.at[8 * indices_i + 6].get() * r.at[:, 2].get(),
            -couplets.at[8 * indices_i + 1].get() * r.at[:, 0].get()
            - couplets.at[8 * indices_i + 4].get() * r.at[:, 1].get()
            - couplets.at[8 * indices_i + 7].get() * r.at[:, 2].get(),
            -couplets.at[8 * indices_i + 2].get() * r.at[:, 0].get()
            - couplets.at[8 * indices_i + 3].get() * r.at[:, 1].get()
            + (couplets.at[8 * indices_i + 0].get() + couplets.at[8 * indices_i + 4].get())
            * r.at[:, 2].get(),
        ]
    )

    rdotC_jj_dotr = (
        r.at[:, 0].get() * Cj_dotr.at[0, :].get()
        + r.at[:, 1].get() * Cj_dotr.at[1, :].get()
        + r.at[:, 2].get() * Cj_dotr.at[2, :].get()
    )
    mrdotC_ii_dotmr = -(
        r.at[:, 0].get() * Ci_dotmr.at[0, :].get()
        + r.at[:, 1].get() * Ci_dotmr.at[1, :].get()
        + r.at[:, 2].get() * Ci_dotmr.at[2, :].get()
    )

    # Compute Velocity for particles i
    r_lin_velocities = r_lin_velocities.at[indices_i, 0].add(
        f1 * forces.at[3 * indices_j].get() + (f2 - f1) * rdotf_j * r.at[:, 0].get()
    )
    r_lin_velocities = r_lin_velocities.at[indices_i, 1].add(
        f1 * forces.at[3 * indices_j + 1].get() + (f2 - f1) * rdotf_j * r.at[:, 1].get()
    )
    r_lin_velocities = r_lin_velocities.at[indices_i, 2].add(
        f1 * forces.at[3 * indices_j + 2].get() + (f2 - f1) * rdotf_j * r.at[:, 2].get()
    )
    r_lin_velocities = r_lin_velocities.at[indices_i, 0].add(
        g1 * (Cj_dotr.at[0, :].get() - rdotC_jj_dotr * r.at[:, 0].get())
        + g2 * (rdotC_j.at[0, :].get() - 4.0 * rdotC_jj_dotr * r.at[:, 0].get())
    )
    r_lin_velocities = r_lin_velocities.at[indices_i, 1].add(
        g1 * (Cj_dotr.at[1, :].get() - rdotC_jj_dotr * r.at[:, 1].get())
        + g2 * (rdotC_j.at[1, :].get() - 4.0 * rdotC_jj_dotr * r.at[:, 1].get())
    )
    r_lin_velocities = r_lin_velocities.at[indices_i, 2].add(
        g1 * (Cj_dotr.at[2, :].get() - rdotC_jj_dotr * r.at[:, 2].get())
        + g2 * (rdotC_j.at[2, :].get() - 4.0 * rdotC_jj_dotr * r.at[:, 2].get())
    )
    # Compute Velocity for particles j
    r_lin_velocities = r_lin_velocities.at[indices_j, 0].add(
        f1 * forces.at[3 * indices_i].get() - (f2 - f1) * mrdotf_i * r.at[:, 0].get()
    )
    r_lin_velocities = r_lin_velocities.at[indices_j, 1].add(
        f1 * forces.at[3 * indices_i + 1].get() - (f2 - f1) * mrdotf_i * r.at[:, 1].get()
    )
    r_lin_velocities = r_lin_velocities.at[indices_j, 2].add(
        f1 * forces.at[3 * indices_i + 2].get() - (f2 - f1) * mrdotf_i * r.at[:, 2].get()
    )
    r_lin_velocities = r_lin_velocities.at[indices_j, 0].add(
        g1 * (Ci_dotmr.at[0, :].get() + mrdotC_ii_dotmr * r.at[:, 0].get())
        + g2 * (mrdotC_i.at[0, :].get() + 4.0 * mrdotC_ii_dotmr * r.at[:, 0].get())
    )
    r_lin_velocities = r_lin_velocities.at[indices_j, 1].add(
        g1 * (Ci_dotmr.at[1, :].get() + mrdotC_ii_dotmr * r.at[:, 1].get())
        + g2 * (mrdotC_i.at[1, :].get() + 4.0 * mrdotC_ii_dotmr * r.at[:, 1].get())
    )
    r_lin_velocities = r_lin_velocities.at[indices_j, 2].add(
        g1 * (Ci_dotmr.at[2, :].get() + mrdotC_ii_dotmr * r.at[:, 2].get())
        + g2 * (mrdotC_i.at[2, :].get() + 4.0 * mrdotC_ii_dotmr * r.at[:, 2].get())
    )

    # Compute Velocity Gradient for particles i and j
    r_velocity_gradient = r_velocity_gradient.at[indices_i, 0].add(
        (-1)
        * g1
        * (
            r.at[:, 0].get() * forces.at[3 * indices_j + 0].get()
            - rdotf_j * r.at[:, 0].get() * r.at[:, 0].get()
        )
        + (-1)
        * g2
        * (
            rdotf_j
            + forces.at[3 * indices_j + 0].get() * r.at[:, 0].get()
            - 4.0 * rdotf_j * r.at[:, 0].get() * r.at[:, 0].get()
        )
    )

    r_velocity_gradient = r_velocity_gradient.at[indices_j, 0].add(
        (-1)
        * g1
        * (
            -r.at[:, 0].get() * forces.at[3 * indices_i + 0].get()
            - mrdotf_i * r.at[:, 0].get() * r.at[:, 0].get()
        )
        + (-1)
        * g2
        * (
            mrdotf_i
            - forces.at[3 * indices_i + 0].get() * r.at[:, 0].get()
            - 4.0 * mrdotf_i * r.at[:, 0].get() * r.at[:, 0].get()
        )
    )

    r_velocity_gradient = r_velocity_gradient.at[indices_i, 0].add(
        h1 * (couplets.at[8 * indices_j + 0].get() - 4.0 * couplets.at[8 * indices_j + 0].get())
        + h2
        * (
            r.at[:, 0].get() * Cj_dotr.at[0, :].get()
            - rdotC_jj_dotr * r.at[:, 0].get() * r.at[:, 0].get()
        )
        + h3
        * (
            rdotC_jj_dotr
            + Cj_dotr.at[0, :].get() * r.at[:, 0].get()
            + r.at[:, 0].get() * rdotC_j.at[0, :].get()
            + rdotC_j.at[0, :].get() * r.at[:, 0].get()
            - 6.0 * rdotC_jj_dotr * r.at[:, 0].get() * r.at[:, 0].get()
            - couplets.at[8 * indices_j + 0].get()
        )
    )
    r_velocity_gradient = r_velocity_gradient.at[indices_j, 0].add(
        h1 * (couplets.at[8 * indices_i + 0].get() - 4.0 * couplets.at[8 * indices_i + 0].get())
        + h2
        * (
            -r.at[:, 0].get() * Ci_dotmr.at[0, :].get()
            - mrdotC_ii_dotmr * r.at[:, 0].get() * r.at[:, 0].get()
        )
        + h3
        * (
            mrdotC_ii_dotmr
            - Ci_dotmr.at[0, :].get() * r.at[:, 0].get()
            - r.at[:, 0].get() * mrdotC_i.at[0, :].get()
            - mrdotC_i.at[0, :].get() * r.at[:, 0].get()
            - 6.0 * mrdotC_ii_dotmr * r.at[:, 0].get() * r.at[:, 0].get()
            - couplets.at[8 * indices_i + 0].get()
        )
    )

    r_velocity_gradient = r_velocity_gradient.at[indices_i, 1].add(
        (-1)
        * g1
        * (
            r.at[:, 1].get() * forces.at[3 * indices_j + 0].get()
            - rdotf_j * r.at[:, 1].get() * r.at[:, 0].get()
        )
        + (-1)
        * g2
        * (
            forces.at[3 * indices_j + 1].get() * r.at[:, 0].get()
            - 4.0 * rdotf_j * r.at[:, 1].get() * r.at[:, 0].get()
        )
    )

    r_velocity_gradient = r_velocity_gradient.at[indices_j, 1].add(
        (-1)
        * g1
        * (
            -r.at[:, 1].get() * forces.at[3 * indices_i + 0].get()
            - mrdotf_i * r.at[:, 1].get() * r.at[:, 0].get()
        )
        + (-1)
        * g2
        * (
            -forces.at[3 * indices_i + 1].get() * r.at[:, 0].get()
            - 4.0 * mrdotf_i * r.at[:, 1].get() * r.at[:, 0].get()
        )
    )

    r_velocity_gradient = r_velocity_gradient.at[indices_i, 1].add(
        h1 * (couplets.at[8 * indices_j + 5].get() - 4.0 * couplets.at[8 * indices_j + 1].get())
        + h2
        * (
            r.at[:, 1].get() * Cj_dotr.at[0, :].get()
            - rdotC_jj_dotr * r.at[:, 1].get() * r.at[:, 0].get()
        )
        + h3
        * (
            Cj_dotr.at[1, :].get() * r.at[:, 0].get()
            + r.at[:, 1].get() * rdotC_j.at[0, :].get()
            + rdotC_j.at[1, :].get() * r.at[:, 0].get()
            - 6.0 * rdotC_jj_dotr * r.at[:, 1].get() * r.at[:, 0].get()
            - couplets.at[8 * indices_j + 1].get()
        )
    )

    r_velocity_gradient = r_velocity_gradient.at[indices_j, 1].add(
        h1 * (couplets.at[8 * indices_i + 5].get() - 4.0 * couplets.at[8 * indices_i + 1].get())
        + h2
        * (
            -r.at[:, 1].get() * Ci_dotmr.at[0, :].get()
            - mrdotC_ii_dotmr * r.at[:, 1].get() * r.at[:, 0].get()
        )
        + h3
        * (
            -Ci_dotmr.at[1, :].get() * r.at[:, 0].get()
            - r.at[:, 1].get() * mrdotC_i.at[0, :].get()
            - mrdotC_i.at[1, :].get() * r.at[:, 0].get()
            - 6.0 * mrdotC_ii_dotmr * r.at[:, 1].get() * r.at[:, 0].get()
            - couplets.at[8 * indices_i + 1].get()
        )
    )

    r_velocity_gradient = r_velocity_gradient.at[indices_i, 2].add(
        (-1)
        * g1
        * (
            r.at[:, 2].get() * forces.at[3 * indices_j + 0].get()
            - rdotf_j * r.at[:, 2].get() * r.at[:, 0].get()
        )
        + (-1)
        * g2
        * (
            forces.at[3 * indices_j + 2].get() * r.at[:, 0].get()
            - 4.0 * rdotf_j * r.at[:, 2].get() * r.at[:, 0].get()
        )
    )

    r_velocity_gradient = r_velocity_gradient.at[indices_j, 2].add(
        (-1)
        * g1
        * (
            -r.at[:, 2].get() * forces.at[3 * indices_i + 0].get()
            - mrdotf_i * r.at[:, 2].get() * r.at[:, 0].get()
        )
        + (-1)
        * g2
        * (
            -forces.at[3 * indices_i + 2].get() * r.at[:, 0].get()
            - 4.0 * mrdotf_i * r.at[:, 2].get() * r.at[:, 0].get()
        )
    )
    r_velocity_gradient = r_velocity_gradient.at[indices_i, 2].add(
        h1 * (couplets.at[8 * indices_j + 6].get() - 4.0 * couplets.at[8 * indices_j + 2].get())
        + h2
        * (
            r.at[:, 2].get() * Cj_dotr.at[0, :].get()
            - rdotC_jj_dotr * r.at[:, 2].get() * r.at[:, 0].get()
        )
        + h3
        * (
            Cj_dotr.at[2, :].get() * r.at[:, 0].get()
            + r.at[:, 2].get() * rdotC_j.at[0, :].get()
            + rdotC_j.at[2, :].get() * r.at[:, 0].get()
            - 6.0 * rdotC_jj_dotr * r.at[:, 2].get() * r.at[:, 0].get()
            - couplets.at[8 * indices_j + 2].get()
        )
    )
    r_velocity_gradient = r_velocity_gradient.at[indices_j, 2].add(
        h1 * (couplets.at[8 * indices_i + 6].get() - 4.0 * couplets.at[8 * indices_i + 2].get())
        + h2
        * (
            r.at[:, 2].get() * Ci_dotmr.at[0, :].get() * (-1)
            - mrdotC_ii_dotmr * r.at[:, 2].get() * r.at[:, 0].get()
        )
        + h3
        * (
            -Ci_dotmr.at[2, :].get() * r.at[:, 0].get()
            - r.at[:, 2].get() * mrdotC_i.at[0, :].get()
            - mrdotC_i.at[2, :].get() * r.at[:, 0].get()
            - 6.0 * mrdotC_ii_dotmr * r.at[:, 2].get() * r.at[:, 0].get()
            - couplets.at[8 * indices_i + 2].get()
        )
    )

    r_velocity_gradient = r_velocity_gradient.at[indices_i, 3].add(
        (-1)
        * g1
        * (
            r.at[:, 2].get() * forces.at[3 * indices_j + 1].get()
            - rdotf_j * r.at[:, 2].get() * r.at[:, 1].get()
        )
        + (-1)
        * g2
        * (
            forces.at[3 * indices_j + 2].get() * r.at[:, 1].get()
            - 4.0 * rdotf_j * r.at[:, 2].get() * r.at[:, 1].get()
        )
    )

    r_velocity_gradient = r_velocity_gradient.at[indices_j, 3].add(
        (-1)
        * g1
        * (
            -r.at[:, 2].get() * forces.at[3 * indices_i + 1].get()
            - mrdotf_i * r.at[:, 2].get() * r.at[:, 1].get()
        )
        + (-1)
        * g2
        * (
            -forces.at[3 * indices_i + 2].get() * r.at[:, 1].get()
            - 4.0 * mrdotf_i * r.at[:, 2].get() * r.at[:, 1].get()
        )
    )

    r_velocity_gradient = r_velocity_gradient.at[indices_i, 3].add(
        h1 * (couplets.at[8 * indices_j + 7].get() - 4.0 * couplets.at[8 * indices_j + 3].get())
        + h2
        * (
            r.at[:, 2].get() * Cj_dotr.at[1, :].get()
            - rdotC_jj_dotr * r.at[:, 2].get() * r.at[:, 1].get()
        )
        + h3
        * (
            Cj_dotr.at[2, :].get() * r.at[:, 1].get()
            + r.at[:, 2].get() * rdotC_j.at[1, :].get()
            + rdotC_j.at[2, :].get() * r.at[:, 1].get()
            - 6.0 * rdotC_jj_dotr * r.at[:, 2].get() * r.at[:, 1].get()
            - couplets.at[8 * indices_j + 3].get()
        )
    )

    r_velocity_gradient = r_velocity_gradient.at[indices_j, 3].add(
        h1 * (couplets.at[8 * indices_i + 7].get() - 4.0 * couplets.at[8 * indices_i + 3].get())
        + h2
        * (
            -r.at[:, 2].get() * Ci_dotmr.at[1, :].get()
            - mrdotC_ii_dotmr * r.at[:, 2].get() * r.at[:, 1].get()
        )
        + h3
        * (
            -Ci_dotmr.at[2, :].get() * r.at[:, 1].get()
            - r.at[:, 2].get() * mrdotC_i.at[1, :].get()
            - mrdotC_i.at[2, :].get() * r.at[:, 1].get()
            - 6.0 * mrdotC_ii_dotmr * r.at[:, 2].get() * r.at[:, 1].get()
            - couplets.at[8 * indices_i + 3].get()
        )
    )

    r_velocity_gradient = r_velocity_gradient.at[indices_i, 4].add(
        (-1)
        * g1
        * (
            r.at[:, 1].get() * forces.at[3 * indices_j + 1].get()
            - rdotf_j * r.at[:, 1].get() * r.at[:, 1].get()
        )
        + (-1)
        * g2
        * (
            rdotf_j
            + forces.at[3 * indices_j + 1].get() * r.at[:, 1].get()
            - 4.0 * rdotf_j * r.at[:, 1].get() * r.at[:, 1].get()
        )
    )

    r_velocity_gradient = r_velocity_gradient.at[indices_j, 4].add(
        (-1)
        * g1
        * (
            -r.at[:, 1].get() * forces.at[3 * indices_i + 1].get()
            - mrdotf_i * r.at[:, 1].get() * r.at[:, 1].get()
        )
        + (-1)
        * g2
        * (
            mrdotf_i
            - forces.at[3 * indices_i + 1].get() * r.at[:, 1].get()
            - 4.0 * mrdotf_i * r.at[:, 1].get() * r.at[:, 1].get()
        )
    )

    r_velocity_gradient = r_velocity_gradient.at[indices_i, 4].add(
        h1 * (couplets.at[8 * indices_j + 4].get() - 4.0 * couplets.at[8 * indices_j + 4].get())
        + h2
        * (
            r.at[:, 1].get() * Cj_dotr.at[1, :].get()
            - rdotC_jj_dotr * r.at[:, 1].get() * r.at[:, 1].get()
        )
        + h3
        * (
            rdotC_jj_dotr
            + Cj_dotr.at[1, :].get() * r.at[:, 1].get()
            + r.at[:, 1].get() * rdotC_j.at[1, :].get()
            + rdotC_j.at[1, :].get() * r.at[:, 1].get()
            - 6.0 * rdotC_jj_dotr * r.at[:, 1].get() * r.at[:, 1].get()
            - couplets.at[8 * indices_j + 4].get()
        )
    )

    r_velocity_gradient = r_velocity_gradient.at[indices_j, 4].add(
        h1 * (couplets.at[8 * indices_i + 4].get() - 4.0 * couplets.at[8 * indices_i + 4].get())
        + h2
        * (
            -r.at[:, 1].get() * Ci_dotmr.at[1, :].get()
            - mrdotC_ii_dotmr * r.at[:, 1].get() * r.at[:, 1].get()
        )
        + h3
        * (
            mrdotC_ii_dotmr
            - Ci_dotmr.at[1, :].get() * r.at[:, 1].get()
            - r.at[:, 1].get() * mrdotC_i.at[1, :].get()
            - mrdotC_i.at[1, :].get() * r.at[:, 1].get()
            - 6.0 * mrdotC_ii_dotmr * r.at[:, 1].get() * r.at[:, 1].get()
            - couplets.at[8 * indices_i + 4].get()
        )
    )

    r_velocity_gradient = r_velocity_gradient.at[indices_i, 5].add(
        (-1)
        * g1
        * (
            r.at[:, 0].get() * forces.at[3 * indices_j + 1].get()
            - rdotf_j * r.at[:, 0].get() * r.at[:, 1].get()
        )
        + (-1)
        * g2
        * (
            forces.at[3 * indices_j + 0].get() * r.at[:, 1].get()
            - 4.0 * rdotf_j * r.at[:, 0].get() * r.at[:, 1].get()
        )
    )

    r_velocity_gradient = r_velocity_gradient.at[indices_j, 5].add(
        (-1)
        * g1
        * (
            -r.at[:, 0].get() * forces.at[3 * indices_i + 1].get()
            - mrdotf_i * r.at[:, 0].get() * r.at[:, 1].get()
        )
        + (-1)
        * g2
        * (
            -forces.at[3 * indices_i + 0].get() * r.at[:, 1].get()
            - 4.0 * mrdotf_i * r.at[:, 0].get() * r.at[:, 1].get()
        )
    )

    r_velocity_gradient = r_velocity_gradient.at[indices_i, 5].add(
        h1 * (couplets.at[8 * indices_j + 1].get() - 4.0 * couplets.at[8 * indices_j + 5].get())
        + h2
        * (
            r.at[:, 0].get() * Cj_dotr.at[1, :].get()
            - rdotC_jj_dotr * r.at[:, 0].get() * r.at[:, 1].get()
        )
        + h3
        * (
            Cj_dotr.at[0, :].get() * r.at[:, 1].get()
            + r.at[:, 0].get() * rdotC_j.at[1, :].get()
            + rdotC_j.at[0, :].get() * r.at[:, 1].get()
            - 6.0 * rdotC_jj_dotr * r.at[:, 0].get() * r.at[:, 1].get()
            - couplets.at[8 * indices_j + 5].get()
        )
    )

    r_velocity_gradient = r_velocity_gradient.at[indices_j, 5].add(
        h1 * (couplets.at[8 * indices_i + 1].get() - 4.0 * couplets.at[8 * indices_i + 5].get())
        + h2
        * (
            -r.at[:, 0].get() * Ci_dotmr.at[1, :].get()
            - mrdotC_ii_dotmr * r.at[:, 0].get() * r.at[:, 1].get()
        )
        + h3
        * (
            -Ci_dotmr.at[0, :].get() * r.at[:, 1].get()
            - r.at[:, 0].get() * mrdotC_i.at[1, :].get()
            - mrdotC_i.at[0, :].get() * r.at[:, 1].get()
            - 6.0 * mrdotC_ii_dotmr * r.at[:, 0].get() * r.at[:, 1].get()
            - couplets.at[8 * indices_i + 5].get()
        )
    )

    r_velocity_gradient = r_velocity_gradient.at[indices_i, 6].add(
        (-1)
        * g1
        * (
            r.at[:, 0].get() * forces.at[3 * indices_j + 2].get()
            - rdotf_j * r.at[:, 0].get() * r.at[:, 2].get()
        )
        + (-1)
        * g2
        * (
            forces.at[3 * indices_j + 0].get() * r.at[:, 2].get()
            - 4.0 * rdotf_j * r.at[:, 0].get() * r.at[:, 2].get()
        )
    )

    r_velocity_gradient = r_velocity_gradient.at[indices_j, 6].add(
        (-1)
        * g1
        * (
            -r.at[:, 0].get() * forces.at[3 * indices_i + 2].get()
            - mrdotf_i * r.at[:, 0].get() * r.at[:, 2].get()
        )
        + (-1)
        * g2
        * (
            -forces.at[3 * indices_i + 0].get() * r.at[:, 2].get()
            - 4.0 * mrdotf_i * r.at[:, 0].get() * r.at[:, 2].get()
        )
    )

    r_velocity_gradient = r_velocity_gradient.at[indices_i, 6].add(
        h1 * (couplets.at[8 * indices_j + 2].get() - 4.0 * couplets.at[8 * indices_j + 6].get())
        + h2
        * (
            r.at[:, 0].get() * Cj_dotr.at[2, :].get()
            - rdotC_jj_dotr * r.at[:, 0].get() * r.at[:, 2].get()
        )
        + h3
        * (
            Cj_dotr.at[0, :].get() * r.at[:, 2].get()
            + r.at[:, 0].get() * rdotC_j.at[2, :].get()
            + rdotC_j.at[0, :].get() * r.at[:, 2].get()
            - 6.0 * rdotC_jj_dotr * r.at[:, 0].get() * r.at[:, 2].get()
            - couplets.at[8 * indices_j + 6].get()
        )
    )

    r_velocity_gradient = r_velocity_gradient.at[indices_j, 6].add(
        h1 * (couplets.at[8 * indices_i + 2].get() - 4.0 * couplets.at[8 * indices_i + 6].get())
        + h2
        * (
            -r.at[:, 0].get() * Ci_dotmr.at[2, :].get()
            - mrdotC_ii_dotmr * r.at[:, 0].get() * r.at[:, 2].get()
        )
        + h3
        * (
            -Ci_dotmr.at[0, :].get() * r.at[:, 2].get()
            - r.at[:, 0].get() * mrdotC_i.at[2, :].get()
            - mrdotC_i.at[0, :].get() * r.at[:, 2].get()
            - 6.0 * mrdotC_ii_dotmr * r.at[:, 0].get() * r.at[:, 2].get()
            - couplets.at[8 * indices_i + 6].get()
        )
    )

    r_velocity_gradient = r_velocity_gradient.at[indices_i, 7].add(
        (-1)
        * g1
        * (
            r.at[:, 1].get() * forces.at[3 * indices_j + 2].get()
            - rdotf_j * r.at[:, 1].get() * r.at[:, 2].get()
        )
        + (-1)
        * g2
        * (
            forces.at[3 * indices_j + 1].get() * r.at[:, 2].get()
            - 4.0 * rdotf_j * r.at[:, 1].get() * r.at[:, 2].get()
        )
    )

    r_velocity_gradient = r_velocity_gradient.at[indices_j, 7].add(
        (-1)
        * g1
        * (
            -r.at[:, 1].get() * forces.at[3 * indices_i + 2].get()
            - mrdotf_i * r.at[:, 1].get() * r.at[:, 2].get()
        )
        + (-1)
        * g2
        * (
            -forces.at[3 * indices_i + 1].get() * r.at[:, 2].get()
            - 4.0 * mrdotf_i * r.at[:, 1].get() * r.at[:, 2].get()
        )
    )

    r_velocity_gradient = r_velocity_gradient.at[indices_i, 7].add(
        h1 * (couplets.at[8 * indices_j + 3].get() - 4.0 * couplets.at[8 * indices_j + 7].get())
        + h2
        * (
            r.at[:, 1].get() * Cj_dotr.at[2, :].get()
            - rdotC_jj_dotr * r.at[:, 1].get() * r.at[:, 2].get()
        )
        + h3
        * (
            Cj_dotr.at[1, :].get() * r.at[:, 2].get()
            + r.at[:, 1].get() * rdotC_j.at[2, :].get()
            + rdotC_j.at[1, :].get() * r.at[:, 2].get()
            - 6.0 * rdotC_jj_dotr * r.at[:, 1].get() * r.at[:, 2].get()
            - couplets.at[8 * indices_j + 7].get()
        )
    )

    r_velocity_gradient = r_velocity_gradient.at[indices_j, 7].add(
        h1 * (couplets.at[8 * indices_i + 3].get() - 4.0 * couplets.at[8 * indices_i + 7].get())
        + h2
        * (
            -r.at[:, 1].get() * Ci_dotmr.at[2, :].get()
            - mrdotC_ii_dotmr * r.at[:, 1].get() * r.at[:, 2].get()
        )
        + h3
        * (
            -Ci_dotmr.at[1, :].get() * r.at[:, 2].get()
            - r.at[:, 1].get() * mrdotC_i.at[2, :].get()
            - mrdotC_i.at[1, :].get() * r.at[:, 2].get()
            - 6.0 * mrdotC_ii_dotmr * r.at[:, 1].get() * r.at[:, 2].get()
            - couplets.at[8 * indices_i + 7].get()
        )
    )

    ##########################################################################################################################################
    ##########################################################################################################################################

    # Add wave and real space part together
    lin_vel = w_lin_velocities + r_lin_velocities
    velocity_gradient = w_velocity_gradient + r_velocity_gradient

    # # Convert to angular velocities and rate of strain
    ang_vel_and_strain = jnp.zeros((N, 8))
    ang_vel_and_strain = ang_vel_and_strain.at[:, 0].set(
        (velocity_gradient.at[:, 3].get() - velocity_gradient.at[:, 7].get()) * 0.5
    )
    ang_vel_and_strain = ang_vel_and_strain.at[:, 1].set(
        (velocity_gradient.at[:, 6].get() - velocity_gradient.at[:, 2].get()) * 0.5
    )
    ang_vel_and_strain = ang_vel_and_strain.at[:, 2].set(
        (velocity_gradient.at[:, 1].get() - velocity_gradient.at[:, 5].get()) * 0.5
    )
    ang_vel_and_strain = ang_vel_and_strain.at[:, 3].set(
        2 * velocity_gradient.at[:, 0].get() + velocity_gradient.at[:, 4].get()
    )
    ang_vel_and_strain = ang_vel_and_strain.at[:, 4].set(
        velocity_gradient.at[:, 1].get() + velocity_gradient.at[:, 5].get()
    )
    ang_vel_and_strain = ang_vel_and_strain.at[:, 5].set(
        velocity_gradient.at[:, 2].get() + velocity_gradient.at[:, 6].get()
    )
    ang_vel_and_strain = ang_vel_and_strain.at[:, 6].set(
        velocity_gradient.at[:, 3].get() + velocity_gradient.at[:, 7].get()
    )
    ang_vel_and_strain = ang_vel_and_strain.at[:, 7].set(
        velocity_gradient.at[:, 0].get() + 2 * velocity_gradient.at[:, 4].get()
    )

    # Convert to Generalized Velocities+Strain
    generalized_velocities = jnp.zeros(
        11 * N
    )  # First 6N entries for U and last 5N for strain rates

    generalized_velocities = generalized_velocities.at[0 : 6 * N : 6].set(lin_vel.at[:, 0].get())
    generalized_velocities = generalized_velocities.at[1 : 6 * N : 6].set(lin_vel.at[:, 1].get())
    generalized_velocities = generalized_velocities.at[2 : 6 * N : 6].set(lin_vel.at[:, 2].get())
    generalized_velocities = generalized_velocities.at[3 : 6 * N : 6].set(
        ang_vel_and_strain.at[:, 0].get()
    )
    generalized_velocities = generalized_velocities.at[4 : 6 * N : 6].set(
        ang_vel_and_strain.at[:, 1].get()
    )
    generalized_velocities = generalized_velocities.at[5 : 6 * N : 6].set(
        ang_vel_and_strain.at[:, 2].get()
    )
    generalized_velocities = generalized_velocities.at[(6 * N + 0) :: 5].set(
        ang_vel_and_strain.at[:, 3].get()
    )
    generalized_velocities = generalized_velocities.at[(6 * N + 1) :: 5].set(
        ang_vel_and_strain.at[:, 4].get()
    )
    generalized_velocities = generalized_velocities.at[(6 * N + 2) :: 5].set(
        ang_vel_and_strain.at[:, 5].get()
    )
    generalized_velocities = generalized_velocities.at[(6 * N + 3) :: 5].set(
        ang_vel_and_strain.at[:, 6].get()
    )
    generalized_velocities = generalized_velocities.at[(6 * N + 4) :: 5].set(
        ang_vel_and_strain.at[:, 7].get()
    )

    return generalized_velocities


@partial(jit, static_argnums=[0, 1, 2, 3, 4])
def Mobility_periodic(
    N: int,
    Nx: int,
    Ny: int,
    Nz: int,
    gaussP: int,
    gridk: ArrayLike,
    m_self: ArrayLike,
    all_indices_x: ArrayLike,
    all_indices_y: ArrayLike,
    all_indices_z: ArrayLike,
    gaussian_grid_spacing1: ArrayLike,
    gaussian_grid_spacing2: ArrayLike,
    r: ArrayLike,
    indices_i: ArrayLike,
    indices_j: ArrayLike,
    f1: ArrayLike,
    f2: ArrayLike,
    g1: ArrayLike,
    g2: ArrayLike,
    h1: ArrayLike,
    h2: ArrayLike,
    h3: ArrayLike,
    generalized_forces: ArrayLike,
) -> Array:
    """
    Compute the matrix-vector product of the mobility matrix with a generalized force vector, in periodic boundary conditions..

    Parameters
    ----------

    N: (int)
        Number of particles
    Nx: (int)
        Number of grid points in x direction
    Ny: (int)
        Number of grid points in y direction
    Nz: (int)
        Number of grid points in z direction
    gaussP: (int)
        Gaussian support size for wave space calculation
    gridk: (float)
        Array (Nx,Ny,Nz,4) containing wave vectors and scaling factors for far-field wavespace calculation
    m_self: (float)
        Array (,2) containing mobility self contributions
    all_indices_x: (int)
        Array (,N*gaussP*gaussP*gaussP) containing all the x-indices of wave grid points overlapping with each particle Gaussian support
    all_indices_y: (int)
        Array (,N*gaussP*gaussP*gaussP) containing all the y-indices of wave grid points overlapping with each particle Gaussian support
    all_indices_z: (int)
        Array (,N*gaussP*gaussP*gaussP) containing all the z-indices of wave grid points overlapping with each particle Gaussian support
    gaussian_grid_spacing1: (float)
        Array (,gaussP*gaussP*gaussP) containing scaled distances from support center to each gridpoint in the gaussian support (for FFT)
    gaussian_grid_spacing2: (float)
        Array (,gaussP*gaussP*gaussP) containing scaled distances from support center to each gridpoint in the gaussian support (for inverse FFT)
    r: (float)
        Array (n_pair_ff,3) containing units vectors connecting each pair of particles in the far-field neighbor list
    indices_i: (int)
        Array (n_pair_ff) of indices of first particle in neighbor list pairs
    indices_j: (int)
        Array (n_pair_ff) of indices of second particle in neighbor list pairs
    f1: (float)
        Array (,n_pair_ff) containing mobility scalar function evaluated for the current particle configuration
    f2: (float)
        Array (,n_pair_ff) containing mobility scalar function evaluated for the current particle configuration
    g1: (float)
        Array (,n_pair_ff) containing mobility scalar function evaluated for the current particle configuration
    g2: (float)
        Array (,n_pair_ff) containing mobility scalar function evaluated for the current particle configuration
    h1: (float)
        Array (,n_pair_ff) containing mobility scalar function evaluated for the current particle configuration
    h2: (float)
        Array (,n_pair_ff) containing mobility scalar function evaluated for the current particle configuration
    h3: (float)
        Array (,n_pair_ff) containing mobility scalar function evaluated for the current particle configuration
    generalized_forces: (float)
        Array (,6*N) containing input generalized forces (force/torque)

    Returns
    -------
    generalized_velocities (linear/angular velocities)

    """

    # Helper function
    def swap_real_imag(cplx_arr: ArrayLike) -> Array:
        """Perform an operation on a complex number.

        Take a complex number as input and return a complex number with real part equal to (minus) the imaginary part of the input,
        and an imaginary part equal to the real part of the input.

        Parameters
        ----------
        cplx_arr: (complex)
            Array of complex values

        Returns
        -------
        -jnp.imag(cplx_arr) + 1j * jnp.real(cplx_arr)

        """
        return -jnp.imag(cplx_arr) + 1j * jnp.real(cplx_arr)

    # Get forces,torques,couplets from generalized forces (3*N vector: f1x,f1y,f1z, ... , fNx,fNy,fNz and same for torque, while couplet is 5N)
    forces = jnp.zeros(3 * N)
    forces = forces.at[0::3].set(generalized_forces.at[0 : (6 * N) : 6].get())
    forces = forces.at[1::3].set(generalized_forces.at[1 : (6 * N) : 6].get())
    forces = forces.at[2::3].set(generalized_forces.at[2 : (6 * N) : 6].get())
    torques = jnp.zeros(3 * N)
    torques = torques.at[0::3].set(generalized_forces.at[3 : (6 * N) : 6].get())
    torques = torques.at[1::3].set(generalized_forces.at[4 : (6 * N) : 6].get())
    torques = torques.at[2::3].set(generalized_forces.at[5 : (6 * N) : 6].get())

    ######################################## WAVE SPACE CONTRIBUTION #########################################################################

    # Create Grids for current iteration
    gridX = jnp.zeros((Nx, Ny, Nz))
    gridY = jnp.zeros((Nx, Ny, Nz))
    gridZ = jnp.zeros((Nx, Ny, Nz))

    gridX = gridX.at[all_indices_x, all_indices_y, all_indices_z].add(
        jnp.ravel(
            (
                jnp.swapaxes(
                    jnp.swapaxes(
                        jnp.swapaxes(
                            gaussian_grid_spacing1
                            * jnp.resize(forces.at[0::3].get(), (gaussP, gaussP, gaussP, N)),
                            3,
                            2,
                        ),
                        2,
                        1,
                    ),
                    1,
                    0,
                )
            )
        )
    )
    gridY = gridY.at[all_indices_x, all_indices_y, all_indices_z].add(
        jnp.ravel(
            (
                jnp.swapaxes(
                    jnp.swapaxes(
                        jnp.swapaxes(
                            gaussian_grid_spacing1
                            * jnp.resize(forces.at[1::3].get(), (gaussP, gaussP, gaussP, N)),
                            3,
                            2,
                        ),
                        2,
                        1,
                    ),
                    1,
                    0,
                )
            )
        )
    )
    gridZ = gridZ.at[all_indices_x, all_indices_y, all_indices_z].add(
        jnp.ravel(
            (
                jnp.swapaxes(
                    jnp.swapaxes(
                        jnp.swapaxes(
                            gaussian_grid_spacing1
                            * jnp.resize(forces.at[2::3].get(), (gaussP, gaussP, gaussP, N)),
                            3,
                            2,
                        ),
                        2,
                        1,
                    ),
                    1,
                    0,
                )
            )
        )
    )

    # Apply FFT
    gridX = jnp.fft.fftn(gridX)
    gridY = jnp.fft.fftn(gridY)
    gridZ = jnp.fft.fftn(gridZ)

    gridk_sqr = (
        gridk.at[:, :, :, 0].get() * gridk.at[:, :, :, 0].get()
        + gridk.at[:, :, :, 1].get() * gridk.at[:, :, :, 1].get()
        + gridk.at[:, :, :, 2].get() * gridk.at[:, :, :, 2].get()
    )
    gridk_mod = jnp.sqrt(gridk_sqr)

    kdF = jnp.where(
        gridk_mod > 0,
        (
            gridk.at[:, :, :, 0].get() * gridX
            + gridk.at[:, :, :, 1].get() * gridY
            + gridk.at[:, :, :, 2].get() * gridZ
        )
        / gridk_sqr,
        0,
    )

    # UF part
    B = jnp.where(
        gridk_mod > 0,
        gridk.at[:, :, :, 3].get()
        * (jnp.sin(gridk_mod) / gridk_mod)
        * (jnp.sin(gridk_mod) / gridk_mod),
        0,
    )  # scaling factor
    gridX = B * (gridX - gridk.at[:, :, :, 0].get() * kdF)
    gridY = B * (gridY - gridk.at[:, :, :, 1].get() * kdF)
    gridZ = B * (gridZ - gridk.at[:, :, :, 2].get() * kdF)

    # Inverse FFT
    gridX = jnp.real(jnp.fft.ifftn(gridX, norm="forward"))
    gridY = jnp.real(jnp.fft.ifftn(gridY, norm="forward"))
    gridZ = jnp.real(jnp.fft.ifftn(gridZ, norm="forward"))

    # Compute Linear velocities and Velocity gradients (from which we can extract angular velocities and rate of strain)
    w_lin_velocities = jnp.zeros((N, 3), float)
    w_velocity_gradient = jnp.zeros((N, 8), float)

    w_lin_velocities = w_lin_velocities.at[:, 0].add(
        jnp.sum(
            gaussian_grid_spacing2
            * jnp.reshape(
                gridX.at[all_indices_x, all_indices_y, all_indices_z].get(),
                (N, gaussP, gaussP, gaussP),
            ),
            axis=(1, 2, 3),
        )
    )
    w_lin_velocities = w_lin_velocities.at[:, 1].add(
        jnp.sum(
            gaussian_grid_spacing2
            * jnp.reshape(
                gridY.at[all_indices_x, all_indices_y, all_indices_z].get(),
                (N, gaussP, gaussP, gaussP),
            ),
            axis=(1, 2, 3),
        )
    )
    w_lin_velocities = w_lin_velocities.at[:, 2].add(
        jnp.sum(
            gaussian_grid_spacing2
            * jnp.reshape(
                gridZ.at[all_indices_x, all_indices_y, all_indices_z].get(),
                (N, gaussP, gaussP, gaussP),
            ),
            axis=(1, 2, 3),
        )
    )

    ######################################## REAL SPACE CONTRIBUTION #########################################################################

    # Allocate arrays for Linear velocities and Velocity gradients (from which we can extract angular velocities and rate of strain)
    r_lin_velocities = jnp.zeros((N, 3), float)
    r_velocity_gradient = jnp.zeros((N, 8), float)

    # SELF CONTRIBUTIONS
    r_lin_velocities = r_lin_velocities.at[:, 0].set(m_self.at[0].get() * forces.at[0::3].get())
    r_lin_velocities = r_lin_velocities.at[:, 1].set(m_self.at[0].get() * forces.at[1::3].get())
    r_lin_velocities = r_lin_velocities.at[:, 2].set(m_self.at[0].get() * forces.at[2::3].get())

    # # Pair contributions

    # # Geometric quantities
    rdotf_j = (
        r.at[:, 0].get() * forces.at[3 * indices_j + 0].get()
        + r.at[:, 1].get() * forces.at[3 * indices_j + 1].get()
        + r.at[:, 2].get() * forces.at[3 * indices_j + 2].get()
    )
    mrdotf_i = -(
        r.at[:, 0].get() * forces.at[3 * indices_i + 0].get()
        + r.at[:, 1].get() * forces.at[3 * indices_i + 1].get()
        + r.at[:, 2].get() * forces.at[3 * indices_i + 2].get()
    )

    # Compute Velocity for particles i
    r_lin_velocities = r_lin_velocities.at[indices_i, 0].add(
        f1 * forces.at[3 * indices_j].get() + (f2 - f1) * rdotf_j * r.at[:, 0].get()
    )
    r_lin_velocities = r_lin_velocities.at[indices_i, 1].add(
        f1 * forces.at[3 * indices_j + 1].get() + (f2 - f1) * rdotf_j * r.at[:, 1].get()
    )
    r_lin_velocities = r_lin_velocities.at[indices_i, 2].add(
        f1 * forces.at[3 * indices_j + 2].get() + (f2 - f1) * rdotf_j * r.at[:, 2].get()
    )
    # Compute Velocity for particles j
    r_lin_velocities = r_lin_velocities.at[indices_j, 0].add(
        f1 * forces.at[3 * indices_i].get() - (f2 - f1) * mrdotf_i * r.at[:, 0].get()
    )
    r_lin_velocities = r_lin_velocities.at[indices_j, 1].add(
        f1 * forces.at[3 * indices_i + 1].get() - (f2 - f1) * mrdotf_i * r.at[:, 1].get()
    )
    r_lin_velocities = r_lin_velocities.at[indices_j, 2].add(
        f1 * forces.at[3 * indices_i + 2].get() - (f2 - f1) * mrdotf_i * r.at[:, 2].get()
    )

    # Compute Velocity Gradient for particles i and j
    r_velocity_gradient = r_velocity_gradient.at[indices_i, 0].add(
        (-1)
        * g1
        * (
            r.at[:, 0].get() * forces.at[3 * indices_j + 0].get()
            - rdotf_j * r.at[:, 0].get() * r.at[:, 0].get()
        )
        + (-1)
        * g2
        * (
            rdotf_j
            + forces.at[3 * indices_j + 0].get() * r.at[:, 0].get()
            - 4.0 * rdotf_j * r.at[:, 0].get() * r.at[:, 0].get()
        )
    )

    r_velocity_gradient = r_velocity_gradient.at[indices_j, 0].add(
        (-1)
        * g1
        * (
            -r.at[:, 0].get() * forces.at[3 * indices_i + 0].get()
            - mrdotf_i * r.at[:, 0].get() * r.at[:, 0].get()
        )
        + (-1)
        * g2
        * (
            mrdotf_i
            - forces.at[3 * indices_i + 0].get() * r.at[:, 0].get()
            - 4.0 * mrdotf_i * r.at[:, 0].get() * r.at[:, 0].get()
        )
    )

    r_velocity_gradient = r_velocity_gradient.at[indices_i, 1].add(
        (-1)
        * g1
        * (
            r.at[:, 1].get() * forces.at[3 * indices_j + 0].get()
            - rdotf_j * r.at[:, 1].get() * r.at[:, 0].get()
        )
        + (-1)
        * g2
        * (
            forces.at[3 * indices_j + 1].get() * r.at[:, 0].get()
            - 4.0 * rdotf_j * r.at[:, 1].get() * r.at[:, 0].get()
        )
    )

    r_velocity_gradient = r_velocity_gradient.at[indices_j, 1].add(
        (-1)
        * g1
        * (
            -r.at[:, 1].get() * forces.at[3 * indices_i + 0].get()
            - mrdotf_i * r.at[:, 1].get() * r.at[:, 0].get()
        )
        + (-1)
        * g2
        * (
            -forces.at[3 * indices_i + 1].get() * r.at[:, 0].get()
            - 4.0 * mrdotf_i * r.at[:, 1].get() * r.at[:, 0].get()
        )
    )

    r_velocity_gradient = r_velocity_gradient.at[indices_i, 2].add(
        (-1)
        * g1
        * (
            r.at[:, 2].get() * forces.at[3 * indices_j + 0].get()
            - rdotf_j * r.at[:, 2].get() * r.at[:, 0].get()
        )
        + (-1)
        * g2
        * (
            forces.at[3 * indices_j + 2].get() * r.at[:, 0].get()
            - 4.0 * rdotf_j * r.at[:, 2].get() * r.at[:, 0].get()
        )
    )

    r_velocity_gradient = r_velocity_gradient.at[indices_j, 2].add(
        (-1)
        * g1
        * (
            -r.at[:, 2].get() * forces.at[3 * indices_i + 0].get()
            - mrdotf_i * r.at[:, 2].get() * r.at[:, 0].get()
        )
        + (-1)
        * g2
        * (
            -forces.at[3 * indices_i + 2].get() * r.at[:, 0].get()
            - 4.0 * mrdotf_i * r.at[:, 2].get() * r.at[:, 0].get()
        )
    )

    r_velocity_gradient = r_velocity_gradient.at[indices_i, 3].add(
        (-1)
        * g1
        * (
            r.at[:, 2].get() * forces.at[3 * indices_j + 1].get()
            - rdotf_j * r.at[:, 2].get() * r.at[:, 1].get()
        )
        + (-1)
        * g2
        * (
            forces.at[3 * indices_j + 2].get() * r.at[:, 1].get()
            - 4.0 * rdotf_j * r.at[:, 2].get() * r.at[:, 1].get()
        )
    )

    r_velocity_gradient = r_velocity_gradient.at[indices_j, 3].add(
        (-1)
        * g1
        * (
            -r.at[:, 2].get() * forces.at[3 * indices_i + 1].get()
            - mrdotf_i * r.at[:, 2].get() * r.at[:, 1].get()
        )
        + (-1)
        * g2
        * (
            -forces.at[3 * indices_i + 2].get() * r.at[:, 1].get()
            - 4.0 * mrdotf_i * r.at[:, 2].get() * r.at[:, 1].get()
        )
    )

    r_velocity_gradient = r_velocity_gradient.at[indices_i, 4].add(
        (-1)
        * g1
        * (
            r.at[:, 1].get() * forces.at[3 * indices_j + 1].get()
            - rdotf_j * r.at[:, 1].get() * r.at[:, 1].get()
        )
        + (-1)
        * g2
        * (
            rdotf_j
            + forces.at[3 * indices_j + 1].get() * r.at[:, 1].get()
            - 4.0 * rdotf_j * r.at[:, 1].get() * r.at[:, 1].get()
        )
    )

    r_velocity_gradient = r_velocity_gradient.at[indices_j, 4].add(
        (-1)
        * g1
        * (
            -r.at[:, 1].get() * forces.at[3 * indices_i + 1].get()
            - mrdotf_i * r.at[:, 1].get() * r.at[:, 1].get()
        )
        + (-1)
        * g2
        * (
            mrdotf_i
            - forces.at[3 * indices_i + 1].get() * r.at[:, 1].get()
            - 4.0 * mrdotf_i * r.at[:, 1].get() * r.at[:, 1].get()
        )
    )

    r_velocity_gradient = r_velocity_gradient.at[indices_i, 5].add(
        (-1)
        * g1
        * (
            r.at[:, 0].get() * forces.at[3 * indices_j + 1].get()
            - rdotf_j * r.at[:, 0].get() * r.at[:, 1].get()
        )
        + (-1)
        * g2
        * (
            forces.at[3 * indices_j + 0].get() * r.at[:, 1].get()
            - 4.0 * rdotf_j * r.at[:, 0].get() * r.at[:, 1].get()
        )
    )

    r_velocity_gradient = r_velocity_gradient.at[indices_j, 5].add(
        (-1)
        * g1
        * (
            -r.at[:, 0].get() * forces.at[3 * indices_i + 1].get()
            - mrdotf_i * r.at[:, 0].get() * r.at[:, 1].get()
        )
        + (-1)
        * g2
        * (
            -forces.at[3 * indices_i + 0].get() * r.at[:, 1].get()
            - 4.0 * mrdotf_i * r.at[:, 0].get() * r.at[:, 1].get()
        )
    )

    r_velocity_gradient = r_velocity_gradient.at[indices_i, 6].add(
        (-1)
        * g1
        * (
            r.at[:, 0].get() * forces.at[3 * indices_j + 2].get()
            - rdotf_j * r.at[:, 0].get() * r.at[:, 2].get()
        )
        + (-1)
        * g2
        * (
            forces.at[3 * indices_j + 0].get() * r.at[:, 2].get()
            - 4.0 * rdotf_j * r.at[:, 0].get() * r.at[:, 2].get()
        )
    )

    r_velocity_gradient = r_velocity_gradient.at[indices_j, 6].add(
        (-1)
        * g1
        * (
            -r.at[:, 0].get() * forces.at[3 * indices_i + 2].get()
            - mrdotf_i * r.at[:, 0].get() * r.at[:, 2].get()
        )
        + (-1)
        * g2
        * (
            -forces.at[3 * indices_i + 0].get() * r.at[:, 2].get()
            - 4.0 * mrdotf_i * r.at[:, 0].get() * r.at[:, 2].get()
        )
    )

    r_velocity_gradient = r_velocity_gradient.at[indices_i, 7].add(
        (-1)
        * g1
        * (
            r.at[:, 1].get() * forces.at[3 * indices_j + 2].get()
            - rdotf_j * r.at[:, 1].get() * r.at[:, 2].get()
        )
        + (-1)
        * g2
        * (
            forces.at[3 * indices_j + 1].get() * r.at[:, 2].get()
            - 4.0 * rdotf_j * r.at[:, 1].get() * r.at[:, 2].get()
        )
    )

    r_velocity_gradient = r_velocity_gradient.at[indices_j, 7].add(
        (-1)
        * g1
        * (
            -r.at[:, 1].get() * forces.at[3 * indices_i + 2].get()
            - mrdotf_i * r.at[:, 1].get() * r.at[:, 2].get()
        )
        + (-1)
        * g2
        * (
            -forces.at[3 * indices_i + 1].get() * r.at[:, 2].get()
            - 4.0 * mrdotf_i * r.at[:, 1].get() * r.at[:, 2].get()
        )
    )

    ##########################################################################################################################################
    ##########################################################################################################################################

    # Add wave and real space part together
    lin_vel = w_lin_velocities + r_lin_velocities
    velocity_gradient = w_velocity_gradient + r_velocity_gradient

    # # Convert to angular velocities and rate of strain
    ang_vel_and_strain = jnp.zeros((N, 3))
    ang_vel_and_strain = ang_vel_and_strain.at[:, 0].set(
        (velocity_gradient.at[:, 3].get() - velocity_gradient.at[:, 7].get()) * 0.5
    )
    ang_vel_and_strain = ang_vel_and_strain.at[:, 1].set(
        (velocity_gradient.at[:, 6].get() - velocity_gradient.at[:, 2].get()) * 0.5
    )
    ang_vel_and_strain = ang_vel_and_strain.at[:, 2].set(
        (velocity_gradient.at[:, 1].get() - velocity_gradient.at[:, 5].get()) * 0.5
    )

    # Convert to Generalized Velocities
    generalized_velocities = jnp.zeros(
        6 * N
    )  # First 6N entries for U (linear and angular velocities)

    generalized_velocities = generalized_velocities.at[0 : 6 * N : 6].set(lin_vel.at[:, 0].get())
    generalized_velocities = generalized_velocities.at[1 : 6 * N : 6].set(lin_vel.at[:, 1].get())
    generalized_velocities = generalized_velocities.at[2 : 6 * N : 6].set(lin_vel.at[:, 2].get())
    generalized_velocities = generalized_velocities.at[3 : 6 * N : 6].set(
        ang_vel_and_strain.at[:, 0].get()
    )
    generalized_velocities = generalized_velocities.at[4 : 6 * N : 6].set(
        ang_vel_and_strain.at[:, 1].get()
    )
    generalized_velocities = generalized_velocities.at[5 : 6 * N : 6].set(
        ang_vel_and_strain.at[:, 2].get()
    )

    # Clean Grids for next iteration
    gridX = jnp.zeros((Nx, Ny, Nz))
    gridY = jnp.zeros((Nx, Ny, Nz))
    gridZ = jnp.zeros((Nx, Ny, Nz))

    return generalized_velocities


@partial(jit, static_argnums=[0])
def GeneralizedMobility_open(
    N: int,
    r: ArrayLike,
    indices_i: ArrayLike,
    indices_j: ArrayLike,
    generalized_forces: ArrayLike,
    mobil_scal: ArrayLike,
) -> Array:
    """Compute the matrix-vector product of the grandmobility matrix with a generalized force vector (and stresslet), in open boundary conditions.

    Parameters
    ----------
    N: (int)
        Number of particles
    r: (float)
        Array (N*(N-1)/2) ,3) containing the interparticle unit vectors for each pair of particle
    indices_i: (int)
        Array (,N*(N-1)/2) of indices of first particle in open boundaries list of pairs
    indices_j: (int)
        Array (,N*(N-1)/2) of indices of second particle in open boundaries list of pairs
    generalized_forces: (float)
        Array (,11*N) containing input generalized forces (force/torque/stresslet)
    mobil_scal: (float)
        Array (11,N*(N-1)/2)) containing mobility functions evaluated for the current particle configuration

    Returns
    -------
    generalized_velocities (linear/angular velocities and rateOfStrain)

    """
    strain = jnp.zeros((N, 5), float)
    velocities = jnp.zeros((N, 6), float)

    forces_torques = generalized_forces[: 6 * N]
    forces_torques = -forces_torques
    ft_i = (jnp.reshape(forces_torques, (N, 6))).at[indices_i].get()
    ft_j = (jnp.reshape(forces_torques, (N, 6))).at[indices_j].get()

    stresslets = generalized_forces[
        6 * N :
    ]  # stresslet in vector form has the format [Sxx,Sxy,Sxz,Syz,Syy]
    # stresslets = -stresslets

    s_i = (jnp.reshape(stresslets, (N, 5))).at[indices_i].get()
    # s_i = jnp.array([[(1.0/3.0) * ( 2.0 * s_i[:,0] - s_i[:,4] ) ,    0.5 * s_i[:,1]                 ,     0.5 * s_i[:,2] ],
    #                   [0.5 * s_i[:,1]                        ,(1.0/3.0)*(-s_i[:,0]+2.0*s_i[:,4])    ,     0.5 * s_i[:,3] ],
    #                   [0.5 * s_i[:,2]                        ,0.5 * s_i[:,3]                     ,  (-1.0/3.0) * ( s_i[:,0] + s_i[:,4] ) ]])
    s_i = jnp.array(
        [
            [s_i[:, 0], s_i[:, 1], s_i[:, 2]],
            [s_i[:, 1], s_i[:, 4], s_i[:, 3]],
            [s_i[:, 2], s_i[:, 3], -s_i[:, 0] - s_i[:, 4]],
        ]
    )

    s_j = (jnp.reshape(stresslets, (N, 5))).at[indices_j].get()
    s_j = jnp.array(
        [
            [s_j[:, 0], s_j[:, 1], s_j[:, 2]],
            [s_j[:, 1], s_j[:, 4], s_j[:, 3]],
            [s_j[:, 2], s_j[:, 3], -s_j[:, 0] - s_j[:, 4]],
        ]
    )
    r = -r

    # Dot product of levi-civita-symbol and r
    epsr = jnp.array(
        [
            [jnp.zeros(int(N * (N - 1) / 2)), r[:, 2], -r[:, 1]],
            [-r[:, 2], jnp.zeros(int(N * (N - 1) / 2)), r[:, 0]],
            [r[:, 1], -r[:, 0], jnp.zeros(int(N * (N - 1) / 2))],
        ]
    )

    # Dot product of r and U, i.e. axisymmetric projection
    rdfi = (
        r.at[:, 0].get() * ft_i.at[:, 0].get()
        + r.at[:, 1].get() * ft_i.at[:, 1].get()
        + r.at[:, 2].get() * ft_i.at[:, 2].get()
    )
    rdfj = (
        r.at[:, 0].get() * ft_j.at[:, 0].get()
        + r.at[:, 1].get() * ft_j.at[:, 1].get()
        + r.at[:, 2].get() * ft_j.at[:, 2].get()
    )
    rdti = (
        r.at[:, 0].get() * ft_i.at[:, 3].get()
        + r.at[:, 1].get() * ft_i.at[:, 4].get()
        + r.at[:, 2].get() * ft_i.at[:, 5].get()
    )
    rdtj = (
        r.at[:, 0].get() * ft_j.at[:, 3].get()
        + r.at[:, 1].get() * ft_j.at[:, 4].get()
        + r.at[:, 2].get() * ft_j.at[:, 5].get()
    )

    # Cross product of U and r, i.e. eps_ijk*r_k*U_j = Px dot U, (eps_ijk is the Levi-Civita symbol)
    epsrdfi = jnp.array(
        [
            r.at[:, 2].get() * ft_i.at[:, 1].get() - r.at[:, 1].get() * ft_i.at[:, 2].get(),
            -r.at[:, 2].get() * ft_i.at[:, 0].get() + r.at[:, 0].get() * ft_i.at[:, 2].get(),
            r.at[:, 1].get() * ft_i.at[:, 0].get() - r.at[:, 0].get() * ft_i.at[:, 1].get(),
        ]
    )

    epsrdti = jnp.array(
        [
            r.at[:, 2].get() * ft_i.at[:, 4].get() - r.at[:, 1].get() * ft_i.at[:, 5].get(),
            -r.at[:, 2].get() * ft_i.at[:, 3].get() + r.at[:, 0].get() * ft_i.at[:, 5].get(),
            r.at[:, 1].get() * ft_i.at[:, 3].get() - r.at[:, 0].get() * ft_i.at[:, 4].get(),
        ]
    )

    epsrdfj = jnp.array(
        [
            r.at[:, 2].get() * ft_j.at[:, 1].get() - r.at[:, 1].get() * ft_j.at[:, 2].get(),
            -r.at[:, 2].get() * ft_j.at[:, 0].get() + r.at[:, 0].get() * ft_j.at[:, 2].get(),
            r.at[:, 1].get() * ft_j.at[:, 0].get() - r.at[:, 0].get() * ft_j.at[:, 1].get(),
        ]
    )

    epsrdtj = jnp.array(
        [
            r.at[:, 2].get() * ft_j.at[:, 4].get() - r.at[:, 1].get() * ft_j.at[:, 5].get(),
            -r.at[:, 2].get() * ft_j.at[:, 3].get() + r.at[:, 0].get() * ft_j.at[:, 5].get(),
            r.at[:, 1].get() * ft_j.at[:, 3].get() - r.at[:, 0].get() * ft_j.at[:, 4].get(),
        ]
    )

    Sdri = jnp.array(
        [
            s_i.at[0, 0, :].get() * r.at[:, 0].get()
            + s_i.at[0, 1, :].get() * r.at[:, 1].get()
            + s_i.at[0, 2, :].get() * r.at[:, 2].get(),
            s_i.at[1, 0, :].get() * r.at[:, 0].get()
            + s_i.at[1, 1, :].get() * r.at[:, 1].get()
            + s_i.at[1, 2, :].get() * r.at[:, 2].get(),
            s_i.at[2, 0, :].get() * r.at[:, 0].get()
            + s_i.at[2, 1, :].get() * r.at[:, 1].get()
            + s_i.at[2, 2, :].get() * r.at[:, 2].get(),
        ]
    )

    Sdrj = jnp.array(
        [
            s_j.at[0, 0, :].get() * r.at[:, 0].get()
            + s_j.at[0, 1, :].get() * r.at[:, 1].get()
            + s_j.at[0, 2, :].get() * r.at[:, 2].get(),
            s_j.at[1, 0, :].get() * r.at[:, 0].get()
            + s_j.at[1, 1, :].get() * r.at[:, 1].get()
            + s_j.at[1, 2, :].get() * r.at[:, 2].get(),
            s_j.at[2, 0, :].get() * r.at[:, 0].get()
            + s_j.at[2, 1, :].get() * r.at[:, 1].get()
            + s_j.at[2, 2, :].get() * r.at[:, 2].get(),
        ]
    )

    rdSdri = (
        r.at[:, 0].get() * Sdri.at[0].get()
        + r.at[:, 1].get() * Sdri.at[1].get()
        + r.at[:, 2].get() * Sdri.at[2].get()
    )
    rdSdrj = (
        r.at[:, 0].get() * Sdrj.at[0].get()
        + r.at[:, 1].get() * Sdrj.at[1].get()
        + r.at[:, 2].get() * Sdrj.at[2].get()
    )

    epsrdSdri = jnp.array(
        [
            epsr[0, 0, :] * Sdri[0, :] + epsr[0, 1, :] * Sdri[1, :] + epsr[0, 2, :] * Sdri[2, :],
            epsr[1, 0, :] * Sdri[0, :] + epsr[1, 1, :] * Sdri[1, :] + epsr[1, 2, :] * Sdri[2, :],
            epsr[2, 0, :] * Sdri[0, :] + epsr[2, 1, :] * Sdri[1, :] + epsr[2, 2, :] * Sdri[2, :],
        ]
    )
    epsrdSdrj = jnp.array(
        [
            epsr[0, 0, :] * Sdrj[0, :] + epsr[0, 1, :] * Sdrj[1, :] + epsr[0, 2, :] * Sdrj[2, :],
            epsr[1, 0, :] * Sdrj[0, :] + epsr[1, 1, :] * Sdrj[1, :] + epsr[1, 2, :] * Sdrj[2, :],
            epsr[2, 0, :] * Sdrj[0, :] + epsr[2, 1, :] * Sdrj[1, :] + epsr[2, 2, :] * Sdrj[2, :],
        ]
    )

    xa12 = mobil_scal[0]
    ya12 = mobil_scal[1]
    yb12 = mobil_scal[2]
    xc12 = mobil_scal[3]
    yc12 = mobil_scal[4]
    xm12 = mobil_scal[5]
    ym12 = mobil_scal[6]
    zm12 = mobil_scal[7]
    xg12 = mobil_scal[8]
    yg12 = mobil_scal[9]
    yh12 = mobil_scal[10]
    n_pairs = int(N * (N - 1) / 2)

    # normalize self terms (avoid double counting)
    normaliz_factor = jnp.where(N > 1, N - 1, 1)

    # M_UF * F

    # Compute the contributions to the force for particles i (Fi = A11*Ui + A12*Uj + BT11*Wi + BT12*Wj)
    u = (
        (jnp.ones(n_pairs)).at[:, None].get() * (ft_i.at[:, :3].get() / normaliz_factor)
        + (xa12 - ya12).at[:, None].get() * rdfj.at[:, None].get() * r
        + ya12.at[:, None].get() * ft_j.at[:, :3].get()
        + (-yb12).at[:, None].get() * (-epsrdtj.T)
    )
    velocities = velocities.at[indices_i, :3].add(u)
    # Compute the contributions to the force for particles j (Fj = A11*Uj + A12*Ui + BT11*Wj + BT12*Wi)
    u = (
        (jnp.ones(n_pairs)).at[:, None].get() * (ft_j.at[:, :3].get() / normaliz_factor)
        + (xa12 - ya12).at[:, None].get() * rdfi.at[:, None].get() * r
        + ya12.at[:, None].get() * ft_i.at[:, :3].get()
        + (-yb12).at[:, None].get() * (epsrdti.T)
    )
    velocities = velocities.at[indices_j, :3].add(u)
    # Compute the contributions to the torque for particles i (Li = B11*Ui + B12*Uj + C11*Wi + C12*Wj)
    w = (
        (3 / 4 * jnp.ones(n_pairs)).at[:, None].get() * (ft_i.at[:, 3:].get() / normaliz_factor)
        + yb12.at[:, None].get() * epsrdfj.T
        + (xc12 - yc12).at[:, None].get() * rdtj.at[:, None].get() * r
        + yc12.at[:, None].get() * ft_j.at[:, 3:].get()
    )
    velocities = velocities.at[indices_i, 3:].add(w)
    # Compute the contributions to the torque for particles j (Lj = B11*Uj + B12*Ui + C11*Wj + C12*Wi)
    w = (
        (3 / 4 * jnp.ones(n_pairs)).at[:, None].get() * (ft_j.at[:, 3:].get() / normaliz_factor)
        - yb12.at[:, None].get() * epsrdfi.T
        + (xc12 - yc12).at[:, None].get() * rdti.at[:, None].get() * r
        + yc12.at[:, None].get() * ft_i.at[:, 3:].get()
    )
    velocities = velocities.at[indices_j, 3:].add(w)

    # M_US * S

    u = ((-xg12) - 2.0 * (-yg12)).at[:, None].get() * (rdSdrj.at[:, None].get()) * r + 2.0 * (
        -yg12
    ).at[:, None].get() * (Sdrj.T)
    w = yh12.at[:, None].get() * (2.0 * epsrdSdrj.T)

    velocities = velocities.at[indices_i, :3].add(u)
    velocities = velocities.at[indices_i, 3:].add(w)

    u = ((-xg12) - 2.0 * (-yg12)).at[:, None].get() * (rdSdri.at[:, None].get()) * (-r) + 2.0 * (
        -yg12
    ).at[:, None].get() * (-Sdri.T)
    w = yh12.at[:, None].get() * (2.0 * epsrdSdri.T)

    velocities = velocities.at[indices_j, :3].add(u)
    velocities = velocities.at[indices_j, 3:].add(w)

    # #M_EF * F
    # translational part

    # strain_xx component
    strain_xx_i = +xg12 * (r[:, 0] * r[:, 0] - 1.0 / 3.0) * rdfj + yg12 * (
        2 * ft_j[:, 0] * r[:, 0] - 2.0 * r[:, 0] * r[:, 0] * rdfj
    )
    # strain_xy component
    strain_xy_i = +xg12 * (r[:, 0] * r[:, 1]) * rdfj + yg12 * (
        ft_j[:, 0] * r[:, 1] + r[:, 0] * ft_j[:, 1] - 2.0 * r[:, 0] * r[:, 1] * rdfj
    )
    # strain_xz component
    strain_xz_i = +xg12 * (r[:, 0] * r[:, 2]) * rdfj + yg12 * (
        ft_j[:, 0] * r[:, 2] + r[:, 0] * ft_j[:, 2] - 2.0 * r[:, 0] * r[:, 2] * rdfj
    )
    # strain_yz component
    strain_yz_i = +xg12 * (r[:, 1] * r[:, 2]) * rdfj + yg12 * (
        ft_j[:, 1] * r[:, 2] + r[:, 1] * ft_j[:, 2] - 2.0 * r[:, 1] * r[:, 2] * rdfj
    )
    # strain_yy component
    strain_yy_i = +xg12 * (r[:, 1] * r[:, 1] - 1.0 / 3.0) * rdfj + yg12 * (
        2 * ft_j[:, 1] * r[:, 1] - 2.0 * r[:, 1] * r[:, 1] * rdfj
    )
    # compute strain for particles j
    # strain_xx component
    strain_xx_j = +xg12 * (r[:, 0] * r[:, 0] - 1.0 / 3.0) * (-rdfi) + yg12 * (
        2.0 * ft_i[:, 0] * (-r[:, 0]) - 2.0 * r[:, 0] * r[:, 0] * (-rdfi)
    )
    # strain_xy component
    strain_xy_j = +xg12 * (r[:, 0] * r[:, 1]) * (-rdfi) + yg12 * (
        ft_i[:, 0] * (-r[:, 1]) + (-r[:, 0]) * ft_i[:, 1] - 2.0 * r[:, 0] * r[:, 1] * (-rdfi)
    )
    # strain_xz component
    strain_xz_j = +xg12 * (r[:, 0] * r[:, 2]) * (-rdfi) + yg12 * (
        ft_i[:, 0] * (-r[:, 2]) + (-r[:, 0]) * ft_i[:, 2] - 2.0 * r[:, 0] * r[:, 2] * (-rdfi)
    )
    # strain_yz component
    strain_yz_j = +xg12 * (r[:, 1] * r[:, 2]) * (-rdfi) + yg12 * (
        ft_i[:, 1] * (-r[:, 2]) + (-r[:, 1]) * ft_i[:, 2] - 2.0 * r[:, 1] * r[:, 2] * (-rdfi)
    )
    # strain_yy component
    strain_yy_j = +xg12 * (r[:, 1] * r[:, 1] - 1.0 / 3.0) * (-rdfi) + yg12 * (
        2.0 * ft_i[:, 1] * (-r[:, 1]) - 2.0 * r[:, 1] * r[:, 1] * (-rdfi)
    )

    # rotational part

    # compute strain for particles i
    strain_xx_i += yh12 * (r[:, 0] * epsrdtj[0] + epsrdtj[0] * r[:, 0])
    strain_xy_i += yh12 * (r[:, 0] * epsrdtj[1] + epsrdtj[0] * r[:, 1])
    strain_xz_i += yh12 * (r[:, 0] * epsrdtj[2] + epsrdtj[0] * r[:, 2])
    strain_yz_i += yh12 * (r[:, 1] * epsrdtj[2] + epsrdtj[1] * r[:, 2])
    strain_yy_i += yh12 * (r[:, 1] * epsrdtj[1] + epsrdtj[1] * r[:, 1])

    # compute strain for particles j
    strain_xx_j += yh12 * (r[:, 0] * epsrdti[0] + epsrdti[0] * r[:, 0])
    strain_xy_j += yh12 * (r[:, 0] * epsrdti[1] + epsrdti[0] * r[:, 1])
    strain_xz_j += yh12 * (r[:, 0] * epsrdti[2] + epsrdti[0] * r[:, 2])
    strain_yz_j += yh12 * (r[:, 1] * epsrdti[2] + epsrdti[1] * r[:, 2])
    strain_yy_j += yh12 * (r[:, 1] * epsrdti[1] + epsrdti[1] * r[:, 1])

    strain = strain.at[indices_i, 0].add((2.0 * strain_xx_i + strain_yy_i))
    strain = strain.at[indices_i, 1].add(2.0 * strain_xy_i)
    strain = strain.at[indices_i, 2].add(2.0 * strain_xz_i)
    strain = strain.at[indices_i, 3].add(2.0 * strain_yz_i)
    strain = strain.at[indices_i, 4].add((strain_xx_i + 2.0 * strain_yy_i))

    strain = strain.at[indices_j, 0].add((2.0 * strain_xx_j + strain_yy_j))
    strain = strain.at[indices_j, 1].add(2.0 * strain_xy_j)
    strain = strain.at[indices_j, 2].add(2.0 * strain_xz_j)
    strain = strain.at[indices_j, 3].add(2.0 * strain_yz_j)
    strain = strain.at[indices_j, 4].add((strain_xx_j + 2.0 * strain_yy_j))

    # M_ES * S

    # compute strain for particles i
    # strain_xx component
    strain_xx_i = (
        9 / 10 * jnp.ones(n_pairs) * (s_i[0, 0, :]) / normaliz_factor
        + 1.5 * xm12 * (r[:, 0] * r[:, 0] - 1.0 / 3.0) * rdSdrj
        + 0.5 * ym12 * (4.0 * r[:, 0] * Sdrj[0] - 4.0 * rdSdrj * r[:, 0] * r[:, 0])
        + 0.5
        * zm12
        * (2.0 * s_j[0, 0, :] + (1.0 + r[:, 0] * r[:, 0]) * rdSdrj - 4.0 * r[:, 0] * Sdrj[0])
    )
    strain_xx_j = (
        9 / 10 * jnp.ones(n_pairs) * (s_j[0, 0, :]) / normaliz_factor
        + 1.5 * xm12 * (r[:, 0] * r[:, 0] - 1.0 / 3.0) * rdSdri
        + 0.5 * ym12 * (4.0 * r[:, 0] * Sdri[0] - 4.0 * rdSdri * r[:, 0] * r[:, 0])
        + 0.5
        * zm12
        * (2.0 * s_i[0, 0, :] + (1.0 + r[:, 0] * r[:, 0]) * rdSdri - 4.0 * r[:, 0] * Sdri[0])
    )

    # strain_xy component
    strain_xy_i = (
        9 / 10 * jnp.ones(n_pairs) * (s_i[0, 1, :]) / normaliz_factor
        + 1.5 * xm12 * (r[:, 0] * r[:, 1]) * rdSdrj
        + 0.5
        * ym12
        * (2.0 * r[:, 0] * Sdrj[1] + 2.0 * r[:, 1] * Sdrj[0] - 4.0 * rdSdrj * r[:, 0] * r[:, 1])
        + 0.5
        * zm12
        * (
            2.0 * s_j[0, 1, :]
            + (r[:, 0] * r[:, 1]) * rdSdrj
            - 2.0 * r[:, 0] * Sdrj[1]
            - 2.0 * r[:, 1] * Sdrj[0]
        )
    )
    strain_xy_j = (
        9 / 10 * jnp.ones(n_pairs) * (s_j[0, 1, :]) / normaliz_factor
        + 1.5 * xm12 * (r[:, 0] * r[:, 1]) * rdSdri
        + 0.5
        * ym12
        * (2.0 * r[:, 0] * Sdri[1] + 2.0 * r[:, 1] * Sdri[0] - 4.0 * rdSdri * r[:, 0] * r[:, 1])
        + 0.5
        * zm12
        * (
            2.0 * s_i[0, 1, :]
            + (r[:, 0] * r[:, 1]) * rdSdri
            - 2.0 * r[:, 0] * Sdri[1]
            - 2.0 * r[:, 1] * Sdri[0]
        )
    )

    # strain_xz component
    strain_xz_i = (
        9 / 10 * jnp.ones(n_pairs) * (s_i[0, 2, :]) / normaliz_factor
        + 1.5 * xm12 * (r[:, 0] * r[:, 2]) * rdSdrj
        + 0.5
        * ym12
        * (2.0 * r[:, 0] * Sdrj[2] + 2.0 * r[:, 2] * Sdrj[0] - 4.0 * rdSdrj * r[:, 0] * r[:, 2])
        + 0.5
        * zm12
        * (
            2.0 * s_j[0, 2, :]
            + (r[:, 0] * r[:, 2]) * rdSdrj
            - 2.0 * r[:, 0] * Sdrj[2]
            - 2.0 * r[:, 2] * Sdrj[0]
        )
    )
    strain_xz_j = (
        9 / 10 * jnp.ones(n_pairs) * (s_j[0, 2, :]) / normaliz_factor
        + 1.5 * xm12 * (r[:, 0] * r[:, 2]) * rdSdri
        + 0.5
        * ym12
        * (2.0 * r[:, 0] * Sdri[2] + 2.0 * r[:, 2] * Sdri[0] - 4.0 * rdSdri * r[:, 0] * r[:, 2])
        + 0.5
        * zm12
        * (
            2.0 * s_i[0, 2, :]
            + (r[:, 0] * r[:, 2]) * rdSdri
            - 2.0 * r[:, 0] * Sdri[2]
            - 2.0 * r[:, 2] * Sdri[0]
        )
    )

    # strain_yz component
    strain_yz_i = (
        9 / 10 * jnp.ones(n_pairs) * (s_i[1, 2, :]) / normaliz_factor
        + 1.5 * xm12 * (r[:, 1] * r[:, 2]) * rdSdrj
        + 0.5
        * ym12
        * (2.0 * r[:, 1] * Sdrj[2] + 2.0 * r[:, 2] * Sdrj[1] - 4.0 * rdSdrj * r[:, 1] * r[:, 2])
        + 0.5
        * zm12
        * (
            2.0 * s_j[1, 2, :]
            + (r[:, 1] * r[:, 2]) * rdSdrj
            - 2.0 * r[:, 1] * Sdrj[2]
            - 2.0 * r[:, 2] * Sdrj[1]
        )
    )
    strain_yz_j = (
        9 / 10 * jnp.ones(n_pairs) * (s_j[1, 2, :]) / normaliz_factor
        + 1.5 * xm12 * (r[:, 1] * r[:, 2]) * rdSdri
        + 0.5
        * ym12
        * (2.0 * r[:, 1] * Sdri[2] + 2.0 * r[:, 2] * Sdri[1] - 4.0 * rdSdri * r[:, 1] * r[:, 2])
        + 0.5
        * zm12
        * (
            2.0 * s_i[1, 2, :]
            + (r[:, 1] * r[:, 2]) * rdSdri
            - 2.0 * r[:, 1] * Sdri[2]
            - 2.0 * r[:, 2] * Sdri[1]
        )
    )

    # strain_yy component
    strain_yy_i = (
        9 / 10 * jnp.ones(n_pairs) * (s_i[1, 1, :]) / normaliz_factor
        + 1.5 * xm12 * (r[:, 1] * r[:, 1] - 1.0 / 3.0) * rdSdrj
        + 0.5 * ym12 * (4.0 * r[:, 1] * Sdrj[1] - 4.0 * rdSdrj * r[:, 1] * r[:, 1])
        + 0.5
        * zm12
        * (2.0 * s_j[1, 1, :] + (1.0 + r[:, 1] * r[:, 1]) * rdSdrj - 4.0 * r[:, 1] * Sdrj[1])
    )
    strain_yy_j = (
        9 / 10 * jnp.ones(n_pairs) * (s_j[1, 1, :]) / normaliz_factor
        + 1.5 * xm12 * (r[:, 1] * r[:, 1] - 1.0 / 3.0) * rdSdri
        + 0.5 * ym12 * (4.0 * r[:, 1] * Sdri[1] - 4.0 * rdSdri * r[:, 1] * r[:, 1])
        + 0.5
        * zm12
        * (2.0 * s_i[1, 1, :] + (1.0 + r[:, 1] * r[:, 1]) * rdSdri - 4.0 * r[:, 1] * Sdri[1])
    )

    strain = strain.at[indices_i, 0].add((2.0 * strain_xx_i + strain_yy_i))
    strain = strain.at[indices_i, 1].add(2.0 * strain_xy_i)
    strain = strain.at[indices_i, 2].add(2.0 * strain_xz_i)
    strain = strain.at[indices_i, 3].add(2.0 * strain_yz_i)
    strain = strain.at[indices_i, 4].add((strain_xx_i + 2.0 * strain_yy_i))

    strain = strain.at[indices_j, 0].add((2.0 * strain_xx_j + strain_yy_j))
    strain = strain.at[indices_j, 1].add(2.0 * strain_xy_j)
    strain = strain.at[indices_j, 2].add(2.0 * strain_xz_j)
    strain = strain.at[indices_j, 3].add(2.0 * strain_yz_j)
    strain = strain.at[indices_j, 4].add((strain_xx_j + 2.0 * strain_yy_j))

    r = -r  # reset sign to original
    velocities = jnp.ravel(velocities)
    strain = jnp.ravel(strain)
    gen_vel = jnp.zeros(11 * N)
    # mobility is build to return the particle velocity (instead of -velocity)
    # and minus the ambient rate of strain (instead of the ambient rate of strain)
    # this is because of the input 'r' 'stresslets' and 'forces/torques'
    gen_vel = gen_vel.at[: 6 * N].set(-velocities)
    gen_vel = gen_vel.at[6 * N :].set(strain)

    return gen_vel


@partial(jit, static_argnums=[0])
def Mobility_open(
    N: int,
    r: ArrayLike,
    indices_i: ArrayLike,
    indices_j: ArrayLike,
    generalized_forces: ArrayLike,
    mobil_scal: ArrayLike,
) -> Array:
    """Compute the matrix-vector product of the mobility matrix with a generalized force vector, in open boundary conditions.

    Parameters
    ----------
    N: (int)
        Number of particles
    r: (float)
        Array (N*(N-1)/2) ,3) containing the interparticle unit vectors for each pair of particle
    indices_i: (int)
        Array (,N*(N-1)/2) of indices of first particle in open boundaries list of pairs
    indices_j: (int)
        Array (,N*(N-1)/2) of indices of second particle in open boundaries list of pairs
    generalized_forces: (float)
        Array (,6*N) containing input generalized forces (force/torque)
    mobil_scal: (float)
        Array (11,N*(N-1)/2)) containing mobility functions evaluated for the current particle configuration

    Returns
    -------
    generalized_velocities (linear/angular velocities)

    """

    velocities = jnp.zeros((N, 6), float)
    forces_torques = generalized_forces[: 6 * N]
    ft_i = (jnp.reshape(forces_torques, (N, 6))).at[indices_i].get()
    ft_j = (jnp.reshape(forces_torques, (N, 6))).at[indices_j].get()

    # Dot product of r and U, i.e. axisymmetric projection (minus sign of rj is taken into account at the end of calculation)
    rdfi = (
        r.at[:, 0].get() * ft_i.at[:, 0].get()
        + r.at[:, 1].get() * ft_i.at[:, 1].get()
        + r.at[:, 2].get() * ft_i.at[:, 2].get()
    )
    rdfj = (
        r.at[:, 0].get() * ft_j.at[:, 0].get()
        + r.at[:, 1].get() * ft_j.at[:, 1].get()
        + r.at[:, 2].get() * ft_j.at[:, 2].get()
    )
    rdti = (
        r.at[:, 0].get() * ft_i.at[:, 3].get()
        + r.at[:, 1].get() * ft_i.at[:, 4].get()
        + r.at[:, 2].get() * ft_i.at[:, 5].get()
    )
    rdtj = (
        r.at[:, 0].get() * ft_j.at[:, 3].get()
        + r.at[:, 1].get() * ft_j.at[:, 4].get()
        + r.at[:, 2].get() * ft_j.at[:, 5].get()
    )

    # Cross product of U and r, i.e. eps_ijk*r_k*U_j = Px dot U, (eps_ijk is the Levi-Civita symbol)
    epsrdfi = jnp.array(
        [
            r.at[:, 2].get() * ft_i.at[:, 1].get() - r.at[:, 1].get() * ft_i.at[:, 2].get(),
            -r.at[:, 2].get() * ft_i.at[:, 0].get() + r.at[:, 0].get() * ft_i.at[:, 2].get(),
            r.at[:, 1].get() * ft_i.at[:, 0].get() - r.at[:, 0].get() * ft_i.at[:, 1].get(),
        ]
    )

    epsrdti = jnp.array(
        [
            r.at[:, 2].get() * ft_i.at[:, 4].get() - r.at[:, 1].get() * ft_i.at[:, 5].get(),
            -r.at[:, 2].get() * ft_i.at[:, 3].get() + r.at[:, 0].get() * ft_i.at[:, 5].get(),
            r.at[:, 1].get() * ft_i.at[:, 3].get() - r.at[:, 0].get() * ft_i.at[:, 4].get(),
        ]
    )

    epsrdfj = jnp.array(
        [
            r.at[:, 2].get() * ft_j.at[:, 1].get() - r.at[:, 1].get() * ft_j.at[:, 2].get(),
            -r.at[:, 2].get() * ft_j.at[:, 0].get() + r.at[:, 0].get() * ft_j.at[:, 2].get(),
            r.at[:, 1].get() * ft_j.at[:, 0].get() - r.at[:, 0].get() * ft_j.at[:, 1].get(),
        ]
    )

    epsrdtj = jnp.array(
        [
            r.at[:, 2].get() * ft_j.at[:, 4].get() - r.at[:, 1].get() * ft_j.at[:, 5].get(),
            -r.at[:, 2].get() * ft_j.at[:, 3].get() + r.at[:, 0].get() * ft_j.at[:, 5].get(),
            r.at[:, 1].get() * ft_j.at[:, 3].get() - r.at[:, 0].get() * ft_j.at[:, 4].get(),
        ]
    )

    # normalize self terms (avoid double counting)
    normaliz_factor = jnp.where(N > 1, N - 1, 1)

    xa12 = mobil_scal[0]
    ya12 = mobil_scal[1]
    yb12 = mobil_scal[2]
    xc12 = mobil_scal[3]
    yc12 = mobil_scal[4]
    n_pairs = int(N * (N - 1) / 2)

    # M_UF * F

    # Compute the contributions to the force for particles i (Fi = A11*Ui + A12*Uj + BT11*Wi + BT12*Wj)
    u = (
        (jnp.ones(n_pairs)).at[:, None].get() * (ft_i.at[:, :3].get() / normaliz_factor)
        + (xa12 - ya12).at[:, None].get() * rdfj.at[:, None].get() * r
        + ya12.at[:, None].get() * ft_j.at[:, :3].get()
        + (-yb12).at[:, None].get() * (-epsrdtj.T)
    )
    velocities = velocities.at[indices_i, :3].add(u)
    # Compute the contributions to the force for particles j (Fj = A11*Uj + A12*Ui + BT11*Wj + BT12*Wi)
    u = (
        (jnp.ones(n_pairs)).at[:, None].get() * (ft_j.at[:, :3].get() / normaliz_factor)
        + (xa12 - ya12).at[:, None].get() * rdfi.at[:, None].get() * r
        + ya12.at[:, None].get() * ft_i.at[:, :3].get()
        + (-yb12).at[:, None].get() * (epsrdti.T)
    )
    velocities = velocities.at[indices_j, :3].add(u)
    # Compute the contributions to the torque for particles i (Li = B11*Ui + B12*Uj + C11*Wi + C12*Wj)
    w = (
        (3 / 4 * jnp.ones(n_pairs)).at[:, None].get() * (ft_i.at[:, 3:].get() / normaliz_factor)
        + yb12.at[:, None].get() * epsrdfj.T
        + (xc12 - yc12).at[:, None].get() * rdtj.at[:, None].get() * r
        + yc12.at[:, None].get() * ft_j.at[:, 3:].get()
    )
    velocities = velocities.at[indices_i, 3:].add(w)
    # Compute the contributions to the torque for particles j (Lj = B11*Uj + B12*Ui + C11*Wj + C12*Wi)
    w = (
        (3 / 4 * jnp.ones(n_pairs)).at[:, None].get() * (ft_j.at[:, 3:].get() / normaliz_factor)
        - yb12.at[:, None].get() * epsrdfi.T
        + (xc12 - yc12).at[:, None].get() * rdti.at[:, None].get() * r
        + yc12.at[:, None].get() * ft_i.at[:, 3:].get()
    )
    velocities = velocities.at[indices_j, 3:].add(w)

    return jnp.ravel(velocities)
