from functools import partial

import jax.numpy as jnp
import jax.scipy as jscipy
from jax import Array, jit
from jax.typing import ArrayLike

from jfsd import mobility, resistance


@partial(jit, static_argnums=[0, 5, 6, 7, 8])
def solve_linear_system(
    num_particles: int,
    rhs: ArrayLike,
    gridk: ArrayLike,
    rfu_pre_low_tri: ArrayLike,
    precomputed: tuple,
    grid_nx: int,
    grid_ny: int,
    grid_nz: int,
    gauss_support: int,
    m_self: ArrayLike,
) -> tuple[Array, int]:
    """Solve the linear system Ax=b.

    With:
        A contains the saddle point matrix,
        b contains applied_forces, thermal_noise, R_SU and strain terms
        x contains particle linear/angular velocities and stresslet

        With HIs_flag = 0 --> the solver is not called, as the system is already diagonalized
        With HIs_flag = 1 --> the solver is not called, as the system is already diagonalized
        With HIs_flag = 2 --> A is a 17x17 matrix reproducing hydrodynamic interaction at the SD level

    Parameters
    ----------
    num_particles: (int)
        Number of particles
    rhs: (float)
        Array (17*num_particles) containing right-hand side vector of the linear system Ax=b
    gridk: (float)
        Array (grid_nx,grid_ny,grid_nz,4) containing wave vectors and scaling factors for far-field wavespace calculation
    rfu_pre_low_tri: (float)
        Array (6*num_particles,6*num_particles) containing lower triangular Cholesky factor of R_FU (built only from particle pairs very close)
    precomputed: (float)
        Tuples containing quantities needed to iteratively solve the linear system, computed only once
    grid_nx: (int)
        Number of grid points in x direction
    grid_ny: (int)
        Number of grid points in y direction
    grid_nz: (int)
        Number of grid points in z direction
    gauss_support: (int)
        Gaussian support size for wave space calculation
    m_self: (float)
        Array (,2) containing mobility self contributions
    dist_ff: (float)
        Array (,num_particles*(num_particles-1)/2) containing interparticle distances for each pair in the far-field regime

    Returns
    -------
    x, exitCode

    """

    def compute_saddle_sd(x: ArrayLike) -> Array:
        """Construct the saddle point operator A.

        This acts on x and returns A*x (without using A in matrix representation).

        Parameters
        ----------
        x: (float)
            Array (,17*num_particles) containing unknown particle linear/angular velocities, stresslets and hydrodynamic forces

        Returns
        -------
        ax

        """
        # set output to zero to start
        ax = jnp.zeros(num_particles * 17, float)

        # compute far-field contribution (M * F): output velocities+torques (first 6N) and strain rates (last 5N)

        ax = ax.at[: 11 * num_particles].set(
            mobility.generalized_mobility_periodic(
                num_particles,
                (grid_nx),
                (grid_ny),
                (grid_nz),
                (gauss_support),
                gridk,
                m_self,
                all_indices_x,
                all_indices_y,
                all_indices_z,
                gaussian_grid_spacing1,
                gaussian_grid_spacing2,
                r,
                indices_i,
                indices_j,
                f1,
                f2,
                g1,
                g2,
                h1,
                h2,
                h3,
                x.at[: 11 * num_particles].get(),
            )
        )

        # add B*U to M*F (modify only the first 6N entries of output because of projector B)
        ax = ax.at[: 6 * num_particles].add(x.at[11 * num_particles :].get())
        ax = ax.at[11 * num_particles :].set(
            (
                resistance.compute_lubrication_fu(
                    x[11 * num_particles :], indices_i_lub, indices_j_lub, res_functions, r_lub, num_particles
                )
            )
            * (-1)
        )

        # Add (B^T * F) to (- R^nf_FU * U): modify the last 6N entries of output
        ax = ax.at[11 * num_particles :].add(x.at[: 6 * num_particles].get())

        return ax

    def compute_precond_sd(x: ArrayLike) -> Array:
        """Construct precondition operator P that approximate the action of A^(-1)

        Parameters
        ----------
        x:
            Array (17*num_particles)

        Returns
        -------
        px

        """
        # set output to zero to start
        px = jnp.zeros(17 * num_particles, float)

        # action of precondition matrix on the first 11*num_particles entries of x is the same as the
        # identity (indeed, the identity is the precondition matrix for the far field granmobility M)
        px = px.at[: 11 * num_particles].set(x[: 11 * num_particles])

        # action of resistance matrix (full, not just lubrication)
        # -R_FU^-1 * x[:6N]
        # First solve L^T * y = x, for y, where L^T is the lower triangular Chol. factor of R^nf_FU
        buffer = jscipy.linalg.solve_triangular(rfu_pre_low_tri, x.at[: 6 * num_particles].get(), lower=True)
        # Then solve L * z = y, for z, where L is the upper triangular Chol. factor of R^nf_FU
        buffer = jscipy.linalg.solve_triangular(jnp.transpose(rfu_pre_low_tri), buffer, lower=False)
        px = px.at[: 6 * num_particles].add(-buffer)
        px = px.at[11 * num_particles :].set(buffer)
        # -R_FU^-1 * x[11N:]
        # First solve L^T * y = x, for y, where L^T is the lower triangular Chol. factor of R^nf_FU
        buffer = jscipy.linalg.solve_triangular(rfu_pre_low_tri, x.at[11 * num_particles :].get(), lower=True)
        # Then solve L * z = y, for z, where L is the upper triangular Chol. factor of R^nf_FU
        buffer = jscipy.linalg.solve_triangular(jnp.transpose(rfu_pre_low_tri), buffer, lower=False)
        px = px.at[: 6 * num_particles].add(buffer)
        px = px.at[11 * num_particles :].add(-buffer)

        return px

    # Extract the quantities for the calculation, from input
    (
        all_indices_x,
        all_indices_y,
        all_indices_z,
        gaussian_grid_spacing1,
        gaussian_grid_spacing2,
        r,
        indices_i,
        indices_j,
        f1,
        f2,
        g1,
        g2,
        h1,
        h2,
        h3,
        r_lub,
        indices_i_lub,
        indices_j_lub,
        res_functions,
    ) = precomputed

    # Solve the linear system Ax= b
    x, exitCode = jscipy.sparse.linalg.gmres(
        A=compute_saddle_sd, b=rhs, tol=1e-5, restart=25, M=compute_precond_sd
    )

    return x, exitCode


@partial(jit, static_argnums=[0])
def solve_linear_system_open(
    num_particles: int, rhs: ArrayLike, rfu_pre_low_tri: ArrayLike, precomputed: tuple, dist_ff: ArrayLike
) -> tuple[Array, int]:
    def compute_distances():
        return jnp.linalg.norm(dist_ff, axis=1)

    def compute_saddle_sd_open(x: ArrayLike) -> Array:
        """Construct the saddle point operator A.

        This acts on x and returns A*x (without using A in matrix representation).

        Parameters
        ----------
        x: (float)
            Array (,17*num_particles) containing unknown particle linear/angular velocities, stresslets and hydrodynamic forces

        Returns
        -------
        ax

        """
        # set output to zero to start
        ax = jnp.zeros(num_particles * 17, float)

        # compute far-field contribution (M * F): output velocities+torques (first 6N) and strain rates (last 5N)

        ax = ax.at[: 11 * num_particles].add(
            mobility.generalized_mobility_open(
                num_particles, r, indices_i, indices_j, x.at[: 11 * num_particles].get(), mobil_funct
            )
        )

        # add B*U to M*F (modify only the first 6N entries of output because of projector B)
        ax = ax.at[: 6 * num_particles].add(x.at[11 * num_particles :].get())

        ax = ax.at[11 * num_particles :].add(
            (
                resistance.ComputeLubricationFU(
                    x[11 * num_particles :], indices_i_lub, indices_j_lub, res_functions, r_lub, num_particles
                )
            )
            * (-1)
        )

        # Add (B^T * F) to (- R^nf_FU * U): modify the last 6N entries of output
        ax = ax.at[11 * num_particles :].add(x.at[: 6 * num_particles].get())

        return ax

    def compute_precond_sd_open(x: ArrayLike) -> Array:
        """Construct precondition operator P that approximate the action of A^(-1)

        Parameters
        ----------
        x:
            Array (17*num_particles)

        Returns
        -------
        px

        """
        # set output to zero to start
        px = jnp.zeros(17 * num_particles, float)

        # action of precondition matrix on the first 11*num_particles entries of x is the same as the
        # identity (indeed, the identity is the precondition matrix for the far field granmobility M)
        px = px.at[: 11 * num_particles].set(x[: 11 * num_particles])

        # action of resistance matrix (full, not just lubrication)
        # -R_FU^-1 * x[:6N]
        # First solve L^T * y = x, for y, where L^T is the lower triangular Chol. factor of R^nf_FU
        buffer = jscipy.linalg.solve_triangular(rfu_pre_low_tri, x.at[: 6 * num_particles].get(), lower=True)
        # Then solve L * z = y, for z, where L is the upper triangular Chol. factor of R^nf_FU
        buffer = jscipy.linalg.solve_triangular(jnp.transpose(rfu_pre_low_tri), buffer, lower=False)
        px = px.at[: 6 * num_particles].add(-buffer)
        px = px.at[11 * num_particles :].add(buffer)
        # -R_FU^-1 * x[11N:]
        # First solve L^T * y = x, for y, where L^T is the lower triangular Chol. factor of R^nf_FU
        buffer = jscipy.linalg.solve_triangular(rfu_pre_low_tri, x.at[11 * num_particles :].get(), lower=True)
        # Then solve L * z = y, for z, where L is the upper triangular Chol. factor of R^nf_FU
        buffer = jscipy.linalg.solve_triangular(jnp.transpose(rfu_pre_low_tri), buffer, lower=False)
        px = px.at[: 6 * num_particles].add(buffer)
        px = px.at[11 * num_particles :].add(-buffer)

        return px

    # Extract the quantities for the calculation, from input
    (r, indices_i, indices_j, r_lub, indices_i_lub, indices_j_lub, res_functions, mobil_funct) = (
        precomputed
    )

    # Solve the linear system Ax= b
    x, exitCode = jscipy.sparse.linalg.gmres(
        A=compute_saddle_sd_open, b=rhs, tol=1e-5, restart=25, M=compute_precond_sd_open
    )

    return x, exitCode