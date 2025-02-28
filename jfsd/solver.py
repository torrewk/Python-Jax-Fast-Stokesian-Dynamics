from functools import partial
import jax.numpy as jnp
import jax.scipy as jscipy
from jax import Array, jit, vmap
from jax.typing import ArrayLike
from jfsd import mobility, resistance

@partial(jit, static_argnums=[0, 4, 5, 6, 7, 9])
def solve_linear_system(
    num_particles: int,
    rhs: ArrayLike,
    gridk: ArrayLike,
    precomputed: tuple,
    grid_nx: int,
    grid_ny: int,
    grid_nz: int,
    gauss_support: int,
    m_self: ArrayLike,
    max_nonzero_per_row: int,
    r_prec: ArrayLike,
    initial_guess: ArrayLike = None,
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
    max_nonzero_per_row: (int)
        Max number of non zero element in each row of the precondition resistnce FU matrix
    r_prec: (tuple)
        Approximation of the resistance matrix in sparse format
    initial_guess: (float)
        Array (17*num_particles) containing initial guess x0 for the linear system Ax=b
    
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
                    x.at[11 * num_particles :].get(), indices_i_lub, indices_j_lub, res_functions, r_lub, num_particles
                )
            )
            * (-1)
        )

        # Add (B^T * F) to (- R^nf_FU * U): modify the last 6N entries of output
        ax = ax.at[11 * num_particles :].add(x.at[: 6 * num_particles].get())

        return ax

    def compute_precond_sd_kwt(x: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the preconditioner by solving a single system R x = b using JAX CG.
        
        Parameters
        ----------
        x : jnp.ndarray
            Input vector containing unknown particle linear/angular velocities and stresslets.
            
        Returns
        -------
        ppx : jnp.ndarray
            Preconditioned output vector.
        """
        # Initialize output
        ppx = jnp.zeros(17 * num_particles, float)
        # Copy over the first 11*num_particles entries from x
        ppx = ppx.at[:11 * num_particles].set(x[:11 * num_particles])
        
        # Compute right-hand side b
        b = -x[:6 * num_particles] + x[11 * num_particles:]
        
        # --- Define a single linear operator that applies L and U ---
        def apply_R(v: jnp.ndarray) -> jnp.ndarray:
            """Applies R to a vector v."""
            def apply_L(i):
                return jnp.dot(row_values[i, :max_nonzero_per_row], 
                            v[row_indices[i, :max_nonzero_per_row]])

            return vmap(apply_L)(jnp.arange(6 * num_particles))

        # Solve R x = b in one CG call
        x_solved, exitCode = jscipy.sparse.linalg.cg(apply_R, b, tol=1e-3)
        
        # Build the full preconditioner output
        ppx = ppx.at[:6 * num_particles].set(x_solved)
        ppx = ppx.at[11 * num_particles:].set(-x_solved)
        ppx = ppx.at[:11 * num_particles].add(x[:11 * num_particles])

        return ppx


    row_indices, row_values = r_prec
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

    x, exitCode = jscipy.sparse.linalg.gmres(
        x0=initial_guess, A=compute_saddle_sd, b=rhs, tol=1e-5, restart=25, M=compute_precond_sd_kwt
    )
    return x, exitCode


@partial(jit, static_argnums=[0,4])
def solve_linear_system_open(
        num_particles: int, rhs: ArrayLike, precomputed: tuple, dist_ff: ArrayLike, max_nonzero_per_row: int,
        r_prec: ArrayLike, initial_guess: ArrayLike = None
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
                resistance.compute_lubrication_fu(
                    x[11 * num_particles :], indices_i_lub, indices_j_lub, res_functions, r_lub, num_particles
                )
            )
            * (-1)
        )

        # Add (B^T * F) to (- R^nf_FU * U): modify the last 6N entries of output
        ax = ax.at[11 * num_particles :].add(x.at[: 6 * num_particles].get())

        return ax
    
    def compute_precond_sd_kwt(x: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the preconditioner by solving a single system R x = b using JAX CG.
        
        Parameters
        ----------
        x : jnp.ndarray
            Input vector containing unknown particle linear/angular velocities and stresslets.
            
        Returns
        -------
        ppx : jnp.ndarray
            Preconditioned output vector.
        """
        # Initialize output
        ppx = jnp.zeros(17 * num_particles, float)
        # Copy over the first 11*num_particles entries from x
        ppx = ppx.at[:11 * num_particles].set(x[:11 * num_particles])
                
        # --- Define a single linear operator that applies L and U ---
        def apply_R(v: jnp.ndarray) -> jnp.ndarray:
            """Applies R to a vector v."""
            def apply_L(i):
                return jnp.dot(row_values[i, :max_nonzero_per_row], 
                            v[row_indices[i, :max_nonzero_per_row]])

            return vmap(apply_L)(jnp.arange(6 * num_particles))

        # Solve R x = b in one CG call
        x_solved, _ = jscipy.sparse.linalg.cg(apply_R, -x[:6 * num_particles] + x[11 * num_particles:], tol=1e-3)
        
        # Build the full preconditioner output
        ppx = ppx.at[:6 * num_particles].set(x_solved)
        ppx = ppx.at[11 * num_particles:].set(-x_solved)
        ppx = ppx.at[:11 * num_particles].add(x[:11 * num_particles])

        return ppx
    
    row_indices, row_values = r_prec
    # Extract the quantities for the calculation, from input
    (r, indices_i, indices_j, r_lub, indices_i_lub, indices_j_lub, res_functions, mobil_funct) = (
        precomputed
    )

    # Solve the linear system Ax= b
    x, exitCode = jscipy.sparse.linalg.gmres(
        A=compute_saddle_sd_open, b=rhs, tol=1e-5, restart=25, M=compute_precond_sd_kwt
    )

    return x, exitCode
