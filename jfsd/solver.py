import jax.numpy as jnp
import jax.scipy as jscipy
from functools import partial
from jax.typing import ArrayLike
from jax import jit
from jfsd import mobility, resistance

@partial(jit, static_argnums=[0,6,7,8,9])
def solverSD(
    N: int,
    HIs_flag: int,
    rhs: ArrayLike,
    gridk: ArrayLike,
    RFU_pre_low_tri: ArrayLike,
    precomputed: tuple,
    Nx: int,
    Ny: int,
    Nz: int,
    gaussP: int,
    m_self: ArrayLike) -> tuple:                               

    """Solve the linear system Ax=b
        A contains the saddle point matrix,
        b contains applied_forces, thermal_noise, R_SU and strain terms
        x contains particle linear/angular velocities and stresslet
    
        With HIs_flag = 0 --> the solver is not called, as the system is already diagonalized 
        With HIs_flag = 1 --> A is a 17x17 matrix reproducing hydrodynamic interaction at the SD level   
    
    Parameters
    ----------
    N:
        Number of particles
    HIs_flag:
        Flag used to set level of hydrodynamic interaction.
    rhs:
        Right-hand side vector of the linear system Ax=b
    gridk:
        Array (Nx,Ny,Nz,4) of wave vectors and scaling factors for far-field wavespace calculation
    RFU_pre_low_tri:
        Lower triangular Cholesky factor of R_FU (built only from particle pairs very close)
    precomputed:
        Quantities needed to iteratively solve the linear system, computed only once
    Nx:
        Number of grid points in x direction
    Ny:
        Number of grid points in y direction
    Nz:
        Number of grid points in z direction
    gaussP:
        Gaussian support size for wave space calculation 
    m_self:
        Mobility self contribution

    Returns
    -------
    x, exitCode
    
    """   
        
    def compute_saddleSD(
            x: ArrayLike) -> ArrayLike:
        
        """Construct the saddle point operator A,
            which acts on x and returns A*x (without using A in matrix representation)
    
        Parameters
        ----------
        x:
            Unknown particle linear/angular velocities and stresslet
            
        Returns
        -------
        Ax
    
        """   
        
        # set output to zero to start
        Ax = jnp.zeros(N*17, float)

        # compute far-field contribution (M * F): output velocities+torques (first 6N) and strain rates (last 5N)
        
        Ax = Ax.at[:11*N].add(mobility.GeneralizedMobility(N, (Nx), (Ny), (Nz),
                                                            (gaussP), gridk, m_self,
                                                            all_indices_x, all_indices_y, all_indices_z, gaussian_grid_spacing1, gaussian_grid_spacing2,
                                                            r, indices_i, indices_j, f1, f2, g1, g2, h1, h2, h3,
                                                            x.at[:11*N].get()))
        
        # add B*U to M*F (modify only the first 6N entries of output because of projector B)
        # Ax = Ax.at[:6*N].add(x.at[11*N:].get())
        Ax = Ax.at[:6*N].add(x.at[11*N:].get())
        
        Ax = Ax.at[11*N:].add((resistance.ComputeLubricationFU(x[11*N:],
                   indices_i_lub, indices_j_lub, ResFunctions, r_lub, N)) * (-1))
        
        # Add (B^T * F) to (- R^nf_FU * U): modify the last 6N entries of output
        Ax = Ax.at[11*N:].add(x.at[:6*N].get())

        return Ax
    
    def compute_precondSD(
            x: ArrayLike) -> ArrayLike:
        
        """Construct precondition operator P that approximate the action of A^(-1)
        
        Parameters
        ----------
        x:
            Unknown particle linear/angular velocities and stresslet

        Returns
        -------
        Px
        
        """ 
        
        # set output to zero to start
        Px = jnp.zeros(17*N, float)
        
        # action of precondition matrix on the first 11*N entries of x is the same as the
        # identity (indeed, the identity is the precondition matrix for the far field granmobility M)
        Px = Px.at[:11*N].set(x[:11*N])
        
        #action of resistance matrix (full, not just lubrication)
        # -R_FU^-1 * x[:6N]
        # First solve L^T * y = x, for y, where L^T is the lower triangular Chol. factor of R^nf_FU
        buffer = jscipy.linalg.solve_triangular(
            RFU_pre_low_tri, x.at[:6*N].get(), lower=True)
        # Then solve L * z = y, for z, where L is the upper triangular Chol. factor of R^nf_FU
        buffer = jscipy.linalg.solve_triangular(
            jnp.transpose(RFU_pre_low_tri), buffer, lower=False)
        Px = Px.at[:6*N].add(-buffer)
        Px = Px.at[11*N:].add(buffer)
        # -R_FU^-1 * x[11N:]
        # First solve L^T * y = x, for y, where L^T is the lower triangular Chol. factor of R^nf_FU
        buffer = jscipy.linalg.solve_triangular(
            RFU_pre_low_tri, x.at[11*N:].get(), lower=True)
        # Then solve L * z = y, for z, where L is the upper triangular Chol. factor of R^nf_FU
        buffer = jscipy.linalg.solve_triangular(
            jnp.transpose(RFU_pre_low_tri), buffer, lower=False)
        Px = Px.at[:6*N].add(buffer)
        Px = Px.at[11*N:].add(-buffer)       

        return Px

    #Extract the quantities for the calculation, from input
    (all_indices_x, all_indices_y, all_indices_z, gaussian_grid_spacing1, gaussian_grid_spacing2,
     r, indices_i, indices_j, f1, f2, g1, g2, h1, h2, h3,
     r_lub, indices_i_lub, indices_j_lub, ResFunctions) = precomputed

    #Solve the linear system Ax= b
    x, exitCode = jscipy.sparse.linalg.gmres(
        A=compute_saddleSD, b=rhs, tol=1e-5, restart=25, M=compute_precondSD)

    return x, exitCode