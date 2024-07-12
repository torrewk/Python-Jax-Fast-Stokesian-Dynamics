import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false' # avoid JAX allocating most of the GPU memory even if not needed
from functools import partial
from jax import jit
import jax.numpy as jnp
from jax.config import config
from jax.typing import ArrayLike
config.update("jax_enable_x64", False) #disable double precision

@partial(jit, static_argnums=[0,1,2,3,4])
def GeneralizedMobility(
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
        generalized_forces: ArrayLike) -> ArrayLike:
    
    """Construct the saddle point operator A,
        which acts on x and returns A*x (without using A in matrix representation)

    Parameters
    ----------
    N:
        Number of particles
    Nx:
        Number of grid points in x direction
    Ny:
        Number of grid points in y direction
    Nz:
        Number of grid points in z direction
    gaussP:
        Gaussian support size for wave space calculation 
    gridk:
        Wave number values in the grid
    m_self:
        Mobility self contribution
    all_indices_x:
        All indices (x) of wave grid points for each particle
    all_indices_y:
        All indices (y) of wave grid points for each particle
    all_indices_z:
        All indices (z) of wave grid points for each particle
    gaussian_grid_spacing1:
        Scaled distances from support center to each gridpoint, for FFT  
    gaussian_grid_spacing2:
        Scaled distances from support center to each gridpoint, for inverse FFT  
    r:
        Units vectors connecting each pair of particles in the far-field neighbor list
    indices_i:
        Indices of first particle in far-field neighbor list pairs 
    indices_j:
        Indices of second particle in far-field neighbor list pairs 
    f1:
        Mobility scalar function 1
    f2:
        Mobility scalar function 2
    g1:
        Mobility scalar function 3
    g2:
        Mobility scalar function 4
    h1:
        Mobility scalar function 5
    h2:
        Mobility scalar function 6
    h3:
        Mobility scalar function 7
    generalized_forces:
        Input generalized forces (force/torque/stresslet)
        
    Returns
    -------
    generalized_velocities (linear/angular velocities and rateOfStrain) 

    """
    
    # Helper function
    def swap_real_imag(
            cplx_arr: ArrayLike) -> ArrayLike:
        """Take a comple number as input and return a complex number with real part equal to (minus) 
        the imaginary part of the input and an imaginary part equal to the real part of the input. 

        Parameters
        ----------
        cplx_arr:
            Array of complex values
            
        Returns
        -------
        -jnp.imag(cplx_arr) + 1j * jnp.real(cplx_arr)

        """
        return -jnp.imag(cplx_arr) + 1j * jnp.real(cplx_arr)

    # Get forces,torques,couplets from generalized forces (3*N vector: f1x,f1y,f1z, ... , fNx,fNy,fNz and same for torque, while couplet is 5N) 
    forces = jnp.zeros(3*N)
    forces = forces.at[0::3].set(generalized_forces.at[0:(6*N):6].get())
    forces = forces.at[1::3].set(generalized_forces.at[1:(6*N):6].get())
    forces = forces.at[2::3].set(generalized_forces.at[2:(6*N):6].get())
    torques = jnp.zeros(3*N)
    torques = torques.at[0::3].set(generalized_forces.at[3:(6*N):6].get())
    torques = torques.at[1::3].set(generalized_forces.at[4:(6*N):6].get())
    torques = torques.at[2::3].set(generalized_forces.at[5:(6*N):6].get())
    stresslet = jnp.zeros(5*N)
    stresslet = stresslet.at[0::5].set(generalized_forces.at[(6*N+0)::5].get()) #Sxx
    stresslet = stresslet.at[1::5].set(generalized_forces.at[(6*N+1)::5].get()) #Sxy
    stresslet = stresslet.at[2::5].set(generalized_forces.at[(6*N+2)::5].get()) #Sxz
    stresslet = stresslet.at[3::5].set(generalized_forces.at[(6*N+3)::5].get()) #Syz
    stresslet = stresslet.at[4::5].set(generalized_forces.at[(6*N+4)::5].get()) #Syy

    # Get 'couplet' from generalized forces (8*N vector)    
    couplets = jnp.zeros(8*N)
    couplets = couplets.at[::8].set(stresslet.at[::5].get())  # C[0] = S[0]
    couplets = couplets.at[1::8].set(
        stresslet.at[1::5].get()+torques.at[2::3].get()*0.5)  # C[1] = S[1] + L[2]/2
    couplets = couplets.at[2::8].set(
        stresslet.at[2::5].get()-torques.at[1::3].get()*0.5)  # C[2] = S[2] - L[1]/2
    couplets = couplets.at[3::8].set(
        stresslet.at[3::5].get()+torques.at[::3].get()*0.5)  # C[3] = S[3] + L[0]/2
    couplets = couplets.at[4::8].set(stresslet.at[4::5].get())  # C[4] = S[4]
    couplets = couplets.at[5::8].set(
        stresslet.at[1::5].get()-torques.at[2::3].get()*0.5)  # C[5] = S[1] - L[2]/2
    couplets = couplets.at[6::8].set(
        stresslet.at[2::5].get()+torques.at[1::3].get()*0.5)  # C[6] = S[2] + L[1]/2
    couplets = couplets.at[7::8].set(
        stresslet.at[3::5].get()-torques.at[::3].get()*0.5)  # C[7] = S[3] - L[0]/2

    ##########################################################################################################################################
    ######################################## WAVE SPACE CONTRIBUTION #########################################################################
    ##########################################################################################################################################
    
    #Create Grids for current iteration
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
    
    gridX = gridX.at[all_indices_x, all_indices_y, all_indices_z].add(jnp.ravel((jnp.swapaxes(jnp.swapaxes(jnp.swapaxes(gaussian_grid_spacing1*jnp.resize(forces.at[0::3].get(), (gaussP,gaussP,gaussP,N) ),3,2),2,1),1,0))))
    gridY = gridY.at[all_indices_x, all_indices_y, all_indices_z].add(jnp.ravel((jnp.swapaxes(jnp.swapaxes(jnp.swapaxes(gaussian_grid_spacing1*jnp.resize(forces.at[1::3].get(), (gaussP,gaussP,gaussP,N) ),3,2),2,1),1,0))))            
    gridZ = gridZ.at[all_indices_x, all_indices_y, all_indices_z].add(jnp.ravel((jnp.swapaxes(jnp.swapaxes(jnp.swapaxes(gaussian_grid_spacing1*jnp.resize(forces.at[2::3].get(), (gaussP,gaussP,gaussP,N) ),3,2),2,1),1,0))))
    gridXX = gridXX.at[all_indices_x, all_indices_y, all_indices_z].add(jnp.ravel((jnp.swapaxes(jnp.swapaxes(jnp.swapaxes(gaussian_grid_spacing1*jnp.resize(couplets.at[0::8].get(), (gaussP,gaussP,gaussP,N) ),3,2),2,1),1,0))))
    gridXY = gridXY.at[all_indices_x, all_indices_y, all_indices_z].add(jnp.ravel((jnp.swapaxes(jnp.swapaxes(jnp.swapaxes(gaussian_grid_spacing1*jnp.resize(couplets.at[1::8].get(), (gaussP,gaussP,gaussP,N) ),3,2),2,1),1,0))))
    gridXZ = gridXZ.at[all_indices_x, all_indices_y, all_indices_z].add(jnp.ravel((jnp.swapaxes(jnp.swapaxes(jnp.swapaxes(gaussian_grid_spacing1*jnp.resize(couplets.at[2::8].get(), (gaussP,gaussP,gaussP,N) ),3,2),2,1),1,0)))) 
    gridYZ = gridYZ.at[all_indices_x, all_indices_y, all_indices_z].add(jnp.ravel((jnp.swapaxes(jnp.swapaxes(jnp.swapaxes(gaussian_grid_spacing1*jnp.resize(couplets.at[3::8].get(), (gaussP,gaussP,gaussP,N) ),3,2),2,1),1,0))))
    gridYY = gridYY.at[all_indices_x, all_indices_y, all_indices_z].add(jnp.ravel((jnp.swapaxes(jnp.swapaxes(jnp.swapaxes(gaussian_grid_spacing1*jnp.resize(couplets.at[4::8].get(), (gaussP,gaussP,gaussP,N) ),3,2),2,1),1,0))))
    gridYX = gridYX.at[all_indices_x, all_indices_y, all_indices_z].add(jnp.ravel((jnp.swapaxes(jnp.swapaxes(jnp.swapaxes(gaussian_grid_spacing1*jnp.resize(couplets.at[5::8].get(), (gaussP,gaussP,gaussP,N) ),3,2),2,1),1,0))))
    gridZX = gridZX.at[all_indices_x, all_indices_y, all_indices_z].add(jnp.ravel((jnp.swapaxes(jnp.swapaxes(jnp.swapaxes(gaussian_grid_spacing1*jnp.resize(couplets.at[6::8].get(), (gaussP,gaussP,gaussP,N) ),3,2),2,1),1,0))))
    gridZY = gridZY.at[all_indices_x, all_indices_y, all_indices_z].add(jnp.ravel((jnp.swapaxes(jnp.swapaxes(jnp.swapaxes(gaussian_grid_spacing1*jnp.resize(couplets.at[7::8].get(), (gaussP,gaussP,gaussP,N) ),3,2),2,1),1,0))))

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
    gridZZ = - gridXX - gridYY
    
    gridk_sqr = (gridk.at[:, :, :, 0].get()*gridk.at[:, :, :, 0].get()+gridk.at[:, :, :, 1].get()
                 * gridk.at[:, :, :, 1].get()+gridk.at[:, :, :, 2].get()*gridk.at[:, :, :, 2].get())
    gridk_mod = jnp.sqrt(gridk_sqr)
    
    kdF = jnp.where(gridk_mod > 0, (gridk.at[:, :, :, 0].get()*gridX +
                    gridk.at[:, :, :, 1].get()*gridY+gridk.at[:, :, :, 2].get()*gridZ)/gridk_sqr, 0)

    Cdkx = (gridk.at[:, :, :, 0].get()*gridXX + gridk.at[:, :, :, 1].get()*gridXY + gridk.at[:, :, :, 2].get()*gridXZ)
    Cdky = (gridk.at[:, :, :, 0].get()*gridYX + gridk.at[:, :, :, 1].get()*gridYY + gridk.at[:, :, :, 2].get()*gridYZ)
    Cdkz = (gridk.at[:, :, :, 0].get()*gridZX + gridk.at[:, :, :, 1].get()*gridZY + gridk.at[:, :, :, 2].get()*gridZZ)

    kdcdk = jnp.where(
        gridk_mod > 0, ( gridk.at[:, :, :, 0].get()*Cdkx
                        +gridk.at[:, :, :, 1].get()*Cdky
                        +gridk.at[:, :, :, 2].get()*Cdkz)/gridk_sqr, 0)
    
    Fkxx = (gridk.at[:, :, :, 0].get()*gridX)
    Fkxy = (gridk.at[:, :, :, 1].get()*gridX)
    Fkxz = (gridk.at[:, :, :, 2].get()*gridX)
    Fkyx = (gridk.at[:, :, :, 0].get()*gridY)
    Fkyy = (gridk.at[:, :, :, 1].get()*gridY)
    Fkyz = (gridk.at[:, :, :, 2].get()*gridY)
    Fkzx = (gridk.at[:, :, :, 0].get()*gridZ)
    Fkzy = (gridk.at[:, :, :, 1].get()*gridZ)
    kkxx = gridk.at[:, :, :, 0].get()*gridk.at[:, :, :, 0].get()
    kkxy = gridk.at[:, :, :, 0].get()*gridk.at[:, :, :, 1].get()
    kkxz = gridk.at[:, :, :, 0].get()*gridk.at[:, :, :, 2].get()
    kkyx = gridk.at[:, :, :, 1].get()*gridk.at[:, :, :, 0].get()
    kkyy = gridk.at[:, :, :, 1].get()*gridk.at[:, :, :, 1].get()
    kkyz = gridk.at[:, :, :, 1].get()*gridk.at[:, :, :, 2].get()
    kkzx = gridk.at[:, :, :, 2].get()*gridk.at[:, :, :, 0].get()
    kkzy = gridk.at[:, :, :, 2].get()*gridk.at[:, :, :, 1].get()
    Cdkkxx = gridk.at[:, :, :, 0].get() * Cdkx
    Cdkkxy = gridk.at[:, :, :, 1].get() * Cdkx
    Cdkkxz = gridk.at[:, :, :, 2].get() * Cdkx
    Cdkkyx = gridk.at[:, :, :, 0].get() * Cdky
    Cdkkyy = gridk.at[:, :, :, 1].get() * Cdky
    Cdkkyz = gridk.at[:, :, :, 2].get() * Cdky
    Cdkkzx = gridk.at[:, :, :, 0].get() * Cdkz
    Cdkkzy = gridk.at[:, :, :, 1].get() * Cdkz

    # UF part
    B = jnp.where(gridk_mod > 0, gridk.at[:, :, :, 3].get()*(jnp.sin(gridk_mod)/gridk_mod)*(
        jnp.sin(gridk_mod)/gridk_mod), 0)  # scaling factor
    gridX = B * (gridX - gridk.at[:, :, :, 0].get() * kdF)
    gridY = B * (gridY - gridk.at[:, :, :, 1].get() * kdF)
    gridZ = B * (gridZ - gridk.at[:, :, :, 2].get() * kdF)

    # UC part (here B is imaginary so we absorb the imaginary unit in the funtion 'swap_real_imag()' which returns -Im(c)+i*Re(c)
    B = jnp.where(gridk_mod > 0, gridk.at[:, :, :, 3].get()*(jnp.sin(gridk_mod)/gridk_mod)*(3 * (jnp.sin(
        gridk_mod) - gridk_mod*jnp.cos(gridk_mod)) / (gridk_mod*gridk_sqr)), 0)  # scaling factor

    gridX += B * swap_real_imag( (Cdkx - kdcdk * gridk.at[:, :, :, 0].get()))
    gridY += B * swap_real_imag( (Cdky - kdcdk * gridk.at[:, :, :, 1].get()))
    gridZ += B * swap_real_imag( (Cdkz - kdcdk * gridk.at[:, :, :, 2].get()))

    # DF part
    B = jnp.where(gridk_mod > 0, gridk.at[:, :, :, 3].get()*(-1)*(jnp.sin(gridk_mod)/gridk_mod)*(3 * (
        jnp.sin(gridk_mod) - gridk_mod*jnp.cos(gridk_mod)) / (gridk_mod*gridk_sqr)), 0)  # scaling factor
    gridXX = B * swap_real_imag((Fkxx - kkxx * kdF))
    gridXY = B * swap_real_imag((Fkxy - kkxy * kdF))
    gridXZ = B * swap_real_imag((Fkxz - kkxz * kdF))
    gridYX = B * swap_real_imag((Fkyx - kkyx * kdF))
    gridYY = B * swap_real_imag((Fkyy - kkyy * kdF))
    gridYZ = B * swap_real_imag((Fkyz - kkyz * kdF))
    gridZX = B * swap_real_imag((Fkzx - kkzx * kdF))
    gridZY = B * swap_real_imag((Fkzy - kkzy * kdF))

    # DC part
    B = jnp.where(gridk_mod > 0, gridk.at[:, :, :, 3].get()*(9)*((jnp.sin(gridk_mod) - gridk_mod*jnp.cos(gridk_mod)) / (
        gridk_mod*gridk_sqr)) * ((jnp.sin(gridk_mod) - gridk_mod*jnp.cos(gridk_mod)) / (gridk_mod*gridk_sqr)), 0)  # scaling factor
    gridXX += B * (Cdkkxx - kkxx * kdcdk)
    gridXY += B * (Cdkkxy - kkxy * kdcdk)
    gridXZ += B * (Cdkkxz - kkxz * kdcdk)
    gridYX += B * (Cdkkyx - kkyx * kdcdk)
    gridYY += B * (Cdkkyy - kkyy * kdcdk)
    gridYZ += B * (Cdkkyz - kkyz * kdcdk)
    gridZX += B * (Cdkkzx - kkzx * kdcdk)
    gridZY += B * (Cdkkzy - kkzy * kdcdk)
    
    # Inverse FFT
    gridX = jnp.real(jnp.fft.ifftn(gridX,norm='forward'))
    gridY = jnp.real(jnp.fft.ifftn(gridY,norm='forward'))
    gridZ = jnp.real(jnp.fft.ifftn(gridZ,norm='forward'))
    gridXX = jnp.real(jnp.fft.ifftn(gridXX,norm='forward'))
    gridXY = jnp.real(jnp.fft.ifftn(gridXY,norm='forward'))
    gridXZ = jnp.real(jnp.fft.ifftn(gridXZ,norm='forward'))
    gridYX = jnp.real(jnp.fft.ifftn(gridYX,norm='forward'))
    gridYY = jnp.real(jnp.fft.ifftn(gridYY,norm='forward'))
    gridYZ = jnp.real(jnp.fft.ifftn(gridYZ,norm='forward'))
    gridZX = jnp.real(jnp.fft.ifftn(gridZX,norm='forward'))
    gridZY = jnp.real(jnp.fft.ifftn(gridZY,norm='forward'))
    
    # Compute Linear velocities and Velocity gradients (from which we can extract angular velocities and rate of strain)
    w_lin_velocities = jnp.zeros((N, 3),float)
    w_velocity_gradient = jnp.zeros((N, 8),float)
       
     
    w_lin_velocities = w_lin_velocities.at[:, 0].add(
        jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
            gridX.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3)))
    w_lin_velocities = w_lin_velocities.at[:, 1].add(
        jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
            gridY.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3))) 
    w_lin_velocities = w_lin_velocities.at[:, 2].add(
        jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
            gridZ.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3)))              


    w_velocity_gradient = w_velocity_gradient.at[:, 0].add(
        jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
            gridXX.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3)))
    
    w_velocity_gradient = w_velocity_gradient.at[:, 5].add(
        jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
            gridXY.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3))) 

    w_velocity_gradient = w_velocity_gradient.at[:, 6].add(
          jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
            gridXZ.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3)))  

    w_velocity_gradient = w_velocity_gradient.at[:, 7].add(
        jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
            gridYZ.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3)))
    
    
    w_velocity_gradient = w_velocity_gradient.at[:, 4].add(
        jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
            gridYY.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3))) 
    
    w_velocity_gradient = w_velocity_gradient.at[:, 1].add(
        jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
            gridYX.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3)))     

    w_velocity_gradient = w_velocity_gradient.at[:, 2].add(
          jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
            gridZX.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3))) 

    w_velocity_gradient = w_velocity_gradient.at[:, 3].add(
        jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
            gridZY.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3)))              
    
    
    ##########################################################################################################################################
    ######################################## REAL SPACE CONTRIBUTION #########################################################################
    ##########################################################################################################################################
    
    
    # Allocate arrays for Linear velocities and Velocity gradients (from which we can extract angular velocities and rate of strain)
    r_lin_velocities = jnp.zeros((N, 3),float)
    r_velocity_gradient = jnp.zeros((N, 8),float)
    
    # SELF CONTRIBUTIONS
    r_lin_velocities = r_lin_velocities.at[:, 0].set(
        m_self.at[0].get() * forces.at[0::3].get())
    r_lin_velocities = r_lin_velocities.at[:, 1].set(
        m_self.at[0].get() * forces.at[1::3].get())
    r_lin_velocities = r_lin_velocities.at[:, 2].set(
        m_self.at[0].get() * forces.at[2::3].get())
    
    r_velocity_gradient = r_velocity_gradient.at[:, 0].set(
        m_self.at[1].get()*(couplets.at[0::8].get() - 4 * couplets.at[0::8].get()))
    r_velocity_gradient = r_velocity_gradient.at[:, 5].set(
        m_self.at[1].get()*(couplets.at[1::8].get() - 4 * couplets.at[5::8].get()))
    r_velocity_gradient = r_velocity_gradient.at[:, 6].set(
        m_self.at[1].get()*(couplets.at[2::8].get() - 4 * couplets.at[6::8].get()))
    r_velocity_gradient = r_velocity_gradient.at[:, 7].set(
        m_self.at[1].get()*(couplets.at[3::8].get() - 4 * couplets.at[7::8].get()))
    
    r_velocity_gradient = r_velocity_gradient.at[:, 4].set(
        m_self.at[1].get()*(couplets.at[4::8].get() - 4 * couplets.at[4::8].get()))
    r_velocity_gradient = r_velocity_gradient.at[:, 1].set(
        m_self.at[1].get()*(couplets.at[5::8].get() - 4 * couplets.at[1::8].get()))
    r_velocity_gradient = r_velocity_gradient.at[:, 2].set(
        m_self.at[1].get()*(couplets.at[6::8].get() - 4 * couplets.at[2::8].get()))
    r_velocity_gradient = r_velocity_gradient.at[:, 3].set(
        m_self.at[1].get()*(couplets.at[7::8].get() - 4 * couplets.at[3::8].get()))
    
    
    
    # # Pair contributions

    # # Geometric quantities
    rdotf_j =   (r.at[:,0].get() * forces.at[3*indices_j + 0].get() + r.at[:,1].get() * forces.at[3*indices_j + 1].get() + r.at[:,2].get() * forces.at[3*indices_j + 2].get())
    mrdotf_i = -(r.at[:,0].get() * forces.at[3*indices_i + 0].get() + r.at[:,1].get() * forces.at[3*indices_i + 1].get() + r.at[:,2].get() * forces.at[3*indices_i + 2].get())
    
    Cj_dotr = jnp.array( [couplets.at[8*indices_j + 0].get() * r.at[:,0].get() + couplets.at[8*indices_j + 1].get() * r.at[:,1].get() + couplets.at[8*indices_j + 2].get() * r.at[:,2].get(),
                          couplets.at[8*indices_j + 5].get() * r.at[:,0].get() + couplets.at[8*indices_j + 4].get() * r.at[:,1].get() + couplets.at[8*indices_j + 3].get() * r.at[:,2].get(),
                          couplets.at[8*indices_j + 6].get() * r.at[:,0].get() + couplets.at[8*indices_j + 7].get() * r.at[:,1].get() -(couplets.at[8*indices_j + 0].get() + couplets.at[8*indices_j + 4].get()) * r.at[:,2].get()])
    
    Ci_dotmr=jnp.array( [-couplets.at[8*indices_i + 0].get() * r.at[:,0].get() - couplets.at[8*indices_i + 1].get() * r.at[:,1].get() - couplets.at[8*indices_i + 2].get() * r.at[:,2].get(),
                          -couplets.at[8*indices_i + 5].get() * r.at[:,0].get() - couplets.at[8*indices_i + 4].get() * r.at[:,1].get() - couplets.at[8*indices_i + 3].get() * r.at[:,2].get(),
                          -couplets.at[8*indices_i + 6].get() * r.at[:,0].get() - couplets.at[8*indices_i + 7].get() * r.at[:,1].get() +(couplets.at[8*indices_i + 0].get() + couplets.at[8*indices_i + 4].get()) * r.at[:,2].get()])


    rdotC_j = jnp.array([couplets.at[8*indices_j + 0].get() * r.at[:,0].get() + couplets.at[8*indices_j + 5].get() * r.at[:,1].get() + couplets.at[8*indices_j + 6].get() * r.at[:,2].get(),
                          couplets.at[8*indices_j + 1].get() * r.at[:,0].get() + couplets.at[8*indices_j + 4].get() * r.at[:,1].get() + couplets.at[8*indices_j + 7].get() * r.at[:,2].get(),
                          couplets.at[8*indices_j + 2].get() * r.at[:,0].get() + couplets.at[8*indices_j + 3].get() * r.at[:,1].get() -(couplets.at[8*indices_j + 0].get() + couplets.at[8*indices_j + 4].get()) * r.at[:,2].get()])
    
    mrdotC_i=jnp.array([-couplets.at[8*indices_i + 0].get() * r.at[:,0].get() - couplets.at[8*indices_i + 5].get() * r.at[:,1].get() - couplets.at[8*indices_i + 6].get() * r.at[:,2].get(),
                          -couplets.at[8*indices_i + 1].get() * r.at[:,0].get() - couplets.at[8*indices_i + 4].get() * r.at[:,1].get() - couplets.at[8*indices_i + 7].get() * r.at[:,2].get(),
                          -couplets.at[8*indices_i + 2].get() * r.at[:,0].get() - couplets.at[8*indices_i + 3].get() * r.at[:,1].get() +(couplets.at[8*indices_i + 0].get() + couplets.at[8*indices_i + 4].get()) * r.at[:,2].get()])
                
    rdotC_jj_dotr   =  (r.at[:,0].get()*Cj_dotr.at[0,:].get()  + r.at[:,1].get()*Cj_dotr.at[1,:].get()  + r.at[:,2].get()*Cj_dotr.at[2,:].get())
    mrdotC_ii_dotmr = -(r.at[:,0].get()*Ci_dotmr.at[0,:].get() + r.at[:,1].get()*Ci_dotmr.at[1,:].get() + r.at[:,2].get()*Ci_dotmr.at[2,:].get())
    
    
    # Compute Velocity for particles i
    r_lin_velocities = r_lin_velocities.at[indices_i, 0].add(f1 * forces.at[3*indices_j].get() + (f2 - f1) * rdotf_j * r.at[:,0].get())
    r_lin_velocities = r_lin_velocities.at[indices_i, 1].add(f1 * forces.at[3*indices_j+1].get() + (f2 - f1) * rdotf_j * r.at[:,1].get())
    r_lin_velocities = r_lin_velocities.at[indices_i, 2].add(f1 * forces.at[3*indices_j+2].get() + (f2 - f1) * rdotf_j * r.at[:,2].get())
    r_lin_velocities = r_lin_velocities.at[indices_i, 0].add(g1 * (Cj_dotr.at[0,:].get() - rdotC_jj_dotr * r.at[:,0].get()) + g2 * (rdotC_j.at[0,:].get() - 4.*rdotC_jj_dotr * r.at[:,0].get()))
    r_lin_velocities = r_lin_velocities.at[indices_i, 1].add(g1 * (Cj_dotr.at[1,:].get() - rdotC_jj_dotr * r.at[:,1].get()) + g2 * (rdotC_j.at[1,:].get() - 4.*rdotC_jj_dotr * r.at[:,1].get()))
    r_lin_velocities = r_lin_velocities.at[indices_i, 2].add(g1 * (Cj_dotr.at[2,:].get() - rdotC_jj_dotr * r.at[:,2].get()) + g2 * (rdotC_j.at[2,:].get() - 4.*rdotC_jj_dotr * r.at[:,2].get()))
    # Compute Velocity for particles j
    r_lin_velocities = r_lin_velocities.at[indices_j, 0].add(f1 * forces.at[3*indices_i].get() - (f2 - f1) * mrdotf_i * r.at[:,0].get())
    r_lin_velocities = r_lin_velocities.at[indices_j, 1].add(f1 * forces.at[3*indices_i+1].get() - (f2 - f1) * mrdotf_i * r.at[:,1].get())
    r_lin_velocities = r_lin_velocities.at[indices_j, 2].add(f1 * forces.at[3*indices_i+2].get() - (f2 - f1) * mrdotf_i * r.at[:,2].get())
    r_lin_velocities = r_lin_velocities.at[indices_j, 0].add(g1 * (Ci_dotmr.at[0,:].get() + mrdotC_ii_dotmr * r.at[:,0].get()) + g2 * (mrdotC_i.at[0,:].get() + 4.*mrdotC_ii_dotmr * r.at[:,0].get()))
    r_lin_velocities = r_lin_velocities.at[indices_j, 1].add(g1 * (Ci_dotmr.at[1,:].get() + mrdotC_ii_dotmr * r.at[:,1].get()) + g2 * (mrdotC_i.at[1,:].get() + 4.*mrdotC_ii_dotmr * r.at[:,1].get()))
    r_lin_velocities = r_lin_velocities.at[indices_j, 2].add(g1 * (Ci_dotmr.at[2,:].get() + mrdotC_ii_dotmr * r.at[:,2].get()) + g2 * (mrdotC_i.at[2,:].get() + 4.*mrdotC_ii_dotmr * r.at[:,2].get()))
    
    
    # Compute Velocity Gradient for particles i and j
    r_velocity_gradient = r_velocity_gradient.at[indices_i, 0].add(
        (-1) * g1 * (r.at[:,0].get() * forces.at[3*indices_j+0].get() - rdotf_j * r.at[:,0].get()*r.at[:,0].get())
        +(-1)*g2 * (rdotf_j + forces.at[3*indices_j+0].get()*r.at[:,0].get() - 4.*rdotf_j * r.at[:,0].get()*r.at[:,0].get()))
    
    r_velocity_gradient = r_velocity_gradient.at[indices_j, 0].add(
        (-1) * g1 * (-r.at[:,0].get() * forces.at[3*indices_i+0].get() - mrdotf_i * r.at[:,0].get()*r.at[:,0].get()) 
        +(-1)*g2 * (mrdotf_i - forces.at[3*indices_i+0].get()*r.at[:,0].get() - 4.*mrdotf_i * r.at[:,0].get()*r.at[:,0].get()))
    
    r_velocity_gradient = r_velocity_gradient.at[indices_i, 0].add(
        h1 * (couplets.at[8*indices_j+0].get() - 4. * couplets.at[8*indices_j+0].get()) 
        + h2 * (r.at[:,0].get()*Cj_dotr.at[0,:].get() - rdotC_jj_dotr * r.at[:,0].get()*r.at[:,0].get()) 
        + h3 * (rdotC_jj_dotr + Cj_dotr.at[0,:].get()*r.at[:,0].get() + r.at[:,0].get()*rdotC_j.at[0,:].get() + rdotC_j.at[0,:].get()*r.at[:,0].get() 
                                                                - 6.*rdotC_jj_dotr*r.at[:,0].get()*r.at[:,0].get() - couplets.at[8*indices_j+0].get()))
    r_velocity_gradient = r_velocity_gradient.at[indices_j, 0].add(
        h1 * (couplets.at[8*indices_i+0].get() - 4. * couplets.at[8*indices_i+0].get()) 
        + h2 * (-r.at[:,0].get()*Ci_dotmr.at[0,:].get() - mrdotC_ii_dotmr * r.at[:,0].get()*r.at[:,0].get())
        + h3 * (mrdotC_ii_dotmr - Ci_dotmr.at[0,:].get()*r.at[:,0].get() - r.at[:,0].get()*mrdotC_i.at[0,:].get() - mrdotC_i.at[0,:].get()*r.at[:,0].get() 
                                                                - 6.*mrdotC_ii_dotmr*r.at[:,0].get()*r.at[:,0].get() - couplets.at[8*indices_i+0].get()))


    r_velocity_gradient = r_velocity_gradient.at[indices_i, 1].add(
        (-1) * g1 * (r.at[:,1].get() * forces.at[3*indices_j+0].get() - rdotf_j * r.at[:,1].get()*r.at[:,0].get()) 
        + (-1)*g2 * (forces.at[3*indices_j+1].get()*r.at[:,0].get() - 4.*rdotf_j * r.at[:,1].get()*r.at[:,0].get()))
    
    r_velocity_gradient = r_velocity_gradient.at[indices_j, 1].add(
        (-1) * g1 * (-r.at[:,1].get() * forces.at[3*indices_i+0].get() - mrdotf_i * r.at[:,1].get()*r.at[:,0].get()) 
        + (-1)*g2 * (-forces.at[3*indices_i+1].get()*r.at[:,0].get() - 4.*mrdotf_i * r.at[:,1].get()*r.at[:,0].get()))
    
    r_velocity_gradient = r_velocity_gradient.at[indices_i, 1].add(
        h1 * (couplets.at[8*indices_j+5].get() - 4. * couplets.at[8*indices_j+1].get()) 
        + h2 * (r.at[:,1].get()*Cj_dotr.at[0,:].get() - rdotC_jj_dotr * r.at[:,1].get()*r.at[:,0].get())
        + h3 * (Cj_dotr.at[1,:].get()*r.at[:,0].get() + r.at[:,1].get()*rdotC_j.at[0,:].get() + rdotC_j.at[1,:].get()*r.at[:,0].get()
                                            - 6.*rdotC_jj_dotr*r.at[:,1].get()*r.at[:,0].get() - couplets.at[8*indices_j+1].get()))
    
    r_velocity_gradient = r_velocity_gradient.at[indices_j, 1].add(
        h1 * (couplets.at[8*indices_i+5].get() - 4. * couplets.at[8*indices_i+1].get())
        + h2 * (-r.at[:,1].get()*Ci_dotmr.at[0,:].get() - mrdotC_ii_dotmr * r.at[:,1].get()*r.at[:,0].get())
        + h3 * (-Ci_dotmr.at[1,:].get()*r.at[:,0].get() - r.at[:,1].get()*mrdotC_i.at[0,:].get() - mrdotC_i.at[1,:].get()*r.at[:,0].get()
                                                - 6.*mrdotC_ii_dotmr*r.at[:,1].get()*r.at[:,0].get() - couplets.at[8*indices_i+1].get()))



    r_velocity_gradient = r_velocity_gradient.at[indices_i, 2].add(
              (-1) * g1 * (r.at[:,2].get() * forces.at[3*indices_j+0].get() - rdotf_j * r.at[:,2].get()*r.at[:,0].get()) 
              + (-1)*g2 * (forces.at[3*indices_j+2].get()*r.at[:,0].get() - 4.*rdotf_j * r.at[:,2].get()*r.at[:,0].get()))
    
    r_velocity_gradient = r_velocity_gradient.at[indices_j, 2].add(
              (-1) * g1 * (-r.at[:,2].get() * forces.at[3*indices_i+0].get() - mrdotf_i * r.at[:,2].get()*r.at[:,0].get()) 
              + (-1)*g2 * (-forces.at[3*indices_i+2].get()*r.at[:,0].get() - 4.*mrdotf_i * r.at[:,2].get()*r.at[:,0].get()))
    r_velocity_gradient = r_velocity_gradient.at[indices_i, 2].add(
        h1 * (couplets.at[8*indices_j+6].get() - 4. * couplets.at[8*indices_j+2].get())
        + h2 * (r.at[:,2].get()*Cj_dotr.at[0,:].get() - rdotC_jj_dotr * r.at[:,2].get()*r.at[:,0].get())
        + h3 * (Cj_dotr.at[2,:].get()*r.at[:,0].get() + r.at[:,2].get()*rdotC_j.at[0,:].get() + rdotC_j.at[2,:].get()*r.at[:,0].get()
                - 6.*rdotC_jj_dotr*r.at[:,2].get()*r.at[:,0].get() - couplets.at[8*indices_j+2].get()))
    r_velocity_gradient = r_velocity_gradient.at[indices_j, 2].add(
        h1 * (couplets.at[8*indices_i+6].get() - 4. * couplets.at[8*indices_i+2].get())
        + h2 * (r.at[:,2].get()*Ci_dotmr.at[0,:].get()*(-1) - mrdotC_ii_dotmr * r.at[:,2].get()*r.at[:,0].get())
        + h3 * (-Ci_dotmr.at[2,:].get()*r.at[:,0].get() - r.at[:,2].get()*mrdotC_i.at[0,:].get() - mrdotC_i.at[2,:].get()*r.at[:,0].get()
                - 6.*mrdotC_ii_dotmr*r.at[:,2].get()*r.at[:,0].get() - couplets.at[8*indices_i+2].get()))
   



    r_velocity_gradient = r_velocity_gradient.at[indices_i, 3].add(
            (-1) * g1 * (r.at[:,2].get() * forces.at[3*indices_j+1].get() - rdotf_j * r.at[:,2].get()*r.at[:,1].get())
            + (-1)*g2 * (forces.at[3*indices_j+2].get()*r.at[:,1].get() - 4.*rdotf_j * r.at[:,2].get()*r.at[:,1].get()))
    
    r_velocity_gradient = r_velocity_gradient.at[indices_j, 3].add(
            (-1) * g1 * (-r.at[:,2].get() * forces.at[3*indices_i+1].get() - mrdotf_i * r.at[:,2].get()*r.at[:,1].get())
            + (-1)*g2 * (-forces.at[3*indices_i+2].get()*r.at[:,1].get() - 4.*mrdotf_i * r.at[:,2].get()*r.at[:,1].get()))
    
    r_velocity_gradient = r_velocity_gradient.at[indices_i, 3].add(
        h1 * (couplets.at[8*indices_j+7].get() - 4. * couplets.at[8*indices_j+3].get())
        + h2 * (r.at[:,2].get()*Cj_dotr.at[1,:].get() - rdotC_jj_dotr * r.at[:,2].get()*r.at[:,1].get())
        + h3 * (Cj_dotr.at[2,:].get()*r.at[:,1].get() + r.at[:,2].get()*rdotC_j.at[1,:].get() + rdotC_j.at[2,:].get()*r.at[:,1].get()
                - 6.*rdotC_jj_dotr*r.at[:,2].get()*r.at[:,1].get() - couplets.at[8*indices_j+3].get()))

    r_velocity_gradient = r_velocity_gradient.at[indices_j, 3].add(
        h1 * (couplets.at[8*indices_i+7].get() - 4. * couplets.at[8*indices_i+3].get())
        + h2 * (-r.at[:,2].get()*Ci_dotmr.at[1,:].get() - mrdotC_ii_dotmr * r.at[:,2].get()*r.at[:,1].get()) 
        + h3 * (-Ci_dotmr.at[2,:].get()*r.at[:,1].get() - r.at[:,2].get()*mrdotC_i.at[1,:].get() - mrdotC_i.at[2,:].get()*r.at[:,1].get()
                - 6.*mrdotC_ii_dotmr*r.at[:,2].get()*r.at[:,1].get() - couplets.at[8*indices_i+3].get()))



    r_velocity_gradient = r_velocity_gradient.at[indices_i, 4].add(
            (-1) * g1 * (r.at[:,1].get() * forces.at[3*indices_j+1].get() - rdotf_j * r.at[:,1].get()*r.at[:,1].get())
            + (-1)*g2 * (rdotf_j + forces.at[3*indices_j+1].get()*r.at[:,1].get() - 4.*rdotf_j * r.at[:,1].get()*r.at[:,1].get()))
    
    r_velocity_gradient = r_velocity_gradient.at[indices_j, 4].add(
            (-1) * g1 * (-r.at[:,1].get() * forces.at[3*indices_i+1].get() - mrdotf_i * r.at[:,1].get()*r.at[:,1].get())
            + (-1)*g2 * (mrdotf_i - forces.at[3*indices_i+1].get()*r.at[:,1].get() - 4.*mrdotf_i * r.at[:,1].get()*r.at[:,1].get()))
     
    r_velocity_gradient = r_velocity_gradient.at[indices_i, 4].add(
        h1 * (couplets.at[8*indices_j+4].get() - 4. * couplets.at[8*indices_j+4].get())
        + h2 * (r.at[:,1].get()*Cj_dotr.at[1,:].get() - rdotC_jj_dotr * r.at[:,1].get()*r.at[:,1].get()) 
        + h3 * (rdotC_jj_dotr + Cj_dotr.at[1,:].get()*r.at[:,1].get() + r.at[:,1].get()*rdotC_j.at[1,:].get() + rdotC_j.at[1,:].get()*r.at[:,1].get()
                - 6.*rdotC_jj_dotr*r.at[:,1].get()*r.at[:,1].get() - couplets.at[8*indices_j+4].get()))
    
    r_velocity_gradient = r_velocity_gradient.at[indices_j, 4].add(
        h1 * (couplets.at[8*indices_i+4].get() - 4. * couplets.at[8*indices_i+4].get())
        + h2 * (-r.at[:,1].get()*Ci_dotmr.at[1,:].get() - mrdotC_ii_dotmr * r.at[:,1].get()*r.at[:,1].get())
        + h3 * (mrdotC_ii_dotmr - Ci_dotmr.at[1,:].get()*r.at[:,1].get() - r.at[:,1].get()*mrdotC_i.at[1,:].get() - mrdotC_i.at[1,:].get()*r.at[:,1].get()
                - 6.*mrdotC_ii_dotmr*r.at[:,1].get()*r.at[:,1].get() - couplets.at[8*indices_i+4].get()))


    r_velocity_gradient = r_velocity_gradient.at[indices_i, 5].add(
            (-1) * g1 * (r.at[:,0].get() * forces.at[3*indices_j+1].get() - rdotf_j * r.at[:,0].get()*r.at[:,1].get())
            + (-1)*g2 * (forces.at[3*indices_j+0].get()*r.at[:,1].get() - 4.*rdotf_j * r.at[:,0].get()*r.at[:,1].get()))
    
    r_velocity_gradient = r_velocity_gradient.at[indices_j, 5].add(
            (-1) * g1 * (-r.at[:,0].get() * forces.at[3*indices_i+1].get() - mrdotf_i * r.at[:,0].get()*r.at[:,1].get())
            + (-1)*g2 * (-forces.at[3*indices_i+0].get()*r.at[:,1].get() - 4.*mrdotf_i * r.at[:,0].get()*r.at[:,1].get()))
    
    r_velocity_gradient = r_velocity_gradient.at[indices_i, 5].add(
        h1 * (couplets.at[8*indices_j+1].get() - 4. * couplets.at[8*indices_j+5].get())
        + h2 * (r.at[:,0].get()*Cj_dotr.at[1,:].get() - rdotC_jj_dotr * r.at[:,0].get()*r.at[:,1].get())
        + h3 * (Cj_dotr.at[0,:].get()*r.at[:,1].get() + r.at[:,0].get()*rdotC_j.at[1,:].get() + rdotC_j.at[0,:].get()*r.at[:,1].get()
                - 6.*rdotC_jj_dotr*r.at[:,0].get()*r.at[:,1].get() - couplets.at[8*indices_j+5].get()))
    
    r_velocity_gradient = r_velocity_gradient.at[indices_j, 5].add(
        h1 * (couplets.at[8*indices_i+1].get() - 4. * couplets.at[8*indices_i+5].get())
        + h2 * (-r.at[:,0].get()*Ci_dotmr.at[1,:].get() - mrdotC_ii_dotmr * r.at[:,0].get()*r.at[:,1].get())
        + h3 * (-Ci_dotmr.at[0,:].get()*r.at[:,1].get() - r.at[:,0].get()*mrdotC_i.at[1,:].get() - mrdotC_i.at[0,:].get()*r.at[:,1].get()
                - 6.*mrdotC_ii_dotmr*r.at[:,0].get()*r.at[:,1].get() - couplets.at[8*indices_i+5].get()))



    r_velocity_gradient = r_velocity_gradient.at[indices_i, 6].add(
              (-1) * g1 * (r.at[:,0].get() * forces.at[3*indices_j+2].get() - rdotf_j * r.at[:,0].get()*r.at[:,2].get())
              + (-1)*g2 * (forces.at[3*indices_j+0].get()*r.at[:,2].get() - 4.*rdotf_j * r.at[:,0].get()*r.at[:,2].get()))
    
    r_velocity_gradient = r_velocity_gradient.at[indices_j, 6].add(
              (-1) * g1 * (-r.at[:,0].get() * forces.at[3*indices_i+2].get() - mrdotf_i * r.at[:,0].get()*r.at[:,2].get())
              + (-1)*g2 * (-forces.at[3*indices_i+0].get()*r.at[:,2].get() - 4.*mrdotf_i * r.at[:,0].get()*r.at[:,2].get()))
     
    r_velocity_gradient = r_velocity_gradient.at[indices_i, 6].add(
        h1 * (couplets.at[8*indices_j+2].get() - 4. * couplets.at[8*indices_j+6].get())
        + h2 * (r.at[:,0].get()*Cj_dotr.at[2,:].get() - rdotC_jj_dotr * r.at[:,0].get()*r.at[:,2].get())
        + h3 * (Cj_dotr.at[0,:].get()*r.at[:,2].get() + r.at[:,0].get()*rdotC_j.at[2,:].get() + rdotC_j.at[0,:].get()*r.at[:,2].get()
                - 6.*rdotC_jj_dotr*r.at[:,0].get()*r.at[:,2].get() - couplets.at[8*indices_j+6].get()))
    
    r_velocity_gradient = r_velocity_gradient.at[indices_j, 6].add(
        h1 * (couplets.at[8*indices_i+2].get() - 4. * couplets.at[8*indices_i+6].get())
        + h2 * (-r.at[:,0].get()*Ci_dotmr.at[2,:].get() - mrdotC_ii_dotmr * r.at[:,0].get()*r.at[:,2].get())
        + h3 * (-Ci_dotmr.at[0,:].get()*r.at[:,2].get() - r.at[:,0].get()*mrdotC_i.at[2,:].get() - mrdotC_i.at[0,:].get()*r.at[:,2].get()
                - 6.*mrdotC_ii_dotmr*r.at[:,0].get()*r.at[:,2].get() - couplets.at[8*indices_i+6].get()))
    
    
    
    r_velocity_gradient = r_velocity_gradient.at[indices_i, 7].add(
            (-1) * g1 * (r.at[:,1].get() * forces.at[3*indices_j+2].get() - rdotf_j * r.at[:,1].get()*r.at[:,2].get())
            + (-1)*g2 * (forces.at[3*indices_j+1].get()*r.at[:,2].get() - 4.*rdotf_j * r.at[:,1].get()*r.at[:,2].get()))
    
    r_velocity_gradient = r_velocity_gradient.at[indices_j, 7].add(
            (-1) * g1 * (-r.at[:,1].get() * forces.at[3*indices_i+2].get() - mrdotf_i * r.at[:,1].get()*r.at[:,2].get())
            + (-1)*g2 * (-forces.at[3*indices_i+1].get()*r.at[:,2].get() - 4.*mrdotf_i * r.at[:,1].get()*r.at[:,2].get()))
    
    r_velocity_gradient = r_velocity_gradient.at[indices_i, 7].add(
        h1 * (couplets.at[8*indices_j+3].get() - 4. * couplets.at[8*indices_j+7].get())
        + h2 * (r.at[:,1].get()*Cj_dotr.at[2,:].get() - rdotC_jj_dotr * r.at[:,1].get()*r.at[:,2].get())
        + h3 * (Cj_dotr.at[1,:].get()*r.at[:,2].get() + r.at[:,1].get()*rdotC_j.at[2,:].get() + rdotC_j.at[1,:].get()*r.at[:,2].get()
                - 6.*rdotC_jj_dotr*r.at[:,1].get()*r.at[:,2].get() - couplets.at[8*indices_j+7].get()))
    
    r_velocity_gradient = r_velocity_gradient.at[indices_j, 7].add(
        h1 * (couplets.at[8*indices_i+3].get() - 4. * couplets.at[8*indices_i+7].get())
        + h2 * (-r.at[:,1].get()*Ci_dotmr.at[2,:].get() - mrdotC_ii_dotmr * r.at[:,1].get()*r.at[:,2].get())
        + h3 * (-Ci_dotmr.at[1,:].get()*r.at[:,2].get() - r.at[:,1].get()*mrdotC_i.at[2,:].get() - mrdotC_i.at[1,:].get()*r.at[:,2].get()
                - 6.*mrdotC_ii_dotmr*r.at[:,1].get()*r.at[:,2].get() - couplets.at[8*indices_i+7].get()))
    


    ##########################################################################################################################################
    ##########################################################################################################################################

    # Add wave and real space part together
    lin_vel = w_lin_velocities + r_lin_velocities
    velocity_gradient = w_velocity_gradient + r_velocity_gradient

    # # Convert to angular velocities and rate of strain
    ang_vel_and_strain = jnp.zeros((N, 8))
    ang_vel_and_strain = ang_vel_and_strain.at[:, 0].set((velocity_gradient.at[:, 3].get()-velocity_gradient.at[:, 7].get()) * 0.5)
    ang_vel_and_strain = ang_vel_and_strain.at[:, 1].set((velocity_gradient.at[:, 6].get()-velocity_gradient.at[:, 2].get()) * 0.5)
    ang_vel_and_strain = ang_vel_and_strain.at[:, 2].set((velocity_gradient.at[:, 1].get()-velocity_gradient.at[:, 5].get()) * 0.5)
    ang_vel_and_strain = ang_vel_and_strain.at[:, 3].set(2*velocity_gradient.at[:, 0].get()+velocity_gradient.at[:, 4].get())
    ang_vel_and_strain = ang_vel_and_strain.at[:, 4].set(velocity_gradient.at[:, 1].get()+velocity_gradient.at[:, 5].get())
    ang_vel_and_strain = ang_vel_and_strain.at[:, 5].set(velocity_gradient.at[:, 2].get()+velocity_gradient.at[:, 6].get())
    ang_vel_and_strain = ang_vel_and_strain.at[:, 6].set(velocity_gradient.at[:, 3].get()+velocity_gradient.at[:, 7].get())
    ang_vel_and_strain = ang_vel_and_strain.at[:, 7].set(velocity_gradient.at[:, 0].get()+2*velocity_gradient.at[:, 4].get())

    # Convert to Generalized Velocities+Strain 
    generalized_velocities = jnp.zeros(11*N) #First 6N entries for U and last 5N for strain rates

    generalized_velocities = generalized_velocities.at[0:6*N:6].set(
        lin_vel.at[:, 0].get())
    generalized_velocities = generalized_velocities.at[1:6*N:6].set(
        lin_vel.at[:, 1].get())
    generalized_velocities = generalized_velocities.at[2:6*N:6].set(
        lin_vel.at[:, 2].get())
    generalized_velocities = generalized_velocities.at[3:6*N:6].set(
        ang_vel_and_strain.at[:, 0].get())
    generalized_velocities = generalized_velocities.at[4:6*N:6].set(
        ang_vel_and_strain.at[:, 1].get())
    generalized_velocities = generalized_velocities.at[5:6*N:6].set(
        ang_vel_and_strain.at[:, 2].get())
    generalized_velocities = generalized_velocities.at[(6*N+0)::5].set(
        ang_vel_and_strain.at[:, 3].get())
    generalized_velocities = generalized_velocities.at[(6*N+1)::5].set(
        ang_vel_and_strain.at[:, 4].get())
    generalized_velocities = generalized_velocities.at[(6*N+2)::5].set(
        ang_vel_and_strain.at[:, 5].get())
    generalized_velocities = generalized_velocities.at[(6*N+3)::5].set(
        ang_vel_and_strain.at[:, 6].get())
    generalized_velocities = generalized_velocities.at[(6*N+4)::5].set(
        ang_vel_and_strain.at[:, 7].get())
    
    #Clean Grids for next iteration
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
    
    return generalized_velocities

@partial(jit, static_argnums=[0,1,2,3,4])
def Mobility(
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
        generalized_forces: ArrayLike) -> ArrayLike:
    
    """Construct the saddle point operator A,
        which acts on x and returns A*x (without using A in matrix representation)

    Parameters
    ----------
    N:
        Number of particles
    Nx:
        Number of grid points in x direction
    Ny:
        Number of grid points in y direction
    Nz:
        Number of grid points in z direction
    gaussP:
        Gaussian support size for wave space calculation 
    gridk:
        Wave number values in the grid
    m_self:
        Mobility self contribution
    all_indices_x:
        All indices (x) of wave grid points for each particle
    all_indices_y:
        All indices (y) of wave grid points for each particle
    all_indices_z:
        All indices (z) of wave grid points for each particle
    gaussian_grid_spacing1:
        Scaled distances from support center to each gridpoint, for FFT  
    gaussian_grid_spacing2:
        Scaled distances from support center to each gridpoint, for inverse FFT  
    r:
        Units vectors connecting each pair of particles in the far-field neighbor list
    indices_i:
        Indices of first particle in far-field neighbor list pairs 
    indices_j:
        Indices of second particle in far-field neighbor list pairs 
    f1:
        Mobility scalar function 1
    f2:
        Mobility scalar function 2
    g1:
        Mobility scalar function 3
    g2:
        Mobility scalar function 4
    h1:
        Mobility scalar function 5
    h2:
        Mobility scalar function 6
    h3:
        Mobility scalar function 7
    generalized_forces:
        Input generalized forces (force/torque/stresslet)
        
    Returns
    -------
    generalized_velocities (linear/angular velocities and rateOfStrain) 

    """
    
    # Helper function
    def swap_real_imag(cplx_arr):
        return -jnp.imag(cplx_arr) + 1j * jnp.real(cplx_arr)

    # Get forces,torques,couplets from generalized forces (3*N vector: f1x,f1y,f1z, ... , fNx,fNy,fNz and same for torque, while couplet is 5N) 
    forces = jnp.zeros(3*N)
    forces = forces.at[0::3].set(generalized_forces.at[0:(6*N):6].get())
    forces = forces.at[1::3].set(generalized_forces.at[1:(6*N):6].get())
    forces = forces.at[2::3].set(generalized_forces.at[2:(6*N):6].get())
    torques = jnp.zeros(3*N)
    torques = torques.at[0::3].set(generalized_forces.at[3:(6*N):6].get())
    torques = torques.at[1::3].set(generalized_forces.at[4:(6*N):6].get())
    torques = torques.at[2::3].set(generalized_forces.at[5:(6*N):6].get())
    # stresslet = jnp.zeros(5*N)
    # stresslet = stresslet.at[0::5].set(generalized_forces.at[(6*N+0)::5].get()) #Sxx
    # stresslet = stresslet.at[1::5].set(generalized_forces.at[(6*N+1)::5].get()) #Sxy
    # stresslet = stresslet.at[2::5].set(generalized_forces.at[(6*N+2)::5].get()) #Sxz
    # stresslet = stresslet.at[3::5].set(generalized_forces.at[(6*N+3)::5].get()) #Syz
    # stresslet = stresslet.at[4::5].set(generalized_forces.at[(6*N+4)::5].get()) #Syy

    # # Get 'couplet' from generalized forces (8*N vector)    
    # couplets = jnp.zeros(8*N)
    # couplets = couplets.at[::8].set(stresslet.at[::5].get())  # C[0] = S[0]
    # couplets = couplets.at[1::8].set(
    #     stresslet.at[1::5].get()+torques.at[2::3].get()*0.5)  # C[1] = S[1] + L[2]/2
    # couplets = couplets.at[2::8].set(
    #     stresslet.at[2::5].get()-torques.at[1::3].get()*0.5)  # C[2] = S[2] - L[1]/2
    # couplets = couplets.at[3::8].set(
    #     stresslet.at[3::5].get()+torques.at[::3].get()*0.5)  # C[3] = S[3] + L[0]/2
    # couplets = couplets.at[4::8].set(stresslet.at[4::5].get())  # C[4] = S[4]
    # couplets = couplets.at[5::8].set(
    #     stresslet.at[1::5].get()-torques.at[2::3].get()*0.5)  # C[5] = S[1] - L[2]/2
    # couplets = couplets.at[6::8].set(
    #     stresslet.at[2::5].get()+torques.at[1::3].get()*0.5)  # C[6] = S[2] + L[1]/2
    # couplets = couplets.at[7::8].set(
    #     stresslet.at[3::5].get()-torques.at[::3].get()*0.5)  # C[7] = S[3] - L[0]/2

    ##########################################################################################################################################
    ######################################## WAVE SPACE CONTRIBUTION #########################################################################
    ##########################################################################################################################################
    
    #Create Grids for current iteration
    gridX = jnp.zeros((Nx, Ny, Nz))
    gridY = jnp.zeros((Nx, Ny, Nz))
    gridZ = jnp.zeros((Nx, Ny, Nz))
    # gridXX = jnp.zeros((Nx, Ny, Nz))
    # gridXY = jnp.zeros((Nx, Ny, Nz))
    # gridXZ = jnp.zeros((Nx, Ny, Nz))
    # gridYX = jnp.zeros((Nx, Ny, Nz))
    # gridYY = jnp.zeros((Nx, Ny, Nz))
    # gridYZ = jnp.zeros((Nx, Ny, Nz))
    # gridZX = jnp.zeros((Nx, Ny, Nz))
    # gridZY = jnp.zeros((Nx, Ny, Nz))
    
    gridX = gridX.at[all_indices_x, all_indices_y, all_indices_z].add(jnp.ravel((jnp.swapaxes(jnp.swapaxes(jnp.swapaxes(gaussian_grid_spacing1*jnp.resize(forces.at[0::3].get(), (gaussP,gaussP,gaussP,N) ),3,2),2,1),1,0))))
    gridY = gridY.at[all_indices_x, all_indices_y, all_indices_z].add(jnp.ravel((jnp.swapaxes(jnp.swapaxes(jnp.swapaxes(gaussian_grid_spacing1*jnp.resize(forces.at[1::3].get(), (gaussP,gaussP,gaussP,N) ),3,2),2,1),1,0))))            
    gridZ = gridZ.at[all_indices_x, all_indices_y, all_indices_z].add(jnp.ravel((jnp.swapaxes(jnp.swapaxes(jnp.swapaxes(gaussian_grid_spacing1*jnp.resize(forces.at[2::3].get(), (gaussP,gaussP,gaussP,N) ),3,2),2,1),1,0))))
    # gridXX = gridXX.at[all_indices_x, all_indices_y, all_indices_z].add(jnp.ravel((jnp.swapaxes(jnp.swapaxes(jnp.swapaxes(gaussian_grid_spacing1*jnp.resize(couplets.at[0::8].get(), (gaussP,gaussP,gaussP,N) ),3,2),2,1),1,0))))
    # gridXY = gridXY.at[all_indices_x, all_indices_y, all_indices_z].add(jnp.ravel((jnp.swapaxes(jnp.swapaxes(jnp.swapaxes(gaussian_grid_spacing1*jnp.resize(couplets.at[1::8].get(), (gaussP,gaussP,gaussP,N) ),3,2),2,1),1,0))))
    # gridXZ = gridXZ.at[all_indices_x, all_indices_y, all_indices_z].add(jnp.ravel((jnp.swapaxes(jnp.swapaxes(jnp.swapaxes(gaussian_grid_spacing1*jnp.resize(couplets.at[2::8].get(), (gaussP,gaussP,gaussP,N) ),3,2),2,1),1,0)))) 
    # gridYZ = gridYZ.at[all_indices_x, all_indices_y, all_indices_z].add(jnp.ravel((jnp.swapaxes(jnp.swapaxes(jnp.swapaxes(gaussian_grid_spacing1*jnp.resize(couplets.at[3::8].get(), (gaussP,gaussP,gaussP,N) ),3,2),2,1),1,0))))
    # gridYY = gridYY.at[all_indices_x, all_indices_y, all_indices_z].add(jnp.ravel((jnp.swapaxes(jnp.swapaxes(jnp.swapaxes(gaussian_grid_spacing1*jnp.resize(couplets.at[4::8].get(), (gaussP,gaussP,gaussP,N) ),3,2),2,1),1,0))))
    # gridYX = gridYX.at[all_indices_x, all_indices_y, all_indices_z].add(jnp.ravel((jnp.swapaxes(jnp.swapaxes(jnp.swapaxes(gaussian_grid_spacing1*jnp.resize(couplets.at[5::8].get(), (gaussP,gaussP,gaussP,N) ),3,2),2,1),1,0))))
    # gridZX = gridZX.at[all_indices_x, all_indices_y, all_indices_z].add(jnp.ravel((jnp.swapaxes(jnp.swapaxes(jnp.swapaxes(gaussian_grid_spacing1*jnp.resize(couplets.at[6::8].get(), (gaussP,gaussP,gaussP,N) ),3,2),2,1),1,0))))
    # gridZY = gridZY.at[all_indices_x, all_indices_y, all_indices_z].add(jnp.ravel((jnp.swapaxes(jnp.swapaxes(jnp.swapaxes(gaussian_grid_spacing1*jnp.resize(couplets.at[7::8].get(), (gaussP,gaussP,gaussP,N) ),3,2),2,1),1,0))))

    # Apply FFT
    gridX = jnp.fft.fftn(gridX)
    gridY = jnp.fft.fftn(gridY)
    gridZ = jnp.fft.fftn(gridZ)
    # gridXX = jnp.fft.fftn(gridXX)
    # gridXY = jnp.fft.fftn(gridXY)
    # gridXZ = jnp.fft.fftn(gridXZ)
    # gridYZ = jnp.fft.fftn(gridYZ)
    # gridYY = jnp.fft.fftn(gridYY)
    # gridYX = jnp.fft.fftn(gridYX)
    # gridZX = jnp.fft.fftn(gridZX)
    # gridZY = jnp.fft.fftn(gridZY)
    # gridZZ = - gridXX - gridYY
    
    gridk_sqr = (gridk.at[:, :, :, 0].get()*gridk.at[:, :, :, 0].get()+gridk.at[:, :, :, 1].get()
                 * gridk.at[:, :, :, 1].get()+gridk.at[:, :, :, 2].get()*gridk.at[:, :, :, 2].get())
    gridk_mod = jnp.sqrt(gridk_sqr)
    
    kdF = jnp.where(gridk_mod > 0, (gridk.at[:, :, :, 0].get()*gridX +
                    gridk.at[:, :, :, 1].get()*gridY+gridk.at[:, :, :, 2].get()*gridZ)/gridk_sqr, 0)

    # Cdkx = (gridk.at[:, :, :, 0].get()*gridXX + gridk.at[:, :, :, 1].get()*gridXY + gridk.at[:, :, :, 2].get()*gridXZ)
    # Cdky = (gridk.at[:, :, :, 0].get()*gridYX + gridk.at[:, :, :, 1].get()*gridYY + gridk.at[:, :, :, 2].get()*gridYZ)
    # Cdkz = (gridk.at[:, :, :, 0].get()*gridZX + gridk.at[:, :, :, 1].get()*gridZY + gridk.at[:, :, :, 2].get()*gridZZ)

    # kdcdk = jnp.where(
    #     gridk_mod > 0, ( gridk.at[:, :, :, 0].get()*Cdkx
    #                     +gridk.at[:, :, :, 1].get()*Cdky
    #                     +gridk.at[:, :, :, 2].get()*Cdkz)/gridk_sqr, 0)
    
    # Fkxx = (gridk.at[:, :, :, 0].get()*gridX)
    # Fkxy = (gridk.at[:, :, :, 1].get()*gridX)
    # Fkxz = (gridk.at[:, :, :, 2].get()*gridX)
    # Fkyx = (gridk.at[:, :, :, 0].get()*gridY)
    # Fkyy = (gridk.at[:, :, :, 1].get()*gridY)
    # Fkyz = (gridk.at[:, :, :, 2].get()*gridY)
    # Fkzx = (gridk.at[:, :, :, 0].get()*gridZ)
    # Fkzy = (gridk.at[:, :, :, 1].get()*gridZ)
    # kkxx = gridk.at[:, :, :, 0].get()*gridk.at[:, :, :, 0].get()
    # kkxy = gridk.at[:, :, :, 0].get()*gridk.at[:, :, :, 1].get()
    # kkxz = gridk.at[:, :, :, 0].get()*gridk.at[:, :, :, 2].get()
    # kkyx = gridk.at[:, :, :, 1].get()*gridk.at[:, :, :, 0].get()
    # kkyy = gridk.at[:, :, :, 1].get()*gridk.at[:, :, :, 1].get()
    # kkyz = gridk.at[:, :, :, 1].get()*gridk.at[:, :, :, 2].get()
    # kkzx = gridk.at[:, :, :, 2].get()*gridk.at[:, :, :, 0].get()
    # kkzy = gridk.at[:, :, :, 2].get()*gridk.at[:, :, :, 1].get()
    # Cdkkxx = gridk.at[:, :, :, 0].get() * Cdkx
    # Cdkkxy = gridk.at[:, :, :, 1].get() * Cdkx
    # Cdkkxz = gridk.at[:, :, :, 2].get() * Cdkx
    # Cdkkyx = gridk.at[:, :, :, 0].get() * Cdky
    # Cdkkyy = gridk.at[:, :, :, 1].get() * Cdky
    # Cdkkyz = gridk.at[:, :, :, 2].get() * Cdky
    # Cdkkzx = gridk.at[:, :, :, 0].get() * Cdkz
    # Cdkkzy = gridk.at[:, :, :, 1].get() * Cdkz

    # UF part
    B = jnp.where(gridk_mod > 0, gridk.at[:, :, :, 3].get()*(jnp.sin(gridk_mod)/gridk_mod)*(
        jnp.sin(gridk_mod)/gridk_mod), 0)  # scaling factor
    gridX = B * (gridX - gridk.at[:, :, :, 0].get() * kdF)
    gridY = B * (gridY - gridk.at[:, :, :, 1].get() * kdF)
    gridZ = B * (gridZ - gridk.at[:, :, :, 2].get() * kdF)

    # UC part (here B is imaginary so we absorb the imaginary unit in the funtion 'swap_real_imag()' which returns -Im(c)+i*Re(c)
    # B = jnp.where(gridk_mod > 0, gridk.at[:, :, :, 3].get()*(jnp.sin(gridk_mod)/gridk_mod)*(3 * (jnp.sin(
    #     gridk_mod) - gridk_mod*jnp.cos(gridk_mod)) / (gridk_mod*gridk_sqr)), 0)  # scaling factor

    # gridX += B * swap_real_imag( (Cdkx - kdcdk * gridk.at[:, :, :, 0].get()))
    # gridY += B * swap_real_imag( (Cdky - kdcdk * gridk.at[:, :, :, 1].get()))
    # gridZ += B * swap_real_imag( (Cdkz - kdcdk * gridk.at[:, :, :, 2].get()))

    # DF part
    # B = jnp.where(gridk_mod > 0, gridk.at[:, :, :, 3].get()*(-1)*(jnp.sin(gridk_mod)/gridk_mod)*(3 * (
        # jnp.sin(gridk_mod) - gridk_mod*jnp.cos(gridk_mod)) / (gridk_mod*gridk_sqr)), 0)  # scaling factor
    # gridXX = B * swap_real_imag((Fkxx - kkxx * kdF))
    # gridXY = B * swap_real_imag((Fkxy - kkxy * kdF))
    # gridXZ = B * swap_real_imag((Fkxz - kkxz * kdF))
    # gridYX = B * swap_real_imag((Fkyx - kkyx * kdF))
    # gridYY = B * swap_real_imag((Fkyy - kkyy * kdF))
    # gridYZ = B * swap_real_imag((Fkyz - kkyz * kdF))
    # gridZX = B * swap_real_imag((Fkzx - kkzx * kdF))
    # gridZY = B * swap_real_imag((Fkzy - kkzy * kdF))

    # DC part
    # B = jnp.where(gridk_mod > 0, gridk.at[:, :, :, 3].get()*(9)*((jnp.sin(gridk_mod) - gridk_mod*jnp.cos(gridk_mod)) / (
    #     gridk_mod*gridk_sqr)) * ((jnp.sin(gridk_mod) - gridk_mod*jnp.cos(gridk_mod)) / (gridk_mod*gridk_sqr)), 0)  # scaling factor
    # gridXX += B * (Cdkkxx - kkxx * kdcdk)
    # gridXY += B * (Cdkkxy - kkxy * kdcdk)
    # gridXZ += B * (Cdkkxz - kkxz * kdcdk)
    # gridYX += B * (Cdkkyx - kkyx * kdcdk)
    # gridYY += B * (Cdkkyy - kkyy * kdcdk)
    # gridYZ += B * (Cdkkyz - kkyz * kdcdk)
    # gridZX += B * (Cdkkzx - kkzx * kdcdk)
    # gridZY += B * (Cdkkzy - kkzy * kdcdk)
    
    # Inverse FFT
    gridX = jnp.real(jnp.fft.ifftn(gridX,norm='forward'))
    gridY = jnp.real(jnp.fft.ifftn(gridY,norm='forward'))
    gridZ = jnp.real(jnp.fft.ifftn(gridZ,norm='forward'))
    # gridXX = jnp.real(jnp.fft.ifftn(gridXX,norm='forward'))
    # gridXY = jnp.real(jnp.fft.ifftn(gridXY,norm='forward'))
    # gridXZ = jnp.real(jnp.fft.ifftn(gridXZ,norm='forward'))
    # gridYX = jnp.real(jnp.fft.ifftn(gridYX,norm='forward'))
    # gridYY = jnp.real(jnp.fft.ifftn(gridYY,norm='forward'))
    # gridYZ = jnp.real(jnp.fft.ifftn(gridYZ,norm='forward'))
    # gridZX = jnp.real(jnp.fft.ifftn(gridZX,norm='forward'))
    # gridZY = jnp.real(jnp.fft.ifftn(gridZY,norm='forward'))
    
    # Compute Linear velocities and Velocity gradients (from which we can extract angular velocities and rate of strain)
    w_lin_velocities = jnp.zeros((N, 3),float)
    w_velocity_gradient = jnp.zeros((N, 8),float)
       
     
    w_lin_velocities = w_lin_velocities.at[:, 0].add(
        jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
            gridX.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3)))
    w_lin_velocities = w_lin_velocities.at[:, 1].add(
        jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
            gridY.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3))) 
    w_lin_velocities = w_lin_velocities.at[:, 2].add(
        jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
            gridZ.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3)))              


    # w_velocity_gradient = w_velocity_gradient.at[:, 0].add(
    #     jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
    #         gridXX.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3)))
    
    # w_velocity_gradient = w_velocity_gradient.at[:, 5].add(
    #     jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
    #         gridXY.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3))) 

    # w_velocity_gradient = w_velocity_gradient.at[:, 6].add(
    #       jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
    #         gridXZ.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3)))  

    # w_velocity_gradient = w_velocity_gradient.at[:, 7].add(
    #     jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
    #         gridYZ.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3)))
    
    
    # w_velocity_gradient = w_velocity_gradient.at[:, 4].add(
    #     jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
    #         gridYY.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3))) 
    
    # w_velocity_gradient = w_velocity_gradient.at[:, 1].add(
    #     jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
    #         gridYX.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3)))     

    # w_velocity_gradient = w_velocity_gradient.at[:, 2].add(
    #       jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
    #         gridZX.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3))) 

    # w_velocity_gradient = w_velocity_gradient.at[:, 3].add(
    #     jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
    #         gridZY.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3)))              
    
    
    ##########################################################################################################################################
    ######################################## REAL SPACE CONTRIBUTION #########################################################################
    ##########################################################################################################################################
    
    
    # Allocate arrays for Linear velocities and Velocity gradients (from which we can extract angular velocities and rate of strain)
    r_lin_velocities = jnp.zeros((N, 3),float)
    r_velocity_gradient = jnp.zeros((N, 8),float)
    
    # SELF CONTRIBUTIONS
    r_lin_velocities = r_lin_velocities.at[:, 0].set(
        m_self.at[0].get() * forces.at[0::3].get())
    r_lin_velocities = r_lin_velocities.at[:, 1].set(
        m_self.at[0].get() * forces.at[1::3].get())
    r_lin_velocities = r_lin_velocities.at[:, 2].set(
        m_self.at[0].get() * forces.at[2::3].get())
    
    # r_velocity_gradient = r_velocity_gradient.at[:, 0].set(
    #     m_self.at[1].get()*(couplets.at[0::8].get() - 4 * couplets.at[0::8].get()))
    # r_velocity_gradient = r_velocity_gradient.at[:, 5].set(
    #     m_self.at[1].get()*(couplets.at[1::8].get() - 4 * couplets.at[5::8].get()))
    # r_velocity_gradient = r_velocity_gradient.at[:, 6].set(
    #     m_self.at[1].get()*(couplets.at[2::8].get() - 4 * couplets.at[6::8].get()))
    # r_velocity_gradient = r_velocity_gradient.at[:, 7].set(
    #     m_self.at[1].get()*(couplets.at[3::8].get() - 4 * couplets.at[7::8].get()))
    
    # r_velocity_gradient = r_velocity_gradient.at[:, 4].set(
    #     m_self.at[1].get()*(couplets.at[4::8].get() - 4 * couplets.at[4::8].get()))
    # r_velocity_gradient = r_velocity_gradient.at[:, 1].set(
    #     m_self.at[1].get()*(couplets.at[5::8].get() - 4 * couplets.at[1::8].get()))
    # r_velocity_gradient = r_velocity_gradient.at[:, 2].set(
    #     m_self.at[1].get()*(couplets.at[6::8].get() - 4 * couplets.at[2::8].get()))
    # r_velocity_gradient = r_velocity_gradient.at[:, 3].set(
    #     m_self.at[1].get()*(couplets.at[7::8].get() - 4 * couplets.at[3::8].get()))
    
    
    
    # # Pair contributions

    # # Geometric quantities
    rdotf_j =   (r.at[:,0].get() * forces.at[3*indices_j + 0].get() + r.at[:,1].get() * forces.at[3*indices_j + 1].get() + r.at[:,2].get() * forces.at[3*indices_j + 2].get())
    mrdotf_i = -(r.at[:,0].get() * forces.at[3*indices_i + 0].get() + r.at[:,1].get() * forces.at[3*indices_i + 1].get() + r.at[:,2].get() * forces.at[3*indices_i + 2].get())
    
    # Cj_dotr = jnp.array( [couplets.at[8*indices_j + 0].get() * r.at[:,0].get() + couplets.at[8*indices_j + 1].get() * r.at[:,1].get() + couplets.at[8*indices_j + 2].get() * r.at[:,2].get(),
    #                       couplets.at[8*indices_j + 5].get() * r.at[:,0].get() + couplets.at[8*indices_j + 4].get() * r.at[:,1].get() + couplets.at[8*indices_j + 3].get() * r.at[:,2].get(),
    #                       couplets.at[8*indices_j + 6].get() * r.at[:,0].get() + couplets.at[8*indices_j + 7].get() * r.at[:,1].get() -(couplets.at[8*indices_j + 0].get() + couplets.at[8*indices_j + 4].get()) * r.at[:,2].get()])
    
    # Ci_dotmr=jnp.array( [-couplets.at[8*indices_i + 0].get() * r.at[:,0].get() - couplets.at[8*indices_i + 1].get() * r.at[:,1].get() - couplets.at[8*indices_i + 2].get() * r.at[:,2].get(),
    #                       -couplets.at[8*indices_i + 5].get() * r.at[:,0].get() - couplets.at[8*indices_i + 4].get() * r.at[:,1].get() - couplets.at[8*indices_i + 3].get() * r.at[:,2].get(),
    #                       -couplets.at[8*indices_i + 6].get() * r.at[:,0].get() - couplets.at[8*indices_i + 7].get() * r.at[:,1].get() +(couplets.at[8*indices_i + 0].get() + couplets.at[8*indices_i + 4].get()) * r.at[:,2].get()])


    # rdotC_j = jnp.array([couplets.at[8*indices_j + 0].get() * r.at[:,0].get() + couplets.at[8*indices_j + 5].get() * r.at[:,1].get() + couplets.at[8*indices_j + 6].get() * r.at[:,2].get(),
    #                       couplets.at[8*indices_j + 1].get() * r.at[:,0].get() + couplets.at[8*indices_j + 4].get() * r.at[:,1].get() + couplets.at[8*indices_j + 7].get() * r.at[:,2].get(),
    #                       couplets.at[8*indices_j + 2].get() * r.at[:,0].get() + couplets.at[8*indices_j + 3].get() * r.at[:,1].get() -(couplets.at[8*indices_j + 0].get() + couplets.at[8*indices_j + 4].get()) * r.at[:,2].get()])
    
    # mrdotC_i=jnp.array([-couplets.at[8*indices_i + 0].get() * r.at[:,0].get() - couplets.at[8*indices_i + 5].get() * r.at[:,1].get() - couplets.at[8*indices_i + 6].get() * r.at[:,2].get(),
    #                       -couplets.at[8*indices_i + 1].get() * r.at[:,0].get() - couplets.at[8*indices_i + 4].get() * r.at[:,1].get() - couplets.at[8*indices_i + 7].get() * r.at[:,2].get(),
    #                       -couplets.at[8*indices_i + 2].get() * r.at[:,0].get() - couplets.at[8*indices_i + 3].get() * r.at[:,1].get() +(couplets.at[8*indices_i + 0].get() + couplets.at[8*indices_i + 4].get()) * r.at[:,2].get()])
                
    # rdotC_jj_dotr   =  (r.at[:,0].get()*Cj_dotr.at[0,:].get()  + r.at[:,1].get()*Cj_dotr.at[1,:].get()  + r.at[:,2].get()*Cj_dotr.at[2,:].get())
    # mrdotC_ii_dotmr = -(r.at[:,0].get()*Ci_dotmr.at[0,:].get() + r.at[:,1].get()*Ci_dotmr.at[1,:].get() + r.at[:,2].get()*Ci_dotmr.at[2,:].get())
    
    
    # Compute Velocity for particles i
    r_lin_velocities = r_lin_velocities.at[indices_i, 0].add(f1 * forces.at[3*indices_j].get() + (f2 - f1) * rdotf_j * r.at[:,0].get())
    r_lin_velocities = r_lin_velocities.at[indices_i, 1].add(f1 * forces.at[3*indices_j+1].get() + (f2 - f1) * rdotf_j * r.at[:,1].get())
    r_lin_velocities = r_lin_velocities.at[indices_i, 2].add(f1 * forces.at[3*indices_j+2].get() + (f2 - f1) * rdotf_j * r.at[:,2].get())
    # r_lin_velocities = r_lin_velocities.at[indices_i, 0].add(g1 * (Cj_dotr.at[0,:].get() - rdotC_jj_dotr * r.at[:,0].get()) + g2 * (rdotC_j.at[0,:].get() - 4.*rdotC_jj_dotr * r.at[:,0].get()))
    # r_lin_velocities = r_lin_velocities.at[indices_i, 1].add(g1 * (Cj_dotr.at[1,:].get() - rdotC_jj_dotr * r.at[:,1].get()) + g2 * (rdotC_j.at[1,:].get() - 4.*rdotC_jj_dotr * r.at[:,1].get()))
    # r_lin_velocities = r_lin_velocities.at[indices_i, 2].add(g1 * (Cj_dotr.at[2,:].get() - rdotC_jj_dotr * r.at[:,2].get()) + g2 * (rdotC_j.at[2,:].get() - 4.*rdotC_jj_dotr * r.at[:,2].get()))
    # Compute Velocity for particles j
    r_lin_velocities = r_lin_velocities.at[indices_j, 0].add(f1 * forces.at[3*indices_i].get() - (f2 - f1) * mrdotf_i * r.at[:,0].get())
    r_lin_velocities = r_lin_velocities.at[indices_j, 1].add(f1 * forces.at[3*indices_i+1].get() - (f2 - f1) * mrdotf_i * r.at[:,1].get())
    r_lin_velocities = r_lin_velocities.at[indices_j, 2].add(f1 * forces.at[3*indices_i+2].get() - (f2 - f1) * mrdotf_i * r.at[:,2].get())
    # r_lin_velocities = r_lin_velocities.at[indices_j, 0].add(g1 * (Ci_dotmr.at[0,:].get() + mrdotC_ii_dotmr * r.at[:,0].get()) + g2 * (mrdotC_i.at[0,:].get() + 4.*mrdotC_ii_dotmr * r.at[:,0].get()))
    # r_lin_velocities = r_lin_velocities.at[indices_j, 1].add(g1 * (Ci_dotmr.at[1,:].get() + mrdotC_ii_dotmr * r.at[:,1].get()) + g2 * (mrdotC_i.at[1,:].get() + 4.*mrdotC_ii_dotmr * r.at[:,1].get()))
    # r_lin_velocities = r_lin_velocities.at[indices_j, 2].add(g1 * (Ci_dotmr.at[2,:].get() + mrdotC_ii_dotmr * r.at[:,2].get()) + g2 * (mrdotC_i.at[2,:].get() + 4.*mrdotC_ii_dotmr * r.at[:,2].get()))
    
    
    # Compute Velocity Gradient for particles i and j
    r_velocity_gradient = r_velocity_gradient.at[indices_i, 0].add(
        (-1) * g1 * (r.at[:,0].get() * forces.at[3*indices_j+0].get() - rdotf_j * r.at[:,0].get()*r.at[:,0].get())
        +(-1)*g2 * (rdotf_j + forces.at[3*indices_j+0].get()*r.at[:,0].get() - 4.*rdotf_j * r.at[:,0].get()*r.at[:,0].get()))
    
    r_velocity_gradient = r_velocity_gradient.at[indices_j, 0].add(
        (-1) * g1 * (-r.at[:,0].get() * forces.at[3*indices_i+0].get() - mrdotf_i * r.at[:,0].get()*r.at[:,0].get()) 
        +(-1)*g2 * (mrdotf_i - forces.at[3*indices_i+0].get()*r.at[:,0].get() - 4.*mrdotf_i * r.at[:,0].get()*r.at[:,0].get()))
    
    # r_velocity_gradient = r_velocity_gradient.at[indices_i, 0].add(
    #     h1 * (couplets.at[8*indices_j+0].get() - 4. * couplets.at[8*indices_j+0].get()) 
    #     + h2 * (r.at[:,0].get()*Cj_dotr.at[0,:].get() - rdotC_jj_dotr * r.at[:,0].get()*r.at[:,0].get()) 
    #     + h3 * (rdotC_jj_dotr + Cj_dotr.at[0,:].get()*r.at[:,0].get() + r.at[:,0].get()*rdotC_j.at[0,:].get() + rdotC_j.at[0,:].get()*r.at[:,0].get() 
                                                                # - 6.*rdotC_jj_dotr*r.at[:,0].get()*r.at[:,0].get() - couplets.at[8*indices_j+0].get()))
    # r_velocity_gradient = r_velocity_gradient.at[indices_j, 0].add(
    #     h1 * (couplets.at[8*indices_i+0].get() - 4. * couplets.at[8*indices_i+0].get()) 
    #     + h2 * (-r.at[:,0].get()*Ci_dotmr.at[0,:].get() - mrdotC_ii_dotmr * r.at[:,0].get()*r.at[:,0].get())
    #     + h3 * (mrdotC_ii_dotmr - Ci_dotmr.at[0,:].get()*r.at[:,0].get() - r.at[:,0].get()*mrdotC_i.at[0,:].get() - mrdotC_i.at[0,:].get()*r.at[:,0].get() 
    #                                                             - 6.*mrdotC_ii_dotmr*r.at[:,0].get()*r.at[:,0].get() - couplets.at[8*indices_i+0].get()))


    r_velocity_gradient = r_velocity_gradient.at[indices_i, 1].add(
        (-1) * g1 * (r.at[:,1].get() * forces.at[3*indices_j+0].get() - rdotf_j * r.at[:,1].get()*r.at[:,0].get()) 
        + (-1)*g2 * (forces.at[3*indices_j+1].get()*r.at[:,0].get() - 4.*rdotf_j * r.at[:,1].get()*r.at[:,0].get()))
    
    r_velocity_gradient = r_velocity_gradient.at[indices_j, 1].add(
        (-1) * g1 * (-r.at[:,1].get() * forces.at[3*indices_i+0].get() - mrdotf_i * r.at[:,1].get()*r.at[:,0].get()) 
        + (-1)*g2 * (-forces.at[3*indices_i+1].get()*r.at[:,0].get() - 4.*mrdotf_i * r.at[:,1].get()*r.at[:,0].get()))
    
    # r_velocity_gradient = r_velocity_gradient.at[indices_i, 1].add(
    #     h1 * (couplets.at[8*indices_j+5].get() - 4. * couplets.at[8*indices_j+1].get()) 
    #     + h2 * (r.at[:,1].get()*Cj_dotr.at[0,:].get() - rdotC_jj_dotr * r.at[:,1].get()*r.at[:,0].get())
    #     + h3 * (Cj_dotr.at[1,:].get()*r.at[:,0].get() + r.at[:,1].get()*rdotC_j.at[0,:].get() + rdotC_j.at[1,:].get()*r.at[:,0].get()
    #                                         - 6.*rdotC_jj_dotr*r.at[:,1].get()*r.at[:,0].get() - couplets.at[8*indices_j+1].get()))
    
    # r_velocity_gradient = r_velocity_gradient.at[indices_j, 1].add(
    #     h1 * (couplets.at[8*indices_i+5].get() - 4. * couplets.at[8*indices_i+1].get())
    #     + h2 * (-r.at[:,1].get()*Ci_dotmr.at[0,:].get() - mrdotC_ii_dotmr * r.at[:,1].get()*r.at[:,0].get())
    #     + h3 * (-Ci_dotmr.at[1,:].get()*r.at[:,0].get() - r.at[:,1].get()*mrdotC_i.at[0,:].get() - mrdotC_i.at[1,:].get()*r.at[:,0].get()
    #                                             - 6.*mrdotC_ii_dotmr*r.at[:,1].get()*r.at[:,0].get() - couplets.at[8*indices_i+1].get()))



    r_velocity_gradient = r_velocity_gradient.at[indices_i, 2].add(
              (-1) * g1 * (r.at[:,2].get() * forces.at[3*indices_j+0].get() - rdotf_j * r.at[:,2].get()*r.at[:,0].get()) 
              + (-1)*g2 * (forces.at[3*indices_j+2].get()*r.at[:,0].get() - 4.*rdotf_j * r.at[:,2].get()*r.at[:,0].get()))
    
    r_velocity_gradient = r_velocity_gradient.at[indices_j, 2].add(
              (-1) * g1 * (-r.at[:,2].get() * forces.at[3*indices_i+0].get() - mrdotf_i * r.at[:,2].get()*r.at[:,0].get()) 
              + (-1)*g2 * (-forces.at[3*indices_i+2].get()*r.at[:,0].get() - 4.*mrdotf_i * r.at[:,2].get()*r.at[:,0].get()))
    # r_velocity_gradient = r_velocity_gradient.at[indices_i, 2].add(
    #     h1 * (couplets.at[8*indices_j+6].get() - 4. * couplets.at[8*indices_j+2].get())
    #     + h2 * (r.at[:,2].get()*Cj_dotr.at[0,:].get() - rdotC_jj_dotr * r.at[:,2].get()*r.at[:,0].get())
    #     + h3 * (Cj_dotr.at[2,:].get()*r.at[:,0].get() + r.at[:,2].get()*rdotC_j.at[0,:].get() + rdotC_j.at[2,:].get()*r.at[:,0].get()
    #             - 6.*rdotC_jj_dotr*r.at[:,2].get()*r.at[:,0].get() - couplets.at[8*indices_j+2].get()))
    # r_velocity_gradient = r_velocity_gradient.at[indices_j, 2].add(
    #     h1 * (couplets.at[8*indices_i+6].get() - 4. * couplets.at[8*indices_i+2].get())
    #     + h2 * (r.at[:,2].get()*Ci_dotmr.at[0,:].get()*(-1) - mrdotC_ii_dotmr * r.at[:,2].get()*r.at[:,0].get())
    #     + h3 * (-Ci_dotmr.at[2,:].get()*r.at[:,0].get() - r.at[:,2].get()*mrdotC_i.at[0,:].get() - mrdotC_i.at[2,:].get()*r.at[:,0].get()
    #             - 6.*mrdotC_ii_dotmr*r.at[:,2].get()*r.at[:,0].get() - couplets.at[8*indices_i+2].get()))
   



    r_velocity_gradient = r_velocity_gradient.at[indices_i, 3].add(
            (-1) * g1 * (r.at[:,2].get() * forces.at[3*indices_j+1].get() - rdotf_j * r.at[:,2].get()*r.at[:,1].get())
            + (-1)*g2 * (forces.at[3*indices_j+2].get()*r.at[:,1].get() - 4.*rdotf_j * r.at[:,2].get()*r.at[:,1].get()))
    
    r_velocity_gradient = r_velocity_gradient.at[indices_j, 3].add(
            (-1) * g1 * (-r.at[:,2].get() * forces.at[3*indices_i+1].get() - mrdotf_i * r.at[:,2].get()*r.at[:,1].get())
            + (-1)*g2 * (-forces.at[3*indices_i+2].get()*r.at[:,1].get() - 4.*mrdotf_i * r.at[:,2].get()*r.at[:,1].get()))
    
    # r_velocity_gradient = r_velocity_gradient.at[indices_i, 3].add(
        # h1 * (couplets.at[8*indices_j+7].get() - 4. * couplets.at[8*indices_j+3].get())
        # + h2 * (r.at[:,2].get()*Cj_dotr.at[1,:].get() - rdotC_jj_dotr * r.at[:,2].get()*r.at[:,1].get())
        # + h3 * (Cj_dotr.at[2,:].get()*r.at[:,1].get() + r.at[:,2].get()*rdotC_j.at[1,:].get() + rdotC_j.at[2,:].get()*r.at[:,1].get()
        #         - 6.*rdotC_jj_dotr*r.at[:,2].get()*r.at[:,1].get() - couplets.at[8*indices_j+3].get()))

    # r_velocity_gradient = r_velocity_gradient.at[indices_j, 3].add(
    #     h1 * (couplets.at[8*indices_i+7].get() - 4. * couplets.at[8*indices_i+3].get())
    #     + h2 * (-r.at[:,2].get()*Ci_dotmr.at[1,:].get() - mrdotC_ii_dotmr * r.at[:,2].get()*r.at[:,1].get()) 
    #     + h3 * (-Ci_dotmr.at[2,:].get()*r.at[:,1].get() - r.at[:,2].get()*mrdotC_i.at[1,:].get() - mrdotC_i.at[2,:].get()*r.at[:,1].get()
    #             - 6.*mrdotC_ii_dotmr*r.at[:,2].get()*r.at[:,1].get() - couplets.at[8*indices_i+3].get()))



    r_velocity_gradient = r_velocity_gradient.at[indices_i, 4].add(
            (-1) * g1 * (r.at[:,1].get() * forces.at[3*indices_j+1].get() - rdotf_j * r.at[:,1].get()*r.at[:,1].get())
            + (-1)*g2 * (rdotf_j + forces.at[3*indices_j+1].get()*r.at[:,1].get() - 4.*rdotf_j * r.at[:,1].get()*r.at[:,1].get()))
    
    r_velocity_gradient = r_velocity_gradient.at[indices_j, 4].add(
            (-1) * g1 * (-r.at[:,1].get() * forces.at[3*indices_i+1].get() - mrdotf_i * r.at[:,1].get()*r.at[:,1].get())
            + (-1)*g2 * (mrdotf_i - forces.at[3*indices_i+1].get()*r.at[:,1].get() - 4.*mrdotf_i * r.at[:,1].get()*r.at[:,1].get()))
     
    # r_velocity_gradient = r_velocity_gradient.at[indices_i, 4].add(
    #     h1 * (couplets.at[8*indices_j+4].get() - 4. * couplets.at[8*indices_j+4].get())
    #     + h2 * (r.at[:,1].get()*Cj_dotr.at[1,:].get() - rdotC_jj_dotr * r.at[:,1].get()*r.at[:,1].get()) 
    #     + h3 * (rdotC_jj_dotr + Cj_dotr.at[1,:].get()*r.at[:,1].get() + r.at[:,1].get()*rdotC_j.at[1,:].get() + rdotC_j.at[1,:].get()*r.at[:,1].get()
    #             - 6.*rdotC_jj_dotr*r.at[:,1].get()*r.at[:,1].get() - couplets.at[8*indices_j+4].get()))
    
    # r_velocity_gradient = r_velocity_gradient.at[indices_j, 4].add(
    #     h1 * (couplets.at[8*indices_i+4].get() - 4. * couplets.at[8*indices_i+4].get())
    #     + h2 * (-r.at[:,1].get()*Ci_dotmr.at[1,:].get() - mrdotC_ii_dotmr * r.at[:,1].get()*r.at[:,1].get())
    #     + h3 * (mrdotC_ii_dotmr - Ci_dotmr.at[1,:].get()*r.at[:,1].get() - r.at[:,1].get()*mrdotC_i.at[1,:].get() - mrdotC_i.at[1,:].get()*r.at[:,1].get()
    #             - 6.*mrdotC_ii_dotmr*r.at[:,1].get()*r.at[:,1].get() - couplets.at[8*indices_i+4].get()))


    r_velocity_gradient = r_velocity_gradient.at[indices_i, 5].add(
            (-1) * g1 * (r.at[:,0].get() * forces.at[3*indices_j+1].get() - rdotf_j * r.at[:,0].get()*r.at[:,1].get())
            + (-1)*g2 * (forces.at[3*indices_j+0].get()*r.at[:,1].get() - 4.*rdotf_j * r.at[:,0].get()*r.at[:,1].get()))
    
    r_velocity_gradient = r_velocity_gradient.at[indices_j, 5].add(
            (-1) * g1 * (-r.at[:,0].get() * forces.at[3*indices_i+1].get() - mrdotf_i * r.at[:,0].get()*r.at[:,1].get())
            + (-1)*g2 * (-forces.at[3*indices_i+0].get()*r.at[:,1].get() - 4.*mrdotf_i * r.at[:,0].get()*r.at[:,1].get()))
    
    # r_velocity_gradient = r_velocity_gradient.at[indices_i, 5].add(
    #     h1 * (couplets.at[8*indices_j+1].get() - 4. * couplets.at[8*indices_j+5].get())
    #     + h2 * (r.at[:,0].get()*Cj_dotr.at[1,:].get() - rdotC_jj_dotr * r.at[:,0].get()*r.at[:,1].get())
    #     + h3 * (Cj_dotr.at[0,:].get()*r.at[:,1].get() + r.at[:,0].get()*rdotC_j.at[1,:].get() + rdotC_j.at[0,:].get()*r.at[:,1].get()
    #             - 6.*rdotC_jj_dotr*r.at[:,0].get()*r.at[:,1].get() - couplets.at[8*indices_j+5].get()))
    
    # r_velocity_gradient = r_velocity_gradient.at[indices_j, 5].add(
    #     h1 * (couplets.at[8*indices_i+1].get() - 4. * couplets.at[8*indices_i+5].get())
    #     + h2 * (-r.at[:,0].get()*Ci_dotmr.at[1,:].get() - mrdotC_ii_dotmr * r.at[:,0].get()*r.at[:,1].get())
    #     + h3 * (-Ci_dotmr.at[0,:].get()*r.at[:,1].get() - r.at[:,0].get()*mrdotC_i.at[1,:].get() - mrdotC_i.at[0,:].get()*r.at[:,1].get()
    #             - 6.*mrdotC_ii_dotmr*r.at[:,0].get()*r.at[:,1].get() - couplets.at[8*indices_i+5].get()))



    r_velocity_gradient = r_velocity_gradient.at[indices_i, 6].add(
              (-1) * g1 * (r.at[:,0].get() * forces.at[3*indices_j+2].get() - rdotf_j * r.at[:,0].get()*r.at[:,2].get())
              + (-1)*g2 * (forces.at[3*indices_j+0].get()*r.at[:,2].get() - 4.*rdotf_j * r.at[:,0].get()*r.at[:,2].get()))
    
    r_velocity_gradient = r_velocity_gradient.at[indices_j, 6].add(
              (-1) * g1 * (-r.at[:,0].get() * forces.at[3*indices_i+2].get() - mrdotf_i * r.at[:,0].get()*r.at[:,2].get())
              + (-1)*g2 * (-forces.at[3*indices_i+0].get()*r.at[:,2].get() - 4.*mrdotf_i * r.at[:,0].get()*r.at[:,2].get()))
     
    # r_velocity_gradient = r_velocity_gradient.at[indices_i, 6].add(
    #     h1 * (couplets.at[8*indices_j+2].get() - 4. * couplets.at[8*indices_j+6].get())
    #     + h2 * (r.at[:,0].get()*Cj_dotr.at[2,:].get() - rdotC_jj_dotr * r.at[:,0].get()*r.at[:,2].get())
    #     + h3 * (Cj_dotr.at[0,:].get()*r.at[:,2].get() + r.at[:,0].get()*rdotC_j.at[2,:].get() + rdotC_j.at[0,:].get()*r.at[:,2].get()
    #             - 6.*rdotC_jj_dotr*r.at[:,0].get()*r.at[:,2].get() - couplets.at[8*indices_j+6].get()))
    
    # r_velocity_gradient = r_velocity_gradient.at[indices_j, 6].add(
    #     h1 * (couplets.at[8*indices_i+2].get() - 4. * couplets.at[8*indices_i+6].get())
    #     + h2 * (-r.at[:,0].get()*Ci_dotmr.at[2,:].get() - mrdotC_ii_dotmr * r.at[:,0].get()*r.at[:,2].get())
    #     + h3 * (-Ci_dotmr.at[0,:].get()*r.at[:,2].get() - r.at[:,0].get()*mrdotC_i.at[2,:].get() - mrdotC_i.at[0,:].get()*r.at[:,2].get()
    #             - 6.*mrdotC_ii_dotmr*r.at[:,0].get()*r.at[:,2].get() - couplets.at[8*indices_i+6].get()))
    
    
    
    r_velocity_gradient = r_velocity_gradient.at[indices_i, 7].add(
            (-1) * g1 * (r.at[:,1].get() * forces.at[3*indices_j+2].get() - rdotf_j * r.at[:,1].get()*r.at[:,2].get())
            + (-1)*g2 * (forces.at[3*indices_j+1].get()*r.at[:,2].get() - 4.*rdotf_j * r.at[:,1].get()*r.at[:,2].get()))
    
    r_velocity_gradient = r_velocity_gradient.at[indices_j, 7].add(
            (-1) * g1 * (-r.at[:,1].get() * forces.at[3*indices_i+2].get() - mrdotf_i * r.at[:,1].get()*r.at[:,2].get())
            + (-1)*g2 * (-forces.at[3*indices_i+1].get()*r.at[:,2].get() - 4.*mrdotf_i * r.at[:,1].get()*r.at[:,2].get()))
    
    # r_velocity_gradient = r_velocity_gradient.at[indices_i, 7].add(
    #     h1 * (couplets.at[8*indices_j+3].get() - 4. * couplets.at[8*indices_j+7].get())
    #     + h2 * (r.at[:,1].get()*Cj_dotr.at[2,:].get() - rdotC_jj_dotr * r.at[:,1].get()*r.at[:,2].get())
    #     + h3 * (Cj_dotr.at[1,:].get()*r.at[:,2].get() + r.at[:,1].get()*rdotC_j.at[2,:].get() + rdotC_j.at[1,:].get()*r.at[:,2].get()
    #             - 6.*rdotC_jj_dotr*r.at[:,1].get()*r.at[:,2].get() - couplets.at[8*indices_j+7].get()))
    
    # r_velocity_gradient = r_velocity_gradient.at[indices_j, 7].add(
    #     h1 * (couplets.at[8*indices_i+3].get() - 4. * couplets.at[8*indices_i+7].get())
    #     + h2 * (-r.at[:,1].get()*Ci_dotmr.at[2,:].get() - mrdotC_ii_dotmr * r.at[:,1].get()*r.at[:,2].get())
    #     + h3 * (-Ci_dotmr.at[1,:].get()*r.at[:,2].get() - r.at[:,1].get()*mrdotC_i.at[2,:].get() - mrdotC_i.at[1,:].get()*r.at[:,2].get()
    #             - 6.*mrdotC_ii_dotmr*r.at[:,1].get()*r.at[:,2].get() - couplets.at[8*indices_i+7].get()))
    


    ##########################################################################################################################################
    ##########################################################################################################################################

    # Add wave and real space part together
    lin_vel = w_lin_velocities + r_lin_velocities
    velocity_gradient = w_velocity_gradient + r_velocity_gradient

    # # Convert to angular velocities and rate of strain
    ang_vel_and_strain = jnp.zeros((N, 3))
    ang_vel_and_strain = ang_vel_and_strain.at[:, 0].set((velocity_gradient.at[:, 3].get()-velocity_gradient.at[:, 7].get()) * 0.5)
    ang_vel_and_strain = ang_vel_and_strain.at[:, 1].set((velocity_gradient.at[:, 6].get()-velocity_gradient.at[:, 2].get()) * 0.5)
    ang_vel_and_strain = ang_vel_and_strain.at[:, 2].set((velocity_gradient.at[:, 1].get()-velocity_gradient.at[:, 5].get()) * 0.5)
    # ang_vel_and_strain = ang_vel_and_strain.at[:, 3].set(2*velocity_gradient.at[:, 0].get()+velocity_gradient.at[:, 4].get())
    # ang_vel_and_strain = ang_vel_and_strain.at[:, 4].set(velocity_gradient.at[:, 1].get()+velocity_gradient.at[:, 5].get())
    # ang_vel_and_strain = ang_vel_and_strain.at[:, 5].set(velocity_gradient.at[:, 2].get()+velocity_gradient.at[:, 6].get())
    # ang_vel_and_strain = ang_vel_and_strain.at[:, 6].set(velocity_gradient.at[:, 3].get()+velocity_gradient.at[:, 7].get())
    # ang_vel_and_strain = ang_vel_and_strain.at[:, 7].set(velocity_gradient.at[:, 0].get()+2*velocity_gradient.at[:, 4].get())

    # Convert to Generalized Velocities 
    generalized_velocities = jnp.zeros(6*N) #First 6N entries for U and last 5N for strain rates

    generalized_velocities = generalized_velocities.at[0:6*N:6].set(
        lin_vel.at[:, 0].get())
    generalized_velocities = generalized_velocities.at[1:6*N:6].set(
        lin_vel.at[:, 1].get())
    generalized_velocities = generalized_velocities.at[2:6*N:6].set(
        lin_vel.at[:, 2].get())
    generalized_velocities = generalized_velocities.at[3:6*N:6].set(
        ang_vel_and_strain.at[:, 0].get())
    generalized_velocities = generalized_velocities.at[4:6*N:6].set(
        ang_vel_and_strain.at[:, 1].get())
    generalized_velocities = generalized_velocities.at[5:6*N:6].set(
        ang_vel_and_strain.at[:, 2].get())
    # generalized_velocities = generalized_velocities.at[(6*N+0)::5].set(
    #     ang_vel_and_strain.at[:, 3].get())
    # generalized_velocities = generalized_velocities.at[(6*N+1)::5].set(
    #     ang_vel_and_strain.at[:, 4].get())
    # generalized_velocities = generalized_velocities.at[(6*N+2)::5].set(
    #     ang_vel_and_strain.at[:, 5].get())
    # generalized_velocities = generalized_velocities.at[(6*N+3)::5].set(
    #     ang_vel_and_strain.at[:, 6].get())
    # generalized_velocities = generalized_velocities.at[(6*N+4)::5].set(
    #     ang_vel_and_strain.at[:, 7].get())
    
    #Clean Grids for next iteration
    gridX = jnp.zeros((Nx, Ny, Nz))
    gridY = jnp.zeros((Nx, Ny, Nz))
    gridZ = jnp.zeros((Nx, Ny, Nz))
    # gridXX = jnp.zeros((Nx, Ny, Nz))
    # gridXY = jnp.zeros((Nx, Ny, Nz))
    # gridXZ = jnp.zeros((Nx, Ny, Nz))
    # gridYX = jnp.zeros((Nx, Ny, Nz))
    # gridYY = jnp.zeros((Nx, Ny, Nz))
    # gridYZ = jnp.zeros((Nx, Ny, Nz))
    # gridZX = jnp.zeros((Nx, Ny, Nz))
    # gridZY = jnp.zeros((Nx, Ny, Nz))
    
    return generalized_velocities
