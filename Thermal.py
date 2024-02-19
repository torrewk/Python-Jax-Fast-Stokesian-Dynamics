import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import numpy as np
from jax import jit
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)
import scipy
from functools import partial
import Lanczos_lax


def Random_force_on_grid_indexing(Nx,Ny,Nz):
    normal_indices = []
    normal_conj_indices = []
    nyquist_indices = []
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                
                if (not(2*k >= Nz+1) and 
                    not((k == 0) and (2*j >= Ny+1)) and 
                    not((k == 0) and (j == 0) and (2*i >= Nx+1)) and 
                    not((k == 0) and (j == 0) and (i == 0))):
                    
                    ii_nyquist = ( ( i == int(Nx/2) ) and ( int(Nx/2) == int((Nx+1)/2) ) )
                    jj_nyquist = ( ( j == int(Ny/2) ) and ( int(Ny/2) == int((Ny+1)/2) ) )
                    kk_nyquist = ( ( k == int(Nz/2) ) and ( int(Nz/2) == int((Nz+1)/2) ) )
                    if (   (i==0 and jj_nyquist and k==0) 
                        or (ii_nyquist and j==0 and k==0) 
                        or (ii_nyquist and jj_nyquist and k==0)
                        or (i==0 and j==0 and kk_nyquist)
                        or (i ==0 and jj_nyquist and kk_nyquist)
                        or (ii_nyquist and j==0 and kk_nyquist)
                        or (ii_nyquist and jj_nyquist and kk_nyquist)):
                        nyquist_indices.append([i,j,k])
                    else:
                        normal_indices.append([i,j,k])
                        if(ii_nyquist or (i==0)):
                            i_conj = i    
                        else:
                            i_conj = Nx - i
                        if(jj_nyquist or (j==0)):
                            j_conj = j    
                        else:
                            j_conj = Ny - j
                        if(kk_nyquist or (k==0)):
                            k_conj = k    
                        else:
                            k_conj = Nz - k
                        normal_conj_indices.append([i_conj,j_conj,k_conj])
    
    normal_indices = jnp.array(normal_indices)
    normal_conj_indices = jnp.array(normal_conj_indices)
    nyquist_indices = jnp.array(nyquist_indices)
    
    normal_indices_x = normal_indices[:,0]
    normal_indices_y = normal_indices[:,1]
    normal_indices_z = normal_indices[:,2]
    normal_conj_indices_x = normal_conj_indices[:,0]
    normal_conj_indices_y = normal_conj_indices[:,1]
    normal_conj_indices_z = normal_conj_indices[:,2]
    nyquist_indices_x = nyquist_indices[:,0]
    nyquist_indices_y = nyquist_indices[:,1]
    nyquist_indices_z = nyquist_indices[:,2]

    return normal_indices_x,normal_indices_y,normal_indices_z, normal_conj_indices_x,normal_conj_indices_y,normal_conj_indices_z, nyquist_indices_x,nyquist_indices_y,nyquist_indices_z

def Number_of_neigh(N, indices_i_lub, indices_j_lub):
    for_brown_lub = np.zeros(N)
    for i in range(N):
        for_brown_lub[i] = np.sum(np.where(indices_i_lub == i,1,0))
        for_brown_lub[i] += np.sum(np.where(indices_j_lub == i,1,0))
    for_brown_lub = jnp.repeat(for_brown_lub,6)
    return for_brown_lub




@partial(jit, static_argnums=[0,2,3,4])
def compute_real_space_slipvelocity(N,m_self,kT,dt,n_iter_Lanczos_ff,
                                    random_array_real,r,indices_i,indices_j,
                                    f1,f2,g1,g2,h1,h2,h3,
                                    ):

    def helper_Mpsi(random_array):
        
        #input is already in the format: [Forces, Torque+Stresslet] (not in the generalized format [Force+Torque,Stresslet] like in the saddle point solver)
        forces = random_array.at[:3*N].get()
        
        couplets = jnp.zeros(8*N)
        couplets = couplets.at[::8].add(random_array.at[(3*N+3)::8].get())  # C[0] = S[0]
        couplets = couplets.at[1::8].add(
            random_array.at[(3*N+4)::8].get()+random_array.at[(3*N+2)::8].get()*0.5)  # C[1] = S[1] + L[2]/2
        couplets = couplets.at[2::8].add(
            random_array.at[(3*N+5)::8].get()-random_array.at[(3*N+1)::8].get()*0.5)  # C[2] = S[2] - L[1]/2
        couplets = couplets.at[3::8].add(
            random_array.at[(3*N+6)::8].get()+random_array.at[(3*N+0)::8].get()*0.5)  # C[3] = S[3] + L[0]/2
        couplets = couplets.at[4::8].add(random_array.at[(3*N+7)::8].get())  # C[4] = S[4]
        couplets = couplets.at[5::8].add(
            random_array.at[(3*N+4)::8].get()-random_array.at[(3*N+2)::8].get()*0.5)  # C[5] = S[1] - L[2]/2
        couplets = couplets.at[6::8].add(
            random_array.at[(3*N+5)::8].get()+random_array.at[(3*N+1)::8].get()*0.5)  # C[6] = S[2] + L[1]/2
        couplets = couplets.at[7::8].add(
            random_array.at[(3*N+6)::8].get()-random_array.at[(3*N+0)::8].get()*0.5)  # C[7] = S[3] - L[0]/2
        
        
        # Allocate arrays for Linear velocities and Velocity gradients (from which we can extract angular velocities and rate of strain)
        r_lin_velocities = jnp.zeros((N, 3),float)
        r_velocity_gradient = jnp.zeros((N, 8),float)
        
        # SELF CONTRIBUTIONS
        r_lin_velocities = r_lin_velocities.at[:, 0].add(
            m_self.at[0].get() * forces.at[0::3].get())
        r_lin_velocities = r_lin_velocities.at[:, 1].add(
            m_self.at[0].get() * forces.at[1::3].get())
        r_lin_velocities = r_lin_velocities.at[:, 2].add(
            m_self.at[0].get() * forces.at[2::3].get())
        
        r_velocity_gradient = r_velocity_gradient.at[:, 0].add(
            m_self.at[1].get()*(couplets.at[0::8].get() - 4 * couplets.at[0::8].get()))
        r_velocity_gradient = r_velocity_gradient.at[:, 5].add(
            m_self.at[1].get()*(couplets.at[1::8].get() - 4 * couplets.at[5::8].get()))
        r_velocity_gradient = r_velocity_gradient.at[:, 6].add(
            m_self.at[1].get()*(couplets.at[2::8].get() - 4 * couplets.at[6::8].get()))
        r_velocity_gradient = r_velocity_gradient.at[:, 7].add(
            m_self.at[1].get()*(couplets.at[3::8].get() - 4 * couplets.at[7::8].get()))
        
        r_velocity_gradient = r_velocity_gradient.at[:, 4].add(
            m_self.at[1].get()*(couplets.at[4::8].get() - 4 * couplets.at[4::8].get()))
        r_velocity_gradient = r_velocity_gradient.at[:, 1].add(
            m_self.at[1].get()*(couplets.at[5::8].get() - 4 * couplets.at[1::8].get()))
        r_velocity_gradient = r_velocity_gradient.at[:, 2].add(
            m_self.at[1].get()*(couplets.at[6::8].get() - 4 * couplets.at[2::8].get()))
        r_velocity_gradient = r_velocity_gradient.at[:, 3].add(
            m_self.at[1].get()*(couplets.at[7::8].get() - 4 * couplets.at[3::8].get()))
        
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
        
        # # Convert to angular velocities and rate of strain
        r_ang_vel_and_strain = jnp.zeros((N, 8))
        r_ang_vel_and_strain = r_ang_vel_and_strain.at[:, 0].add((r_velocity_gradient.at[:, 3].get()-r_velocity_gradient.at[:, 7].get()) * 0.5)
        r_ang_vel_and_strain = r_ang_vel_and_strain.at[:, 1].add((r_velocity_gradient.at[:, 6].get()-r_velocity_gradient.at[:, 2].get()) * 0.5)
        r_ang_vel_and_strain = r_ang_vel_and_strain.at[:, 2].add((r_velocity_gradient.at[:, 1].get()-r_velocity_gradient.at[:, 5].get()) * 0.5)
        r_ang_vel_and_strain = r_ang_vel_and_strain.at[:, 3].add(2*r_velocity_gradient.at[:, 0].get()+r_velocity_gradient.at[:, 4].get())
        r_ang_vel_and_strain = r_ang_vel_and_strain.at[:, 4].add(r_velocity_gradient.at[:, 1].get()+r_velocity_gradient.at[:, 5].get())
        r_ang_vel_and_strain = r_ang_vel_and_strain.at[:, 5].add(r_velocity_gradient.at[:, 2].get()+r_velocity_gradient.at[:, 6].get())
        r_ang_vel_and_strain = r_ang_vel_and_strain.at[:, 6].add(r_velocity_gradient.at[:, 3].get()+r_velocity_gradient.at[:, 7].get())
        r_ang_vel_and_strain = r_ang_vel_and_strain.at[:, 7].add(r_velocity_gradient.at[:, 0].get()+2*r_velocity_gradient.at[:, 4].get())            
        
        slip_velocity = jnp.zeros(11*N)
        slip_velocity = slip_velocity.at[:3*N].add(jnp.reshape(r_lin_velocities,3*N))
        slip_velocity = slip_velocity.at[3*N:].add(jnp.reshape(r_ang_vel_and_strain,8*N))

        return slip_velocity    
    
    def helper_compute_M12psi(n_iter_Lanczos_ff, tridiagonal, vectors,norm):
        betae1 = jnp.zeros(n_iter_Lanczos_ff)
        betae1 = betae1.at[0].add(1*norm)
        
        a,b = jnp.linalg.eigh(tridiagonal)
        a = jnp.dot((jnp.dot(b,jnp.diag(jnp.sqrt(a)))),b.T)
        return jnp.dot(vectors.T,jnp.dot(a,betae1)) * jnp.sqrt(2.0*kT/dt)
    
    random_array_real = (2.*random_array_real-1.)*jnp.sqrt(3.)
    trid, vectors = Lanczos_lax.lanczos_alg(helper_Mpsi, 11*N, n_iter_Lanczos_ff, random_array_real)
     
    psinorm = jnp.linalg.norm(random_array_real)
    psiMpsi = jnp.dot(random_array_real,helper_Mpsi((random_array_real * 1/psinorm))) / (psinorm*psinorm)
    M12_psi_old = helper_compute_M12psi((n_iter_Lanczos_ff-1), trid[:(n_iter_Lanczos_ff-1),:(n_iter_Lanczos_ff-1)], vectors[:(n_iter_Lanczos_ff-1),:],psinorm)
    M12_psi = helper_compute_M12psi(n_iter_Lanczos_ff, trid, vectors,psinorm)
    stepnorm = jnp.sqrt(jnp.linalg.norm((M12_psi-M12_psi_old)) / psiMpsi)

    #combine w_lin_velocities, w_ang_vel_and_strain
    lin_vel = M12_psi.at[:3*N].get()
    ang_vel_and_strain = M12_psi.at[3*N:].get()
    
    return lin_vel, ang_vel_and_strain, stepnorm

@partial(jit, static_argnums=[0,2,3,4,5])
def compute_wave_space_slipvelocity(N,m_self,Nx,Ny,Nz,gaussP,kT,dt,gridh,
                                  #normal_indices,normal_conj_indices,nyquist_indices,
                                  normal_indices_x,normal_indices_y,normal_indices_z,
                                  normal_conj_indices_x,normal_conj_indices_y,normal_conj_indices_z,
                                  nyquist_indices_x,nyquist_indices_y,nyquist_indices_z,
                                  gridk,random_array_wave,all_indices_x,all_indices_y,all_indices_z,
                                  gaussian_grid_spacing1,gaussian_grid_spacing2):
    
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
        
        ### WAVE SPACE part
        gridX = jnp.array(gridX, complex)
        gridY = jnp.array(gridY, complex)
        gridZ = jnp.array(gridZ, complex)
 
        fac = jnp.sqrt( 3.0 * kT / dt / (gridh[0] * gridh[1] * gridh[2]) )
        random_array_wave = (2.*random_array_wave-1) * fac
        
        len_norm_indices = len(normal_indices_x)
        len_nyquist_indices = len(nyquist_indices_x)
        
        ###############################################################################################################################################
        gridX = gridX.at[normal_indices_x,normal_indices_y,normal_indices_z].add(
            random_array_wave.at[:len_norm_indices].get() + 1j * random_array_wave.at[len_norm_indices:2*len_norm_indices].get() )
        gridX = gridX.at[normal_conj_indices_x,normal_conj_indices_y,normal_conj_indices_z].add(
            random_array_wave.at[:len_norm_indices].get() - 1j * random_array_wave.at[len_norm_indices:2*len_norm_indices].get() )
        gridX = gridX.at[nyquist_indices_x,nyquist_indices_y,nyquist_indices_z].add(
            random_array_wave.at[6*len_norm_indices:(6*len_norm_indices+ 1 * len_nyquist_indices)].get() *1.414213562373095 + 0j )

        gridY = gridY.at[normal_indices_x,normal_indices_y,normal_indices_z].add(
            random_array_wave.at[(2*len_norm_indices):(3*len_norm_indices)].get() + 1j * random_array_wave.at[(3*len_norm_indices):(4*len_norm_indices)].get() )
        gridY = gridY.at[normal_conj_indices_x,normal_conj_indices_y,normal_conj_indices_z].add(
            random_array_wave.at[(2*len_norm_indices):(3*len_norm_indices)].get() - 1j * random_array_wave.at[(3*len_norm_indices):(4*len_norm_indices)].get() )
        gridY = gridY.at[nyquist_indices_x,nyquist_indices_y,nyquist_indices_z].add(
            random_array_wave.at[(6*len_norm_indices+ 1 * len_nyquist_indices):(6*len_norm_indices+ 2 * len_nyquist_indices)].get() *1.414213562373095 + 0j )

        gridZ = gridZ.at[normal_indices_x,normal_indices_y,normal_indices_z].add(
            random_array_wave.at[(4*len_norm_indices):(5*len_norm_indices)].get() + 1j * random_array_wave.at[(5*len_norm_indices):(6*len_norm_indices)].get() )
        gridZ = gridZ.at[normal_conj_indices_x,normal_conj_indices_y,normal_conj_indices_z].add(
            random_array_wave.at[(4*len_norm_indices):(5*len_norm_indices)].get() - 1j * random_array_wave.at[(5*len_norm_indices):(6*len_norm_indices)].get() )
        gridZ = gridZ.at[nyquist_indices_x,nyquist_indices_y,nyquist_indices_z].add(
            random_array_wave.at[(6*len_norm_indices+ 2 * len_nyquist_indices):(6*len_norm_indices+ 3 * len_nyquist_indices)].get() *1.414213562373095 + 0j )
        ###############################################################################################################################################
        
        #Compute k^2 and (|| k ||)
        gridk_sqr = (gridk.at[:, :, :, 0].get()*gridk.at[:, :, :, 0].get()+gridk.at[:, :, :, 1].get()
                     * gridk.at[:, :, :, 1].get()+gridk.at[:, :, :, 2].get()*gridk.at[:, :, :, 2].get())
        gridk_mod = jnp.sqrt(gridk_sqr)
        
        #Scaling factors
        B = jnp.where(gridk_mod > 0, jnp.sqrt(gridk.at[:, :, :, 3].get()), 0.)
        SU = jnp.where(gridk_mod > 0, jnp.sin(gridk_mod) / gridk_mod   , 0.)        
        SD = jnp.where(gridk_mod > 0, 3. * (jnp.sin(gridk_mod) - gridk_mod * jnp.cos(gridk_mod)) / (gridk_sqr * gridk_mod)   , 0.)        
        
        #Conjugate
        SD = -1. * SD
        
 		#Square root of Green's function times dW
        kdF = jnp.where(gridk_mod > 0., (gridk.at[:, :, :, 0].get()*gridX +
                        gridk.at[:, :, :, 1].get()*gridY+gridk.at[:, :, :, 2].get()*gridZ)/gridk_sqr, 0)

        BdWx = (gridX - gridk.at[:, :, :, 0].get() * kdF) * B
        BdWy = (gridY - gridk.at[:, :, :, 1].get() * kdF) * B
        BdWz = (gridZ - gridk.at[:, :, :, 2].get() * kdF) * B

        BdWkxx = BdWx * gridk.at[:, :, :, 0].get()
        BdWkxy = BdWx * gridk.at[:, :, :, 1].get()
        BdWkxz = BdWx * gridk.at[:, :, :, 2].get()
        BdWkyx = BdWy * gridk.at[:, :, :, 0].get()
        BdWkyy = BdWy * gridk.at[:, :, :, 1].get()
        BdWkyz = BdWy * gridk.at[:, :, :, 2].get()
        BdWkzx = BdWz * gridk.at[:, :, :, 0].get()
        BdWkzy = BdWz * gridk.at[:, :, :, 1].get()
        
        gridX = SU * BdWx
        gridY = SU * BdWy
        gridZ = SU * BdWz
        gridXX = SD * (-jnp.imag(BdWkxx) + 1j * jnp.real(BdWkxx))
        gridXY = SD * (-jnp.imag(BdWkxy) + 1j * jnp.real(BdWkxy))
        gridXZ = SD * (-jnp.imag(BdWkxz) + 1j * jnp.real(BdWkxz))
        gridYX = SD * (-jnp.imag(BdWkyx) + 1j * jnp.real(BdWkyx))
        gridYY = SD * (-jnp.imag(BdWkyy) + 1j * jnp.real(BdWkyy))
        gridYZ = SD * (-jnp.imag(BdWkyz) + 1j * jnp.real(BdWkyz))
        gridZX = SD * (-jnp.imag(BdWkzx) + 1j * jnp.real(BdWkzx))
        gridZY = SD * (-jnp.imag(BdWkzy) + 1j * jnp.real(BdWkzy))
        
        #Return rescaled forces to real space (Inverse FFT)
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
        
        #index might be 1 instead if 5 (originally is 1, and so on for the ones below)
        w_velocity_gradient = w_velocity_gradient.at[:, 5].add(
            jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
                gridXY.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3))) 
        #index might be 2 instead if 6           
        w_velocity_gradient = w_velocity_gradient.at[:, 6].add(
              jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
                gridXZ.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3)))  
        #index might be 3 instead if 7
        w_velocity_gradient = w_velocity_gradient.at[:, 7].add(
            jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
                gridYZ.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3)))
        
        
        w_velocity_gradient = w_velocity_gradient.at[:, 4].add(
            jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
                gridYY.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3))) 
        
        #index might be 5 instead if 1
        w_velocity_gradient = w_velocity_gradient.at[:, 1].add(
            jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
                gridYX.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3)))     
        #index might be 6 instead if 2                        
        w_velocity_gradient = w_velocity_gradient.at[:, 2].add(
              jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
                gridZX.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3))) 
        #index might be 7 instead if 3
        w_velocity_gradient = w_velocity_gradient.at[:, 3].add(
            jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
                gridZY.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3)))    
        
        # Convert to angular velocities and rate of strain
        w_ang_vel_and_strain = jnp.zeros((N, 8))
        w_ang_vel_and_strain = w_ang_vel_and_strain.at[:, 0].add((w_velocity_gradient.at[:, 3].get()-w_velocity_gradient.at[:, 7].get()) * 0.5)
        w_ang_vel_and_strain = w_ang_vel_and_strain.at[:, 1].add((w_velocity_gradient.at[:, 6].get()-w_velocity_gradient.at[:, 2].get()) * 0.5)
        w_ang_vel_and_strain = w_ang_vel_and_strain.at[:, 2].add((w_velocity_gradient.at[:, 1].get()-w_velocity_gradient.at[:, 5].get()) * 0.5)
        w_ang_vel_and_strain = w_ang_vel_and_strain.at[:, 3].add(2*w_velocity_gradient.at[:, 0].get()+w_velocity_gradient.at[:, 4].get())
        w_ang_vel_and_strain = w_ang_vel_and_strain.at[:, 4].add(w_velocity_gradient.at[:, 1].get()+w_velocity_gradient.at[:, 5].get())
        w_ang_vel_and_strain = w_ang_vel_and_strain.at[:, 5].add(w_velocity_gradient.at[:, 2].get()+w_velocity_gradient.at[:, 6].get())
        w_ang_vel_and_strain = w_ang_vel_and_strain.at[:, 6].add(w_velocity_gradient.at[:, 3].get()+w_velocity_gradient.at[:, 7].get())
        w_ang_vel_and_strain = w_ang_vel_and_strain.at[:, 7].add(w_velocity_gradient.at[:, 0].get()+2*w_velocity_gradient.at[:, 4].get())
        
        #combine w_lin_velocities, w_ang_vel_and_strain into generalized velocities
        w_lin_vel = jnp.reshape(w_lin_velocities,3*N) 
        w_ang_vel_and_strain = jnp.reshape(w_ang_vel_and_strain,8*N) 

        return w_lin_vel, w_ang_vel_and_strain

@partial(jit, static_argnums=[0,21])
def compute_nearfield_brownianforce(N,kT,dt,
        random_array,
        r_lub,indices_i_lub,indices_j_lub,XA11,XA12,YA11,YA12,YB11,YB12,XC11,XC12,YC11,YC12,YB21,
        diagonal_elements_for_brownian,
        R_fu_prec_lower_triang,
        diagonal_zeroes_for_brownian,
        n_iter_Lanczos_nf
        ):
    
    def Precondition_DiagMult_kernel(x, diag, direction):
        return x*jnp.where(direction==1,diag, 1/diag)
    
    def Precondition_Inn_kernel(x,diagonal_zeroes_for_brownian):
        identity = jnp.where((jnp.arange(6*N)-6*(jnp.repeat(jnp.arange(N), 6))) <
                                       3, 1, 1.33333333333)
        x = jnp.where(diagonal_zeroes_for_brownian==0., x*identity ,0.)
        return x

    def Precondition_ImInn_kernel(x, diagonal_zeroes_for_brownian):
        x = jnp.where(diagonal_zeroes_for_brownian==0., 0., x)
        return x
    
    def Precondition_Brownian_RFUmultiply(psi):
        # psi = jscipy.linalg.solve_triangular(jnp.transpose(R_fu_prec_lower_triang), psi, lower=False)
        # psi = Precondition_DiagMult_kernel(psi, diagonal_elements_for_brownian, 1)
        z = ComputeLubricationFU(psi)
        z += Precondition_Inn_kernel(psi,diagonal_zeroes_for_brownian)
        # psi = Precondition_DiagMult_kernel(z,diagonal_elements_for_brownian, 1)
        # return jscipy.linalg.solve_triangular(R_fu_prec_lower_triang, psi, lower=True)
        # return psi
        return z
    
    def Precondition_Brownian_Undo(nf_Brownian_force):
        # nf_Brownian_force = jnp.dot(R_fu_prec_lower_triang,nf_Brownian_force)
        # nf_Brownian_force = Precondition_DiagMult_kernel(nf_Brownian_force,diagonal_elements_for_brownian,-1)
        return Precondition_ImInn_kernel(nf_Brownian_force, diagonal_zeroes_for_brownian)
    
    def ComputeLubricationFU(velocities):

        vel_i = (jnp.reshape(velocities,(N,6))).at[indices_i_lub].get()
        vel_j = (jnp.reshape(velocities,(N,6))).at[indices_j_lub].get()
        
        # Dot product of r and U, i.e. axisymmetric projection
        rdui = r_lub.at[:,0].get()*vel_i.at[:,0].get()+r_lub.at[:,1].get()*vel_i.at[:,1].get()+r_lub.at[:,2].get()*vel_i.at[:,2].get()
        rduj = r_lub.at[:,0].get()*vel_j.at[:,0].get()+r_lub.at[:,1].get()*vel_j.at[:,1].get()+r_lub.at[:,2].get()*vel_j.at[:,2].get()
        rdwi = r_lub.at[:,0].get()*vel_i.at[:,3].get()+r_lub.at[:,1].get()*vel_i.at[:,4].get()+r_lub.at[:,2].get()*vel_i.at[:,5].get()
        rdwj = r_lub.at[:,0].get()*vel_j.at[:,3].get()+r_lub.at[:,1].get()*vel_j.at[:,4].get()+r_lub.at[:,2].get()*vel_j.at[:,5].get()

        # Cross product of U and r, i.e. eps_ijk*r_k*U_j = Px dot U, (eps_ijk is the Levi-Civita symbol)
        epsrdui = jnp.array([r_lub.at[:,2].get() * vel_i.at[:,1].get() - r_lub.at[:,1].get() * vel_i.at[:,2].get(),
                            -r_lub.at[:,2].get() * vel_i.at[:,0].get() + r_lub.at[:,0].get() * vel_i.at[:,2].get(),
                            r_lub.at[:,1].get() * vel_i.at[:,0].get() - r_lub.at[:,0].get() * vel_i.at[:,1].get()])
        
        epsrdwi = jnp.array([r_lub.at[:,2].get() * vel_i.at[:,4].get() - r_lub.at[:,1].get() * vel_i.at[:,5].get(), 
                            -r_lub.at[:,2].get() * vel_i.at[:,3].get() + r_lub.at[:,0].get() * vel_i.at[:,5].get(), 
                            r_lub.at[:,1].get() * vel_i.at[:,3].get() - r_lub.at[:,0].get() * vel_i.at[:,4].get()])
        
        epsrduj = jnp.array([r_lub.at[:,2].get() * vel_j.at[:,1].get() - r_lub.at[:,1].get() * vel_j.at[:,2].get(), 
                            -r_lub.at[:,2].get() * vel_j.at[:,0].get() + r_lub.at[:,0].get() * vel_j.at[:,2].get(), 
                            r_lub.at[:,1].get() * vel_j.at[:,0].get() - r_lub.at[:,0].get() * vel_j.at[:,1].get()])
        
        epsrdwj = jnp.array([r_lub.at[:,2].get() * vel_j.at[:,4].get() - r_lub.at[:,1].get() * vel_j.at[:,5].get(), 
                            -r_lub.at[:,2].get() * vel_j.at[:,3].get() + r_lub.at[:,0].get() * vel_j.at[:,5].get(), 
                            r_lub.at[:,1].get() * vel_j.at[:,3].get() - r_lub.at[:,0].get() * vel_j.at[:,4].get()])
        
        forces = jnp.zeros((N,6),float)
        
        # Compute the contributions to the force for particles i (Fi = A11*Ui + A12*Uj + BT11*Wi + BT12*Wj)
        f = ((XA11 - YA11).at[:,None].get() * rdui.at[:,None].get() * r_lub + YA11.at[:,None].get() * vel_i.at[:,:3].get() 
        + (XA12 - YA12).at[:,None].get() * rduj.at[:,None].get() * r_lub + YA12.at[:,None].get() * vel_j.at[:,:3].get() + YB11.at[:,None].get() * (-epsrdwi.T) + YB21.at[:,None].get() * (-epsrdwj.T))
        forces = forces.at[indices_i_lub, :3].add(f)
        # Compute the contributions to the force for particles j (Fj = A11*Uj + A12*Ui + BT11*Wj + BT12*Wi)
        f = ((XA11 - YA11).at[:,None].get() * rduj.at[:,None].get() * r_lub + YA11.at[:,None].get() * vel_j.at[:,:3].get() 
        + (XA12 - YA12).at[:,None].get() * rdui.at[:,None].get() * r_lub + YA12.at[:,None].get() * vel_i.at[:,:3].get() + YB11.at[:,None].get() * (epsrdwj.T) + YB21.at[:,None].get() * (epsrdwi.T))
        forces = forces.at[indices_j_lub, :3].add(f)
        # Compute the contributions to the torque for particles i (Li = B11*Ui + B12*Uj + C11*Wi + C12*Wj)
        l = (YB11.at[:,None].get() * epsrdui.T + YB12.at[:,None].get() * epsrduj.T 
        + (XC11 - YC11).at[:,None].get() * rdwi.at[:,None].get() * r_lub + YC11.at[:,None].get() * vel_i.at[:,3:].get() 
        + (XC12 - YC12).at[:,None].get() * rdwj.at[:,None].get() * r_lub + YC12.at[:,None].get() * vel_j.at[:,3:].get())
        forces = forces.at[indices_i_lub, 3:].add(l)
        # Compute the contributions to the torque for particles j (Lj = B11*Uj + B12*Ui + C11*Wj + C12*Wi)
        l = (-YB11.at[:,None].get() * epsrduj.T - YB12.at[:,None].get() * epsrdui.T 
        + (XC11 - YC11).at[:,None].get() * rdwj.at[:,None].get() * r_lub + YC11.at[:,None].get() * vel_j.at[:,3:].get() 
        + (XC12 - YC12).at[:,None].get() * rdwi.at[:,None].get() * r_lub + YC12.at[:,None].get() * vel_i.at[:,3:].get())           
        forces = forces.at[indices_j_lub, 3:].add(l)
        
        return jnp.ravel(forces)
    
    
    def helper_compute_R12psi(n_iter_Lanczos_nf, trid, vectors):
        betae1 = jnp.zeros(n_iter_Lanczos_nf)
        betae1 = betae1.at[0].add(1*psinorm)
        
        a,b = jnp.linalg.eigh(trid)
        a = jnp.dot((jnp.dot(b,jnp.diag(jnp.sqrt(a)))),b.T)
        
        return jnp.dot(vectors.T,jnp.dot(a,betae1)) * jnp.sqrt(2.0*kT/dt)
    
    

    
    ############################################################################################################################################################
    
    
    #Scale random numbers from [0,1] to [-sqrt(3),-sqrt(3)]
    random_array = (2*random_array-1)*jnp.sqrt(3.)
    
    # trid, vectors = Lanczos.lanczos_alg(Precondition_Brownian_RFUmultiply, 6*N, n_iter_Lanczos_nf, random_array)
    trid, vectors = Lanczos_lax.lanczos_alg(Precondition_Brownian_RFUmultiply, 6*N, n_iter_Lanczos_nf, random_array)
       
    psinorm = jnp.linalg.norm(random_array)
    # psiRpsi = jnp.dot(random_array,Precondition_Brownian_RFUmultiply((random_array * 1/psinorm))) / (psinorm*psinorm)
    # psiRpsi = jnp.dot(random_array,ComputeLubricationFU((random_array)))
    
    
    R_FU12psi_old = helper_compute_R12psi((n_iter_Lanczos_nf-1), trid[:(n_iter_Lanczos_nf-1),:(n_iter_Lanczos_nf-1)], vectors[:(n_iter_Lanczos_nf-1),:])
    # R_FU12psi_old = Precondition_Brownian_Undo(R_FU12psi_old)
    R_FU12psi = helper_compute_R12psi(n_iter_Lanczos_nf, trid, vectors)
    stepnorm = (jnp.linalg.norm((R_FU12psi-R_FU12psi_old)) / jnp.linalg.norm(R_FU12psi))
    # print(jnp.linalg.norm(R_FU12psi))
    R_FU12psi = Precondition_Brownian_Undo(R_FU12psi)
    # stepnorm = jnp.abs(jnp.dot(R_FU12psi, R_FU12psi) - psiRpsi) / psiRpsi
    # print(jnp.linalg.norm(R_FU12psi))
    return R_FU12psi, stepnorm




def compute_exact_thermals(N,m_self,Nx,Ny,Nz,gaussP,gridh,kT,dt,
        gridk,random_array_wave,all_indices_x,all_indices_y,all_indices_z,
        gaussian_grid_spacing1,gaussian_grid_spacing2,                   
        random_array,random_array_real,
        r,indices_i,indices_j,f1,f2,g1,g2,h1,h2,h3,
        r_lub,indices_i_lub,indices_j_lub,XA11,XA12,YA11,YA12,YB11,YB12,XC11,XC12,YC11,YC12,YB21
        ):
    @jit
    def convert_to_generalized(M12psi):
        #combine w_lin_velocities, w_ang_vel_and_strain
        lin_vel = M12psi.at[:3*N].get()
        ang_vel_and_strain = M12psi.at[3*N:].get()
        # Convert to Generalized Velocities+strain 
        generalized_velocities = jnp.zeros(11*N) #First 6N entries for U and last 5N for strain rates

        generalized_velocities = generalized_velocities.at[0:6*N:6].set(
            lin_vel.at[0::3].get())
        generalized_velocities = generalized_velocities.at[1:6*N:6].set(
            lin_vel.at[1::3].get())
        generalized_velocities = generalized_velocities.at[2:6*N:6].set(
            lin_vel.at[2::3].get())
        generalized_velocities = generalized_velocities.at[3:6*N:6].set(
            ang_vel_and_strain.at[0::8].get())
        generalized_velocities = generalized_velocities.at[4:6*N:6].set(
            ang_vel_and_strain.at[1::8].get())
        generalized_velocities = generalized_velocities.at[5:6*N:6].set(
            ang_vel_and_strain.at[2::8].get())
        generalized_velocities = generalized_velocities.at[(6*N+0)::5].set(
            ang_vel_and_strain.at[3::8].get())
        generalized_velocities = generalized_velocities.at[(6*N+1)::5].set(
            ang_vel_and_strain.at[4::8].get())
        generalized_velocities = generalized_velocities.at[(6*N+2)::5].set(
            ang_vel_and_strain.at[5::8].get())
        generalized_velocities = generalized_velocities.at[(6*N+3)::5].set(
            ang_vel_and_strain.at[6::8].get())
        generalized_velocities = generalized_velocities.at[(6*N+4)::5].set(
            ang_vel_and_strain.at[7::8].get())
        
        return generalized_velocities
    @jit
    def helper_reshape(Mpsi):
        lin_vel = Mpsi[0]
        ang_vel_and_strain = Mpsi[1]
        reshaped_array = jnp.zeros(11*N)
        reshaped_array = reshaped_array.at[:3*N].set(jnp.reshape(lin_vel,3*N))
        reshaped_array = reshaped_array.at[3*N:].set(jnp.reshape(ang_vel_and_strain,8*N))
        return reshaped_array
    
    @jit
    def ComputeLubricationFU(velocities):

        vel_i = (jnp.reshape(velocities,(N,6))).at[indices_i_lub].get()
        vel_j = (jnp.reshape(velocities,(N,6))).at[indices_j_lub].get()
        
        # Dot product of r and U, i.e. axisymmetric projection
        rdui = r_lub.at[:,0].get()*vel_i.at[:,0].get()+r_lub.at[:,1].get()*vel_i.at[:,1].get()+r_lub.at[:,2].get()*vel_i.at[:,2].get()
        rduj = r_lub.at[:,0].get()*vel_j.at[:,0].get()+r_lub.at[:,1].get()*vel_j.at[:,1].get()+r_lub.at[:,2].get()*vel_j.at[:,2].get()
        rdwi = r_lub.at[:,0].get()*vel_i.at[:,3].get()+r_lub.at[:,1].get()*vel_i.at[:,4].get()+r_lub.at[:,2].get()*vel_i.at[:,5].get()
        rdwj = r_lub.at[:,0].get()*vel_j.at[:,3].get()+r_lub.at[:,1].get()*vel_j.at[:,4].get()+r_lub.at[:,2].get()*vel_j.at[:,5].get()

        # Cross product of U and r, i.e. eps_ijk*r_k*U_j = Px dot U, (eps_ijk is the Levi-Civita symbol)
        epsrdui = jnp.array([r_lub.at[:,2].get() * vel_i.at[:,1].get() - r_lub.at[:,1].get() * vel_i.at[:,2].get(),
                            -r_lub.at[:,2].get() * vel_i.at[:,0].get() + r_lub.at[:,0].get() * vel_i.at[:,2].get(),
                            r_lub.at[:,1].get() * vel_i.at[:,0].get() - r_lub.at[:,0].get() * vel_i.at[:,1].get()])
        
        epsrdwi = jnp.array([r_lub.at[:,2].get() * vel_i.at[:,4].get() - r_lub.at[:,1].get() * vel_i.at[:,5].get(), 
                            -r_lub.at[:,2].get() * vel_i.at[:,3].get() + r_lub.at[:,0].get() * vel_i.at[:,5].get(), 
                            r_lub.at[:,1].get() * vel_i.at[:,3].get() - r_lub.at[:,0].get() * vel_i.at[:,4].get()])
        
        epsrduj = jnp.array([r_lub.at[:,2].get() * vel_j.at[:,1].get() - r_lub.at[:,1].get() * vel_j.at[:,2].get(), 
                            -r_lub.at[:,2].get() * vel_j.at[:,0].get() + r_lub.at[:,0].get() * vel_j.at[:,2].get(), 
                            r_lub.at[:,1].get() * vel_j.at[:,0].get() - r_lub.at[:,0].get() * vel_j.at[:,1].get()])
        
        epsrdwj = jnp.array([r_lub.at[:,2].get() * vel_j.at[:,4].get() - r_lub.at[:,1].get() * vel_j.at[:,5].get(), 
                            -r_lub.at[:,2].get() * vel_j.at[:,3].get() + r_lub.at[:,0].get() * vel_j.at[:,5].get(), 
                            r_lub.at[:,1].get() * vel_j.at[:,3].get() - r_lub.at[:,0].get() * vel_j.at[:,4].get()])
        
        forces = jnp.zeros((N,6),float)
        
        # Compute the contributions to the force for particles i (Fi = A11*Ui + A12*Uj + BT11*Wi + BT12*Wj)
        f = ((XA11 - YA11).at[:,None].get() * rdui.at[:,None].get() * r_lub + YA11.at[:,None].get() * vel_i.at[:,:3].get() 
        + (XA12 - YA12).at[:,None].get() * rduj.at[:,None].get() * r_lub + YA12.at[:,None].get() * vel_j.at[:,:3].get() + YB11.at[:,None].get() * (-epsrdwi.T) + YB21.at[:,None].get() * (-epsrdwj.T))
        forces = forces.at[indices_i_lub, :3].add(f)
        # Compute the contributions to the force for particles j (Fj = A11*Uj + A12*Ui + BT11*Wj + BT12*Wi)
        f = ((XA11 - YA11).at[:,None].get() * rduj.at[:,None].get() * r_lub + YA11.at[:,None].get() * vel_j.at[:,:3].get() 
        + (XA12 - YA12).at[:,None].get() * rdui.at[:,None].get() * r_lub + YA12.at[:,None].get() * vel_i.at[:,:3].get() + YB11.at[:,None].get() * (epsrdwj.T) + YB21.at[:,None].get() * (epsrdwi.T))
        forces = forces.at[indices_j_lub, :3].add(f)
        # Compute the contributions to the torque for particles i (Li = B11*Ui + B12*Uj + C11*Wi + C12*Wj)
        l = (YB11.at[:,None].get() * epsrdui.T + YB12.at[:,None].get() * epsrduj.T 
        + (XC11 - YC11).at[:,None].get() * rdwi.at[:,None].get() * r_lub + YC11.at[:,None].get() * vel_i.at[:,3:].get() 
        + (XC12 - YC12).at[:,None].get() * rdwj.at[:,None].get() * r_lub + YC12.at[:,None].get() * vel_j.at[:,3:].get())
        forces = forces.at[indices_i_lub, 3:].add(l)
        # Compute the contributions to the torque for particles j (Lj = B11*Uj + B12*Ui + C11*Wj + C12*Wi)
        l = (-YB11.at[:,None].get() * epsrduj.T - YB12.at[:,None].get() * epsrdui.T 
        + (XC11 - YC11).at[:,None].get() * rdwj.at[:,None].get() * r_lub + YC11.at[:,None].get() * vel_j.at[:,3:].get() 
        + (XC12 - YC12).at[:,None].get() * rdwi.at[:,None].get() * r_lub + YC12.at[:,None].get() * vel_i.at[:,3:].get())           
        forces = forces.at[indices_j_lub, 3:].add(l)
        
        return jnp.ravel(forces)
    
    @jit
    def GeneralizedMobility_ws(
            N,
            Nx,
            Ny,
            Nz,
            gaussP,
            gridk,
            m_self,
            all_indices_x,all_indices_y,all_indices_z,gaussian_grid_spacing1,gaussian_grid_spacing2,
            generalized_forces
            ):
        
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
    
        velocity_gradient = w_velocity_gradient
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
        
        reshaped_array = jnp.zeros(11*N)
        reshaped_array = reshaped_array.at[:3*N].set(jnp.reshape(w_lin_velocities,3*N))
        reshaped_array = reshaped_array.at[3*N:].set(jnp.reshape(ang_vel_and_strain,8*N))
        
        return reshaped_array
    
    @jit
    def helper_Mpsi(random_array_real):
        
        #input is already in the format: [Forces, Torque+Stresslet] (not in the generalized format [Force+Torque,Stresslet] like in the saddle point solver)
        forces = random_array_real.at[:3*N].get()
        
        couplets = jnp.zeros(8*N)
        couplets = couplets.at[::8].set(random_array_real.at[(3*N+3)::8].get())  # C[0] = S[0]
        couplets = couplets.at[1::8].set(
            random_array_real.at[(3*N+4)::8].get()+random_array_real.at[(3*N+2)::8].get()*0.5)  # C[1] = S[1] + L[2]/2
        couplets = couplets.at[2::8].set(
            random_array_real.at[(3*N+5)::8].get()-random_array_real.at[(3*N+1)::8].get()*0.5)  # C[2] = S[2] - L[1]/2
        couplets = couplets.at[3::8].set(
            random_array_real.at[(3*N+6)::8].get()+random_array_real.at[(3*N+0)::8].get()*0.5)  # C[3] = S[3] + L[0]/2
        couplets = couplets.at[4::8].set(random_array_real.at[(3*N+7)::8].get())  # C[4] = S[4]
        couplets = couplets.at[5::8].set(
            random_array_real.at[(3*N+4)::8].get()-random_array_real.at[(3*N+2)::8].get()*0.5)  # C[5] = S[1] - L[2]/2
        couplets = couplets.at[6::8].set(
            random_array_real.at[(3*N+5)::8].get()+random_array_real.at[(3*N+1)::8].get()*0.5)  # C[6] = S[2] + L[1]/2
        couplets = couplets.at[7::8].set(
            random_array_real.at[(3*N+6)::8].get()-random_array_real.at[(3*N+0)::8].get()*0.5)  # C[7] = S[3] - L[0]/2
        
        
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
        
        # # Convert to angular velocities and rate of strain
        r_ang_vel_and_strain = jnp.zeros((N, 8))
        r_ang_vel_and_strain = r_ang_vel_and_strain.at[:, 0].set((r_velocity_gradient.at[:, 3].get()-r_velocity_gradient.at[:, 7].get()) * 0.5)
        r_ang_vel_and_strain = r_ang_vel_and_strain.at[:, 1].set((r_velocity_gradient.at[:, 6].get()-r_velocity_gradient.at[:, 2].get()) * 0.5)
        r_ang_vel_and_strain = r_ang_vel_and_strain.at[:, 2].set((r_velocity_gradient.at[:, 1].get()-r_velocity_gradient.at[:, 5].get()) * 0.5)
        r_ang_vel_and_strain = r_ang_vel_and_strain.at[:, 3].set(2*r_velocity_gradient.at[:, 0].get()+r_velocity_gradient.at[:, 4].get())
        r_ang_vel_and_strain = r_ang_vel_and_strain.at[:, 4].set(r_velocity_gradient.at[:, 1].get()+r_velocity_gradient.at[:, 5].get())
        r_ang_vel_and_strain = r_ang_vel_and_strain.at[:, 5].set(r_velocity_gradient.at[:, 2].get()+r_velocity_gradient.at[:, 6].get())
        r_ang_vel_and_strain = r_ang_vel_and_strain.at[:, 6].set(r_velocity_gradient.at[:, 3].get()+r_velocity_gradient.at[:, 7].get())
        r_ang_vel_and_strain = r_ang_vel_and_strain.at[:, 7].set(r_velocity_gradient.at[:, 0].get()+2*r_velocity_gradient.at[:, 4].get())            
        
        return r_lin_velocities, r_ang_vel_and_strain
    
    
        
    
    #obtain matrix form of linear operator Mpsi, by computing Mpsi(e_i) with e_i basis vectors (1,0,...,0), (0,1,0,...) ...
    random_array = (2*random_array-1)*jnp.sqrt(3.)
    R_FU_Matrix = np.zeros((6*N,6*N))
    basis_vectors = np.eye(6*N, dtype = float)
    for iii in range(6*N):
        Rei = ComputeLubricationFU(basis_vectors[iii,:])
        R_FU_Matrix[:,iii] =  Rei
    sqrt_R_FU = scipy.linalg.sqrtm(R_FU_Matrix) #EXTEMELY NOT EFFICIENT! need to be replaced with faster method
    R_FU12psi_correct = jnp.dot(sqrt_R_FU,random_array*np.sqrt(2. * kT / dt))
    
    random_array_real = (2*random_array_real-1)*jnp.sqrt(3.)
    Matrix_M = np.zeros((11*N,11*N)); basis_vectors = np.eye(11*N, dtype = float)
    for iii in range(11*N):
        a = helper_Mpsi(basis_vectors[iii,:]); Mei = helper_reshape(a); Matrix_M[:,iii] =  Mei
    sqrt_M = scipy.linalg.sqrtm(Matrix_M); 
    M12psi_debug = jnp.dot(sqrt_M,random_array_real* jnp.sqrt(2.0*kT/dt))
    
    return convert_to_generalized(M12psi_debug), R_FU12psi_correct





@partial(jit, static_argnums=[0])
def convert_to_generalized(N, ws_lin_vel, rs_lin_vel, ws_ang_vel_strain,rs_ang_vel_strain):
    
    lin_vel = ws_lin_vel + rs_lin_vel
    ang_vel_and_strain = ws_ang_vel_strain + rs_ang_vel_strain
    
    # Convert to Generalized Velocities+strain 
    generalized_velocities = jnp.zeros(11*N) #First 6N entries for U and last 5N for strain rates

    generalized_velocities = generalized_velocities.at[0:6*N:6].add(
        lin_vel.at[0::3].get())
    generalized_velocities = generalized_velocities.at[1:6*N:6].add(
        lin_vel.at[1::3].get())
    generalized_velocities = generalized_velocities.at[2:6*N:6].add(
        lin_vel.at[2::3].get())
    generalized_velocities = generalized_velocities.at[3:6*N:6].add(
        ang_vel_and_strain.at[0::8].get())
    generalized_velocities = generalized_velocities.at[4:6*N:6].add(
        ang_vel_and_strain.at[1::8].get())
    generalized_velocities = generalized_velocities.at[5:6*N:6].add(
        ang_vel_and_strain.at[2::8].get())
    generalized_velocities = generalized_velocities.at[(6*N+0)::5].add(
        ang_vel_and_strain.at[3::8].get())
    generalized_velocities = generalized_velocities.at[(6*N+1)::5].add(
        ang_vel_and_strain.at[4::8].get())
    generalized_velocities = generalized_velocities.at[(6*N+2)::5].add(
        ang_vel_and_strain.at[5::8].get())
    generalized_velocities = generalized_velocities.at[(6*N+3)::5].add(
        ang_vel_and_strain.at[6::8].get())
    generalized_velocities = generalized_velocities.at[(6*N+4)::5].add(
        ang_vel_and_strain.at[7::8].get())
    
    return generalized_velocities

