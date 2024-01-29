import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import numpy as np
from jax import jit
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)
import scipy
import jax.scipy as jscipy
import time
from functools import partial
import Lanczos

def compute_nearfield_brownianforce(N,kT,dt,
        random_array,
        r_lub,indices_i_lub,indices_j_lub,XA11,XA12,YA11,YA12,YB11,YB12,XC11,XC12,YC11,YC12,YB21,
        diagonal_elements_for_brownian,
        R_fu_prec_lower_triang,
        diagonal_zeroes_for_brownian,
        n_iter_Lanczos_nf
        ):

    def number_of_neigh(diagonal_zeroes_for_brownian):
        diagonal_zeroes_for_brownian = np.zeros(N)
        for i in range(N):
            diagonal_zeroes_for_brownian[i] = np.sum(np.where(indices_i_lub == i,1,0))
            diagonal_zeroes_for_brownian[i] += np.sum(np.where(indices_j_lub == i,1,0))
        diagonal_zeroes_for_brownian = jnp.repeat(diagonal_zeroes_for_brownian,6)
        return diagonal_zeroes_for_brownian
    @jit
    def Precondition_DiagMult_kernel(x, diag, direction):
        return x*jnp.where(direction==1,diag, 1/diag)
    
    @jit
    def Precondition_Inn_kernel(x,diagonal_zeroes_for_brownian):
        identity = jnp.where((jnp.arange(6*N)-6*(jnp.repeat(jnp.arange(N), 6))) <
                                       3, 1, 1.33333333333)
        x = jnp.where(diagonal_zeroes_for_brownian==0., x*identity ,0.)
        return x

    @jit
    def Precondition_ImInn_kernel(x, diagonal_zeroes_for_brownian):
        x = jnp.where(diagonal_zeroes_for_brownian==0., 0., x)
        return x
    
    @jit
    def Precondition_Brownian_RFUmultiply(psi):
        # psi = jscipy.linalg.solve_triangular(jnp.transpose(R_fu_prec_lower_triang), psi, lower=False)
        # psi = Precondition_DiagMult_kernel(psi, diagonal_elements_for_brownian, 1)
        z = ComputeLubricationFU(psi)
        z += Precondition_Inn_kernel(psi,diagonal_zeroes_for_brownian)
        # psi = Precondition_DiagMult_kernel(z,diagonal_elements_for_brownian, 1)
        # return jscipy.linalg.solve_triangular(R_fu_prec_lower_triang, z, lower=True)
        return z
    
    @jit
    def Precondition_Brownian_Undo(nf_Brownian_force):
        # nf_Brownian_force = jnp.dot(R_fu_prec_lower_triang,nf_Brownian_force)
        # nf_Brownian_force = Precondition_DiagMult_kernel(nf_Brownian_force,diagonal_elements_for_brownian,-1)
        return Precondition_ImInn_kernel(nf_Brownian_force, diagonal_zeroes_for_brownian)
    
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
    def helper_dot(a,b):
        return jnp.dot(a,b)
    
    @jit
    def helper_sqrtm(M):
        a,b = jnp.linalg.eigh(M)
        a = jnp.dot((jnp.dot(b,jnp.diag(jnp.sqrt(a)))),b.T)
        return a
    
    @jit
    def helper_sqrt(a):
        return jnp.sqrt(a)
    @jit
    def helper_multip(a,b):
        return a*b      
    @jit
    def helper_lincomb(a,b,c,d):
        return a*b +c*d
    @jit
    def helper_norm(v):
        return jnp.linalg.norm(v)

    def Lanczos_sqrtm(psi,k,n_iter_Lanczos_nf, diagonal_zeroes_for_brownian):
        Lv=np.zeros((len(psi),len(psi))) #Creates matrix for Lanczos vectors
        R_kril=np.zeros((len(psi),len(psi))) #Creates matrix for the Mobility in Krylov subspace (tridiagonal matrix)
        
        psinorm = helper_norm(psi)
        
        Lv[0] = helper_multip(psi, 1/psinorm)
        
        #Performs the first iteration step of the Lanczos algorithm
        w = Precondition_Brownian_RFUmultiply(Lv[0])
        psiRpsi = helper_dot(psi,w) / (psinorm*psinorm)
        # print(psiRpsi)

        a= helper_dot(w,Lv[0])
        w= helper_lincomb(1,w,-a,Lv[0]) 
        R_kril[0,0]=a
        #Performs first few iterations always the iterative steps of the Lanczos algorithm
        mm = max(2,n_iter_Lanczos_nf)
        for j in range(1,mm):
            b= helper_sqrt(helper_dot(w,(w))) 
            Lv[j]= helper_multip(w,1/b)
            w = Precondition_Brownian_RFUmultiply(Lv[j])
            a= helper_dot(w,Lv[j]) 
            w= helper_lincomb(1,helper_lincomb(1,w,-a,Lv[j]),-b,Lv[j-1]) 
            
            #Creates tridiagonal matrix R_kril using a and b values
            R_kril[j,j]=a
            R_kril[j-1,j]=b
            R_kril[j,j-1]=b
        
        #Compute sqrt(M)*psi for the fist iterations and define stepnorm    
        betae1 = np.zeros(j+1)
        betae1[0] = 1*psinorm
        R12_psi = helper_dot(Lv[:(j+1),:].T,helper_dot(helper_sqrtm(R_kril[:(j+1),:(j+1)]),betae1))
        R12_psi_old = R12_psi
        stepnorm = 1.
        while(stepnorm>0.001):
            j += 1
            if((j>k) or (j==(6*N-1))):
                print('Max Lanczos iterations reached, stepnorm is ',stepnorm)
                break
            
            b= helper_sqrt(helper_dot(w,(w)))
            Lv[j]= helper_multip(w,1/b)
            w = Precondition_Brownian_RFUmultiply(Lv[j])
            a= helper_dot(w,Lv[j]) 
            w= helper_lincomb(1,helper_lincomb(1,w,-a,Lv[j]),-b,Lv[j-1]) 
            
            #Creates tridiagonal matrix R_kril using a and b values
            R_kril[j,j]=a
            R_kril[j-1,j]=b
            R_kril[j,j-1]=(b)
        
            #Compute sqrt(M)*psi for the fist iterations and define stepnorm    
            betae1 = np.zeros(j+1)
            betae1[0] = 1*psinorm
            
            # print((Lv[(j+1),:]).shape, (helper_sqrtm(R_kril[:(j+1),:(j+1)])).shape, betae1.shape)
            R12_psi = helper_dot(Lv[:(j+1),:].T,helper_dot(helper_sqrtm(R_kril[:(j+1),:(j+1)]),betae1))
            stepnorm = np.sqrt(helper_norm(helper_lincomb(1,R12_psi, -1, R12_psi_old)) / psiRpsi)
            # stepnorm = np.sqrt(helper_norm(helper_lincomb(1,R12_psi, -1, R12_psi_old)) / helper_norm(R12_psi))
            R12_psi_old = R12_psi
            #DEBUG
            # print(j, stepnorm, 'this are the iteration number and the associated stepnorm')
        
        n_iter_Lanczos_nf = j
        R12_psi = Precondition_Brownian_Undo(R12_psi)
        
        #DEBUG
        # print(j, stepnorm, 'ITERATION NUMBER and STEPNORM')
        
        return helper_multip(R12_psi, np.sqrt(2.0*kT/dt)), n_iter_Lanczos_nf
    ############################################################################################################################################################
    
    
    #Scale random numbers from [0,1] to [-sqrt(3),-sqrt(3)]
    random_array = (2*random_array-1)*jnp.sqrt(3.)
    diagonal_zeroes_for_brownian = number_of_neigh(diagonal_zeroes_for_brownian)
    
    start_time = time.time()
    
    
    #obtain matrix form of linear operator Mpsi, by computing Mpsi(e_i) with e_i basis vectors (1,0,...,0), (0,1,0,...) ...
    R_FU_Matrix = np.zeros((6*N,6*N))
    basis_vectors = np.eye(6*N, dtype = float)
    for iii in range(6*N):
        Rei = ComputeLubricationFU(basis_vectors[iii,:])
        R_FU_Matrix[:,iii] =  Rei
    sqrt_R_FU = scipy.linalg.sqrtm(R_FU_Matrix) #EXTEMELY NOT EFFICIENT! need to be replaced with faster method
    R_FU12psi_correct = helper_dot(sqrt_R_FU,random_array*np.sqrt(2. * kT / dt))

    
    R_FU12psi, n_iter_Lanczos_nf = Lanczos_sqrtm(random_array, 100, n_iter_Lanczos_nf, diagonal_zeroes_for_brownian)
    
    end_time = time.time()
    print('Time Lubric Brown: ', end_time-start_time, ' error=', np.linalg.norm(R_FU12psi_correct-R_FU12psi)/np.linalg.norm(R_FU12psi_correct))
    
    
    return R_FU12psi, n_iter_Lanczos_nf


def compute_farfield_slipvelocity(N,m_self,Nx,Ny,Nz,gaussP,kT,dt,gridh,
                                  normal_indices,normal_conj_indices,nyquist_indices,
                                  n_iter_Lanczos_ff,gridk,random_array_wave,random_array_real,
                                  all_indices_x,all_indices_y,all_indices_z,gaussian_grid_spacing1,gaussian_grid_spacing2,
                                  r,indices_i,indices_j,f1,f2,g1,g2,h1,h2,h3
                                  ):

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
    
    @jit
    def helper_wavespace_calc(random_array_wave):
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
        gridY = jnp.array(gridX, complex)
        gridZ = jnp.array(gridX, complex)
 
        fac = jnp.sqrt( 3.0 * kT / dt / (gridh[0] * gridh[1] * gridh[2]) )
        random_array_wave = (2*random_array_wave-1) * fac

        gridX = gridX.at[normal_indices[:,0],normal_indices[:,1],normal_indices[:,2]].set(
            random_array_wave.at[:len(normal_indices)].get() + 1j * random_array_wave.at[len(normal_indices):2*len(normal_indices)].get() )
        gridX = gridX.at[normal_conj_indices[:,0],normal_conj_indices[:,1],normal_conj_indices[:,2]].set(
            random_array_wave.at[:len(normal_indices)].get() - 1j * random_array_wave.at[len(normal_indices):2*len(normal_indices)].get() )
        gridX = gridX.at[nyquist_indices[:,0],nyquist_indices[:,1],nyquist_indices[:,2]].set(
            random_array_wave.at[6*len(normal_indices):(6*len(normal_indices)+ 1 * len(nyquist_indices))].get() *1.414213562373095 + 0j )

        gridY = gridY.at[normal_indices[:,0],normal_indices[:,1],normal_indices[:,2]].set(
            random_array_wave.at[(2*len(normal_indices)):(3*len(normal_indices))].get() + 1j * random_array_wave.at[(3*len(normal_indices)):(4*len(normal_indices))].get() )
        gridY = gridY.at[normal_conj_indices[:,0],normal_conj_indices[:,1],normal_conj_indices[:,2]].set(
            random_array_wave.at[(2*len(normal_indices)):(3*len(normal_indices))].get() - 1j * random_array_wave.at[(3*len(normal_indices)):(4*len(normal_indices))].get() )
        gridY = gridY.at[nyquist_indices[:,0],nyquist_indices[:,1],nyquist_indices[:,2]].set(
            random_array_wave.at[(6*len(normal_indices)+ 1 * len(nyquist_indices)):(6*len(normal_indices)+ 2 * len(nyquist_indices))].get() *1.414213562373095 + 0j )

        gridZ = gridZ.at[normal_indices[:,0],normal_indices[:,1],normal_indices[:,2]].set(
            random_array_wave.at[(4*len(normal_indices)):(5*len(normal_indices))].get() + 1j * random_array_wave.at[(5*len(normal_indices)):(6*len(normal_indices))].get() )
        gridZ = gridZ.at[normal_conj_indices[:,0],normal_conj_indices[:,1],normal_conj_indices[:,2]].set(
            random_array_wave.at[(4*len(normal_indices)):(5*len(normal_indices))].get() - 1j * random_array_wave.at[(5*len(normal_indices)):(6*len(normal_indices))].get() )
        gridZ = gridZ.at[nyquist_indices[:,0],nyquist_indices[:,1],nyquist_indices[:,2]].set(
            random_array_wave.at[(6*len(normal_indices)+ 2 * len(nyquist_indices)):(6*len(normal_indices)+ 3 * len(nyquist_indices))].get() *1.414213562373095 + 0j )
        
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
        w_ang_vel_and_strain = w_ang_vel_and_strain.at[:, 0].set((w_velocity_gradient.at[:, 3].get()-w_velocity_gradient.at[:, 7].get()) * 0.5)
        w_ang_vel_and_strain = w_ang_vel_and_strain.at[:, 1].set((w_velocity_gradient.at[:, 6].get()-w_velocity_gradient.at[:, 2].get()) * 0.5)
        w_ang_vel_and_strain = w_ang_vel_and_strain.at[:, 2].set((w_velocity_gradient.at[:, 1].get()-w_velocity_gradient.at[:, 5].get()) * 0.5)
        w_ang_vel_and_strain = w_ang_vel_and_strain.at[:, 3].set(2*w_velocity_gradient.at[:, 0].get()+w_velocity_gradient.at[:, 4].get())
        w_ang_vel_and_strain = w_ang_vel_and_strain.at[:, 4].set(w_velocity_gradient.at[:, 1].get()+w_velocity_gradient.at[:, 5].get())
        w_ang_vel_and_strain = w_ang_vel_and_strain.at[:, 5].set(w_velocity_gradient.at[:, 2].get()+w_velocity_gradient.at[:, 6].get())
        w_ang_vel_and_strain = w_ang_vel_and_strain.at[:, 6].set(w_velocity_gradient.at[:, 3].get()+w_velocity_gradient.at[:, 7].get())
        w_ang_vel_and_strain = w_ang_vel_and_strain.at[:, 7].set(w_velocity_gradient.at[:, 0].get()+2*w_velocity_gradient.at[:, 4].get())
        
        return w_lin_velocities, w_ang_vel_and_strain
    
    @jit
    def helper_reshape(Mpsi):
        lin_vel = Mpsi[0]
        ang_vel_and_strain = Mpsi[1]
        reshaped_array = jnp.zeros(11*N)
        reshaped_array = reshaped_array.at[:3*N].set(jnp.reshape(lin_vel,3*N))
        reshaped_array = reshaped_array.at[3*N:].set(jnp.reshape(ang_vel_and_strain,8*N))
        return reshaped_array
    
    @jit
    def helper_dot(a,b):
        return jnp.dot(a,b)
    @jit
    def helper_sqrtm(M):
        a,b = jnp.linalg.eigh(M)
        a = jnp.dot((jnp.dot(b,jnp.diag(jnp.sqrt(a)))),b.T)
        return a
    
    @jit
    def helper_sqrt(a):
        return jnp.sqrt(a)
    @jit
    def helper_multip(a,b):
        return a*b      
    @jit
    def helper_lincomb(a,b,c,d):
        return a*b +c*d
    @jit
    def helper_norm(v):
        return jnp.linalg.norm(v)
    
    @jit
    def convert_to_generalized(lin_velocities,ang_vel_and_strain,M12psi):
        #combine w_lin_velocities, w_ang_vel_and_strain
        lin_vel = jnp.reshape(lin_velocities,3*N) + M12psi.at[:3*N].get()
        ang_vel_and_strain = jnp.reshape(ang_vel_and_strain,8*N) + M12psi.at[3*N:].get()
        
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
    
    def Lanczos_sqrtm(psi,k,n_iter_Lanczos_ff):
        Lv=np.zeros((len(psi),len(psi))) #Creates matrix for Lanczos vectors
        M_kril=np.zeros((len(psi),len(psi))) #Creates matrix for the Mobility in Krylov subspace (tridiagonal matrix)
        
        psinorm = helper_norm(psi)
        
        Lv[0] = helper_multip(psi, 1/psinorm)
        
        #Performs the first iteration step of the Lanczos algorithm
        w = helper_reshape(helper_Mpsi(Lv[0]))
        
        psiMpsi = helper_dot(psi,w) / (psinorm*psinorm)
        
        a= helper_dot(w,Lv[0]) 
        w= helper_lincomb(1,w,-a,Lv[0]) 
        M_kril[0,0]=a
        #Performs first few iterations always the iterative steps of the Lanczos algorithm
        mm = max(2,n_iter_Lanczos_ff)
        for j in range(1,mm):
            # b=(np.dot(w,np.transpose(w)))**0.5
            b= helper_sqrt(helper_dot(w,(w)))
            Lv[j]= helper_multip(w,1/b)
            w = helper_reshape(helper_Mpsi(Lv[j])) 
            a= helper_dot(w,Lv[j]) 
            w= helper_lincomb(1,helper_lincomb(1,w,-a,Lv[j]),-b,Lv[j-1]) 
            
            #Creates tridiagonal matrix M_kril using a and b values
            M_kril[j,j]=a
            M_kril[j-1,j]=b
            M_kril[j,j-1]=b
        
        #Compute sqrt(M)*psi for the fist iterations and define stepnorm    
        # print()
        # print()
        
        # print(Lv)
        # print()
        # print(M_kril)
        # print()
        # print()
        
        betae1 = np.zeros(j+1)
        betae1[0] = 1*psinorm
        M12_psi_old = helper_dot(Lv[:(j+1),:].T,helper_dot(helper_sqrtm(M_kril[:(j+1),:(j+1)]),betae1))
        stepnorm = 1.
        
        while(stepnorm>0.001):
            j += 1
            if(j>k):
                print('Max Lanczos iterations reached, stepnorm is ',stepnorm)
                break
            
            b= helper_sqrt(helper_dot(w,(w)))
            Lv[j]= helper_multip(w,1/b)
            w = helper_reshape(helper_Mpsi(Lv[j]))
            a= helper_dot(w,Lv[j]) 
            w= helper_lincomb(1,helper_lincomb(1,w,-a,Lv[j]),-b,Lv[j-1]) 
            
            #Creates tridiagonal matrix M_kril using a and b values
            M_kril[j,j]=a
            M_kril[j-1,j]=b
            M_kril[j,j-1]=(b)
        
            #Compute sqrt(M)*psi for the fist iterations and define stepnorm    
            betae1 = np.zeros(j+1)
            betae1[0] = 1*psinorm
            
            # print((Lv[(j+1),:]).shape, (helper_sqrtm(M_kril[:(j+1),:(j+1)])).shape, betae1.shape)
            M12_psi = helper_dot(Lv[:(j+1),:].T,helper_dot(helper_sqrtm(M_kril[:(j+1),:(j+1)]),betae1))
            stepnorm = np.sqrt(helper_norm(helper_lincomb(1,M12_psi, -1, M12_psi_old)) / psiMpsi)
            M12_psi_old = M12_psi
            #DEBUG
            # print(j, stepnorm, 'this are the iteration number and the associated stepnorm')
        
        n_iter_Lanczos_ff = j
        
        #DEBUG
        # print(j, stepnorm, 'ITERATION NUMBER and STEPNORM')
    
        return helper_multip(M12_psi, np.sqrt(2.0*kT/dt)), n_iter_Lanczos_ff
    
    ############################################################################################################################################################
    # start_time = time.time()
    
    
    #Wave Space Part
    lin_velocities, ang_vel_and_strain = helper_wavespace_calc(random_array_wave)

    
    # end_time = time.time()
    # print('time needed for wavewspace thermal velocity is ', end_time-start_time)
    ############################################################################################################################################################
    ### Real Space part
    # start_time = time.time()
    
    # #Scale random numbers from [0,1] to [-sqrt(3),sqrt(3)]
    # random_array_real = (2*random_array_real-1)*jnp.sqrt(3.)

    # Matrix_M = np.zeros((11*N,11*N)); basis_vectors = np.eye(11*N, dtype = float)
    # for iii in range(11*N):
    #     a = helper_Mpsi(basis_vectors[iii,:]); Mei = helper_reshape(a); Matrix_M[:,iii] =  Mei
    # sqrt_M = scipy.linalg.sqrtm(Matrix_M); 
    # M12psi_debug = helper_dot(sqrt_M,random_array_real* jnp.sqrt(2.0*kT/dt))
    
    M12psi, n_iter_Lanczos_ff = Lanczos_sqrtm(random_array_real, 100, n_iter_Lanczos_ff)
    
    # end_time = time.time()
    # print('Time RealSpace Brown: ', end_time-start_time, ' error=', np.linalg.norm(M12psi_debug-M12psi)/np.linalg.norm(M12psi_debug))
    ############################################################################################################################################################
    
    
    
    return convert_to_generalized(lin_velocities,ang_vel_and_strain,M12psi), n_iter_Lanczos_ff





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
    # return jnp.array(normal_indices, int),jnp.array(normal_conj_indices, int),jnp.array(nyquist_indices, int)
    # return jnp.asarray(normal_indices, dtype=int),jnp.asarray(normal_conj_indices, dtype=int),jnp.asarray(nyquist_indices, dtype=int)
    # return np.asarray(normal_indices, dtype=int),np.asarray(normal_conj_indices, dtype=int),np.asarray(nyquist_indices, dtype=int)
    
    # print(((np.array(normal_indices)).T))
    # print(((np.array(normal_indices))))
    
    return np.array(normal_indices),np.array(normal_conj_indices),np.array(nyquist_indices)

