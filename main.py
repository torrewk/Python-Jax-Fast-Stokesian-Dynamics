import math, time, os
import numpy as np
import jax.numpy as jnp
import jax.scipy as jscipy
from jax_md import space
from jax.config import config
from jax import random, jit
import Resistance, Mobility, Thermal, Helper, AppliedForces, Shear, gmres, EwaldTables
config.update("jax_enable_x64", True)
np.set_printoptions(precision=8,suppress=True)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false' #line needed to avoid JAX allocating most of the GPU memory even if not needed



#This function wraps all the components and integrate the particles equation of motions forward in time
#While the simulation is performed, trajectories data are saved into a .npy file
def main(
         Nsteps,          # int (number of timesteps)
         writing_period,  # int (period for writing to file) 
         dt,              # float (timestep)
         Lx, Ly, Lz,      # float (Box sizes)
         N,               # int (number of particles)
         max_strain,      # float (max strain applied to the box)
         T,               # float (termal energy)
         a,               # float (particle radius)
         xi,              # float (Ewald split parameter)
         error,           # float (tolerance error)
         U,               # float (interaction strength)
         U_cutoff,        # float (distance cutoff for interacting particles)
         potential,       # string (type of interactions among particles)
         positions,       # float ( (N,3) array of particles initial positions)
         seed_RFD,        # int (seed for Brownian Drift calculation)
         seed_ffwave,     # int (seed for wave space part of far-field velocity slip
         seed_ffreal,     # int (seed for real space part of far-field velocity slip
         seed_nf,         # int (seed for near-field random forces 
         shear_rate_0,    # float (axisymmetric shear rate amplitude)
         shear_freq,      # float (frequency of shear, set to zero to have simple shear)
         output           # string (file name for output)   
         ):

    
    
    #Update particle positions and neighbor lists
    @jit
    def update_positions(shear_rate,                        # shear rate at current time
                         positions,                         # (N,3) array of particles positions
                         displacements_vector_matrix,       # (N,N,3) array of relative displacements between particles 
                         nbrs_lub, nbrs_lub_prec, nbrs_ff,  # Neighbor lists 
                         net_vel,                           # (6*N) array of linear/angular velocities relative to the background flow
                         dt):                               # timestep
        
        #Define array of displacement r(t+dt)-r(t)
        dR = jnp.zeros((N,3),float)
        #Compute actual displacement due to velocities (relative to background flow)
        dR = dR.at[:,0].set(dt*net_vel.at[(0)::6].get())
        dR = dR.at[:,1].set(dt*net_vel.at[(1)::6].get())
        dR = dR.at[:,2].set(dt*net_vel.at[(2)::6].get())
        #Apply displacement and compute wrapped shift (Lees Edwards boundary conditions)
        positions = shift(positions+jnp.array([Lx,Ly,Lz])/2,dR) #shift system origin to (0,0,0)
        positions = positions - jnp.array([Lx,Ly,Lz])*0.5 #re-shift origin back to box center
        
        #Define array of displacement r(t+dt)-r(t) (this time for displacement given by background flow)
        dR = jnp.zeros((N,3),float)
        dR = dR.at[:,0].set(dt * shear_rate * positions.at[:,1].get()) #Assuming y:gradient direction, x:background flow direction
        positions = shift(positions+jnp.array([Lx,Ly,Lz])/2,dR) #Shift system origin to (0,0,0)
        positions = positions - jnp.array([Lx,Ly,Lz]) * 0.5 #re-shift origin back to box center
        
        #Compute new relative displacements between particles 
        displacements_vector_matrix = (
            space.map_product(displacement))(positions, positions)
        
        #Update neighbor lists
        nbrs_lub = nbrs_lub.update(positions)
        nbrs_lub_prec = nbrs_lub_prec.update(positions)
        nbrs_ff = nbrs_ff.update(positions)
         
        return positions, displacements_vector_matrix, nbrs_lub, nbrs_lub_prec, nbrs_ff
    
    
    
    #Solve the linear system Ax=b 
        #A: saddle point matrix
        #b: applied_forces, thermal_noise, R_SU and strain terms
        #x: particle linear/angular velocities and stresslet
    @jit
    def solver(
            rhs,              # RHS vector of the linear system Ax=b
            gridk,            # (Nx,Ny,Nz,4) array of wave vectors and scaling factors for far-field wavespace calculation
            RFU_pre_low_tri,  # lower triangular Cholesky factor of R_fu (built only from particle pairs very close)
            precomputed       # quantities needed to iteratively solve the linear system, compute only once
            ):
       
        
        #Operator acting on x and returning A*x (without A in matrix form) 
        def compute_saddle(x):

            # set output to zero to start
            Ax = jnp.zeros(N*axis,float)

            # compute far-field contribution (M * F): output velocities+torques (first 6N) and strain rates (last 5N)
            Ax = Ax.at[:11*N].set(Mobility.GeneralizedMobility(N,int(Nx),int(Ny),int(Nz),
                                                               int(gaussP),gridk,m_self,
                                                               all_indices_x,all_indices_y,all_indices_z,gaussian_grid_spacing1,gaussian_grid_spacing2,
                                                               r,indices_i,indices_j,f1,f2,g1,g2,h1,h2,h3,
                                                               x.at[:11*N].get()))

            # Add B*U (M*F + B*U): modify the first 6N entries of output
            Ax = Ax.at[:6*N].add(x.at[11*N:].get())
                       
            # compute near-field contribution (- R^nf_FU * U)
            Ax = Ax.at[11*N:].set((Resistance.ComputeLubricationFU(x[11*N:],indices_i_lub,indices_j_lub,ResFunctions,r_lub,N )) * (-1))
            
            # Add (B^T * F - RFU * U): modify the last 6N entries of output
            Ax = Ax.at[11*N:].add(x.at[:6*N].get())
            
            return Ax


        #Precondition operator that approximate the action of A^(-1)
        def compute_precond(x):

            # set output to zero to start
            Px = jnp.zeros(17*N,float)

            # action of precondition matrix on the first 11*N entries of x is the same as the 
            # identity (indeed, the identity is the precondition matrix for the far field granmobility M)
            Px = Px.at[:11*N].set(x[:11*N])

            # -R_FU^-1 * x[:6N]
            # # First solve L^T * y = x, for y, where L^T is the lower triangular Chol. factor of R^nf_FU
            buffer = jscipy.linalg.solve_triangular(RFU_pre_low_tri, x.at[:6*N].get(), lower=True)
            # # Then solve L * z = y, for z, where L is the upper triangular Chol. factor of R^nf_FU
            buffer = jscipy.linalg.solve_triangular(jnp.transpose(RFU_pre_low_tri), buffer, lower=False)
            Px = Px.at[:6*N].add(-buffer)
            Px = Px.at[11*N:].add(buffer)

            # -R_FU^-1 * x[11N:]
            # # First solve L^T * y = x, for y, where L^T is the lower triangular Chol. factor of R^nf_FU
            buffer = jscipy.linalg.solve_triangular(RFU_pre_low_tri, x.at[11*N:].get(), lower=True)
            # # Then solve L * z = y, for z, where L is the upper triangular Chol. factor of R^nf_FU
            buffer = jscipy.linalg.solve_triangular(jnp.transpose(RFU_pre_low_tri), buffer, lower=False)
            Px = Px.at[:6*N].add(buffer)
            Px = Px.at[11*N:].add(-buffer)
            
            return Px
        
        
        #Extract the quantities for the calculation from input 
        (all_indices_x,all_indices_y,all_indices_z,gaussian_grid_spacing1,gaussian_grid_spacing2,
        r,indices_i,indices_j,f1,f2,g1,g2,h1,h2,h3,
        r_lub,indices_i_lub,indices_j_lub,ResFunctions) = precomputed
        
        
        #Define size of saddle point matrix
        axis = 17 #TODO: make 'axis' input variable in order to switch between BD-RPY-SD (or a combination of them)

        #Solve the linear system Ax= b 
        # actually this solves M*Ax = M*b with M the precondition matrix ( M approx A^(-1))
        x = gmres.gmres(A=compute_saddle, b=rhs, x0=None, n=5, M=compute_precond)    
            
        return x


    
    
    start_time = time.time() #Needed to evaluate Time-Steps-per-Second (TPS)
    #######################################################################
    # Compute the Real Space cutoff for the Ewald Summation in the Far-Field computation
    ewald_cut = jnp.sqrt(- jnp.log(error)) / xi
    # parameter needed to make Chol. decomposition of R_FU converge (initially to 1)
    ichol_relaxer = 1.0
    kmax = int(2.0 * jnp.sqrt(- jnp.log(error)) * xi) + 1  # Max wave number
    # Set number of Lanczos iterations (initially) to 5, for both far- and near-field
    n_iter_Lanczos_ff = 2
    n_iter_Lanczos_nf = 2
    xisq = xi * xi
    # Compute number of grid points in k space
    Nx, Ny, Nz = Helper.Compute_k_gridpoint_number(kmax,Lx,Ly,Lz)
    gridh = jnp.array([Lx, Ly, Lz]) / jnp.array([Nx, Ny, Nz])  # Set Grid Spacing
    quadW = gridh[0] * gridh[1] * gridh[2]
    xy = 0. #set box tilt factor to zero to begin (unsheared box)
    
    # Check that ewald_cut is small enough to avoid interaction with periodic images)
    Helper.Check_ewald_cut(ewald_cut,Lx,Ly,Lz,error)
    
    # (Check) Maximum eigenvalue of A'*A to scale support, P, for spreading on deformed grids
    eta, gaussP = Helper.Check_max_shear(gridh, xisq, Nx, Ny, Nz, max_strain, error)
    prefac = (2.0 * xisq / 3.1415926536 / eta) * jnp.sqrt(2.0 * xisq / 3.1415926536 / eta)
    expfac = 2.0 * xisq / eta
    gaussPd2 = jnp.array(gaussP/2, int)
    
    # Get list of reciprocal space vectors, and scaling factor for the wave space calculation
    gridk = Shear.compute_sheared_grid(int(Nx),int(Ny),int(Nz),xy,Lx,Ly,Lz,eta,xisq)   
    #######################################################################

    ## Set INITIAL Periodic Space and Displacement Metric
    #Box is defined by an upper triangular matrix of the form:
    #    [Lx  Ly*xy  Lz*xz]
    #    [0   Ly     Lz*yz]
    #    [0   0      Lz   ]
    displacement, shift = space.periodic_general(
        jnp.array([[Lx, Ly*xy, Lz*0.],[0., Ly, Lz*0.],[0., 0., Lz]]), fractional_coordinates=False)
    
    # compute matrix of INITIAL displacements between particles (each element is a vector from particle j to i)
    displacements_vector_matrix = (
        space.map_product(displacement))(positions, positions)
    
    #######################################################################
    # Store the coefficients for the real space part of Ewald summation
    # Will precompute scaling factors for real space component of summation for a given discretization to speed up GPU calculations
    # NOTE: Potential sensitivity of real space functions at small xi, tabulation computed in double prec., then truncated to single

    ewald_dr = 0.001  # Distance resolution
    ewald_n = ewald_cut / ewald_dr - 1  # Number of entries in tabulation
    
    # Factors needed to compute self contribution
    pi12 = 1.77245385091  # square root of pi
    pi = 3.1415926536  # pi
    a = a *1.0 # radius
    axi = a * xi  # a * xi
    axi2 = axi * axi  # ( a * xi )^2

    # Compute self contribution
    m_self = jnp.zeros(2,float)
    m_self = m_self.at[0].set(
        (1 + 4*pi12*axi*math.erfc(2.*axi) - math.exp(-4.*axi2))/(4.*pi12*axi*a))
    
    
    m_self = m_self.at[1].set( (-3.*math.erfc(2.*a*xi)*math.pow(a,-3.))/10. - (3.*math.pow(a,-6.)*math.pow(pi,-0.5)*math.pow(xi,-3.))/80.
                              -  (9.*math.pow(a, -4)*math.pow(pi, -0.5)*math.pow(xi, -1))/40 
                              + (3.*math.exp(-4 * math.pow(a, 2)*math.pow(xi, 2))*math.pow(a, -6)*math.pow(pi, -0.5)*math.pow(xi, -3)
                                  *(1+10 * math.pow(a, 2)*math.pow(xi, 2))) / 80)
    #######################################################################
    # Create real space Ewald table
    nR = int(ewald_n + 1)  # number of entries in ewald table
    ewaldC1 = EwaldTables.Compute_real_space_ewald_table(nR,a,xi) #this version uses numpy long float
    ewaldC1 = jnp.array(ewaldC1)
    
    #######################################################################
    # Load resistance table
    ResTable_dist = jnp.load('ResTableDist.npy')
    ResTable_vals = jnp.load('ResTableVals.npy')
    ResTable_min = 0.0001000000000000
    ResTable_dr = 0.0043050000000000 # Table discretization (log space), i.e. ind * dr.set(log10( h[ind] / min )
    #######################################################################
    AppliedForce = jnp.zeros(3*N,float)
    AppliedTorques = jnp.zeros(3*N,float)
    ######################################################################
    # Initialize neighborlists
    lub_neighbor_fn, prec_lub_neighbor_fn, ff_neighbor_fn, pot_neighbor_fn = Helper.initialize_neighborlist(
        U_cutoff, Lx, Ly, Lz, displacement, ewald_cut)

    # Allocate lists for first time - they will be updated at each timestep and if needed, re-allocated too.
    nbrs_lub = lub_neighbor_fn.allocate(positions+jnp.array([Lx,Ly,Lz])/2)
    nbrs_lub_prec = prec_lub_neighbor_fn.allocate(positions+jnp.array([Lx,Ly,Lz])/2)
    nbrs_ff = ff_neighbor_fn.allocate(positions+jnp.array([Lx,Ly,Lz])/2)
    nl_lub = np.array(nbrs_lub.idx)
    nl_ff = np.array(nbrs_ff.idx)
    #######################################################################  
    trajectory = np.zeros((int(Nsteps/writing_period + 1),N,3),float)
    
    # precompute grid distances for FFT (same for each gaussian support)
    gaussian_grid_spacing = Helper.Precompute_grid_distancing(gaussP, gridh[0], xy, positions,N,Nx,Ny,Nz,Lx,Ly,Lz)

    #create RNG states
    key_RFD = random.PRNGKey(seed_RFD)
    key_ffwave = random.PRNGKey(seed_ffwave)
    key_ffreal = random.PRNGKey(seed_ffreal)
    key_nf = random.PRNGKey(seed_nf)
    
    #define epsilon for RFD
    epsilon = error
    
    # #create indices list for spreading random forces on the wave space grid (far field brownian velocity, wave space calculation)    
    (normal_indices_x,normal_indices_y,normal_indices_z,
    normal_conj_indices_x,normal_conj_indices_y,normal_conj_indices_z,
    nyquist_indices_x,nyquist_indices_y,nyquist_indices_z) = Thermal.Random_force_on_grid_indexing(Nx,Ny,Nz)
    
    end_time = time.time(); elapsed_time = end_time - start_time
    print('Compilation of first part took ',elapsed_time, 'seconds')
    
    total_time_start = time.time()
    
    for step in range(Nsteps):
        
        full_start_time = time.time()

        #initialize generalized velocity (6*N array, for linear and angular components)
        general_velocity = jnp.zeros(6*N, float)
        
        #Define arrays for the saddle point solvers (solve linear system Ax=b, and A is the saddle point matrix)
        saddle_b = jnp.zeros(17*N,float)
        
        # Precompute quantities for far-field and near-field calculation
        (all_indices_x,all_indices_y,all_indices_z,gaussian_grid_spacing1,gaussian_grid_spacing2,
        r,indices_i,indices_j,f1,f2,g1,g2,h1,h2,h3,
        r_lub,indices_i_lub,indices_j_lub,ResFunction) = Helper.precompute(positions,gaussian_grid_spacing,nl_ff,nl_lub,displacements_vector_matrix,xy,
                                                                    N,Lx,Ly,Lz,Nx,Ny,Nz,
                                                                    prefac,expfac,quadW,
                                                                    int(gaussP),gaussPd2,
                                                                    ewald_n,ewald_dr,ewald_cut,ewaldC1,
                                                                    ResTable_min,ResTable_dr,ResTable_dist,ResTable_vals)
        
        diagonal_zeroes_for_brownian = Thermal.Number_of_neigh(N, indices_i_lub, indices_j_lub)
        #compute precondition resistance lubrication matrix
        R_fu_prec_lower_triang, diagonal_elements_for_brownian = Resistance.RFU_Precondition(
            ichol_relaxer,
            displacements_vector_matrix.at[np.array(nbrs_lub_prec.idx)[0,:],np.array(nbrs_lub_prec.idx)[1,:]].get(),
            N,
            len(nbrs_lub_prec.idx[0]),
            np.array(nbrs_lub_prec.idx)
            )

        #compute shear-rate for current timestep: simple(shear_freq=0) or oscillatory(shear_freq>0)
        shear_rate = Shear.update_shear_rate(dt, step, shear_rate_0, shear_freq, phase=0)
        
        # If T>0, compute Brownian Drift and use it to initialize the velocity
        if(T>0): 
            
            key_RFD, random_array = Helper.generate_random_array(key_RFD,6*N) #get array of random variables
            random_array = -((2*random_array-1)*jnp.sqrt(3))
            #Add random displacement to RHS of linear system
            saddle_b = saddle_b.at[11*N:].set(random_array)
            
            #Perform a displacement in the positive random directions and save it to a buffer
            buffer_positions, buffer_displacements_vector_matrix, buffer_nbrs_lub, buffer_nbrs_lub_prec, buffer_nbrs_ff = update_positions(shear_rate,positions, displacements_vector_matrix, nbrs_lub, nbrs_lub_prec, nbrs_ff, random_array,epsilon/2.)
            buffer_gaussian_grid_spacing = Helper.Precompute_grid_distancing(gaussP, gridh[0], xy, buffer_positions,N,Nx,Ny,Nz,Lx,Ly,Lz)
            buffer_nl_lub = np.array(buffer_nbrs_lub.idx)
            buffer_nl_ff = np.array(buffer_nbrs_ff.idx)     

            #Solve the saddle point problem in the positive direction
            saddle_x = solver(
                saddle_b,  # rhs vector of the linear system
                gridk,
                R_fu_prec_lower_triang,
                Helper.precompute(buffer_positions,buffer_gaussian_grid_spacing,buffer_nl_ff,buffer_nl_lub,buffer_displacements_vector_matrix,xy,
                           N,Lx,Ly,Lz,Nx,Ny,Nz,
                           prefac,expfac,quadW,
                           gaussP,gaussPd2,
                           ewald_n,ewald_dr,ewald_cut,ewaldC1,
                           ResTable_min,ResTable_dr,ResTable_dist,ResTable_vals)[::]
                )
            general_velocity = saddle_x.at[11*N:].get()

            
            #Perform a displacement in the negative random directions and save it to a buffer
            buffer_positions, buffer_displacements_vector_matrix, buffer_nbrs_lub, buffer_nbrs_lub_prec, buffer_nbrs_ff = update_positions(shear_rate,positions, displacements_vector_matrix, nbrs_lub, nbrs_lub_prec, nbrs_ff, random_array, -epsilon/2.)
            buffer_gaussian_grid_spacing = Helper.Precompute_grid_distancing(gaussP, gridh[0], xy, buffer_positions,N,Nx,Ny,Nz,Lx,Ly,Lz)
            buffer_nl_lub = np.array(buffer_nbrs_lub.idx)
            buffer_nl_ff = np.array(buffer_nbrs_ff.idx)     
            
            #Solve the saddle point problem in the negative direction
            saddle_x = solver(
                saddle_b,  # rhs vector of the linear system
                gridk,
                R_fu_prec_lower_triang,
                Helper.precompute(buffer_positions,buffer_gaussian_grid_spacing,buffer_nl_ff,buffer_nl_lub,buffer_displacements_vector_matrix,xy,
                                  N,Lx,Ly,Lz,Nx,Ny,Nz,
                                  prefac,expfac,quadW,
                                  gaussP,gaussPd2,
                                  ewald_n,ewald_dr,ewald_cut,ewaldC1,
                                  ResTable_min,ResTable_dr,ResTable_dist,ResTable_vals)
                )
            
            #Take Difference and apply scaling
            general_velocity += (-saddle_x.at[11*N:].get())  
            general_velocity = general_velocity * T / epsilon 
            
            #Reset RHS to zero for next saddle point solver 
            saddle_b = jnp.zeros(17*N,float) 
        
        
        #Add applied forces and conservative (pair potential) forces
        saddle_b = AppliedForces.AppliedForces(N, AppliedForce, AppliedTorques, saddle_b, U,
                                               indices_i_lub,indices_j_lub,displacements_vector_matrix,
                                               U_cutoff)
        
        #Add the (-) ambient rate of strain to the right-hand side
        saddle_b = saddle_b.at[(6*N+1):(11*N):5].add(-shear_rate) #see 'computational tricks' from Andrew Fiore's paper
        
        #Compute near field shear contribution R_FE and add it to the rhs of the system 
        saddle_b = saddle_b.at[11*N:].add(Resistance.compute_RFE(N,shear_rate,r_lub,indices_i_lub,indices_j_lub,ResFunction[11],ResFunction[12],
                                                      ResFunction[13],ResFunction[14],ResFunction[15],ResFunction[16],
                                                      -ResFunction[12],-ResFunction[14],ResFunction[16]))
        
        # If T>0, compute Thermal Fluctuations
        if(T>0):
            
            #Generate random numbers for far-field random forces
            key_ffwave, random_array_wave = Helper.generate_random_array(key_ffwave, (3 * 2 * len(normal_indices_x)+ 3 * len(nyquist_indices_x)) )
            key_ffreal, random_array_real = Helper.generate_random_array(key_ffreal, (11*N) )
            key_nf, random_array_nf = Helper.generate_random_array(key_nf, (6*N) )
            
            
            #Compute far-field (wave space contribution) slip velocity and set in rhs of linear system
            ws_linvel, ws_angvel_strain = Thermal.compute_wave_space_slipvelocity(N,m_self,int(Nx),int(Ny),int(Nz),int(gaussP),T,dt,gridh,
                                              normal_indices_x,normal_indices_y,normal_indices_z,
                                              normal_conj_indices_x,normal_conj_indices_y,normal_conj_indices_z,
                                              nyquist_indices_x,nyquist_indices_y,nyquist_indices_z,
                                              gridk,random_array_wave,all_indices_x,all_indices_y,all_indices_z,
                                              gaussian_grid_spacing1,gaussian_grid_spacing2)
            
            #Compute far-field (real space contribution) slip velocity and set in rhs of linear system
            rs_linvel, rs_angvel_strain, stepnormff = Thermal.compute_real_space_slipvelocity(
                N,m_self,T,dt,int(n_iter_Lanczos_ff),
                random_array_real,r,indices_i,indices_j,
                f1,f2,g1,g2,h1,h2,h3)
            while((stepnormff>0.0001) and (n_iter_Lanczos_ff<100)):
                n_iter_Lanczos_ff += 15
                rs_linvel, rs_angvel_strain, stepnormff = Thermal.compute_real_space_slipvelocity(
                    N,m_self,T,dt,int(n_iter_Lanczos_ff),
                    random_array_real,r,indices_i,indices_j,
                    f1,f2,g1,g2,h1,h2,h3)
            
            saddle_b = saddle_b.at[:11*N].add(Thermal.convert_to_generalized(N, ws_linvel, rs_linvel, ws_angvel_strain, rs_angvel_strain))


            #Compute near-field random forces and set in rhs of linear system
            buffer, stepnormnf = Thermal.compute_nearfield_brownianforce(N,T,dt,
                    random_array_nf,
                    r_lub,indices_i_lub,indices_j_lub,
                    ResFunction[0], ResFunction[1], ResFunction[2], ResFunction[3], ResFunction[4], ResFunction[5], 
                    ResFunction[6], ResFunction[7], ResFunction[8], ResFunction[9], ResFunction[10],
                    diagonal_elements_for_brownian,
                    R_fu_prec_lower_triang,
                    diagonal_zeroes_for_brownian,
                    n_iter_Lanczos_nf)
            while((stepnormnf>0.0001) and (n_iter_Lanczos_nf<100)):
                n_iter_Lanczos_nf += 15
                buffer, stepnormnf = Thermal.compute_nearfield_brownianforce(N,T,dt,
                        random_array_nf,
                        r_lub,indices_i_lub,indices_j_lub,
                        ResFunction[0], ResFunction[1], ResFunction[2], ResFunction[3], ResFunction[4], ResFunction[5], 
                        ResFunction[6], ResFunction[7], ResFunction[8], ResFunction[9], ResFunction[10],
                        diagonal_elements_for_brownian,
                        R_fu_prec_lower_triang,
                        diagonal_zeroes_for_brownian,
                        n_iter_Lanczos_nf)
            saddle_b = saddle_b.at[11*N:].add(buffer)


        #Solve the system Ax=b, where x contains the particle velocities (relative to the background flow) and stresslet
        saddle_x = solver(
            saddle_b,
            gridk,
            R_fu_prec_lower_triang,
            [all_indices_x,all_indices_y,all_indices_z,gaussian_grid_spacing1,gaussian_grid_spacing2,
            r,indices_i,indices_j,f1,f2,g1,g2,h1,h2,h3,
            r_lub,indices_i_lub,indices_j_lub,ResFunction]
            )
        
        #Update positions        
        (positions, displacements_vector_matrix, 
         nbrs_lub, nbrs_lub_prec, nbrs_ff) = update_positions(shear_rate,
                                                              positions,displacements_vector_matrix,
                                                              nbrs_lub, nbrs_lub_prec, nbrs_ff,
                                                              saddle_x.at[11*N:].get()+general_velocity,
                                                              dt)                                                         
        nl_lub = np.array(nbrs_lub.idx) #extract lists in sparse format
        nl_ff = np.array(nbrs_ff.idx) #extract lists in sparse format
        
        #Update grid distances for FFT 
        gaussian_grid_spacing = Helper.Precompute_grid_distancing(gaussP, gridh[0], xy, positions,N,Nx,Ny,Nz,Lx,Ly,Lz)

        #If system is sheared, update wave vectors grid and box tilt factor
        if(shear_rate_0 != 0):
            xy = Shear.update_box_tilt_factor(dt,shear_rate,xy)
            displacement, shift = space.periodic_general(
                jnp.array([[Lx, Ly*xy, Lz*0.],[0., Ly, Lz*0.],[0., 0., Lz]]), fractional_coordinates=False)
            gridk = Shear.compute_sheared_grid(int(Nx),int(Ny),int(Nz),xy,Lx,Ly,Lz,eta,xisq)  
        
        #Compute compilation time 
        if(step==0):
            compilation_time = time.time() - full_start_time
            print('Compilation Time is ',compilation_time)
            
        #Reset Lacnzos number of iteration once in a while (avoid to do too many iterations if not needed, can be tuned for better performance)
        elif((step%100)==0):
            n_iter_Lanczos_ff=5
            n_iter_Lanczos_nf=5
        
        #Write trajectory to file
        if((step%writing_period)==0):
            trajectory[int(step/writing_period),:,:] = positions
            np.save(output, trajectory)  
            overlaps = Helper.check_overlap(displacements_vector_matrix) #DEBUG: check if particles overlap
            print('Step= ',step,' Overlaps are ',jnp.sum(overlaps)-N) 
            
            
    total_time_end = time.time()
    print('Time for ',Nsteps,' steps is ',total_time_end-total_time_start-compilation_time, 'or ', Nsteps/(total_time_end-total_time_start-compilation_time), ' steps per second')
     
    return 


