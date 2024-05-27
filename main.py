import math
import time
import os
import numpy as np
import jax.numpy as jnp
import jax.scipy as jscipy
import jaxmd_space as space
from jax.config import config
from jax import random, jit
import resistance
import mobility
import thermal
import utils
import appliedForces
import shear
import ewaldTables
config.update("jax_enable_x64", False)
np.set_printoptions(precision=8, suppress=True)
# line needed to avoid JAX allocating most of the GPU memory even if not needed
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


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
    buoyancy_flag,   # int   (set to 1 to have gravity acting on colloids)
    U_cutoff,        # float (distance cutoff for interacting particles)
    positions,       # float ( (N,3) array of particles initial positions)
    seed_RFD,        # int (seed for Brownian Drift calculation)
    seed_ffwave,     # int (seed for wave space part of far-field velocity slip
    seed_ffreal,     # int (seed for real space part of far-field velocity slip
    seed_nf,         # int (seed for near-field random forces
    shear_rate_0,    # float (axisymmetric shear rate amplitude)
    # float (frequency of shear, set to zero to have simple shear)
    shear_freq,
    output,          # string (file name for output)
    stresslet_flag,  # bool (to have stresslet in the output)
    velocity_flag,   # bool (to have velocities in the output)
    orient_flag,     # bool (to have particle orientations in the output)
    constant_applied_forces, # float ( (N,3) array of forces)
    constant_applied_torques  # float ( (N,3) array of torques)
):

    #Update particle positions and neighbor lists
    @jit
    def update_positions(shear_rate,                        # shear rate at current time
                         # (N,3) array of particles positions
                         positions,
                         # (N,N,3) array of relative displacements between particles
                         displacements_vector_matrix,
                         nbrs_lub, nbrs_lub_prec, nbrs_ff,  # Neighbor lists
                         # (6*N) array of linear/angular velocities relative to the background flow
                         net_vel,
                         dt):                               # timestep

        #Define array of displacement r(t+dt)-r(t)
        dR = jnp.zeros((N, 3), float)
        #Compute actual displacement due to velocities (relative to background flow)
        dR = dR.at[:, 0].set(dt*net_vel.at[(0)::6].get())
        dR = dR.at[:, 1].set(dt*net_vel.at[(1)::6].get())
        dR = dR.at[:, 2].set(dt*net_vel.at[(2)::6].get())
        #Apply displacement and compute wrapped shift (Lees Edwards boundary conditions)
        # shift system origin to (0,0,0)
        positions = shift(positions+jnp.array([Lx, Ly, Lz])/2, dR)
        # re-shift origin back to box center
        positions = positions - jnp.array([Lx, Ly, Lz])*0.5

        #Define array of displacement r(t+dt)-r(t) (this time for displacement given by background flow)
        dR = jnp.zeros((N, 3), float)
        # Assuming y:gradient direction, x:background flow direction
        dR = dR.at[:, 0].set(dt * shear_rate * positions.at[:, 1].get())
        # Shift system origin to (0,0,0)
        positions = shift(positions+jnp.array([Lx, Ly, Lz])/2, dR)
        # re-shift origin back to box center
        positions = positions - jnp.array([Lx, Ly, Lz]) * 0.5

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
            # (Nx,Ny,Nz,4) array of wave vectors and scaling factors for far-field wavespace calculation
            gridk,
            # lower triangular Cholesky factor of R_fu (built only from particle pairs very close)
            RFU_pre_low_tri,
            precomputed       # quantities needed to iteratively solve the linear system, compute only once
    ):

        #Operator acting on x and returning A*x (without A in matrix form)
        def compute_saddle(x):

            # set output to zero to start
            Ax = jnp.zeros(N*axis, float)

            # compute far-field contribution (M * F): output velocities+torques (first 6N) and strain rates (last 5N)
            Ax = Ax.at[:11*N].set(mobility.GeneralizedMobility(N, int(Nx), int(Ny), int(Nz),
                                                               int(gaussP), gridk, m_self,
                                                               all_indices_x, all_indices_y, all_indices_z, gaussian_grid_spacing1, gaussian_grid_spacing2,
                                                               r, indices_i, indices_j, f1, f2, g1, g2, h1, h2, h3,
                                                               x.at[:11*N].get()))

            # Add B*U (M*F + B*U): modify the first 6N entries of output
            Ax = Ax.at[:6*N].add(x.at[11*N:].get())

            # compute near-field contribution (- R^nf_FU * U)
            Ax = Ax.at[11*N:].set((resistance.ComputeLubricationFU(x[11*N:],
                                  indices_i_lub, indices_j_lub, ResFunctions, r_lub, N)) * (-1))

            # Add (B^T * F - RFU * U): modify the last 6N entries of output
            Ax = Ax.at[11*N:].add(x.at[:6*N].get())

            return Ax

        #Precondition operator that approximate the action of A^(-1)

        def compute_precond(x):

            # set output to zero to start
            Px = jnp.zeros(17*N, float)

            # action of precondition matrix on the first 11*N entries of x is the same as the
            # identity (indeed, the identity is the precondition matrix for the far field granmobility M)
            Px = Px.at[:11*N].set(x[:11*N])

            # -R_FU^-1 * x[:6N]
            # # First solve L^T * y = x, for y, where L^T is the lower triangular Chol. factor of R^nf_FU
            buffer = jscipy.linalg.solve_triangular(
                RFU_pre_low_tri, x.at[:6*N].get(), lower=True)
            # # Then solve L * z = y, for z, where L is the upper triangular Chol. factor of R^nf_FU
            buffer = jscipy.linalg.solve_triangular(
                jnp.transpose(RFU_pre_low_tri), buffer, lower=False)
            Px = Px.at[:6*N].add(-buffer)
            Px = Px.at[11*N:].add(buffer)

            # -R_FU^-1 * x[11N:]
            # # First solve L^T * y = x, for y, where L^T is the lower triangular Chol. factor of R^nf_FU
            buffer = jscipy.linalg.solve_triangular(
                RFU_pre_low_tri, x.at[11*N:].get(), lower=True)
            # # Then solve L * z = y, for z, where L is the upper triangular Chol. factor of R^nf_FU
            buffer = jscipy.linalg.solve_triangular(
                jnp.transpose(RFU_pre_low_tri), buffer, lower=False)
            Px = Px.at[:6*N].add(buffer)
            Px = Px.at[11*N:].add(-buffer)

            return Px

        #Extract the quantities for the calculation from input
        (all_indices_x, all_indices_y, all_indices_z, gaussian_grid_spacing1, gaussian_grid_spacing2,
         r, indices_i, indices_j, f1, f2, g1, g2, h1, h2, h3,
         r_lub, indices_i_lub, indices_j_lub, ResFunctions) = precomputed

        #Define size of saddle point matrix
        # TODO: make 'axis' input variable in order to switch between BD-RPY-SD (or a combination of them)
        axis = 17

        #Solve the linear system Ax= b
        x, exitCode = jscipy.sparse.linalg.gmres(
            A=compute_saddle, b=rhs, tol=1e-5, restart=25, M=compute_precond)

        return x, exitCode

    start_time = time.time()  # Needed to evaluate Time-Steps-per-Second (TPS)
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
    Nx, Ny, Nz = utils.Compute_k_gridpoint_number(kmax, Lx, Ly, Lz)
    gridh = jnp.array([Lx, Ly, Lz]) / \
        jnp.array([Nx, Ny, Nz])  # Set Grid Spacing
    quadW = gridh[0] * gridh[1] * gridh[2]
    xy = 0.  # set box tilt factor to zero to begin (unsheared box)

    # Check that ewald_cut is small enough to avoid interaction with periodic images)
    utils.Check_ewald_cut(ewald_cut, Lx, Ly, Lz, error)

    # (Check) Maximum eigenvalue of A'*A to scale support, P, for spreading on deformed grids
    eta, gaussP = utils.Check_max_shear(
        gridh, xisq, Nx, Ny, Nz, max_strain, error)
    prefac = (2.0 * xisq / jnp.pi / eta) * jnp.sqrt(2.0 * xisq / jnp.pi / eta)
    expfac = 2.0 * xisq / eta
    gaussPd2 = jnp.array(gaussP/2, int)

    # Get list of reciprocal space vectors, and scaling factor for the wave space calculation
    gridk = shear.compute_sheared_grid(
        int(Nx), int(Ny), int(Nz), xy, Lx, Ly, Lz, eta, xisq)
    #######################################################################

    ## Set INITIAL Periodic Space and Displacement Metric
    #Box is defined by an upper triangular matrix of the form:
    #    [Lx  Ly*xy  Lz*xz]
    #    [0   Ly     Lz*yz]
    #    [0   0      Lz   ]
    displacement, shift = space.periodic_general(
        jnp.array([[Lx, Ly*xy, Lz*0.], [0., Ly, Lz*0.], [0., 0., Lz]]), fractional_coordinates=False)

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
    pi = jnp.pi  # pi
    pi12 = jnp.sqrt(pi)  # square root of pi
    a = a * 1.0  # radius
    axi = a * xi  # a * xi
    axi2 = axi * axi  # ( a * xi )^2

    # Compute self contribution
    m_self = jnp.zeros(2, float)
    m_self = m_self.at[0].set(
        (1 + 4*pi12*axi*math.erfc(2.*axi) - math.exp(-4.*axi2))/(4.*pi12*axi*a))

    m_self = m_self.at[1].set((-3.*math.erfc(2.*a*xi)*math.pow(a, -3.))/10. - (3.*math.pow(a, -6.)*math.pow(pi, -0.5)*math.pow(xi, -3.))/80.
                              - (9.*math.pow(a, -4)*math.pow(pi, -0.5)
                                  * math.pow(xi, -1))/40
                              + (3.*math.exp(-4 * math.pow(a, 2)*math.pow(xi, 2))*math.pow(a, -6)*math.pow(pi, -0.5)*math.pow(xi, -3)
                                  * (1+10 * math.pow(a, 2)*math.pow(xi, 2))) / 80)
    #######################################################################
    # Create real space Ewald table
    nR = int(ewald_n + 1)  # number of entries in ewald table
    ewaldC1 = ewaldTables.Compute_real_space_ewald_table(
        nR, a, xi)  # this version uses numpy long float
    ewaldC1 = jnp.array(ewaldC1)  # convert to single precision (32-bit)

    #######################################################################
    # Load resistance table
    ResTable_dist = jnp.load('files/ResTableDist.npy')
    ResTable_vals = jnp.load('files/ResTableVals.npy')
    ResTable_min = 0.0001000000000000
    # Table discretization (log space), i.e. ind * dr.set(log10( h[ind] / min )
    ResTable_dr = 0.0043050000000000
    #######################################################################
    AppliedForce = jnp.zeros(3*N, float)
    AppliedTorques = jnp.zeros(3*N, float)
    if(buoyancy_flag == 1):  # apply buoyancy forces (in z direction)
        AppliedForce = AppliedForce.at[2::3].add(-1.)
    if(np.count_nonzero(constant_applied_forces) > 0):  # apply buoyancy forces (in z direction)
        AppliedForce += jnp.ravel(constant_applied_forces)
    if(np.count_nonzero(constant_applied_torques) > 0):  # apply buoyancy forces (in z direction)
        AppliedTorques += jnp.ravel(constant_applied_torques)
    ######################################################################
    # Initialize neighborlists
    lub_neighbor_fn, prec_lub_neighbor_fn, ff_neighbor_fn, pot_neighbor_fn = utils.initialize_neighborlist(
        U_cutoff, Lx, Ly, Lz, displacement, ewald_cut)

    # Allocate lists for first time - they will be updated at each timestep and if needed, re-allocated too.
    nbrs_lub = lub_neighbor_fn.allocate(positions+jnp.array([Lx, Ly, Lz])/2)
    nbrs_lub_prec = prec_lub_neighbor_fn.allocate(
        positions+jnp.array([Lx, Ly, Lz])/2)
    nbrs_ff = ff_neighbor_fn.allocate(positions+jnp.array([Lx, Ly, Lz])/2)
    nl_lub = np.array(nbrs_lub.idx)
    nl_ff = np.array(nbrs_ff.idx)
    #######################################################################
    trajectory = np.zeros((int(Nsteps/writing_period), N, 3), float)
    stresslet_history = np.zeros((int(Nsteps/writing_period), N, 5), float)
    velocities = np.zeros((int(Nsteps/writing_period), N, 6), float)
    # precompute grid distances for FFT (same for each gaussian support)
    gaussian_grid_spacing = utils.Precompute_grid_distancing(
        gaussP, gridh[0], xy, positions, N, Nx, Ny, Nz, Lx, Ly, Lz)

    #create RNG states
    key_RFD = random.PRNGKey(seed_RFD)
    key_ffwave = random.PRNGKey(seed_ffwave)
    key_ffreal = random.PRNGKey(seed_ffreal)
    key_nf = random.PRNGKey(seed_nf)

    #define epsilon for RFD
    epsilon = error

    # #create indices list for spreading random forces on the wave space grid (far field brownian velocity, wave space calculation)
    if(T > 0):
        (normal_indices_x, normal_indices_y, normal_indices_z,
         normal_conj_indices_x, normal_conj_indices_y, normal_conj_indices_z,
         nyquist_indices_x, nyquist_indices_y, nyquist_indices_z) = thermal.Random_force_on_grid_indexing(Nx, Ny, Nz)

    #initialize stresslet (for output, not needed to integrate e.o.m.) (5*N array)
    stresslet = jnp.zeros((N, 5), float)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Compilation of first part took ', elapsed_time, 'seconds')
    
    total_time_start = time.time()
    
    # kwt debug: check if particles overlap
    overlaps, overlaps2 = utils.check_overlap(
        displacements_vector_matrix)
    print('Starting: initial overlaps are ', jnp.sum(overlaps)-N)
    
    for step in range(Nsteps):

        full_start_time = time.time()
        #initialize generalized velocity (6*N array, for linear and angular components)
        general_velocity = jnp.zeros(6*N, float) #this array actually stores the brownian drift only

        if((stresslet_flag > 0) and ((step % writing_period) == 0)):
            #reset stresslet (for output, not needed to integrate e.o.m.) (5*N array)
            stresslet = jnp.zeros((N, 5), float)

        #Define arrays for the saddle point solvers (solve linear system Ax=b, and A is the saddle point matrix)
        saddle_b = jnp.zeros(17*N, float)

        # Precompute quantities for far-field and near-field calculation
        (all_indices_x, all_indices_y, all_indices_z, gaussian_grid_spacing1, gaussian_grid_spacing2,
         r, indices_i, indices_j, f1, f2, g1, g2, h1, h2, h3,
         r_lub, indices_i_lub, indices_j_lub, ResFunction) = utils.precompute(positions, gaussian_grid_spacing, nl_ff, nl_lub, displacements_vector_matrix, xy,
                                                                              N, Lx, Ly, Lz, Nx, Ny, Nz,
                                                                              prefac, expfac, quadW,
                                                                              int(gaussP), gaussPd2,
                                                                              ewald_n, ewald_dr, ewald_cut, ewaldC1,
                                                                              ResTable_min, ResTable_dr, ResTable_dist, ResTable_vals)

        diagonal_zeroes_for_brownian = thermal.Number_of_neigh(
            N, indices_i_lub, indices_j_lub)
        #compute precondition resistance lubrication matrix
        R_fu_prec_lower_triang, diagonal_elements_for_brownian = resistance.RFU_Precondition(
            ichol_relaxer,
            displacements_vector_matrix.at[np.array(nbrs_lub_prec.idx)[
                0, :], np.array(nbrs_lub_prec.idx)[1, :]].get(),
            N,
            len(nbrs_lub_prec.idx[0]),
            np.array(nbrs_lub_prec.idx)
        )

        #Perform cholesky factorization and obtain lower triangle cholesky factor of R_fu
        R_fu_prec_lower_triang = utils.chol_fac(R_fu_prec_lower_triang)

        #Compute shear-rate for current timestep: simple(shear_freq=0) or oscillatory(shear_freq>0)
        shear_rate = shear.update_shear_rate(
            dt, step, shear_rate_0, shear_freq, phase=0)
        # If temperature is not zero, compute Brownian Drift
        if(T > 0):

            key_RFD, random_array = utils.generate_random_array(
                key_RFD, 6*N)  # get array of random variables
            random_array = -((2*random_array-1)*jnp.sqrt(3))
            #Add random displacement to RHS of linear system
            saddle_b = saddle_b.at[11*N:].set(random_array)

            #Perform a displacement in the positive random directions and save it to a buffer
            buffer_positions, buffer_displacements_vector_matrix, buffer_nbrs_lub, buffer_nbrs_lub_prec, buffer_nbrs_ff = update_positions(
                shear_rate, positions, displacements_vector_matrix, nbrs_lub, nbrs_lub_prec, nbrs_ff, random_array, epsilon/2.)
            buffer_gaussian_grid_spacing = utils.Precompute_grid_distancing(
                gaussP, gridh[0], xy, buffer_positions, N, Nx, Ny, Nz, Lx, Ly, Lz)
            buffer_nl_lub = np.array(buffer_nbrs_lub.idx)
            buffer_nl_ff = np.array(buffer_nbrs_ff.idx)

            #Solve the saddle point problem in the positive direction
            output_precompute = utils.precompute(buffer_positions, buffer_gaussian_grid_spacing, buffer_nl_ff, buffer_nl_lub, buffer_displacements_vector_matrix, xy,
                                                 N, Lx, Ly, Lz, Nx, Ny, Nz,
                                                 prefac, expfac, quadW,
                                                 gaussP, gaussPd2,
                                                 ewald_n, ewald_dr, ewald_cut, ewaldC1,
                                                 ResTable_min, ResTable_dr, ResTable_dist, ResTable_vals)

            saddle_x, exitcode_gmres = solver(
                saddle_b,  # rhs vector of the linear system
                gridk,
                R_fu_prec_lower_triang,
                output_precompute
            )
            if(exitcode_gmres > 0):
                raise ValueError(
                    f"GMRES (RFD) did not converge! Iterations are {exitcode_gmres}. Abort!")
            general_velocity = saddle_x.at[11*N:].get()

            #Compute the near-field hydrodynamic stresslet
            if((stresslet_flag > 0) and ((step % writing_period) == 0)):
                stresslet = resistance.compute_RSU(stresslet, general_velocity,
                                                   indices_i_lub,
                                                   indices_j_lub,
                                                   output_precompute[18],
                                                   r_lub,
                                                   N)

            #Perform a displacement in the negative random directions and save it to a buffer
            buffer_positions, buffer_displacements_vector_matrix, buffer_nbrs_lub, buffer_nbrs_lub_prec, buffer_nbrs_ff = update_positions(
                shear_rate, positions, displacements_vector_matrix, nbrs_lub, nbrs_lub_prec, nbrs_ff, random_array, -epsilon/2.)
            buffer_gaussian_grid_spacing = utils.Precompute_grid_distancing(
                gaussP, gridh[0], xy, buffer_positions, N, Nx, Ny, Nz, Lx, Ly, Lz)
            buffer_nl_lub = np.array(buffer_nbrs_lub.idx)
            buffer_nl_ff = np.array(buffer_nbrs_ff.idx)

            #Solve the saddle point problem in the negative direction
            output_precompute = utils.precompute(buffer_positions, buffer_gaussian_grid_spacing, buffer_nl_ff, buffer_nl_lub, buffer_displacements_vector_matrix, xy,
                                                 N, Lx, Ly, Lz, Nx, Ny, Nz,
                                                 prefac, expfac, quadW,
                                                 gaussP, gaussPd2,
                                                 ewald_n, ewald_dr, ewald_cut, ewaldC1,
                                                 ResTable_min, ResTable_dr, ResTable_dist, ResTable_vals)
            saddle_x, exitcode_gmres = solver(
                saddle_b,  # rhs vector of the linear system
                gridk,
                R_fu_prec_lower_triang,
                output_precompute
            )
            if(exitcode_gmres > 0):
                raise ValueError(
                    f"GMRES (RFD) did not converge! Iterations are {exitcode_gmres}. Abort!")

            #Compute the near-field hydrodynamic stresslet
            if((stresslet_flag > 0) and ((step % writing_period) == 0)):
                buffer_stresslet = (
                    resistance.compute_RSU(
                        jnp.zeros((N, 5), float),
                        saddle_x.at[11*N:].get(),
                        indices_i_lub,
                        indices_j_lub,
                        output_precompute[18],
                        r_lub,
                        N
                    )
                )
                
            #Take Difference and apply scaling
            general_velocity += (-saddle_x.at[11*N:].get())
            general_velocity = -general_velocity * T / epsilon
            if((stresslet_flag > 0) and ((step % writing_period) == 0)):
                stresslet += -buffer_stresslet
                stresslet = stresslet * T / epsilon

            #Reset RHS to zero for next saddle point solver
            saddle_b = jnp.zeros(17*N, float)
        #Add applied forces and conservative (pair potential) forces
        saddle_b = appliedForces.sumAppliedForces(N, AppliedForce, AppliedTorques, saddle_b, U,
                                                  indices_i_lub, indices_j_lub, displacements_vector_matrix,
                                                  U_cutoff)
        
        #Add the (-) ambient rate of strain to the right-hand side
        # see 'computational tricks' from Andrew Fiore's paper
        if(shear_rate_0 != 0):
            saddle_b = saddle_b.at[(6*N+1):(11*N):5].add(-shear_rate)

            #Compute near field shear contribution R_FE and add it to the rhs of the system
            saddle_b = saddle_b.at[11*N:].add(resistance.compute_RFE(N, shear_rate, r_lub, indices_i_lub, indices_j_lub, ResFunction[11], ResFunction[12],
                                                                     ResFunction[13], ResFunction[14], ResFunction[15], ResFunction[16],
                                                                     -ResFunction[12], -ResFunction[14], ResFunction[16]))
        # Compute Thermal Fluctuations only if temperature is not zero
        if(T > 0):

            #Generate random numbers for far-field random forces
            key_ffwave, random_array_wave = utils.generate_random_array(
                key_ffwave, (3 * 2 * len(normal_indices_x) + 3 * len(nyquist_indices_x)))
            key_ffreal, random_array_real = utils.generate_random_array(
                key_ffreal, (11*N))
            key_nf, random_array_nf = utils.generate_random_array(
                key_nf, (6*N))

            #Compute far-field (wave space contribution) slip velocity and set in rhs of linear system
            ws_linvel, ws_angvel_strain = thermal.compute_wave_space_slipvelocity(N, m_self, int(Nx), int(Ny), int(Nz), int(gaussP), T, dt, gridh,
                                                                                  normal_indices_x, normal_indices_y, normal_indices_z,
                                                                                  normal_conj_indices_x, normal_conj_indices_y, normal_conj_indices_z,
                                                                                  nyquist_indices_x, nyquist_indices_y, nyquist_indices_z,
                                                                                  gridk, random_array_wave, all_indices_x, all_indices_y, all_indices_z,
                                                                                  gaussian_grid_spacing1, gaussian_grid_spacing2)

            #Compute far-field (real space contribution) slip velocity and set in rhs of linear system
            rs_linvel, rs_angvel_strain, stepnormff, diag_ff = thermal.compute_real_space_slipvelocity(
                N, m_self, T, dt, int(n_iter_Lanczos_ff),
                random_array_real, r, indices_i, indices_j,
                f1, f2, g1, g2, h1, h2, h3)
            while((stepnormff > 0.0001) and (n_iter_Lanczos_ff < 150)):
                n_iter_Lanczos_ff += 20
                rs_linvel, rs_angvel_strain, stepnormff, diag_ff = thermal.compute_real_space_slipvelocity(
                    N, m_self, T, dt, int(n_iter_Lanczos_ff),
                    random_array_real, r, indices_i, indices_j,
                    f1, f2, g1, g2, h1, h2, h3)

            saddle_b = saddle_b.at[:11*N].add(thermal.convert_to_generalized(
                N, ws_linvel, rs_linvel, ws_angvel_strain, rs_angvel_strain))

            stepnormnf = 0.
            if(N > 1):  # no lubrication for isolated particles
                #Compute near-field random forces and set in rhs of linear system
                buffer, stepnormnf, diag_nf = thermal.compute_nearfield_brownianforce(N, T, dt,
                                                                                      random_array_nf,
                                                                                      r_lub, indices_i_lub, indices_j_lub,
                                                                                      ResFunction[0], ResFunction[1], ResFunction[
                                                                                          2], ResFunction[3], ResFunction[4], ResFunction[5],
                                                                                      ResFunction[6], ResFunction[7], ResFunction[
                                                                                          8], ResFunction[9], ResFunction[10],
                                                                                      diagonal_elements_for_brownian,
                                                                                      R_fu_prec_lower_triang,
                                                                                      diagonal_zeroes_for_brownian,
                                                                                      n_iter_Lanczos_nf
                                                                                      )
                while((stepnormnf > 0.0001) and (n_iter_Lanczos_nf < 150)):
                    n_iter_Lanczos_nf += 20
                    buffer, stepnormnf, diag_nf = thermal.compute_nearfield_brownianforce(N, T, dt,
                                                                                          random_array_nf,
                                                                                          r_lub, indices_i_lub, indices_j_lub,
                                                                                          ResFunction[0], ResFunction[1], ResFunction[
                                                                                              2], ResFunction[3], ResFunction[4], ResFunction[5],
                                                                                          ResFunction[6], ResFunction[7], ResFunction[
                                                                                              8], ResFunction[9], ResFunction[10],
                                                                                          diagonal_elements_for_brownian,
                                                                                          R_fu_prec_lower_triang,
                                                                                          diagonal_zeroes_for_brownian,
                                                                                          n_iter_Lanczos_nf
                                                                                          )
                saddle_b = saddle_b.at[11*N:].add(-buffer)
            #check that thermal fluctuation calculation went well
            if ((not math.isfinite(stepnormff)) or ((n_iter_Lanczos_ff > 150) and (stepnormff > 0.0001))):
                raise ValueError(
                    f"Far-field Lanczos did not converge! Stepnorm is {stepnormff}, iterations are {n_iter_Lanczos_ff}. Eigenvalues of tridiagonal matrix are {diag_ff}. Abort!")

            if ((not math.isfinite(stepnormnf)) or ((n_iter_Lanczos_nf > 150) and (stepnormnf > 0.0001))):
                raise ValueError(
                    f"Near-field Lanczos did not converge! Stepnorm is {stepnormnf}, iterations are {n_iter_Lanczos_nf}. Eigenvalues of tridiagonal matrix are {diag_nf}. Abort!")
        
        
        #Solve the system Ax=b, where x contains the particle velocities (relative to the background flow) and stresslet
        saddle_x, exitcode_gmres = solver(
            saddle_b,
            gridk,
            R_fu_prec_lower_triang,
            [all_indices_x, all_indices_y, all_indices_z, gaussian_grid_spacing1, gaussian_grid_spacing2,
             r, indices_i, indices_j, f1, f2, g1, g2, h1, h2, h3,
             r_lub, indices_i_lub, indices_j_lub, ResFunction]
        )
        if(exitcode_gmres > 0):
            raise ValueError(
                f"GMRES did not converge! Iterations are {exitcode_gmres}. Abort!")
        
        
        #Add the near-field contributions to the stresslet
        if((stresslet_flag > 0) and ((step % writing_period) == 0)):
            #get stresslet out of saddle point solution (and add it to the contribution from the Brownian drift, if temperature>0 )
            stresslet += jnp.reshape(-saddle_x[6*N:11*N], (N, 5))
            stresslet = resistance.compute_RSU(stresslet, saddle_x.at[11*N:].get(),
                                                indices_i_lub,
                                                indices_j_lub,
                                                ResFunction,
                                                r_lub,
                                                N)
            # Add shear near-field contributions to the stresslet
            if(shear_rate_0 != 0):
                stresslet = resistance.compute_RSE(N, shear_rate, r_lub, indices_i_lub, indices_j_lub, ResFunction[17], ResFunction[18],
                                                    ResFunction[19], ResFunction[20], ResFunction[21], ResFunction[22], stresslet)
            #save stresslet
            stresslet_history[int(step/writing_period), :, :] = stresslet
            if(output != 'None'):
                np.save('stresslet_'+output, stresslet_history)

        #Update positions
        (positions, displacements_vector_matrix,
         nbrs_lub, nbrs_lub_prec, nbrs_ff) = update_positions(shear_rate,
                                                              positions, displacements_vector_matrix,
                                                              nbrs_lub, nbrs_lub_prec, nbrs_ff,
                                                              saddle_x.at[11*N:].get() +
                                                              general_velocity,
                                                              dt)
        nl_lub = np.array(nbrs_lub.idx)  # extract lists in sparse format
        nl_ff = np.array(nbrs_ff.idx)  # extract lists in sparse format

        #Update grid distances for FFT
        gaussian_grid_spacing = utils.Precompute_grid_distancing(
            gaussP, gridh[0], xy, positions, N, Nx, Ny, Nz, Lx, Ly, Lz)

        #If system is sheared, update wave-vectors grid and box tilt factor
        if(shear_rate_0 != 0):
            xy = shear.update_box_tilt_factor(
                dt, shear_rate_0, xy, step, shear_freq)
            displacement, shift = space.periodic_general(
                jnp.array([[Lx, Ly*xy, Lz*0.], [0., Ly, Lz*0.], [0., 0., Lz]]), fractional_coordinates=False)
            gridk = shear.compute_sheared_grid(
                int(Nx), int(Ny), int(Nz), xy, Lx, Ly, Lz, eta, xisq)

        #Compute compilation time
        if(step == 0):
            compilation_time = time.time() - full_start_time
            print('Compilation Time is ', compilation_time)

        #Reset Lacnzos number of iteration once in a while (avoid to do too many iterations if not needed, can be tuned for better performance)
        elif((step % 100) == 0):
            n_iter_Lanczos_ff = 5
            n_iter_Lanczos_nf = 5
            
            
        #Write trajectory to file
        if((step % writing_period) == 0):

            #check that the position to save do not contain 'nan' or 'inf'
            if((jnp.isnan(positions)).any() or (jnp.isinf(positions)).any()):
                raise ValueError(
                    "Invalid particles positions. Abort!")

            #save position to trajectory (and to file)
            trajectory[int(step/writing_period), :, :] = positions
            if(output != 'None'):
                np.save(output, trajectory)
                
            #store velocity (linear and angular)
            if( (velocity_flag > 0) ):
                velocities[int(step/writing_period), :, :] = jnp.reshape(saddle_x.at[11*N:].get() + general_velocity, (N,6))
                if (output != 'None'): 
                    np.save('velocities_'+output, velocities)
                    
            #store orientation
            #TODO
            
            # kwt debug: check if particles overlap
            overlaps, overlaps2 = utils.check_overlap(
                displacements_vector_matrix)
            print('Step= ', step, ' Overlaps are ', jnp.sum(overlaps)-N)
                
    total_time_end = time.time()
    print('Time for ', Nsteps, ' steps is ', total_time_end-total_time_start-compilation_time,
          'or ', Nsteps/(total_time_end-total_time_start-compilation_time), ' steps per second')

    return trajectory, stresslet_history, velocities
