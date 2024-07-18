import math
import os
import time

import jax.numpy as jnp

from jfsd import jaxmd_space as space
from jfsd import jaxmd_partition as partition

import numpy as np
from jax import jit, random, Array
from jax.config import config
from jax.typing import ArrayLike

from jfsd import appliedForces, ewaldTables, resistance, shear, thermal, utils, solver, mobility

config.update("jax_enable_x64", False) #disable double precision
np.set_printoptions(precision=8, suppress=True)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false' # avoid JAX allocating most of the GPU memory even if not needed

def main(
    Nsteps: int,
    writing_period: int,
    dt: float,
    Lx: float, Ly: float, Lz: float,
    N: int,
    max_strain: float,
    T: float,
    a: float,
    xi: float,
    error: float,
    U: float,
    buoyancy_flag: int,
    U_cutoff: float,
    positions: ArrayLike,
    seed_RFD: int,
    seed_ffwave: int,
    seed_ffreal: int,
    seed_nf: int,
    shear_rate_0: float,
    shear_freq: float,
    output: str,
    stresslet_flag: bool,
    velocity_flag: bool,
    orient_flag: bool,
    constant_applied_forces: ArrayLike,
    constant_applied_torques: ArrayLike,
    HIs_flag: int,
    thermal_test_flag: int,
    alpha_friction: float,
    ho_friction: float) -> tuple[Array,Array,Array,list[float]]:
    
    """Integrate the particles equation of motions forward in time.

    While the simulation is performed, trajectories data are saved into a .npy file.

    Parameters
    ----------
    Nsteps: (int)
        Number of timesteps
    writing_period: (int)
        Period for writing to file
    dt: (float)
        Timestep
    Lx: (float)
        Box size (x-direction)
    Ly: (float)
        Box size (y-direction)
    Lz: (float)
        Box size (z-direction)
    N: (int)
        Number of particles
    max_strain: (float)
        Max strain applied to the box
    T: (float)
        Thermal energy
    a: (float)
        Particle radius
    xi: (float)
        Ewald split parameter
    error: (float)
        Tolerance error
    U: (float)
        Interaction strength
    buoyancy_flag: (int)
        Set to 1 to have gravity acting on colloids
    U_cutoff: (float)
        Distance cutoff for interacting particles
    positions: (float)
        Array of particles initial positions (N,3)
    seed_RFD: (int)
        Seed for Brownian Drift calculation
    seed_ffwave: (int)
        Seed for wave space part of far-field velocity slip
    seed_ffreal: (int)
        Seed for real space part of far-field velocity slip
    seed_nf: (int)
        Seed for near-field random forces
    shear_rate_0: (float)
        Axisymmetric shear rate amplitude
    shear_freq: (float)
        Frequency of shear, set to zero to have simple shear
    output: (str)
        File name for output
    stresslet_flag: (int)
        To have stresslet in the output
    velocity_flag: (int)
        To have velocities in the output/var/log/nvidia-installer.log
    orient_flag: (int)
        To have particle orientations in the output
    constant_applied_forces: (float)
        Array of external forces (N,3)
    constant_applied_torques: (float)
        Array of external torques (N,3)
    HIs_flag: (int)
        Flag used to set level of hydrodynamic interaction.
    thermal_test_flag: (int)
        Flag used to test thermal fluctuation calculation (1 for far-field real space, 2 for lubrication)
    alpha_friction: (float)
        Strength of hydrodynamic friction
    h0_friction: (float)
        Range of hydrodynamic friction

    Returns
    -------
        trajectory, stresslet_history, velocities, test_result

    """

    @jit
    def update_positions(
        shear_rate: float,
        positions: ArrayLike,
        displacements_vector_matrix: ArrayLike,
        net_vel: ArrayLike,
        dt: float) -> tuple[Array,Array]:    
                           
        """Update particle positions and neighbor lists
        
        Parameters
        ----------
        shear_rate:
            Shear rate at current time step
        positions:
            Array (N,3) of particles positions
        displacements_vector_matrix:
            Array (N,N,3) of relative displacements between particles
        net_vel:
            Array (6*N) of linear/angular velocities relative to the background flow
        dt:
            Discrete time step
        
        Returns
        -------
        positions, displacements_vector_matrix, nbrs_lub, nbrs_lub_prec, nbrs_ff
        
        """   

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

        return positions, displacements_vector_matrix

    @jit
    def update_nlist(
        positions: ArrayLike,
        nbrs: partition.NeighborList) -> partition.NeighborList:    
                           
        """Update particle neighbor list
        
        Parameters
        ----------
        positions:
            Array (N,3) of particles positions
        nbrs:
            Input neighbor lists
        
        Returns
        -------
        nbrs
        
        """   
        
        #Update neighbor list
        nbrs = nbrs.update(positions)
        
        return nbrs

    start_time = time.time()  # perfomances evaluation
    
    # set array for output trajectory, velocities and stresslet in time
    trajectory = np.zeros((int(Nsteps/writing_period), N, 3), float)
    stresslet_history = np.zeros((int(Nsteps/writing_period), N, 5), float)
    velocities = np.zeros((int(Nsteps/writing_period), N, 6), float)
    
    #  set INITIAL Periodic Space and Displacement Metric
    #  box is defined by an upper triangular matrix of the form: (xy,xz,yz are tilt factors)
    #    [Lx  Ly*xy  Lz*xz]
    #    [0   Ly     Lz*yz]
    #    [0   0      Lz   ]
    xy = 0.  # set box tilt factor to zero to begin (unsheared box)
    displacement, shift = space.periodic_general(
        jnp.array([[Lx, Ly*xy, Lz*0.], [0., Ly, Lz*0.], [0., 0., Lz]]), fractional_coordinates=False)
        
    # compute matrix of INITIAL displacements between particles (each element is a vector from particle j to i)
    displacements_vector_matrix = (
        space.map_product(displacement))(positions, positions)
    
    # initialize near-field hydrodynamics neighborlists (also used for pair potential)
    lub_neighbor_fn = utils.initialize_single_neighborlist(
        4., Lx, Ly, Lz, displacement)
    # allocate neighborlist for first time
    nbrs_lub = lub_neighbor_fn.allocate(positions+jnp.array([Lx, Ly, Lz])/2)
    # convert to array
    nl_lub = np.array(nbrs_lub.idx)
    
    # set external applied forces/torques (no pair-interactions, will be added later)
    AppliedForce = jnp.zeros(3*N, float)
    AppliedTorques = jnp.zeros(3*N, float)
    if(buoyancy_flag == 1):  # apply buoyancy forces (in z direction)
        AppliedForce = AppliedForce.at[2::3].add(-1.)
    if(np.count_nonzero(constant_applied_forces) > 0):  # apply external forces
        AppliedForce += jnp.ravel(constant_applied_forces)
    if(np.count_nonzero(constant_applied_torques) > 0):  # apply external torques
        AppliedTorques += jnp.ravel(constant_applied_torques)
    
    # Initialize hydrodynamic calculation
    if (HIs_flag>0):

        # compute the Real Space cutoff for the Ewald Summation in the Far-Field computation
        ewald_cut = jnp.sqrt(- jnp.log(error)) / xi
        # parameter needed to make Chol. decomposition of R_FU converge (initially to 1)
        ichol_relaxer = 1.0
        kmax = int(2.0 * jnp.sqrt(- jnp.log(error)) * xi) + 1  # Max wave number
        # set number of Lanczos iterations (initially) to 5, for both far- and near-field
        n_iter_Lanczos_ff = 2
        n_iter_Lanczos_nf = 2
        xisq = xi * xi
        # compute number of grid points in k space
        Nx, Ny, Nz = utils.Compute_k_gridpoint_number(kmax, Lx, Ly, Lz)
        gridh = jnp.array([Lx, Ly, Lz]) / \
            jnp.array([Nx, Ny, Nz])  # Set Grid Spacing
        quadW = gridh[0] * gridh[1] * gridh[2]
        

        # check that ewald_cut is small enough to avoid interaction with periodic images)
        utils.Check_ewald_cut(ewald_cut, Lx, Ly, Lz, error)

        # check maximum eigenvalue of A'*A to scale support, P, for spreading on deformed grids
        eta, gaussP = utils.Check_max_shear(
            gridh, xisq, Nx, Ny, Nz, max_strain, error)
        prefac = (2.0 * xisq / jnp.pi / eta) * jnp.sqrt(2.0 * xisq / jnp.pi / eta)
        expfac = 2.0 * xisq / eta
        gaussPd2 = jnp.array(gaussP/2, int)

        # get list of reciprocal space vectors, and scaling factor for the wave space calculation
        gridk = shear.compute_sheared_grid(
            int(Nx), int(Ny), int(Nz), xy, Lx, Ly, Lz, eta, xisq)    
        
        # store the coefficients for the real space part of Ewald summation
        # will precompute scaling factors for real space component of summation for a given discretization to speed up GPU calculations
        # NOTE: Potential sensitivity of real space functions at small xi, tabulation computed in double prec., then truncated to single
        ewald_dr = 0.001  # Distance resolution
        ewald_n = ewald_cut / ewald_dr - 1  # Number of entries in tabulation

        # factors needed to compute mobility self contribution
        pi = jnp.pi  # pi
        pi12 = jnp.sqrt(pi)  # square root of pi
        a = a * 1.0  # radius
        axi = a * xi  # a * xi
        axi2 = axi * axi  # ( a * xi )^2

        # compute mobility self contribution
        m_self = jnp.zeros(2, float)
        m_self = m_self.at[0].set(
            (1 + 4*pi12*axi*math.erfc(2.*axi) - math.exp(-4.*axi2))/(4.*pi12*axi*a))

        m_self = m_self.at[1].set((-3.*math.erfc(2.*a*xi)*math.pow(a, -3.))/10. - (3.*math.pow(a, -6.)*math.pow(pi, -0.5)*math.pow(xi, -3.))/80.
                                  - (9.*math.pow(a, -4)*math.pow(pi, -0.5)
                                     * math.pow(xi, -1))/40
                                  + (3.*math.exp(-4 * math.pow(a, 2)*math.pow(xi, 2))*math.pow(a, -6)*math.pow(pi, -0.5)*math.pow(xi, -3)
                                     * (1+10 * math.pow(a, 2)*math.pow(xi, 2))) / 80)
    
        # create real space Ewald table
        nR = int(ewald_n + 1)  # number of entries in ewald table
        ewaldC1 = ewaldTables.Compute_real_space_ewald_table(
            nR, a, xi)  # this version uses numpy long float
        ewaldC1 = jnp.array(ewaldC1)  # convert to single precision (32-bit)

        # initialize far-field hydrodynamics neighborlists
        ff_neighbor_fn = utils.initialize_single_neighborlist(
            ewald_cut, Lx, Ly, Lz, displacement)
        # allocate neighborlist for first time     
        nbrs_ff = ff_neighbor_fn.allocate(positions+jnp.array([Lx, Ly, Lz])/2)
        # convert to array   
        nl_ff = np.array(nbrs_ff.idx)

        # precompute grid distances for FFT (same for each gaussian support)
        gaussian_grid_spacing = utils.Precompute_grid_distancing(
            gaussP, gridh[0], xy, positions, N, Nx, Ny, Nz, Lx, Ly, Lz)
        
        # load resistance table
        ResTable_dist = jnp.load('files/ResTableDist.npy')
        ResTable_vals = jnp.load('files/ResTableVals.npy')
        ResTable_min = 0.0001000000000000
        # table discretization (log space), i.e. ind * dr.set(log10( h[ind] / min )
        ResTable_dr = 0.0043050000000000
        
        # initialize near-field hydrodynamics precondition neighborlists
        prec_lub_neighbor_fn = utils.initialize_single_neighborlist(
            2.1, Lx, Ly, Lz, displacement)
        # allocate neighborlist for first time 
        nbrs_lub_prec = prec_lub_neighbor_fn.allocate(
            positions+jnp.array([Lx, Ly, Lz])/2)
    
        #define epsilon for RFD
        epsilon = error

        # create indices list for spreading random forces on the wave space grid (far field brownian velocity, wave space calculation)
        if(T > 0):
            (normal_indices_x, normal_indices_y, normal_indices_z,
             normal_conj_indices_x, normal_conj_indices_y, normal_conj_indices_z,
             nyquist_indices_x, nyquist_indices_y, nyquist_indices_z) = thermal.Random_force_on_grid_indexing(Nx, Ny, Nz)

        # create RNG states
        key_RFD = random.PRNGKey(seed_RFD)
        key_ffwave = random.PRNGKey(seed_ffwave)
        key_ffreal = random.PRNGKey(seed_ffreal)
    # create RNG states 
    key_nf = random.PRNGKey(seed_nf) #also used for standard Brownian Dynamics


    

    # initialize stresslet (for output, not needed to integrate trajectory in space)
    stresslet = jnp.zeros((N, 5), float)

    #measure compilation time of the first part
    compilation_time = time.time() - start_time
        
    # kwt debug: check if particles overlap
    # overlaps, overlaps2 = utils.check_overlap(
    #     displacements_vector_matrix)
    # print('Starting: initial overlaps are ', jnp.sum(overlaps)-N)
 
    start_time = time.time() #perfomances evaluation
    

    for step in range(Nsteps):

        # initialize generalized velocity (6*N array, for linear and angular components)
        general_velocity = jnp.zeros(6*N, float) #this array stores the velocity for Brownian Dynamics, or the Brownian drift otherwise

        if((stresslet_flag > 0) and ((step % writing_period) == 0)):
            #reset stresslet to zero
            stresslet = jnp.zeros((N, 5), float)

        # define rhs array of the linear system Ax=b (with A the saddle point matrix)
        saddle_b = jnp.zeros(17*N, float)

        if(HIs_flag>1):
            # precompute quantities for far-field and near-field hydrodynamic calculation
            (all_indices_x, all_indices_y, all_indices_z, gaussian_grid_spacing1, gaussian_grid_spacing2,
             r, indices_i, indices_j, f1, f2, g1, g2, h1, h2, h3,
             r_lub, indices_i_lub, indices_j_lub, ResFunction) = utils.precompute(positions, gaussian_grid_spacing, nl_ff, nl_lub, displacements_vector_matrix, xy,
                                                                                  N, Lx, Ly, Lz, Nx, Ny, Nz,
                                                                                  prefac, expfac, quadW,
                                                                                  int(gaussP), gaussPd2,
                                                                                  ewald_n, ewald_dr, ewald_cut, ewaldC1,
                                                                                  ResTable_min, ResTable_dr, ResTable_dist, ResTable_vals,alpha_friction,ho_friction)

            # set projector needed to compute thermal fluctuations given by lubrication
            diagonal_zeroes_for_brownian = thermal.Number_of_neigh(
                N, indices_i_lub, indices_j_lub)
        
            # compute precondition resistance lubrication matrix
            R_fu_prec_lower_triang, diagonal_elements_for_brownian = resistance.RFU_Precondition(
                ichol_relaxer, 
                displacements_vector_matrix.at[np.array(nbrs_lub_prec.idx)[0, :], 
                                               np.array(nbrs_lub_prec.idx)[1, :]].get(), N,
                len(nbrs_lub_prec.idx[0]), np.array(nbrs_lub_prec.idx))

            # perform Cholesky factorization and obtain lower triangle Cholesky factor of R_FU^nf
            R_fu_prec_lower_triang = utils.chol_fac(R_fu_prec_lower_triang)

        elif(HIs_flag==1):
            # precompute quantities for far-field and near-field hydrodynamic calculation
            (all_indices_x, all_indices_y, all_indices_z, gaussian_grid_spacing1, gaussian_grid_spacing2,
             r, indices_i, indices_j, f1, f2, g1, g2, h1, h2, h3,
             indices_i_lub, indices_j_lub) = utils.precomputeRPY(positions, gaussian_grid_spacing, nl_ff, nl_lub, displacements_vector_matrix, xy,
                                                                                  N, Lx, Ly, Lz, Nx, Ny, Nz,
                                                                                  prefac, expfac, quadW,
                                                                                  int(gaussP), gaussPd2,
                                                                                  ewald_n, ewald_dr, ewald_cut, ewaldC1)
        else:
            # precompute quantities for Brownian dynamics calculation
            (r_lub, indices_i_lub, indices_j_lub) = utils.precomputeBD(positions, nl_lub, displacements_vector_matrix, N, Lx, Ly, Lz)

        # compute shear-rate for current timestep: simple(shear_freq=0) or oscillatory(shear_freq>0)
        shear_rate = shear.update_shear_rate(
            dt, step, shear_rate_0, shear_freq, phase=0)
        
        # if temperature is not zero (and full hydrodynamics are switched on), compute Brownian Drift
        if((T > 0) and (HIs_flag>1)):

            # get array of random variables
            key_RFD, random_array = utils.generate_random_array(
                key_RFD, 6*N)  
            random_array = -((2*random_array-1)*jnp.sqrt(3))
            # add random displacement to right-hand side of linear system Ax=b
            saddle_b = saddle_b.at[11*N:].set(random_array)

            # perform a displacement in the positive random directions (and update wave grid and neighbor lists) and save it to a buffer
            buffer_positions, buffer_displacements_vector_matrix = update_positions(
                shear_rate, positions, displacements_vector_matrix, random_array, epsilon/2.)
            buffer_nbrs_lub = update_nlist(buffer_positions,nbrs_lub)
            buffer_nbrs_ff = update_nlist(buffer_positions,nbrs_ff)
            buffer_gaussian_grid_spacing = utils.Precompute_grid_distancing(
                gaussP, gridh[0], xy, buffer_positions, N, Nx, Ny, Nz, Lx, Ly, Lz)
            buffer_nl_lub = np.array(buffer_nbrs_lub.idx)
            buffer_nl_ff = np.array(buffer_nbrs_ff.idx)

            # solve the saddle point problem in the positive direction
            output_precompute = utils.precompute(buffer_positions, buffer_gaussian_grid_spacing, buffer_nl_ff, buffer_nl_lub, buffer_displacements_vector_matrix, xy,
                                                 N, Lx, Ly, Lz, Nx, Ny, Nz,
                                                 prefac, expfac, quadW,
                                                 gaussP, gaussPd2,
                                                 ewald_n, ewald_dr, ewald_cut, ewaldC1,
                                                 ResTable_min, ResTable_dr, ResTable_dist, ResTable_vals,alpha_friction,ho_friction)
            saddle_x, exitcode_gmres = solver.solverSD(
                N,HIs_flag,
                saddle_b,  # rhs vector of the linear system
                gridk,
                R_fu_prec_lower_triang,
                output_precompute,
                Nx,
                Ny,
                Nz,
                gaussP,
                m_self
            )

            if(exitcode_gmres > 0):
                raise ValueError(
                    f"GMRES (RFD) did not converge! Iterations are {exitcode_gmres}. Abort!")
            general_velocity = saddle_x.at[11*N:].get()

            # compute the near-field hydrodynamic stresslet from the saddle point solution velocity
            if((stresslet_flag > 0) and ((step % writing_period) == 0)):
                stresslet = resistance.compute_RSU(stresslet, general_velocity,
                                                   indices_i_lub,
                                                   indices_j_lub,
                                                   output_precompute[18],
                                                   r_lub,
                                                   N)

            # repeat everything but in opposite direction of displacement
            # perform a displacement in the negative random directions (and update wave grid and neighbor lists) and save it to a buffer
            buffer_positions, buffer_displacements_vector_matrix = update_positions(
                shear_rate, positions, displacements_vector_matrix, random_array, -epsilon/2.)
            buffer_nbrs_lub = update_nlist(buffer_positions,nbrs_lub)
            buffer_nbrs_ff = update_nlist(buffer_positions,nbrs_ff)
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
                                                 ResTable_min, ResTable_dr, ResTable_dist, ResTable_vals,alpha_friction,ho_friction)
            saddle_x, exitcode_gmres = solver.solverSD(
                N, HIs_flag,
                saddle_b,  # rhs vector of the linear system
                gridk,
                R_fu_prec_lower_triang,
                output_precompute,
                Nx,
                Ny,
                Nz,
                gaussP,
                m_self
            )
            if(exitcode_gmres > 0):
                raise ValueError(
                    f"GMRES (RFD) did not converge! Iterations are {exitcode_gmres}. Abort!")

            # compute the near-field hydrodynamic stresslet from the saddle point solution velocity
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
                
            # take Difference and apply scaling
            general_velocity += (-saddle_x.at[11*N:].get())
            general_velocity = -general_velocity * T / epsilon
            if((stresslet_flag > 0) and ((step % writing_period) == 0)):
                stresslet += -buffer_stresslet
                stresslet = stresslet * T / epsilon

            # reset RHS to zero for next saddle point solver
            saddle_b = jnp.zeros(17*N, float)
        
        # add applied forces and conservative (pair potential) forces to right-hand side of system
        saddle_b = appliedForces.sumAppliedForces(N, AppliedForce, AppliedTorques, saddle_b, U,
                                                  indices_i_lub, indices_j_lub, displacements_vector_matrix,
                                                  U_cutoff,HIs_flag)
        
        # add (-) the ambient rate of strain to the right-hand side (if full hydrodynamics are switched on)
        if((shear_rate_0 != 0) and (HIs_flag>1)):
            saddle_b = saddle_b.at[(6*N+1):(11*N):5].add(-shear_rate)

            # compute near field shear contribution R_FE and add it to the rhs of the system
            saddle_b = saddle_b.at[11*N:].add(resistance.compute_RFE(N, shear_rate, r_lub, indices_i_lub, indices_j_lub, ResFunction[11], ResFunction[12],
                                                                     ResFunction[13], ResFunction[14], ResFunction[15], ResFunction[16],
                                                                     -ResFunction[12], -ResFunction[14], ResFunction[16]))
        # compute Thermal Fluctuations only if temperature is not zero
        if(T > 0):
            # generate random numbers for the various contributions of thermal noise
            key_nf, random_array_nf = utils.generate_random_array(
                key_nf, (6*N))
            
            if(HIs_flag>0):
                key_ffwave, random_array_wave = utils.generate_random_array(
                    key_ffwave, (3 * 2 * len(normal_indices_x) + 3 * len(nyquist_indices_x)))
                key_ffreal, random_array_real = utils.generate_random_array(
                    key_ffreal, (11*N))

                # compute far-field (wave space contribution) slip velocity and set in rhs of linear system
                ws_linvel, ws_angvel_strain = thermal.compute_wave_space_slipvelocity(N, int(Nx), int(Ny), int(Nz), int(gaussP), T, dt, gridh,
                                                                                  normal_indices_x, normal_indices_y, normal_indices_z,
                                                                                  normal_conj_indices_x, normal_conj_indices_y, normal_conj_indices_z,
                                                                                  nyquist_indices_x, nyquist_indices_y, nyquist_indices_z,
                                                                                  gridk, random_array_wave, all_indices_x, all_indices_y, all_indices_z,
                                                                                  gaussian_grid_spacing1, gaussian_grid_spacing2)

                # compute far-field (real space contribution) slip velocity and set in rhs of linear system
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

                # compute lubrication contribution only if there is more than 1 particle (and full SD is switched on)
                stepnormnf = 0.
                if((N > 1) and (HIs_flag>1)):  
                    # compute near-field random forces and set in rhs of linear system
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

                # check that thermal fluctuation calculation went well
                if ((not math.isfinite(stepnormff)) or ((n_iter_Lanczos_ff > 150) and (stepnormff > 0.0001))):
                    raise ValueError(
                        f"Far-field Lanczos did not converge! Stepnorm is {stepnormff}, iterations are {n_iter_Lanczos_ff}. Eigenvalues of tridiagonal matrix are {diag_ff}. Abort!")

                if ((not math.isfinite(stepnormnf)) or ((n_iter_Lanczos_nf > 150) and (stepnormnf > 0.0001))):
                    raise ValueError(
                        f"Near-field Lanczos did not converge! Stepnorm is {stepnormnf}, iterations are {n_iter_Lanczos_nf}. Eigenvalues of tridiagonal matrix are {diag_nf}. Abort!")

            else:
                #compute random force for Brownian Dynamics 
                random_velocity = thermal.compute_BD_randomforce(N, T, dt,random_array_nf)
                general_velocity += random_velocity

        
        if(HIs_flag>1):
            # solve the system Ax=b, where x contains the unknown particle velocities (relative to the background flow) and stresslet
            saddle_x, exitcode_gmres = solver.solverSD(
                N, HIs_flag,
                saddle_b,
                gridk,
                R_fu_prec_lower_triang,
                [all_indices_x, all_indices_y, all_indices_z, gaussian_grid_spacing1, gaussian_grid_spacing2,
                 r, indices_i, indices_j, f1, f2, g1, g2, h1, h2, h3,
                 r_lub, indices_i_lub, indices_j_lub, ResFunction],
                Nx,
                Ny,
                Nz,
                gaussP,
                m_self)
            if(exitcode_gmres > 0):
                raise ValueError(
                    f"GMRES did not converge! Iterations are {exitcode_gmres}. Abort!")
        
            # add the near-field contributions to the stresslet
            if((stresslet_flag > 0) and ((step % writing_period) == 0)):
                # get stresslet out of saddle point solution (and add it to the contribution from the Brownian drift, if temperature>0 )
                stresslet += jnp.reshape(-saddle_x[6*N:11*N], (N, 5))
                stresslet = resistance.compute_RSU(stresslet, saddle_x.at[11*N:].get(),
                                                indices_i_lub,
                                                indices_j_lub,
                                                ResFunction,
                                                r_lub,
                                                N)
                # add shear (near-field) contributions to the stresslet
                if(shear_rate_0 != 0):
                    stresslet = resistance.compute_RSE(N, shear_rate, r_lub, indices_i_lub, indices_j_lub, ResFunction[17], ResFunction[18],
                                                    ResFunction[19], ResFunction[20], ResFunction[21], ResFunction[22], stresslet)
                # save stresslet
                stresslet_history[int(step/writing_period), :, :] = stresslet
                if(output != 'None'):
                    np.save('stresslet_'+output, stresslet_history)

            # update positions and neighborlists
            (positions, displacements_vector_matrix) = update_positions(shear_rate,
                                                              positions, displacements_vector_matrix,
                                                              saddle_x.at[11*N:].get() +
                                                              general_velocity,
                                                              dt)
            nbrs_lub_prec= update_nlist(positions, nbrs_lub_prec)

            nbrs_ff= update_nlist(positions, nbrs_ff)
            nl_ff = np.array(nbrs_ff.idx)  # extract lists in sparse format
            
            # update grid distances for FFT (needed for wave space calculation of mobility)
            gaussian_grid_spacing = utils.Precompute_grid_distancing(
                gaussP, gridh[0], xy, positions, N, Nx, Ny, Nz, Lx, Ly, Lz)

        elif(HIs_flag==1):
            #add random velocity to total velocity in RPY
            general_velocity += saddle_b[:6*N]
            #add potential force contribution to total velocity in RPY
            general_velocity += mobility.Mobility(N,Nx,Ny,Nz,
                                                  gaussP,gridk,m_self,
                                                  all_indices_x,all_indices_y,all_indices_z,
                                                  gaussian_grid_spacing1,gaussian_grid_spacing2,
                                                  r,indices_i,indices_j,f1,f2,g1,g2,h1,h2,h3,
                                                  -saddle_b[11*N:])
            # update positions and neighborlists
            (positions, displacements_vector_matrix) = update_positions(shear_rate,positions, 
                                                                           displacements_vector_matrix,
                                                                           general_velocity,
                                                                           dt)
            nbrs_ff= update_nlist(positions, nbrs_ff)
            nl_ff = np.array(nbrs_ff.idx)  # extract lists in sparse format
        else:
            #add potential force contribution to total velocity (thermal contribution is already included)
            general_velocity += - saddle_b[11*N:]
            # update positions
            (positions, displacements_vector_matrix) = update_positions(shear_rate,positions, 
                                                                           displacements_vector_matrix,
                                                                           general_velocity,
                                                                           dt)
        nbrs_lub = update_nlist(positions, nbrs_lub)
        nl_lub = np.array(nbrs_lub.idx)  # extract lists in sparse format

        # if system is sheared, strain wave-vectors grid and update box tilt factor
        if(shear_rate_0 != 0):
            xy = shear.update_box_tilt_factor(
                dt, shear_rate_0, xy, step, shear_freq)
            displacement, shift = space.periodic_general(
                jnp.array([[Lx, Ly*xy, Lz*0.], [0., Ly, Lz*0.], [0., 0., Lz]]), fractional_coordinates=False)
            if(HIs_flag>0):
                gridk = shear.compute_sheared_grid(
                    int(Nx), int(Ny), int(Nz), xy, Lx, Ly, Lz, eta, xisq)

        # compute compilation time
        if(step == 0):
            compilation_time2 = (time.time() - start_time)
            print('Compilation Time is ', compilation_time+compilation_time2)

        # reset Lacnzos number of iteration once in a while (avoid to do too many iterations if not needed, can be tuned for better performance)
        elif((step % 100) == 0):
            n_iter_Lanczos_ff = 5
            n_iter_Lanczos_nf = 5
            
        # write trajectory to file
        if((step % writing_period) == 0):

            #check that the position to save do not contain 'nan' or 'inf'
            if((jnp.isnan(positions)).any() or (jnp.isinf(positions)).any()):
                raise ValueError(
                    "Invalid particles positions. Abort!")

            #save trajectory to file
            trajectory[int(step/writing_period), :, :] = positions
            if(output != 'None'):
                np.save(output, trajectory)
                
            #store velocity (linear and angular) to file
            if( (velocity_flag > 0) ):
                velocities[int(step/writing_period), :, :] = jnp.reshape(saddle_x.at[11*N:].get() + general_velocity, (N,6))
                if (output != 'None'): 
                    np.save('velocities_'+output, velocities)
                    
            #store orientation
            #TODO
            
            # kwt debug: check if particles overlap
            # overlaps, overlaps2 = utils.check_overlap(
            #     displacements_vector_matrix)
            # print('Step= ', step, ' Overlaps are ', jnp.sum(overlaps)-N)
            print('Step= ', step)
                
    end_time = time.time()
    print('Time for ', Nsteps, ' steps is ', end_time-start_time-compilation_time2,
          'or ', Nsteps/(end_time-start_time-compilation_time2), ' steps per second')

    #perform thermal test if needed
    test_result = 0.    
    if((T>0) and (thermal_test_flag==1)):
        ff , nf = thermal.compute_exact_thermals(
            N,m_self,
            T,dt,
            random_array_nf,
            random_array_real,
            r,
            indices_i,
            indices_j,
            f1,f2,g1,g2,h1,h2,h3,
            r_lub,
            indices_i_lub,
            indices_j_lub,
            ResFunction[0], ResFunction[1], ResFunction[
                2], ResFunction[3], ResFunction[4], ResFunction[5],
            ResFunction[6], ResFunction[7], ResFunction[
                8], ResFunction[9], ResFunction[10])
        
        test_result = [jnp.linalg.norm(buffer-nf) / jnp.linalg.norm(nf),jnp.linalg.norm(thermal.convert_to_generalized(N, 0, rs_linvel, 0, rs_angvel_strain)-ff)/jnp.linalg.norm(ff)]
                        
            
            

    return trajectory, stresslet_history, velocities, test_result
