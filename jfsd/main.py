import math
import os
from pathlib import Path
from jax.lib import xla_bridge
import jax.numpy as jnp
import numpy as np
from jax import Array, jit, random
from jax.config import config
from jax.typing import ArrayLike
from jfsd import appliedForces, mobility, resistance, shear, solver, thermal, utils
from jfsd import jaxmd_space as space
from tqdm import tqdm

config.update("jax_enable_x64", False)  # disable double precision by default
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false' # avoid JAX allocating most of the GPU memory even if not needed
def main(Nsteps: int,
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
         boundary_flag: int,
         thermal_test_flag: int,
         alpha_friction: float,
         ho_friction: float) -> tuple[Array, Array, Array, list[float]]:
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
    boundary_flag: (int)
        Flag used to set the type of boundary conditions for the hydrodynamic interaction.
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
    trajectory = stresslet_history= velocities=test_result= None
    if(writing_period>Nsteps):
        raise ValueError(
            "Error: writing-to-file period greater than number of simulation steps. Abort!")
    if(boundary_flag==1):
        if((N<2) and (HIs_flag>0)):
            raise ValueError(
                "Error: trying to use open boundaries hydrodynamic for a single particle. Select 'brownian' and re-run. Abort!")
        Lx = Ly = Lz = 999999 #set system size to 'infinite' (does not impact performance)
        config.update("jax_enable_x64", True)  # enable double precision: needed to resolve long-range hydrodynamics completely in real space
        
    print('jfsd is running on device: ',(xla_bridge.get_backend().platform))
        
    if (HIs_flag==0):
        trajectory, velocities = wrap_BD(Nsteps, writing_period,
                 dt,Lx, Ly, Lz, N, T,U,buoyancy_flag,U_cutoff,positions,
                 seed_nf,shear_rate_0,shear_freq,output,
                 velocity_flag,orient_flag,constant_applied_forces,constant_applied_torques)
    elif (HIs_flag==1):
        trajectory, velocities = wrap_RPY(Nsteps, writing_period,
                 dt,Lx, Ly, Lz, N, max_strain,T,xi,error,U,buoyancy_flag,U_cutoff,positions,
                 seed_ffwave,seed_ffreal,shear_rate_0,shear_freq,output,
                 velocity_flag,orient_flag,constant_applied_forces,constant_applied_torques,
                 boundary_flag)
    elif (HIs_flag==2):
        trajectory, stresslet_history, velocities, test_result = wrap_SD(Nsteps, writing_period,
                 dt,Lx, Ly, Lz, N, max_strain,T,a,xi,error,U,buoyancy_flag,U_cutoff,positions,
                 seed_RFD, seed_ffwave,seed_ffreal,seed_nf,shear_rate_0,shear_freq,output,
                 stresslet_flag,velocity_flag,orient_flag,constant_applied_forces,constant_applied_torques,
                 boundary_flag,thermal_test_flag,alpha_friction,ho_friction)
    return trajectory, stresslet_history, velocities, test_result
    
def overlap_monitor(displacements_vector_matrix,N):
    # check that current configuration does not have overlapping particles
    overlaps, overlaps_indices = utils.check_overlap(displacements_vector_matrix, 2., N)
    if (overlaps > 0):
        print('Warning: ', (overlaps), ' particles are overlapping. Reducing the timestep might help prevent unphysical overlaps.')
        print('Indices of overlapping particles are', overlaps_indices[0][:int(overlaps)], overlaps_indices[1][:int(overlaps)] )
        print('Distances of overlapping particles are', displacements_vector_matrix[overlaps_indices[0][:int(overlaps)] ,overlaps_indices[1][:int(overlaps)] ])
    return (overlaps > 0)
    
def wrap_SD(
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
        boundary_flag: int,
        thermal_test_flag: int,
        alpha_friction: float,
        ho_friction: float) -> tuple[Array, Array, Array, list[float]]:
    """Wrap all functions needed to integrate the particles equation of motions forward in time, using Stokesian Dynamics method.

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
    boundary_flag: (int)
        Flag used to set the type of boundary conditions for the hydrodynamic interaction.
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
            dt: float) -> tuple[Array, Array]:
        """Update particle positions and neighbor lists

        Parameters
        ----------
        
        shear_rate: (float)
            Shear rate at current time step
        positions: (float)
            Array (N,3) of particles positions
        displacements_vector_matrix: (float)
            Array (N,N,3) of relative displacements between particles
        net_vel: (float)
            Array (6*N) of linear/angular velocities relative to the background flow
        dt: (float)
            Timestep used to advance positions

        Returns
        -------
        positions, displacements_vector_matrix

        """

        # Define array of displacement r(t+dt)-r(t)
        dR = jnp.zeros((N, 3), float)
        # Compute actual displacement due to velocities (relative to background flow)
        dR = dR.at[:, 0].set(dt*net_vel.at[(0)::6].get())
        dR = dR.at[:, 1].set(dt*net_vel.at[(1)::6].get())
        dR = dR.at[:, 2].set(dt*net_vel.at[(2)::6].get())
        # Apply displacement and compute wrapped shift (Lees Edwards boundary conditions)
        positions = shift(positions+jnp.array([Lx, Ly, Lz])/2, dR) - jnp.array([Lx, Ly, Lz])*0.5 
        
        # Define array of displacement r(t+dt)-r(t) (this time for displacement given by background flow)
        dR = jnp.zeros((N, 3), float)
        dR = dR.at[:, 0].set(dt * shear_rate * positions.at[:, 1].get()) # Assuming y:gradient direction, x:background flow direction
        positions = shift(positions+jnp.array([Lx, Ly, Lz])/2, dR) - jnp.array([Lx, Ly, Lz]) * 0.5 # Apply shift

        # Compute new relative displacements between particles
        displacements_vector_matrix = (space.map_product(displacement))(positions, positions)
        return positions, displacements_vector_matrix

    if output is not None:
        output = Path(output)
        output.mkdir(exist_ok=True, parents=True)

    # set array for output trajectory, velocities and stresslet in time
    trajectory = np.zeros((int(Nsteps/writing_period), N, 3), float)
    stresslet_history = np.zeros((int(Nsteps/writing_period), N, 5), float)
    velocities = np.zeros((int(Nsteps/writing_period), N, 6), float)
       
    # set initial number of Lanczos iterations, for both thermal fluctuations
    n_iter_Lanczos_ff = 2
    n_iter_Lanczos_nf = 2
    
    epsilon = error # define epsilon for RFD   
    xy = 0.  # set box tilt factor to zero to begin (unsheared box)
    ewald_cut = jnp.sqrt(- jnp.log(error)) / xi  #Real Space cutoff for the Ewald Summation in the Far-Field computation
    ichol_relaxer = 1.0 # for Chol. factorization of R_FU^prec (initially to 1)

    # load resistance table
    ResTable_dist = jnp.load('files/ResTableDist.npy')
    ResTable_vals = jnp.load('files/ResTableVals.npy')
    ResTable_min = 0.0001000000000000
    ResTable_dr = 0.0043050000000000 #table discretization (log space), i.e. ind * dr.set(log10( h[ind] / min )
    
    # set INITIAL Periodic Space and Displacement Metric
    displacement, shift = space.periodic_general(jnp.array([[Lx, Ly*xy, Lz*0.], [0., Ly, Lz*0.], [0., 0., Lz]]), fractional_coordinates=False)
    # compute matrix of INITIAL displacements between particles (each element is a vector from particle j to i)
    displacements_vector_matrix = (space.map_product(displacement))(positions, positions)
    
    # initialize near-field hydrodynamics neighborlists (used also by pair potentials)
    lub_neighbor_fn = utils.initialize_single_neighborlist(3.99, Lx, Ly, Lz, displacement) #for near-field hydrodynamics and pair potential
    nbrs_lub = lub_neighbor_fn.allocate(positions+jnp.array([Lx, Ly, Lz])/2) # allocate neighborlist for first time
    nl_lub = np.array(nbrs_lub.idx) # convert to array
    
    # initialize far-field hydrodynamics neighborlists
    ff_neighbor_fn = utils.initialize_single_neighborlist(ewald_cut, Lx, Ly, Lz, displacement) #for far-field hydrodynamics
    nbrs_ff = ff_neighbor_fn.allocate(positions+jnp.array([Lx, Ly, Lz])/2) # allocate neighborlist for first time
    nl_ff = np.array(nbrs_ff.idx) # convert to array
    
    # initialize near-field hydrodynamics precondition neighborlists
    prec_lub_neighbor_fn = utils.initialize_single_neighborlist(2.1, Lx, Ly, Lz, displacement) #for near-field hydrodynamics precondition
    nbrs_lub_prec = prec_lub_neighbor_fn.allocate(positions+jnp.array([Lx, Ly, Lz])/2) # allocate neighborlist for first time
    
    if(boundary_flag==0):
        # Initialize periodic hydrodynamic quantities
        (quadW,prefac,expfac,gaussPd2,gridk,gridh,gaussian_grid_spacing,key_ffwave,
         ewaldC1,m_self,Nx,Ny,Nz,gaussP,
         ewald_n,ewald_dr,eta,xisq,
         wave_bro_ind,wave_bro_nyind) = utils.init_periodic_box(error,xi,
                              Lx,Ly,Lz, ewald_cut, max_strain, xy,
                              positions, N, T, seed_ffwave)
    if(boundary_flag == 1):
        nl_ff = utils.compute_distinct_pairs(N) #compute list of distinct pairs for long range hydrodynamics (not optimized)  
    
    if(T > 0): # create Random Number Generator states
        key_RFD = random.PRNGKey(seed_RFD)
        key_ffreal = random.PRNGKey(seed_ffreal)
        key_nf = random.PRNGKey(seed_nf)

    # set external applied forces/torques (no pair-interactions, will be added later)
    AppliedForce = jnp.zeros(3*N, float)
    AppliedTorques = jnp.zeros(3*N, float)
    if(buoyancy_flag == 1):  # apply buoyancy forces (in z direction)
        AppliedForce = AppliedForce.at[2::3].add(-1.)
    if(np.count_nonzero(constant_applied_forces) > 0):  # apply external forces
        AppliedForce += jnp.ravel(constant_applied_forces)
    if(np.count_nonzero(constant_applied_torques) > 0):  # apply external torques
        AppliedTorques += jnp.ravel(constant_applied_torques)

    # Check if particles overlap
    overlaps, overlaps_indices = utils.check_overlap(
        displacements_vector_matrix, 2.)
    if(overlaps > 0):
        print('Warning: initial overlaps are ', (overlaps))
    print('Starting: compiling the code... This should not take more than 1-2 minutes.')
    for step in tqdm(range(Nsteps),mininterval=0.5): #use tqdm to have progress bar and TPS 
        
        # check that neighborlists buffers did not overflow, if so, re-allocate the lists
        if(nbrs_lub.did_buffer_overflow):
            nbrs_lub = utils.allocate_nlist(positions, lub_neighbor_fn)
            nl_lub = np.array(nbrs_lub.idx)
        if(nbrs_ff.did_buffer_overflow):
            nbrs_ff = utils.allocate_nlist(positions, ff_neighbor_fn)
            nl_ff = np.array(nbrs_ff.idx)
        if(nbrs_lub_prec.did_buffer_overflow):
            nbrs_lub_prec = utils.allocate_nlist(
                positions, prec_lub_neighbor_fn)

        # Initialize Brownian drift (6*N array, for linear and angular components)
        brownian_drift = jnp.zeros(6*N, float)

        if( (stresslet_flag > 0) and ((step % writing_period) == 0) ):
            stresslet = jnp.zeros((N, 5), float) # reset stresslet to zero

        # define rhs array of the linear system Ax=b (with A the saddle point matrix)
        saddle_b = jnp.zeros(17*N, float)

        # precompute quantities for far-field and near-field hydrodynamic calculation
        if(boundary_flag==0):
            (all_indices_x, all_indices_y, all_indices_z, gaussian_grid_spacing1, gaussian_grid_spacing2,
                 r, indices_i, indices_j, f1, f2, g1, g2, h1, h2, h3,
            r_lub, indices_i_lub, indices_j_lub, ResFunction) = utils.precompute(positions, gaussian_grid_spacing, nl_ff, nl_lub, displacements_vector_matrix, xy,
                                                                                  N, Lx, Ly, Lz, Nx, Ny, Nz,
                                                                                  prefac, expfac, quadW,
                                                                                  int(gaussP), gaussPd2,
                                                                                  ewald_n, ewald_dr, ewald_cut, ewaldC1,
                                                                                  ResTable_min, ResTable_dr, ResTable_dist, ResTable_vals, alpha_friction, ho_friction)
        elif(boundary_flag==1):
           (r,indices_i, indices_j,r_lub,indices_i_lub,indices_j_lub,ResFunction,mobil_scalar) = utils.precompute_open(positions,nl_ff, nl_lub, displacements_vector_matrix,                                                                               
                                                                                 ResTable_min, ResTable_dr, ResTable_dist, ResTable_vals, alpha_friction, ho_friction)
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

        # compute shear-rate for current timestep: simple(shear_freq=0) or oscillatory(shear_freq>0)
        shear_rate = shear.update_shear_rate(
            dt, step, shear_rate_0, shear_freq, phase=0)

        # if temperature is not zero (and full hydrodynamics are switched on), compute Brownian Drift
        if(T > 0):

            # get array of random variables
            key_RFD, random_array = utils.generate_random_array(key_RFD, 6*N)
            random_array = -((2*random_array-1)*jnp.sqrt(3))
            # add random displacement to right-hand side of linear system Ax=b
            saddle_b = saddle_b.at[11*N:].set(random_array)

            # SOLVE SADDLE POINT IN THE POSITIVE DIRECTION
            # perform a displacement in the positive random directions (and update wave grid and neighbor lists) and save it to a buffer
            buffer_positions, buffer_displacements_vector_matrix = update_positions(
                shear_rate, positions, displacements_vector_matrix, random_array, epsilon/2.)
            #update neighborlists 
            buffer_nbrs_lub = utils.update_nlist(buffer_positions, nbrs_lub)
            buffer_nl_lub = np.array(buffer_nbrs_lub.idx)
            if(boundary_flag == 0):
                #update wave grid and far-field neighbor list (not needed in open boundaries)
                buffer_gaussian_grid_spacing = utils.Precompute_grid_distancing(
                    gaussP, gridh[0], xy, buffer_positions, N, Nx, Ny, Nz, Lx, Ly, Lz)
                buffer_nbrs_ff = utils.update_nlist(buffer_positions, nbrs_ff)
                buffer_nl_ff = np.array(buffer_nbrs_ff.idx)
                output_precompute = utils.precompute(buffer_positions, buffer_gaussian_grid_spacing, buffer_nl_ff, buffer_nl_lub, buffer_displacements_vector_matrix, xy,
                                                 N, Lx, Ly, Lz, Nx, Ny, Nz,
                                                 prefac, expfac, quadW,
                                                 gaussP, gaussPd2,
                                                 ewald_n, ewald_dr, ewald_cut, ewaldC1,
                                                 ResTable_min, ResTable_dr, ResTable_dist, ResTable_vals, alpha_friction, ho_friction)
                saddle_x, exitcode_gmres = solver.solver(N,saddle_b,  # rhs vector of the linear system
                                                         gridk, R_fu_prec_lower_triang, output_precompute,Nx,Ny,Nz,gaussP,m_self)
            elif(boundary_flag==1):
                output_precompute = utils.precompute_open(buffer_positions,nl_ff, buffer_nl_lub, buffer_displacements_vector_matrix,
                                                          ResTable_min, ResTable_dr, ResTable_dist, ResTable_vals, alpha_friction, ho_friction)
                saddle_x, exitcode_gmres = solver.solver_open(N,saddle_b,  # rhs vector of the linear system
                                                         R_fu_prec_lower_triang, output_precompute,
                                                         displacements_vector_matrix.at[nl_ff[0, :], nl_ff[1, :]].get())
            if(exitcode_gmres > 0):
                raise ValueError(
                    f"GMRES (RFD) did not converge! Iterations are {exitcode_gmres}. Abort!")
            brownian_drift = saddle_x.at[11*N:].get()

            # compute the near-field hydrodynamic stresslet from the saddle point solution velocity
            if((stresslet_flag > 0) and ((step % writing_period) == 0)):
                stresslet = resistance.compute_RSU(stresslet, brownian_drift,
                                                   indices_i_lub,
                                                   indices_j_lub,
                                                   output_precompute[18],
                                                   r_lub,
                                                   N)
            
            # SOLVE SADDLE POINT IN THE NEGATIVE DIRECTION
            buffer_positions, buffer_displacements_vector_matrix = update_positions(
                shear_rate, positions, displacements_vector_matrix, random_array, -epsilon/2.)
            buffer_nbrs_lub = utils.update_nlist(buffer_positions, nbrs_lub)
            buffer_nl_lub = np.array(buffer_nbrs_lub.idx)
            if(boundary_flag == 0):
                buffer_gaussian_grid_spacing = utils.Precompute_grid_distancing(
                    gaussP, gridh[0], xy, buffer_positions, N, Nx, Ny, Nz, Lx, Ly, Lz)
                buffer_nbrs_ff = utils.update_nlist(buffer_positions, nbrs_ff)
                buffer_nl_ff = np.array(buffer_nbrs_ff.idx)
                output_precompute = utils.precompute(buffer_positions, buffer_gaussian_grid_spacing, buffer_nl_ff, buffer_nl_lub, buffer_displacements_vector_matrix, xy,
                                                 N, Lx, Ly, Lz, Nx, Ny, Nz,
                                                 prefac, expfac, quadW,
                                                 gaussP, gaussPd2,
                                                 ewald_n, ewald_dr, ewald_cut, ewaldC1,
                                                 ResTable_min, ResTable_dr, ResTable_dist, ResTable_vals, alpha_friction, ho_friction)
                saddle_x, exitcode_gmres = solver.solver(N,saddle_b,  # rhs vector of the linear system
                                                         gridk, R_fu_prec_lower_triang, output_precompute,Nx,Ny,Nz,gaussP,m_self)

            elif(boundary_flag==1):
                output_precompute = utils.precompute_open(buffer_positions,nl_ff, buffer_nl_lub, buffer_displacements_vector_matrix,
                                                          ResTable_min, ResTable_dr, ResTable_dist, ResTable_vals, alpha_friction, ho_friction)
                saddle_x, exitcode_gmres = solver.solver_open(N,saddle_b,  # rhs vector of the linear system
                                                         R_fu_prec_lower_triang, output_precompute,
                                                         displacements_vector_matrix.at[nl_ff[0, :], nl_ff[1, :]].get())
            if(exitcode_gmres > 0):
                raise ValueError(
                    f"GMRES (RFD) did not converge! Iterations are {exitcode_gmres}. Abort!")

            # compute the near-field hydrodynamic stresslet from the saddle point solution velocity
            if((stresslet_flag > 0) and ((step % writing_period) == 0)):
                buffer_stresslet = resistance.compute_RSU(
                        jnp.zeros((N, 5), float),
                        saddle_x.at[11*N:].get(),
                        indices_i_lub,
                        indices_j_lub,
                        output_precompute[18],
                        r_lub,
                        N)

            # TAKE THE DIFFERENCE AND APPLY SCALING
            brownian_drift += (-saddle_x.at[11*N:].get())
            brownian_drift = -brownian_drift * T / epsilon
            if((stresslet_flag > 0) and ((step % writing_period) == 0)):
                stresslet += -buffer_stresslet
                stresslet = stresslet * T / epsilon

            # reset RHS to zero for next saddle point solver
            saddle_b = jnp.zeros(17*N, float)

        # add applied forces and conservative (pair potential) forces to right-hand side of system
        saddle_b = appliedForces.sumAppliedForces(N, AppliedForce, AppliedTorques, saddle_b, U,
                                                  indices_i_lub, indices_j_lub, displacements_vector_matrix,
                                                  U_cutoff, 2, dt)

        # add (-) the ambient rate of strain to the right-hand side
        if(shear_rate_0 != 0):
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
            key_ffreal, random_array_real = utils.generate_random_array(
                    key_ffreal, (11*N))

            # compute far-field (real space contribution) slip velocity and set in rhs of linear system
            if(boundary_flag == 0):
                    key_ffwave, random_array_wave = utils.generate_random_array(
                            key_ffwave, (3 * 2 * len(wave_bro_ind[:,0,0]) + 3 * len(wave_bro_nyind[:,0])))
                    # compute far-field (wave space contribution) slip velocity and set in rhs of linear system
                    ws_linvel, ws_angvel_strain = thermal.compute_wave_space_slipvelocity(N, int(Nx), int(Ny), int(Nz), int(gaussP), T, dt, gridh,
                                                                                              wave_bro_ind[:,0,0], wave_bro_ind[:,0,1], wave_bro_ind[:,0,2],
                                                                                              wave_bro_ind[:,1,0], wave_bro_ind[:,1,1], wave_bro_ind[:,1,2],
                                                                                              wave_bro_nyind[:,0], wave_bro_nyind[:,1], wave_bro_nyind[:,2],
                                                                                              gridk, random_array_wave, all_indices_x, all_indices_y, all_indices_z,
                                                                                              gaussian_grid_spacing1, gaussian_grid_spacing2)
                    rs_linvel, rs_angvel_strain, stepnormff, diag_ff = thermal.compute_real_space_slipvelocity(
                        N, m_self, T, dt, int(n_iter_Lanczos_ff),
                        random_array_real, r, indices_i, indices_j,
                        f1, f2, g1, g2, h1, h2, h3)
                    while((stepnormff > 1e-3) and (n_iter_Lanczos_ff < 150)):
                        n_iter_Lanczos_ff += 20
                        rs_linvel, rs_angvel_strain, stepnormff, diag_ff = thermal.compute_real_space_slipvelocity(
                            N, m_self, T, dt, int(n_iter_Lanczos_ff),
                            random_array_real, r, indices_i, indices_j,
                            f1, f2, g1, g2, h1, h2, h3)
                    saddle_b = saddle_b.at[:11*N].add(thermal.convert_to_generalized(
                            N, ws_linvel, rs_linvel, ws_angvel_strain, rs_angvel_strain))
                    
            elif(boundary_flag == 1):
                    rs_linvel, rs_angvel_strain, stepnormff, diag_ff = thermal.compute_real_space_slipvelocity_open(
                        N, T, dt, int(n_iter_Lanczos_ff),
                        random_array_real, r, indices_i, indices_j, mobil_scalar)
                    while((stepnormff > 0.01) and (n_iter_Lanczos_ff < 150)):
                        n_iter_Lanczos_ff += 5
                        rs_linvel, rs_angvel_strain, stepnormff, diag_ff = thermal.compute_real_space_slipvelocity_open(
                            N, T, dt, int(n_iter_Lanczos_ff),
                            random_array_real, r, indices_i, indices_j, mobil_scalar)
                    saddle_b = saddle_b.at[:11*N].add(thermal.convert_to_generalized(
                            N, jnp.zeros_like(rs_linvel), rs_linvel, jnp.zeros_like(rs_angvel_strain), rs_angvel_strain))
            
            
            # check that far-field real space thermal fluctuation calculation went well
            if ((not math.isfinite(stepnormff)) or ((n_iter_Lanczos_ff > 150) and (stepnormff > 0.02))):
                overlap_monitor(displacements_vector_matrix,N)
                raise ValueError(
                    f"Far-field Lanczos did not converge! Stepnorm is {stepnormff}, iterations are {n_iter_Lanczos_ff}. Eigenvalues of tridiagonal matrix are {diag_ff}. Abort!")

            # compute lubrication contribution only if there is more than 1 particle
            stepnormnf = 0.
            if(N > 1):
                # compute near-field random forces 
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
                while((stepnormnf > 1e-3) and (n_iter_Lanczos_nf < 250)):
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
                saddle_b = saddle_b.at[11*N:].add(-buffer) # set in rhs of linear system
                # check that far-field real space thermal fluctuation calculation went well
                if ((not math.isfinite(stepnormnf)) or ((n_iter_Lanczos_nf > 250) and (stepnormnf > 1e-3))):
                    overlap_monitor(displacements_vector_matrix,N)
                    raise ValueError(
                        f"Near-field Lanczos did not converge! Stepnorm is {stepnormnf}, iterations are {n_iter_Lanczos_nf}. Eigenvalues of tridiagonal matrix are {diag_nf}. Abort!")

        # solve the system Ax=b, where x contains the unknown particle velocities (relative to the background flow) and stresslet
        if(boundary_flag==0):
            saddle_x, exitcode_gmres = solver.solver(N,saddle_b,  # rhs vector of the linear system
                                                     gridk, R_fu_prec_lower_triang, 
                                                     [all_indices_x, all_indices_y, all_indices_z, gaussian_grid_spacing1, gaussian_grid_spacing2,
                                                      r, indices_i, indices_j, f1, f2, g1, g2, h1, h2, h3,
                                                      r_lub, indices_i_lub, indices_j_lub, ResFunction],
                                                     Nx,Ny,Nz,gaussP,m_self)
        elif(boundary_flag==1):
            saddle_x, exitcode_gmres = solver.solver_open(N,saddle_b,  # rhs vector of the linear system
                                                     R_fu_prec_lower_triang, [r, indices_i, indices_j, r_lub, indices_i_lub, indices_j_lub, ResFunction, mobil_scalar],
                                                     displacements_vector_matrix.at[nl_ff[0, :], nl_ff[1, :]].get())
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
                if output is not None:
                    np.save(output/"stresslet.npy", stresslet_history)

        (positions, displacements_vector_matrix) = update_positions(shear_rate,
                                                                        positions, displacements_vector_matrix,
                                                                        saddle_x.at[11*N:].get() +
                                                                        brownian_drift, dt)
        nbrs_lub_prec = utils.update_nlist(positions, nbrs_lub_prec)
        nbrs_lub = utils.update_nlist(positions, nbrs_lub)
        nl_lub = np.array(nbrs_lub.idx)  # extract lists in sparse format
        if(boundary_flag == 0):
                nbrs_ff = utils.update_nlist(positions, nbrs_ff)
                nl_ff = np.array(nbrs_ff.idx)  # extract lists in sparse format
                # update grid distances for FFT (needed for wave space calculation of mobility)
                gaussian_grid_spacing = utils.Precompute_grid_distancing(
                    gaussP, gridh[0], xy, positions, N, Nx, Ny, Nz, Lx, Ly, Lz)

        # if system is sheared, strain wave-vectors grid and update box tilt factor
        if(shear_rate_0 != 0):
            xy = shear.update_box_tilt_factor(
                dt, shear_rate_0, xy, step, shear_freq)
            displacement, shift = space.periodic_general(
                jnp.array([[Lx, Ly*xy, Lz*0.], [0., Ly, Lz*0.], [0., 0., Lz]]), fractional_coordinates=False)
            if(boundary_flag == 0):
                gridk = shear.compute_sheared_grid(
                    int(Nx), int(Ny), int(Nz), xy, Lx, Ly, Lz, eta, xisq)

        # reset Lanczos number of iteration once in a while (avoid to do too many iterations if not needed, can be tuned for better performance)
        elif((step % 100) == 0):
            n_iter_Lanczos_ff = 5
            n_iter_Lanczos_nf = 5

        # write trajectory to file
        if((step % writing_period) == 0):

            # check that the position to save do not contain 'nan' or 'inf'
            if((jnp.isnan(positions)).any() or (jnp.isinf(positions)).any()):
                raise ValueError(
                    "Invalid particles positions. Abort!")

            # check that current configuration does not have overlapping particles
            overlap_monitor(displacements_vector_matrix,N)

            # save trajectory to file
            trajectory[int(step/writing_period), :, :] = positions
            if output is not None:
                np.save(output/"trajectory.npy", trajectory)

            # store velocity (linear and angular) to file
            if((velocity_flag > 0)):
                velocities[int(step/writing_period), :, :] = jnp.reshape(
                    saddle_x.at[11*N:].get() + brownian_drift, (N, 6))
                if output is not None:
                    np.save(output/'velocities.npy', velocities)

            # store orientation
            # TODO
            
    # perform thermal test if needed
    test_result = 0.
    if((T > 0) and (thermal_test_flag == 1)):
        ff, nf = thermal.compute_exact_thermals(
            N, m_self,
            T, dt,
            random_array_nf,
            random_array_real,
            r,
            indices_i,
            indices_j,
            f1, f2, g1, g2, h1, h2, h3,
            r_lub,
            indices_i_lub,
            indices_j_lub,
            ResFunction[0], ResFunction[1], ResFunction[
                2], ResFunction[3], ResFunction[4], ResFunction[5],
            ResFunction[6], ResFunction[7], ResFunction[
                8], ResFunction[9], ResFunction[10])

        test_result = [jnp.linalg.norm(buffer-nf) / jnp.linalg.norm(nf),
                       jnp.linalg.norm(thermal.convert_to_generalized(
                           N, 0, rs_linvel, 0, rs_angvel_strain)-ff)/jnp.linalg.norm(ff),
                       n_iter_Lanczos_nf,
                       stepnormnf]

    return trajectory, stresslet_history, velocities, test_result

def wrap_RPY(
        Nsteps: int,
        writing_period: int,
        dt: float,
        Lx: float, Ly: float, Lz: float,
        N: int,
        max_strain: float,
        T: float,
        xi: float,
        error: float,
        U: float,
        buoyancy_flag: int,
        U_cutoff: float,
        positions: ArrayLike,
        seed_ffwave: int,
        seed_ffreal: int,
        shear_rate_0: float,
        shear_freq: float,
        output: str,
        velocity_flag: bool,
        orient_flag: bool,
        constant_applied_forces: ArrayLike,
        constant_applied_torques: ArrayLike,
        boundary_flag: int) -> tuple[Array, Array, Array, list[float]]:
    """Wrap all functions needed to integrate the particles equation of motions forward in time, using RPY method.    

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
    U: (float)
        Interaction strength
    buoyancy_flag: (int)
        Set to 1 to have gravity acting on colloids
    U_cutoff: (float)
        Distance cutoff for interacting particles
    positions: (float)
        Array of particles initial positions (N,3)
    seed_ffwave: (int)
        Seed for wave space part of far-field velocity slip
    seed_ffreal: (int)
        Seed for real space part of far-field velocity slip
    shear_rate_0: (float)
        Axisymmetric shear rate amplitude
    shear_freq: (float)
        Frequency of shear, set to zero to have simple shear
    output: (str)
        File name for output
    velocity_flag: (int)
        To have velocities in the output/var/log/nvidia-installer.log
    orient_flag: (int)
        To have particle orientations in the output
    constant_applied_forces: (float)
        Array of external forces (N,3)
    constant_applied_torques: (float)
        Array of external torques (N,3)
    boundary_flag: (int)
        Flag used to set the type of boundary conditions for the hydrodynamic interaction.

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
            dt: float) -> tuple[Array, Array]:
        """Update particle positions and neighbor lists

        Parameters
        ----------
        
        shear_rate: (float)
            Shear rate at current time step
        positions: (float)
            Array (N,3) of particles positions
        displacements_vector_matrix: (float)
            Array (N,N,3) of relative displacements between particles
        net_vel: (float)
            Array (6*N) of linear/angular velocities relative to the background flow
        dt: (float)
            Timestep used to advance positions

        Returns
        -------
        positions, displacements_vector_matrix

        """

        # Define array of displacement r(t+dt)-r(t)
        dR = jnp.zeros((N, 3), float)
        # Compute actual displacement due to velocities (relative to background flow)
        dR = dR.at[:, 0].set(dt*net_vel.at[(0)::6].get())
        dR = dR.at[:, 1].set(dt*net_vel.at[(1)::6].get())
        dR = dR.at[:, 2].set(dt*net_vel.at[(2)::6].get())
        # Apply displacement and compute wrapped shift (Lees Edwards boundary conditions)
        positions = shift(positions+jnp.array([Lx, Ly, Lz])/2, dR) - jnp.array([Lx, Ly, Lz])*0.5 
        
        # Define array of displacement r(t+dt)-r(t) (this time for displacement given by background flow)
        dR = jnp.zeros((N, 3), float)
        dR = dR.at[:, 0].set(dt * shear_rate * positions.at[:, 1].get()) # Assuming y:gradient direction, x:background flow direction
        positions = shift(positions+jnp.array([Lx, Ly, Lz])/2, dR) - jnp.array([Lx, Ly, Lz]) * 0.5 # Apply shift

        # Compute new relative displacements between particles
        displacements_vector_matrix = (space.map_product(displacement))(positions, positions)
        return positions, displacements_vector_matrix
    
    if output is not None:
        output = Path(output)
        output.mkdir(exist_ok=True, parents=True)

    # set array for output trajectory, velocities and stresslet in time
    trajectory = np.zeros((int(Nsteps/writing_period), N, 3), float)
    velocities = np.zeros((int(Nsteps/writing_period), N, 6), float)

    #  set INITIAL Periodic Space and Displacement Metric
    xy = 0.  # set box tilt factor to zero to begin (unsheared box)
    displacement, shift = space.periodic_general(
        jnp.array([[Lx, Ly*xy, Lz*0.], [0., Ly, Lz*0.], [0., 0., Lz]]), fractional_coordinates=False)
    # compute matrix of INITIAL displacements between particles (each element is a vector from particle j to i)
    displacements_vector_matrix = (
        space.map_product(displacement))(positions, positions)

    # set external applied forces/torques (no pair-interactions, will be added later)
    AppliedForce = jnp.zeros(3*N, float)
    AppliedTorques = jnp.zeros(3*N, float)
    if(buoyancy_flag == 1):  # apply buoyancy forces (in z direction)
        AppliedForce = AppliedForce.at[2::3].add(-1.)
    if(np.count_nonzero(constant_applied_forces) > 0):  # apply external forces
        AppliedForce += jnp.ravel(constant_applied_forces)
    if(np.count_nonzero(constant_applied_torques) > 0):  # apply external torques
        AppliedTorques += jnp.ravel(constant_applied_torques)

    # compute the Real Space cutoff for the Ewald Summation in the Far-Field computation (used in general to build a neighborlist)
    ewald_cut = jnp.sqrt(- jnp.log(error)) / xi
    # initialize far-field hydrodynamics neighborlists
    ff_neighbor_fn = utils.initialize_single_neighborlist(
        ewald_cut, Lx, Ly, Lz, displacement)
    # allocate neighborlist for first time
    nbrs_ff = ff_neighbor_fn.allocate(positions+jnp.array([Lx, Ly, Lz])/2)
    # convert to array
    nl_ff = np.array(nbrs_ff.idx)
    n_iter_Lanczos_ff = 2  # set initial Lanczos iterations, for thermal fluctuation calculation
    if(boundary_flag==0):
        (quadW,prefac,expfac,gaussPd2,gridk,gridh,gaussian_grid_spacing,key_ffwave,
         ewaldC1,m_self,Nx,Ny,Nz,gaussP,
         ewald_n,ewald_dr,eta,xisq,
         wave_bro_ind,wave_bro_nyind) = utils.init_periodic_box(error,xi,
                              Lx,Ly,Lz, ewald_cut, max_strain, xy,
                              positions, N, T, seed_ffwave)
    elif(boundary_flag == 1):
        nl_ff = utils.compute_distinct_pairs(N)

    # create RNG states for real space thermal fluctuations
    key_ffreal = random.PRNGKey(seed_ffreal)

    # check if particles overlap
    overlaps, overlaps_indices = utils.check_overlap(
        displacements_vector_matrix, 2.)
    if(overlaps > 0):
        print('Warning: initial overlaps are ', (overlaps))
    print('Starting: compiling the code... This should not take more than 1-2 minutes.')

    for step in tqdm(range(Nsteps),mininterval=0.5):
            
        # initialize generalized velocity (6*N array, for linear and angular components)
        # this array stores the velocity for Brownian Dynamics, or the Brownian drift otherwise
        general_velocity = jnp.zeros(6*N, float)

        # define rhs array of the linear system Ax=b (with A the saddle point matrix)
        saddle_b = jnp.zeros(17*N, float)

        if(boundary_flag == 0):
            if(nbrs_ff.did_buffer_overflow):
                nbrs_ff = utils.allocate_nlist(positions, ff_neighbor_fn)
                nl_ff = np.array(nbrs_ff.idx)
            # precompute quantities for periodic boundaries hydrodynamic calculation
            (all_indices_x, all_indices_y, all_indices_z, gaussian_grid_spacing1, gaussian_grid_spacing2,
             r, indices_i, indices_j, f1, f2, g1, g2, h1, h2, h3) = utils.precomputeRPY(positions, gaussian_grid_spacing, nl_ff, displacements_vector_matrix, xy,
                                                                                        N, Lx, Ly, Lz, Nx, Ny, Nz,
                                                                                        prefac, expfac, quadW,
                                                                                        int(gaussP), gaussPd2,
                                                                                        ewald_n, ewald_dr, ewald_cut, ewaldC1)
        elif(boundary_flag==1):
            (r, indices_i, indices_j,mobil_scalar) = utils.precomputeRPY_open(positions,nl_ff,displacements_vector_matrix)
        # compute shear-rate for current timestep: simple(shear_freq=0) or oscillatory(shear_freq>0)
        shear_rate = shear.update_shear_rate(
            dt, step, shear_rate_0, shear_freq, phase=0)

        # add applied forces and conservative (pair potential) forces to right-hand side of system
        saddle_b = appliedForces.sumAppliedForces(N, AppliedForce, AppliedTorques, saddle_b, U,
                                                  indices_i, indices_j, displacements_vector_matrix,
                                                  U_cutoff, 1, dt)

        # compute Thermal Fluctuations only if temperature is not zero
        if(T > 0):
    
            # compute far-field (real space contribution) slip velocity and set in rhs of linear system
            if(boundary_flag == 0):
                key_ffwave, random_array_wave = utils.generate_random_array(
                    key_ffwave, (3 * 2 * len(wave_bro_ind[:,0,0]) + 3 * len(wave_bro_nyind[:,0])))
                key_ffreal, random_array_real = utils.generate_random_array(
                    key_ffreal, (11*N))

                # compute far-field (wave space contribution) slip velocity and set in rhs of linear system
                ws_linvel, ws_angvel_strain = thermal.compute_wave_space_slipvelocity(N, int(Nx), int(Ny), int(Nz), int(gaussP), T, dt, gridh,
                                                                                      wave_bro_ind[:,0,0],wave_bro_ind[:,0,1],wave_bro_ind[:,0,2],
                                                                                      wave_bro_ind[:,1,0],wave_bro_ind[:,1,1],wave_bro_ind[:,1,2],
                                                                                      wave_bro_nyind[:,0],wave_bro_nyind[:,1],wave_bro_nyind[:,2],
                                                                                      gridk, random_array_wave, all_indices_x, all_indices_y, all_indices_z,
                                                                                      gaussian_grid_spacing1, gaussian_grid_spacing2)
                rs_linvel, rs_angvel_strain, stepnormff, diag_ff = thermal.compute_real_space_slipvelocity(
                    N, m_self, T, dt, int(n_iter_Lanczos_ff),
                    random_array_real, r, indices_i, indices_j,
                    f1, f2, g1, g2, h1, h2, h3)
                while((stepnormff > 1e-3) and (n_iter_Lanczos_ff < 150)):
                        n_iter_Lanczos_ff += 20
                        rs_linvel, rs_angvel_strain, stepnormff, diag_ff = thermal.compute_real_space_slipvelocity(
                            N, m_self, T, dt, int(n_iter_Lanczos_ff),
                            random_array_real, r, indices_i, indices_j,
                            f1, f2, g1, g2, h1, h2, h3)
                #combine real- and wave-space part of thermal fluctuation
                saddle_b = saddle_b.at[:11*N].add(thermal.convert_to_generalized(
                    N, ws_linvel, rs_linvel, ws_angvel_strain, rs_angvel_strain))
                
            elif(boundary_flag == 1):
                rs_linvel, rs_angvel_strain, stepnormff, diag_ff = thermal.compute_real_space_slipvelocity_open(
                    N, T, dt, int(n_iter_Lanczos_ff),
                    random_array_real, r,indices_i, indices_j,mobil_scalar)
                while((stepnormff > 0.003) and (n_iter_Lanczos_ff < 150)):
                    n_iter_Lanczos_ff += 20
                    rs_linvel, rs_angvel_strain, stepnormff, diag_ff = thermal.compute_real_space_slipvelocity_open(
                        N, T, dt, int(n_iter_Lanczos_ff),
                        random_array_real, r,indices_i, indices_j,mobil_scalar)
                #convert real-space part of thermal fluctuation into a generalized velocity
                saddle_b = saddle_b.at[:11*N].add(thermal.convert_to_generalized(
                    N, jnp.zeros_like(rs_linvel), rs_linvel, jnp.zeros_like(rs_angvel_strain), rs_angvel_strain))

            # check that thermal fluctuation calculation went well
            if ((not math.isfinite(stepnormff)) or ((n_iter_Lanczos_ff > 150) and (stepnormff >  0.003))):
                overlap_monitor(displacements_vector_matrix,N)
                raise ValueError(
                    f"Far-field Lanczos did not converge! Stepnorm is {stepnormff}, iterations are {n_iter_Lanczos_ff}. Eigenvalues of tridiagonal matrix are {diag_ff}. Abort!")

        # add random velocity to total velocity in RPY
        general_velocity += saddle_b[:6*N]
        # add potential force contribution to total velocity in RPY
        if(boundary_flag == 0):
            general_velocity += mobility.Mobility_periodic(N, Nx, Ny, Nz,
                                                           gaussP, gridk, m_self,
                                                           all_indices_x, all_indices_y, all_indices_z,
                                                           gaussian_grid_spacing1, gaussian_grid_spacing2,
                                                           r, indices_i, indices_j, f1, f2, g1, g2, h1, h2, h3,
                                                           -saddle_b[11*N:])
        elif(boundary_flag == 1):
            general_velocity += mobility.Mobility_open(N, r, displacements_vector_matrix.at[nl_ff[0, :], nl_ff[1, :]].get(),
                                                           indices_i, indices_j,-saddle_b[11*N:],mobil_scalar)
        # update positions and neighborlists
        (positions, displacements_vector_matrix) = update_positions(shear_rate, positions,
                                                                        displacements_vector_matrix,
                                                                        general_velocity,
                                                                        dt)
        if(boundary_flag == 0):
            nbrs_ff = utils.update_nlist(positions, nbrs_ff)
            nl_ff = np.array(nbrs_ff.idx)  # extract lists in sparse format

        # if system is sheared, strain wave-vectors grid and update box tilt factor
        if(shear_rate_0 != 0):
            xy = shear.update_box_tilt_factor(
                dt, shear_rate_0, xy, step, shear_freq)
            displacement, shift = space.periodic_general(
                jnp.array([[Lx, Ly*xy, Lz*0.], [0., Ly, Lz*0.], [0., 0., Lz]]), fractional_coordinates=False)
            if(boundary_flag == 0):
                gridk = shear.compute_sheared_grid(
                    int(Nx), int(Ny), int(Nz), xy, Lx, Ly, Lz, eta, xisq)

        # reset Lanczos number of iteration once in a while (avoid to do too many iterations if not needed, can be tuned for better performance)
        elif((step % 100) == 0):
            n_iter_Lanczos_ff = 5
            
        # write trajectory to file
        if((step % writing_period) == 0):

            # check that the position to save do not contain 'nan' or 'inf'
            if((jnp.isnan(positions)).any() or (jnp.isinf(positions)).any()):
                raise ValueError(
                    "Invalid particles positions. Abort!")

            # check that current configuration does not have overlapping particles
            overlap_monitor(displacements_vector_matrix,N)

            # save trajectory to file
            trajectory[int(step/writing_period), :, :] = positions
            if output is not None:
                np.save(output/"trajectory.npy", trajectory)

            # store velocity (linear and angular) to file
            if((velocity_flag > 0)):
                velocities[int(step/writing_period), :, :] = jnp.reshape(general_velocity, (N, 6))
                if output is not None:
                    np.save(output/'velocities.npy', velocities)
                    
            # store orientation
            # TODO            
    return trajectory, velocities

def wrap_BD(
        Nsteps: int,
        writing_period: int,
        dt: float,
        Lx: float, Ly: float, Lz: float,
        N: int,
        T: float,
        U: float,
        buoyancy_flag: int,
        U_cutoff: float,
        positions: ArrayLike,
        seed: int,
        shear_rate_0: float,
        shear_freq: float,
        output: str,
        velocity_flag: bool,
        orient_flag: bool,
        constant_applied_forces: ArrayLike,
        constant_applied_torques: ArrayLike) -> tuple[Array, Array]:
    """Wrap all functions needed to integrate the particles equation of motions forward in time, using Brownian Dynamics method.

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
    T: (float)
        Thermal energy
    U: (float)
        Interaction strength
    buoyancy_flag: (int)
        Set to 1 to have gravity acting on colloids
    U_cutoff: (float)
        Distance cutoff for interacting particles
    positions: (float)
        Array of particles initial positions (N,3)
    seed: (int)
        Seed for random forces
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

    Returns
    -------
        trajectory, , velocities

    """
    
    @jit
    def update_positions(
            shear_rate: float,
            positions: ArrayLike,
            displacements_vector_matrix: ArrayLike,
            net_vel: ArrayLike,
            dt: float) -> tuple[Array, Array]:
        """Update particle positions and neighbor lists

        Parameters
        ----------
        
        shear_rate: (float)
            Shear rate at current time step
        positions: (float)
            Array (N,3) of particles positions
        displacements_vector_matrix: (float)
            Array (N,N,3) of relative displacements between particles
        net_vel: (float)
            Array (6*N) of linear/angular velocities relative to the background flow
        dt: (float)
            Timestep used to advance positions

        Returns
        -------
        positions, displacements_vector_matrix

        """

        # Define array of displacement r(t+dt)-r(t)
        dR = jnp.zeros((N, 3), float)
        # Compute actual displacement due to velocities (relative to background flow)
        dR = dR.at[:, 0].set(dt*net_vel.at[(0)::6].get())
        dR = dR.at[:, 1].set(dt*net_vel.at[(1)::6].get())
        dR = dR.at[:, 2].set(dt*net_vel.at[(2)::6].get())
        # Apply displacement and compute wrapped shift (Lees Edwards boundary conditions)
        positions = shift(positions+jnp.array([Lx, Ly, Lz])/2, dR) - jnp.array([Lx, Ly, Lz])*0.5 
        
        # Define array of displacement r(t+dt)-r(t) (this time for displacement given by background flow)
        dR = jnp.zeros((N, 3), float)
        dR = dR.at[:, 0].set(dt * shear_rate * positions.at[:, 1].get()) # Assuming y:gradient direction, x:background flow direction
        positions = shift(positions+jnp.array([Lx, Ly, Lz])/2, dR) - jnp.array([Lx, Ly, Lz]) * 0.5 # Apply shift

        # Compute new relative displacements between particles
        displacements_vector_matrix = (space.map_product(displacement))(positions, positions)
        return positions, displacements_vector_matrix

    if output is not None:
        output = Path(output)
        output.mkdir(exist_ok=True, parents=True)

    # set array for output trajectory, velocities and stresslet in time
    trajectory = np.zeros((int(Nsteps/writing_period), N, 3), float)
    velocities = np.zeros((int(Nsteps/writing_period), N, 6), float)

    #  set INITIAL Periodic Space and Displacement Metric
    xy = 0.  # set box tilt factor to zero to begin (unsheared box)
    displacement, shift = space.periodic_general(
        jnp.array([[Lx, Ly*xy, Lz*0.], [0., Ly, Lz*0.], [0., 0., Lz]]), fractional_coordinates=False)

    # compute matrix of INITIAL displacements between particles (each element is a vector from particle j to i)
    displacements_vector_matrix = (
        space.map_product(displacement))(positions, positions)

    # initialize near-field hydrodynamics neighborlists (also used for pair potential)
    neighbor_fn = utils.initialize_single_neighborlist(
        4., Lx, Ly, Lz, displacement)
    # allocate neighborlist for first time
    nbrs = neighbor_fn.allocate(positions+jnp.array([Lx, Ly, Lz])/2)
    # convert to array
    nl = np.array(nbrs.idx)

    # set external applied forces/torques (no pair-interactions, will be added later)
    AppliedForce = jnp.zeros(3*N, float)
    AppliedTorques = jnp.zeros(3*N, float)
    if(buoyancy_flag == 1):  # apply buoyancy forces (in z direction)
        AppliedForce = AppliedForce.at[2::3].add(-1.)
    if(np.count_nonzero(constant_applied_forces) > 0):  # apply external forces
        AppliedForce += jnp.ravel(constant_applied_forces)
    if(np.count_nonzero(constant_applied_torques) > 0):  # apply external torques
        AppliedTorques += jnp.ravel(constant_applied_torques)

    # create RNG states
    key = random.PRNGKey(seed)

    # check if particles overlap
    overlaps, overlaps_indices = utils.check_overlap(
        displacements_vector_matrix, 2.)
    if(overlaps > 0):
        print('Warning: initial overlaps are ', (overlaps))
    print('Starting: compiling the code... This should not take more than 1-2 minutes.')

    for step in tqdm(range(Nsteps),mininterval=0.5):
        # check that neighborlists buffers did not overflow, if so, re-allocate the lists
        if(nbrs.did_buffer_overflow):
            nbrs = utils.allocate_nlist(positions, neighbor_fn)
            nl = np.array(nbrs.idx)

        # initialize generalized velocity (6*N array, for linear and angular components)
        # this array stores the velocity for Brownian Dynamics, or the Brownian drift otherwise
        general_velocity = jnp.zeros(6*N, float)

        # define rhs array of the linear system Ax=b (with A the saddle point matrix)
        saddle_b = jnp.zeros(17*N, float)

        # precompute quantities for Brownian dynamics calculation
        (r, indices_i, indices_j) = utils.precomputeBD(
                positions, nl, displacements_vector_matrix, N, Lx, Ly, Lz)

        # compute shear-rate for current timestep: simple(shear_freq=0) or oscillatory(shear_freq>0)
        shear_rate = shear.update_shear_rate(
            dt, step, shear_rate_0, shear_freq, phase=0)

        # add applied forces and conservative (pair potential) forces to right-hand side of system
        saddle_b = appliedForces.sumAppliedForces(N, AppliedForce, AppliedTorques, saddle_b, U,
                                                  indices_i, indices_j, displacements_vector_matrix,
                                                  U_cutoff, 0, dt)
        
        # compute Thermal Fluctuations only if temperature is not zero
        if(T > 0):
            # generate random numbers for the various contributions of thermal noise
            key, random_array = utils.generate_random_array(
                key, (6*N))

            # compute random force for Brownian Dynamics
            random_velocity = thermal.compute_BD_randomforce(
                    N, T, dt, random_array)
            general_velocity += random_velocity

        # add potential force contribution to total velocity (thermal contribution is already included)
        general_velocity += - saddle_b[11*N:]
        # update positions
        (positions, displacements_vector_matrix) = update_positions(shear_rate, positions,
                                                                        displacements_vector_matrix,
                                                                        general_velocity,dt)

        nbrs = utils.update_nlist(positions, nbrs)
        nl = np.array(nbrs.idx)  # extract lists in sparse format

        # if system is sheared, strain wave-vectors grid and update box tilt factor
        if(shear_rate_0 != 0):
            xy = shear.update_box_tilt_factor(
                dt, shear_rate_0, xy, step, shear_freq)
            displacement, shift = space.periodic_general(
                jnp.array([[Lx, Ly*xy, Lz*0.], [0., Ly, Lz*0.], [0., 0., Lz]]), fractional_coordinates=False)

        # write trajectory to file
        if((step % writing_period) == 0):

            # check that the position to save do not contain 'nan' or 'inf'
            if((jnp.isnan(positions)).any() or (jnp.isinf(positions)).any()):
                raise ValueError(
                    "Invalid particles positions. Abort!")

            # check that current configuration does not have overlapping particles
            overlap_monitor(displacements_vector_matrix,N)

            # save trajectory to file
            trajectory[int(step/writing_period), :, :] = positions
            if output is not None:
                np.save(output/"trajectory.npy", trajectory)

            # store velocity (linear and angular) to file
            if((velocity_flag > 0)):
                velocities[int(step/writing_period), :, :] = jnp.reshape(
                    general_velocity, (N, 6))
                if output is not None:
                    np.save(output/'velocities.npy', velocities)

            # store orientation (TODO)
    return trajectory, velocities


