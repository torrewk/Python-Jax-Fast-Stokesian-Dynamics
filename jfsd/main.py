import math
import os
from pathlib import Path
import scipy.sparse as sp

import jax.numpy as jnp
import numpy as np
from jax import Array, jit, random
from jax.config import config
from jax.lib import xla_bridge
from jax.typing import ArrayLike

from tqdm import tqdm

from jfsd import applied_forces, mobility, resistance, shear, solver, thermal, utils
from jfsd import jaxmd_space as space

config.update("jax_enable_x64", False)  # Disable double precision by default
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # Avoid JAX preallocating most GPU memory

def main(
    num_steps: int,
    writing_period: int,
    time_step: float,
    lx: float,
    ly: float,
    lz: float,
    num_particles: int,
    max_strain: float,
    temperature: float,
    particle_radius: float,
    ewald_xi: float,
    error_tolerance: float,
    interaction_strength: float,
    buoyancy_flag: int,
    interaction_cutoff: float,
    positions: ArrayLike,
    seed_rfd: int,
    seed_ffwave: int,
    seed_ffreal: int,
    seed_nf: int,
    shear_rate_0: float,
    shear_frequency: float,
    output: str,
    store_stresslet: bool,
    store_velocity: bool,
    store_orientation: bool,
    constant_applied_forces: ArrayLike,
    constant_applied_torques: ArrayLike,
    hydrodynamic_interaction_flag: int,
    boundary_flag: int,
    thermal_test_flag: int,
    friction_coefficient: float,
    friction_range: float,
) -> tuple[Array, Array, Array, list[float]]:
    """Integrate the particles' equations of motion forward in time.
    
    While the simulation runs, trajectory data are saved into a .npy file.
    
    Parameters
    ----------
    num_steps : int
        Number of simulation timesteps.
    writing_period : int
        Frequency of writing output to file.
    time_step : float
        Simulation timestep.
    lx : float
        Length of the simulation box in x-direction.
    ly : float
        Length of the simulation box in y-direction.
    lz : float
        Length of the simulation box in z-direction.
    num_particles : int
        Number of particles in the system.
    max_strain : float
        Maximum strain applied to the box.
    temperature : float
        Thermal energy.
    particle_radius : float
        Radius of each particle.
    ewald_xi : float
        Ewald splitting parameter.
    error_tolerance : float
        Error tolerance for numerical computations.
    interaction_strength : float
        Strength of pairwise interactions.
    buoyancy_flag : int
        If 1, applies gravitational forces to the particles.
    interaction_cutoff : float
        Cutoff distance for interaction forces.
    positions : ArrayLike
        Initial positions of the particles, shape (num_particles, 3).
    seed_rfd : int
        Seed for Brownian Drift computation.
    seed_ffwave : int
        Seed for wave-space component of far-field velocity.
    seed_ffreal : int
        Seed for real-space component of far-field velocity.
    seed_nf : int
        Seed for near-field random forces.
    shear_rate_0 : float
        Initial shear rate amplitude.
    shear_frequency : float
        Frequency of applied shear (0 for simple shear).
    output : str
        Name of the output file.
    store_stresslet : bool
        Whether to store the stresslet data.
    store_velocity : bool
        Whether to store velocity data.
    store_orientation : bool
        Whether to store particle orientation data.
    constant_applied_forces : ArrayLike
        External forces applied to particles, shape (num_particles, 3).
    constant_applied_torques : ArrayLike
        External torques applied to particles, shape (num_particles, 3).
    hydrodynamic_interaction_flag : int
        Flag determining the level of hydrodynamic interactions (0: BD, 1: RPY, 2: SD).
    boundary_flag : int
        Flag determining boundary conditions (0: periodic, 1: open).
    thermal_test_flag : int
        Flag to test thermal fluctuations (1 for far-field real-space, 2 for lubrication).
    friction_coefficient : float
        Coefficient for hydrodynamic friction.
    friction_range : float
        Range of hydrodynamic friction effects.
    
    Returns
    -------
    tuple
        (trajectory, stresslet_history, velocities, test_results)
    """
    trajectory = stresslet_history = velocities = test_results = None
    
    if writing_period > num_steps:
        raise ValueError(
            "Error: writing-to-file period is greater than the total number of simulation steps."
        )
    
    if boundary_flag == 1:
        if num_particles < 2 and hydrodynamic_interaction_flag > 0:
            raise ValueError(
                "Error: Open boundary hydrodynamics cannot be used for a single particle. "
                "Select 'brownian' instead."
            )
        lx = ly = lz = 999999  # Effectively infinite box size.
        config.update("jax_enable_x64", True)  # Enable double precision for long-range interactions.
    
    print("jfsd is running on device:", xla_bridge.get_backend().platform)
    
    if hydrodynamic_interaction_flag == 0:
        trajectory, velocities = wrap_bd(
            num_steps,
            writing_period,
            time_step,
            lx,
            ly,
            lz,
            num_particles,
            temperature,
            interaction_strength,
            buoyancy_flag,
            interaction_cutoff,
            positions,
            seed_nf,
            shear_rate_0,
            shear_frequency,
            output,
            store_velocity,
            store_orientation,
            constant_applied_forces,
            constant_applied_torques,
        )
    elif hydrodynamic_interaction_flag == 1:
        trajectory, velocities = wrap_rpy(
            num_steps,
            writing_period,
            time_step,
            lx,
            ly,
            lz,
            num_particles,
            max_strain,
            temperature,
            ewald_xi,
            error_tolerance,
            interaction_strength,
            buoyancy_flag,
            interaction_cutoff,
            positions,
            seed_ffwave,
            seed_ffreal,
            shear_rate_0,
            shear_frequency,
            output,
            store_velocity,
            store_orientation,
            constant_applied_forces,
            constant_applied_torques,
            boundary_flag,
        )
    elif hydrodynamic_interaction_flag == 2:
        trajectory, stresslet_history, velocities, test_results = wrap_sd(
            num_steps,
            writing_period,
            time_step,
            lx,
            ly,
            lz,
            num_particles,
            max_strain,
            temperature,
            particle_radius,
            ewald_xi,
            error_tolerance,
            interaction_strength,
            buoyancy_flag,
            interaction_cutoff,
            positions,
            seed_rfd,
            seed_ffwave,
            seed_ffreal,
            seed_nf,
            shear_rate_0,
            shear_frequency,
            output,
            store_stresslet,
            store_velocity,
            store_orientation,
            constant_applied_forces,
            constant_applied_torques,
            boundary_flag,
            thermal_test_flag,
            friction_coefficient,
            friction_range,
        )
    return trajectory, stresslet_history, velocities, test_results


def check_overlap(positions: ArrayLike, num_particles: int, 
                  nlist: ArrayLike, box: ArrayLike) -> bool:
    """Check for overlapping particles in the current configuration.

    Prints indices and distances of overlapping pairs.

    Parameters
    ----------
    positions : ArrayLike
        Array (num_particles, 3) of particle positions.
    num_particles : int
        Number of particles.
    nlist : ArrayLike
        Neighbor list.
    Box : ArrayLike
        Array (3, 3) defining the box in 3 dimensions. 
        
    Returns
    -------
    bool
        True if overlaps are present, False otherwise.
    """
    overlaps = utils.find_overlaps(
        positions, 2.0, num_particles, nlist, box
    )
    if overlaps > 0:
        print(
            f"Warning: {overlaps} particles are overlapping. "
            "Reducing the timestep might help prevent unphysical overlaps."
        )
    return overlaps > 0



def wrap_sd(
    num_steps: int,
    writing_period: int,
    time_step: float,
    lx: float,
    ly: float,
    lz: float,
    num_particles: int,
    max_strain: float,
    temperature: float,
    particle_radius: float,
    ewald_xi: float,
    error_tolerance: float,
    interaction_strength: float,
    buoyancy_flag: int,
    interaction_cutoff: float,
    positions: ArrayLike,
    seed_rfd: int,
    seed_ffwave: int,
    seed_ffreal: int,
    seed_nf: int,
    shear_rate_0: float,
    shear_frequency: float,
    output: str,
    store_stresslet: bool,
    store_velocity: bool,
    store_orientation: bool,
    constant_applied_forces: ArrayLike,
    constant_applied_torques: ArrayLike,
    boundary_flag: int,
    thermal_test_flag: int,
    friction_coefficient: float,
    friction_range: float,
    max_nonzero_per_row: int = 100,
) -> tuple[Array, Array, Array, list[float]]:
    """Wrap all functions needed to integrate the particles equation of motions forward in time, using Stokesian Dynamics method.

    Parameters
    ----------
    num_steps: (int)
        Number of timesteps
    writing_period: (int)
        Period for writing to file
    time_step: (float)
        Timestep
    lx: (float)
        Box size (x-direction)
    ly: (float)
        Box size (y-direction)
    lz: (float)
        Box size (z-direction)
    num_particles: (int)
        Number of particles
    max_strain: (float)
        Max strain applied to the box
    temperature: (float)
        Thermal energy
    particle_radius: (float)
        Particle radius
    ewald_xi: (float)
        Ewald split parameter
    error_tolerance: (float)
        Tolerance error
    interaction_strength: (float)
        Interaction strength
    buoyancy_flag: (int)
        Set to 1 to have gravity acting on colloids
    interaction_cutoff: (float)
        Distance cutoff for interacting particles
    positions: (float)
        Array of particles initial positions (num_particles,3)
    seed_rfd: (int)
        Seed for Brownian Drift calculation
    seed_ffwave: (int)
        Seed for wave space part of far-field velocity slip
    seed_ffreal: (int)
        Seed for real space part of far-field velocity slip
    seed_nf: (int)
        Seed for near-field random forces
    shear_rate_0: (float)
        Axisymmetric shear rate amplitude
    shear_frequency: (float)
        Frequency of shear, set to zero to have simple shear
    output: (str)
        File name for output
    store_stresslet: (int)
        To have stresslet in the output
    store_velocity: (int)
        To have velocities in the output/var/log/nvidia-installer.log
    store_orientation: (int)
        To have particle orientations in the output
    constant_applied_forces: (float)
        Array of external forces (num_particles,3)
    constant_applied_torques: (float)
        Array of external torques (num_particles,3)
    boundary_flag: (int)
        Flag used to set the type of boundary conditions for the hydrodynamic interaction.
    thermal_test_flag: (int)
        Flag used to test thermal fluctuation calculation (1 for far-field real space, 2 for lubrication)
    friction_coefficient: (float)
        Strength of hydrodynamic friction
    h0_friction: (float)
        Range of hydrodynamic friction
    max_nonzero_per_row: (int)
        Max number of non-zero entries per row in the sparse near-field precondition resistance matrix.

    Returns
    -------
        trajectory, stresslet_history, velocities, test_result

    """

    @jit
    def update_positions(
        shear_rate: float,
        positions: ArrayLike,
        net_vel: ArrayLike,
        time_step: float,
    ) -> tuple[Array, Array]:
        """Update particle positions and neighbor lists

        Parameters
        ----------
        shear_rate: (float)
            Shear rate at current time step
        positions: (float)
            Array (num_particles,3) of particles positions
        net_vel: (float)
            Array (6*num_particles) of linear/angular velocities relative to the background flow
        time_step: (float)
            Timestep used to advance positions

        Returns
        -------
        positions (in-place update)

        """
        # Define array of displacement r(t+time_step)-r(t)
        dR = jnp.zeros((num_particles, 3), float)
        # Compute actual displacement due to velocities (relative to background flow)
        dR = dR.at[:, 0].set(time_step * net_vel[(0)::6])
        dR = dR.at[:, 1].set(time_step * net_vel[(1)::6])
        dR = dR.at[:, 2].set(time_step * net_vel[(2)::6])
        # Apply displacement and compute wrapped shift (Lees Edwards boundary conditions)
        positions = (
            shift_fn(positions + jnp.array([lx, ly, lz]) / 2, dR) - jnp.array([lx, ly, lz]) * 0.5
        )

        # Define array of displacement r(t+time_step)-r(t) (this time for displacement given by background flow)
        dR = jnp.zeros((num_particles, 3), float)
        dR = dR.at[:, 0].set(
            time_step * shear_rate * positions[:, 1]
        )  # Assuming y:gradient direction, x:background flow direction
        positions = (
            shift_fn(positions + jnp.array([lx, ly, lz]) / 2, dR) - jnp.array([lx, ly, lz]) * 0.5
        )  # Apply shift

        return positions
    
    if output is not None:
        output = Path(output)
        output.mkdir(exist_ok=True, parents=True)

    # set array for output trajectory, velocities and stresslet in time
    trajectory = np.zeros((int(num_steps / writing_period), num_particles, 3), float)
    stresslet_history = np.zeros((int(num_steps / writing_period), num_particles, 5), float)
    velocities = np.zeros((int(num_steps / writing_period), num_particles, 6), float)

    # set initial number of Lanczos iterations, for both thermal fluctuations
    n_iter_Lanczos_ff = 2
    n_iter_Lanczos_nf = 2

    epsilon = error_tolerance  # define epsilon for RFD
    xy = 0.0  # set box tilt factor to zero to begin (unsheared box)
    ewald_cut = (
        jnp.sqrt(-jnp.log(error_tolerance)) / ewald_xi
    )  # Real Space cutoff for the Ewald Summation in the Far-Field computation
    
    # load resistance table
    ResTable_dist = jnp.load("files/ResTableDist.npy")
    ResTable_vals = jnp.load("files/ResTableVals.npy")
    ResTable_min = 0.0001000000000000
    ResTable_dr = 0.0043050000000000  # table discretization (log space), i.e. ind * dr.set(log10( h[ind] / min )

    # set INITIAL Periodic Space and Displacement Metric
    box = jnp.array([[lx, ly * xy, lz * 0.0], [0.0, ly, lz * 0.0], [0.0, 0.0, lz]])
    _, shift_fn = space.periodic_general(box, fractional_coordinates=False)
    # compute neighbor lists 
    unique_pairs, nl_ff, nl_lub, nl_prec, nl_safety_margin = utils.cpu_nlist(positions, np.array([lx,ly,lz]), ewald_cut, 3.99, 2.1, xy)

    if boundary_flag == 0:
        # Initialize periodic hydrodynamic quantities
        (
            quadW,
            prefac,
            expfac,
            gaussPd2,
            gridk,
            gridh,
            gaussian_grid_spacing,
            key_ffwave,
            ewaldC1,
            m_self,
            grid_x,
            grid_y,
            grid_z,
            gauss_support,
            ewald_n,
            ewald_dr,
            eta,
            xisq,
            wave_bro_ind,
            wave_bro_nyind,
        ) = utils.init_periodic_box(
            error_tolerance, ewald_xi, lx, ly, lz, ewald_cut, max_strain, xy, positions, num_particles, temperature, seed_ffwave
        )
    if boundary_flag == 1:
        nl_open = utils.compute_distinct_pairs(
            num_particles
        )  # compute list of distinct pairs for long range hydrodynamics (not optimized)

    if temperature > 0:  # create Random Number Generator states
        key_rfd = random.PRNGKey(seed_rfd)
        key_ffreal = random.PRNGKey(seed_ffreal)
        key_nf = random.PRNGKey(seed_nf)

    # set external applied forces/torques (no pair-interactions, will be added later)
    external_forces = jnp.zeros(3 * num_particles, float)
    external_torques = jnp.zeros(3 * num_particles, float)
    if buoyancy_flag == 1:  # apply buoyancy forces (in z direction)
        external_forces = external_forces.at[2::3].add(-1.0)
    if np.count_nonzero(constant_applied_forces) > 0:  # apply external forces
        external_forces += jnp.ravel(constant_applied_forces)
    if np.count_nonzero(constant_applied_torques) > 0:  # apply external torques
        external_torques += jnp.ravel(constant_applied_torques)
    # Check if particles overlap
    overlaps = utils.find_overlaps(positions, 2.0, num_particles, unique_pairs, box)
    if overlaps > 0:
        print("Warning: initial overlaps are ", (overlaps))
    print("Starting: compiling the code... This should not take more than 1-2 minutes.")
    for step in tqdm(range(num_steps), mininterval=0.5):  # use tqdm to have progress bar and TPS
    
        # Initialize Brownian drift (6*num_particlesarray, for linear and angular components)
        brownian_drift = jnp.zeros(6 * num_particles, float)

        if (store_stresslet > 0) and ((step % writing_period) == 0):
            stresslet = jnp.zeros((num_particles, 5), float)  # reset stresslet to zero

        # define rhs array of the linear system Ax=b (with A the saddle point matrix)
        saddle_b = jnp.zeros(17 * num_particles, float)

        # precompute quantities for far-field and near-field hydrodynamic calculation
        if boundary_flag == 0:
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
                ResFunction,
            ) = utils.precompute(
                positions,
                gaussian_grid_spacing,
                nl_ff,
                nl_lub,
                xy,
                num_particles,
                lx,
                ly,
                lz,
                grid_x,
                grid_y,
                grid_z,
                prefac,
                expfac,
                quadW,
                int(gauss_support),
                gaussPd2,
                ewald_n,
                ewald_dr,
                ewald_cut,
                ewaldC1,
                ResTable_min,
                ResTable_dr,
                ResTable_dist,
                ResTable_vals,
                friction_coefficient,
                friction_range,
            )
        elif boundary_flag == 1:
            (
                r,
                indices_i,
                indices_j,
                r_lub,
                indices_i_lub,
                indices_j_lub,
                ResFunction,
                mobil_scalar,
            ) = utils.precompute_open(
                positions,
                nl_open,
                nl_lub,
                ResTable_min,
                ResTable_dr,
                ResTable_dist,
                ResTable_vals,
                friction_coefficient,
                friction_range,
            )
                
        # compute shear-rate for current timestep: simple(shear_frequency=0) or oscillatory(shear_frequency>0)
        shear_rate = shear.update_shear_rate(time_step, step, shear_rate_0, shear_frequency, phase=0)

        # set projector needed to compute thermal fluctuations given by lubrication
        diagonal_zeroes_for_brownian = thermal.number_of_neigh_jit(num_particles, indices_i_lub, indices_j_lub)
        
        row, col, data = resistance.rfu_sparse_precondition(positions, num_particles, len(nl_prec[1,:]), nl_prec, box)
        idx_buffer = np.where((row < 6 * num_particles))
        spp_rfu = sp.csr_matrix((data[idx_buffer], (row[idx_buffer], col[idx_buffer])), shape=(6 * num_particles, 6 * num_particles), dtype=np.float64())
        
        spp_rfu = spp_rfu + spp_rfu.T - sp.diags(spp_rfu.diagonal())
        
        diag = np.where((np.arange(6 * num_particles) - 6 * (np.repeat(jnp.arange(num_particles), 6))) < 3, 1 , 4/3)
        spp_rfu = spp_rfu + sp.diags(diag,format='csr')
        
        r_prec_sparse = utils.preprocess_sparse_triangular((spp_rfu), 6*num_particles, max_nonzero_per_row)
        
        # if temperature is not zero (and full hydrodynamics are switched on), compute Brownian Drift
        if temperature > 0:
            # get array of random variables
            key_rfd, random_array = utils.generate_random_array(key_rfd, 6 * num_particles)
            random_array = -((2 * random_array - 1) * jnp.sqrt(3))
            # add random displacement to right-hand side of linear system Ax=b
            saddle_b = saddle_b.at[11 * num_particles :].set(random_array)

            # SOLVE SADDLE POINT IN THE POSITIVE DIRECTION
            # perform a displacement in the positive random directions (and update wave grid) and save it to a buffer
            buffer_positions = update_positions(
                shear_rate, positions, random_array, epsilon / 2.0
            )
            if boundary_flag == 0:
                # update wave grid and far-field neighbor list (not needed in open boundaries)
                buffer_gaussian_grid_spacing = utils.precompute_grid_distancing(
                    gauss_support, gridh[0], xy, buffer_positions, num_particles, grid_x, grid_y, grid_z, lx, ly, lz
                )
                output_precompute = utils.precompute(
                    buffer_positions,
                    buffer_gaussian_grid_spacing,
                    nl_ff,
                    nl_lub,
                    xy,
                    num_particles,
                    lx,
                    ly,
                    lz,
                    grid_x,
                    grid_y,
                    grid_z,
                    prefac,
                    expfac,
                    quadW,
                    gauss_support,
                    gaussPd2,
                    ewald_n,
                    ewald_dr,
                    ewald_cut,
                    ewaldC1,
                    ResTable_min,
                    ResTable_dr,
                    ResTable_dist,
                    ResTable_vals,
                    friction_coefficient,
                    friction_range,
                )
                saddle_x, exitcode_gmres = solver.solve_linear_system(
                        num_particles,
                        saddle_b,  # rhs vector of the linear system
                        gridk,
                        output_precompute,
                        grid_x,
                        grid_y,
                        grid_z,
                        gauss_support,
                        m_self,
                        max_nonzero_per_row,
                        r_prec_sparse,
                    )
            elif boundary_flag == 1:
                output_precompute = utils.precompute_open(
                    buffer_positions,
                    nl_open,
                    nl_lub,
                    ResTable_min,
                    ResTable_dr,
                    ResTable_dist,
                    ResTable_vals,
                    friction_coefficient,
                    friction_range,
                )
                saddle_x, exitcode_gmres = solver.solve_linear_system_open(
                        num_particles,
                        saddle_b,  # rhs vector of the linear system
                        output_precompute,
                        buffer_positions[nl_open[1,:]] - buffer_positions[nl_open[0,:]],
                        max_nonzero_per_row,
                        r_prec_sparse
                    )
            if exitcode_gmres > 0:
                raise ValueError(
                    "GMRES (RFD) did not converge. Abort!"
                )
            brownian_drift = saddle_x[11 * num_particles :]

            # compute the near-field hydrodynamic stresslet from the saddle point solution velocity
            if (store_stresslet > 0) and ((step % writing_period) == 0):
                stresslet = resistance.compute_rsu(
                    stresslet,
                    brownian_drift,
                    output_precompute[16],
                    output_precompute[17],
                    output_precompute[18],
                    output_precompute[15],
                    num_particles,
                )
        
            # SOLVE SADDLE POINT IN THE NEGATIVE DIRECTION
            buffer_positions = update_positions(
                shear_rate, positions, random_array, -epsilon / 2.0
            )
            if boundary_flag == 0:
                buffer_gaussian_grid_spacing = utils.precompute_grid_distancing(
                    gauss_support, gridh[0], xy, buffer_positions, num_particles, grid_x, grid_y, grid_z, lx, ly, lz
                )
                output_precompute = utils.precompute(
                    buffer_positions,
                    buffer_gaussian_grid_spacing,
                    nl_ff,
                    nl_lub,
                    xy,
                    num_particles,
                    lx,
                    ly,
                    lz,
                    grid_x,
                    grid_y,
                    grid_z,
                    prefac,
                    expfac,
                    quadW,
                    gauss_support,
                    gaussPd2,
                    ewald_n,
                    ewald_dr,
                    ewald_cut,
                    ewaldC1,
                    ResTable_min,
                    ResTable_dr,
                    ResTable_dist,
                    ResTable_vals,
                    friction_coefficient,
                    friction_range,
                )
                saddle_x, exitcode_gmres = solver.solve_linear_system(
                    num_particles,
                    saddle_b,  # rhs vector of the linear system
                    gridk,
                    output_precompute,
                    grid_x,
                    grid_y,
                    grid_z,
                    gauss_support,
                    m_self,
                    max_nonzero_per_row,
                    r_prec_sparse,
                    saddle_x                    
                )

            elif boundary_flag == 1:
                output_precompute = utils.precompute_open(
                    buffer_positions,
                    nl_open,
                    nl_lub,
                    ResTable_min,
                    ResTable_dr,
                    ResTable_dist,
                    ResTable_vals,
                    friction_coefficient,
                    friction_range,
                )
                saddle_x, exitcode_gmres = solver.solve_linear_system_open(
                    num_particles,
                    saddle_b,  # rhs vector of the linear system
                    output_precompute,
                    buffer_positions[nl_open[1,:]] - buffer_positions[nl_open[0,:]],
                    max_nonzero_per_row,
                    r_prec_sparse,
                    saddle_x
                )
            if exitcode_gmres > 0:
                raise ValueError(
                    "GMRES (RFD) did not converge. Abort!"
                )

            # compute the near-field hydrodynamic stresslet from the saddle point solution velocity
            if (store_stresslet > 0) and ((step % writing_period) == 0):
                buffer_stresslet = resistance.compute_rsu(
                    jnp.zeros((num_particles, 5), float),
                    saddle_x[11 * num_particles :],
                    output_precompute[16],
                    output_precompute[17],
                    output_precompute[18],
                    output_precompute[15],
                    num_particles,
                )

            # TAKE THE DIFFERENCE AND APPly SCALING
            brownian_drift += -saddle_x[11 * num_particles :]
            brownian_drift = -brownian_drift * temperature / epsilon
            if (store_stresslet > 0) and ((step % writing_period) == 0):
                stresslet += -buffer_stresslet
                stresslet = stresslet * temperature / epsilon

            # reset RHS to zero for next saddle point solver
            saddle_b = jnp.zeros(17 * num_particles, float)

        # add applied forces and conservative (pair potential) forces to right-hand side of system
        saddle_b = applied_forces.sum_applied_forces(
            num_particles,
            external_forces,
            external_torques,
            saddle_b,
            interaction_strength,
            indices_i_lub,
            indices_j_lub,
            positions,
            interaction_cutoff,
            2,
            time_step,
            box
        )

        # add (-) the ambient rate of strain to the right-hand side
        if shear_rate_0 != 0:
            saddle_b = saddle_b.at[(6 * num_particles + 1) : (11 * num_particles) : 5].add(-shear_rate)
            # compute near field shear contribution R_FE and add it to the rhs of the system
            saddle_b = saddle_b.at[11 * num_particles :].add(
                resistance.compute_rfe(
                    num_particles,
                    shear_rate,
                    r_lub,
                    indices_i_lub,
                    indices_j_lub,
                    ResFunction[11],
                    ResFunction[12],
                    ResFunction[13],
                    ResFunction[14],
                    ResFunction[15],
                    ResFunction[16],
                    -ResFunction[12],
                    -ResFunction[14],
                    ResFunction[16],
                )
            )
        
        # compute Thermal Fluctuations only if temperature is not zero
        if temperature > 0:
            # generate random numbers for the various contributions of thermal noise
            key_nf, random_array_nf = utils.generate_random_array(key_nf, (6 * num_particles))
            key_ffreal, random_array_real = utils.generate_random_array(key_ffreal, (11 * num_particles))

            # compute far-field (real space contribution) slip velocity and set in rhs of linear system
            if boundary_flag == 0:
                key_ffwave, random_array_wave = utils.generate_random_array(
                    key_ffwave, (3 * 2 * len(wave_bro_ind[:, 0, 0]) + 3 * len(wave_bro_nyind[:, 0]))
                )
                # compute far-field (wave space contribution) slip velocity and set in rhs of linear system
                ws_linvel, ws_angvel_strain = thermal.compute_wave_space_slipvelocity(
                    num_particles,
                    int(grid_x),
                    int(grid_y),
                    int(grid_z),
                    int(gauss_support),
                    temperature,
                    time_step,
                    gridh,
                    wave_bro_ind[:, 0, 0],
                    wave_bro_ind[:, 0, 1],
                    wave_bro_ind[:, 0, 2],
                    wave_bro_ind[:, 1, 0],
                    wave_bro_ind[:, 1, 1],
                    wave_bro_ind[:, 1, 2],
                    wave_bro_nyind[:, 0],
                    wave_bro_nyind[:, 1],
                    wave_bro_nyind[:, 2],
                    gridk,
                    random_array_wave,
                    all_indices_x,
                    all_indices_y,
                    all_indices_z,
                    gaussian_grid_spacing1,
                    gaussian_grid_spacing2,
                )
                rs_linvel, rs_angvel_strain, stepnormff, diag_ff = (
                    thermal.compute_real_space_slipvelocity(
                        num_particles,
                        m_self,
                        temperature,
                        time_step,
                        int(n_iter_Lanczos_ff),
                        random_array_real,
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
                    )
                )
                while (stepnormff > 1e-3) and (n_iter_Lanczos_ff < 150):
                    n_iter_Lanczos_ff += 20
                    rs_linvel, rs_angvel_strain, stepnormff, diag_ff = (
                        thermal.compute_real_space_slipvelocity(
                            num_particles,
                            m_self,
                            temperature,
                            time_step,
                            int(n_iter_Lanczos_ff),
                            random_array_real,
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
                        )
                    )
                saddle_b = saddle_b.at[: 11 * num_particles].add(
                    thermal.convert_to_generalized(
                        num_particles, ws_linvel, rs_linvel, ws_angvel_strain, rs_angvel_strain
                    )
                )

            elif boundary_flag == 1:
                rs_linvel, rs_angvel_strain, stepnormff, diag_ff = (
                    thermal.compute_real_space_slipvelocity_open(
                        num_particles,
                        temperature,
                        time_step,
                        int(n_iter_Lanczos_ff),
                        random_array_real,
                        r,
                        indices_i,
                        indices_j,
                        mobil_scalar,
                    )
                )
                while (stepnormff > 0.01) and (n_iter_Lanczos_ff < 150):
                    n_iter_Lanczos_ff += 5
                    rs_linvel, rs_angvel_strain, stepnormff, diag_ff = (
                        thermal.compute_real_space_slipvelocity_open(
                            num_particles,
                            temperature,
                            time_step,
                            int(n_iter_Lanczos_ff),
                            random_array_real,
                            r,
                            indices_i,
                            indices_j,
                            mobil_scalar,
                        )
                    )
                saddle_b = saddle_b.at[: 11 * num_particles].add(
                    thermal.convert_to_generalized(
                        num_particles,
                        jnp.zeros_like(rs_linvel),
                        rs_linvel,
                        jnp.zeros_like(rs_angvel_strain),
                        rs_angvel_strain,
                    )
                )

            # check that far-field real space thermal fluctuation calculation went well
            if (not math.isfinite(stepnormff)) or (
                (n_iter_Lanczos_ff > 150) and (stepnormff > 0.02)
            ):
                check_overlap(positions, num_particles, unique_pairs, box)
                raise ValueError(
                    f"Far-field Lanczos did not converge! Stepnorm is {stepnormff}, iterations are {n_iter_Lanczos_ff}. Eigenvalues of tridiagonal matrix are {diag_ff}. Abort!"
                )

            # compute lubrication contribution only if there is more than 1 particle
            stepnormnf = 0.0
            if num_particles > 1:
                # compute near-field random forces
                buffer, stepnormnf, diag_nf = thermal.compute_nearfield_brownianforce(
                    num_particles,
                    temperature,
                    time_step,
                    random_array_nf,
                    r_lub,
                    indices_i_lub,
                    indices_j_lub,
                    ResFunction[0],
                    ResFunction[1],
                    ResFunction[2],
                    ResFunction[3],
                    ResFunction[4],
                    ResFunction[5],
                    ResFunction[6],
                    ResFunction[7],
                    ResFunction[8],
                    ResFunction[9],
                    ResFunction[10],
                    diagonal_zeroes_for_brownian,
                    n_iter_Lanczos_nf,
                )
                while (stepnormnf > 1e-3) and (n_iter_Lanczos_nf < 250):
                    n_iter_Lanczos_nf += 20
                    buffer, stepnormnf, diag_nf = thermal.compute_nearfield_brownianforce(
                        num_particles,
                        temperature,
                        time_step,
                        random_array_nf,
                        r_lub,
                        indices_i_lub,
                        indices_j_lub,
                        ResFunction[0],
                        ResFunction[1],
                        ResFunction[2],
                        ResFunction[3],
                        ResFunction[4],
                        ResFunction[5],
                        ResFunction[6],
                        ResFunction[7],
                        ResFunction[8],
                        ResFunction[9],
                        ResFunction[10],
                        diagonal_zeroes_for_brownian,
                        n_iter_Lanczos_nf,
                    )
                saddle_b = saddle_b.at[11 * num_particles :].add(-buffer)  # set in rhs of linear system
                # check that far-field real space thermal fluctuation calculation went well
                if (not math.isfinite(stepnormnf)) or (
                    (n_iter_Lanczos_nf > 250) and (stepnormnf > 1e-3)
                ):
                    check_overlap(positions, num_particles, unique_pairs, box)
                    raise ValueError(
                        f"Near-field Lanczos did not converge! Stepnorm is {stepnormnf}, iterations are {n_iter_Lanczos_nf}. Eigenvalues of tridiagonal matrix are {diag_nf}. Abort!"
                    )
        
        # solve the system Ax=b, where x contains the unknown particle velocities (relative to the background flow) and stresslet
        if boundary_flag == 0:
                saddle_x, exitcode_gmres = solver.solve_linear_system(
                    num_particles,
                    saddle_b,  # rhs vector of the linear system
                    gridk,
                    [
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
                        ResFunction,
                    ],
                    grid_x,
                    grid_y,
                    grid_z,
                    gauss_support,
                    m_self,
                    max_nonzero_per_row,
                    r_prec_sparse
                )
        elif boundary_flag == 1:
                saddle_x, exitcode_gmres = solver.solve_linear_system_open(
                    num_particles,
                    saddle_b,  # rhs vector of the linear system
                    [
                        r,
                        indices_i,
                        indices_j,
                        r_lub,
                        indices_i_lub,
                        indices_j_lub,
                        ResFunction,
                        mobil_scalar,
                    ],
                    positions[nl_open[1,:]] - positions[nl_open[0,:]],
                    max_nonzero_per_row,
                    r_prec_sparse
                )
        if exitcode_gmres > 0:
            raise ValueError(f"GMRES did not converge! Iterations are {exitcode_gmres}. Abort!")
        
        # add the near-field contributions to the stresslet
        if (store_stresslet > 0) and ((step % writing_period) == 0):
            # get stresslet out of saddle point solution (and add it to the contribution from the Brownian drift, if temperature>0 )
            stresslet += jnp.reshape(-saddle_x[6 * num_particles : 11 * num_particles], (num_particles, 5))
            stresslet = resistance.compute_rsu(
                stresslet,
                saddle_x[11 * num_particles :],
                indices_i_lub,
                indices_j_lub,
                ResFunction,
                r_lub,
                num_particles,
            )
            # add shear (near-field) contributions to the stresslet
            if shear_rate_0 != 0:
                stresslet = resistance.compute_rse(
                    num_particles,
                    shear_rate,
                    r_lub,
                    indices_i_lub,
                    indices_j_lub,
                    ResFunction[17],
                    ResFunction[18],
                    ResFunction[19],
                    ResFunction[20],
                    ResFunction[21],
                    ResFunction[22],
                    stresslet,
                )
            # save stresslet
            stresslet_history[int(step / writing_period), :, :] = stresslet
            if output is not None:
                np.save(output / "stresslet.npy", stresslet_history)
        # Update positions
        positions = update_positions(
            shear_rate,
            positions,
            saddle_x[11 * num_particles :] + brownian_drift,
            time_step,
        )
        
        # Update neighborlists
        nl_ff, nl_lub, nl_prec, nl_list_bound = utils.update_neighborlist(num_particles, positions, ewald_cut, 3.99, 2.1, unique_pairs, box)
        if not nl_list_bound: # Re-allocate list if number of neighbors exceeded list size.     
            unique_pairs, nl_ff, nl_lub, nl_prec, _ = utils.cpu_nlist(positions, np.array([lx,ly,lz]), ewald_cut, 3.99, 2.1, xy)
        if boundary_flag == 0:
            gaussian_grid_spacing = utils.precompute_grid_distancing(
                gauss_support, gridh[0], xy, positions, num_particles, grid_x, grid_y, grid_z, lx, ly, lz
            )
            
        # if system is sheared, strain wave-vectors grid and update box tilt factor
        if shear_rate_0 != 0:
            xy = shear.update_box_tilt_factor(time_step, shear_rate_0, xy, step, shear_frequency)
            box = jnp.array([[lx, ly * xy, lz * 0.0], [0.0, ly, lz * 0.0], [0.0, 0.0, lz]])
            _, shift_fn = space.periodic_general(box, fractional_coordinates=False
            )
            if boundary_flag == 0:
                gridk = shear.compute_sheared_grid(
                    int(grid_x), int(grid_y), int(grid_z), xy, lx, ly, lz, eta, xisq
                )

        # reset Lanczos number of iteration once in a while (avoid to do too many iterations if not needed, can be tuned for better performance)
        elif (step % 100) == 0:
            n_iter_Lanczos_ff = 5
            n_iter_Lanczos_nf = 5

        # write trajectory to file
        if (step % writing_period) == 0:
            # check that the position to save do not contain 'nan' or 'inf'
            if (jnp.isnan(positions)).any() or (jnp.isinf(positions)).any():
                raise ValueError("Invalid particles positions. Abort!")

            # check that current configuration does not have overlapping particles
            check_overlap(positions, num_particles, unique_pairs, box)

            # save trajectory to file
            trajectory[int(step / writing_period), :, :] = positions
            if output is not None:
                np.save(output / "trajectory.npy", trajectory)

            # store velocity (linear and angular) to file
            if store_velocity > 0:
                velocities[int(step / writing_period), :, :] = jnp.reshape(
                    saddle_x[11 * num_particles :] + brownian_drift, (num_particles, 6)
                )
                if output is not None:
                    np.save(output / "velocities.npy", velocities)

            # store orientation
            # TODO
        
    # perform thermal test if needed
    test_result = 0.0
    if (temperature > 0) and (thermal_test_flag == 1):
        ff, nf = thermal.compute_exact_thermals(
            num_particles,
            m_self,
            temperature,
            time_step,
            random_array_nf,
            random_array_real,
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
            ResFunction[0],
            ResFunction[1],
            ResFunction[2],
            ResFunction[3],
            ResFunction[4],
            ResFunction[5],
            ResFunction[6],
            ResFunction[7],
            ResFunction[8],
            ResFunction[9],
            ResFunction[10],
        )

        test_result = [
            jnp.linalg.norm(buffer - nf) / jnp.linalg.norm(nf),
            jnp.linalg.norm(
                thermal.convert_to_generalized(num_particles, 0, rs_linvel, 0, rs_angvel_strain) - ff
            )
            / jnp.linalg.norm(ff),
            n_iter_Lanczos_nf,
            stepnormnf,
        ]

    return trajectory, stresslet_history, velocities, test_result


def wrap_rpy(
    num_steps: int,
    writing_period: int,
    time_step: float,
    lx: float,
    ly: float,
    lz: float,
    num_particles: int,
    max_strain: float,
    temperature: float,
    ewald_xi: float,
    error_tolerance: float,
    interaction_strength: float,
    buoyancy_flag: int,
    interaction_cutoff: float,
    positions: ArrayLike,
    seed_ffwave: int,
    seed_ffreal: int,
    shear_rate_0: float,
    shear_frequency: float,
    output: str,
    store_velocity: bool,
    store_orientation: bool,
    constant_applied_forces: ArrayLike,
    constant_applied_torques: ArrayLike,
    boundary_flag: int,
) -> tuple[Array, Array, Array, list[float]]:
    """Wrap all functions needed to integrate the particles equation of motions forward in time, using RPY method.

    Parameters
    ----------
    num_steps: (int)
        Number of timesteps
    writing_period: (int)
        Period for writing to file
    time_step: (float)
        Timestep
    lx: (float)
        Box size (x-direction)
    ly: (float)
        Box size (y-direction)
    lz: (float)
        Box size (z-direction)
    num_particles: (int)
        Number of particles
    max_strain: (float)
        Max strain applied to the box
    temperature: (float)
        Thermal energy
    particle_radius: (float)
        Particle radius
    ewald_xi: (float)
        Ewald split parameter
    interaction_strength: (float)
        Interaction strength
    buoyancy_flag: (int)
        Set to 1 to have gravity acting on colloids
    interaction_cutoff: (float)
        Distance cutoff for interacting particles
    positions: (float)
        Array of particles initial positions (num_particles,3)
    seed_ffwave: (int)
        Seed for wave space part of far-field velocity slip
    seed_ffreal: (int)
        Seed for real space part of far-field velocity slip
    shear_rate_0: (float)
        Axisymmetric shear rate amplitude
    shear_frequency: (float)
        Frequency of shear, set to zero to have simple shear
    output: (str)
        File name for output
    store_velocity: (int)
        To have velocities in the output/var/log/nvidia-installer.log
    store_orientation: (int)
        To have particle orientations in the output
    constant_applied_forces: (float)
        Array of external forces (num_particles,3)
    constant_applied_torques: (float)
        Array of external torques (num_particles,3)
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
        net_vel: ArrayLike,
        time_step: float,
    ) -> tuple[Array, Array]:
        """Update particle positions and neighbor lists

        Parameters
        ----------
        shear_rate: (float)
            Shear rate at current time step
        positions: (float)
            Array (num_particles,3) of particles positions
        net_vel: (float)
            Array (6*num_particles) of linear/angular velocities relative to the background flow
        time_step: (float)
            Timestep used to advance positions

        Returns
        -------
        positions (in-place update)

        """
        # Define array of displacement r(t+time_step)-r(t)
        dR = jnp.zeros((num_particles, 3), float)
        # Compute actual displacement due to velocities (relative to background flow)
        dR = dR.at[:, 0].set(time_step * net_vel[(0)::6])
        dR = dR.at[:, 1].set(time_step * net_vel[(1)::6])
        dR = dR.at[:, 2].set(time_step * net_vel[(2)::6])
        # Apply displacement and compute wrapped shift (Lees Edwards boundary conditions)
        positions = (
            shift_fn(positions + jnp.array([lx, ly, lz]) / 2, dR) - jnp.array([lx, ly, lz]) * 0.5
        )

        # Define array of displacement r(t+time_step)-r(t) (this time for displacement given by background flow)
        dR = jnp.zeros((num_particles, 3), float)
        dR = dR.at[:, 0].set(
            time_step * shear_rate * positions[:, 1]
        )  # Assuming y:gradient direction, x:background flow direction
        positions = (
            shift_fn(positions + jnp.array([lx, ly, lz]) / 2, dR) - jnp.array([lx, ly, lz]) * 0.5
        )  # Apply shift

        return positions

    if output is not None:
        output = Path(output)
        output.mkdir(exist_ok=True, parents=True)
    
    # set array for output trajectory, velocities and stresslet in time
    trajectory = np.zeros((int(num_steps / writing_period), num_particles, 3), float)
    velocities = np.zeros((int(num_steps / writing_period), num_particles, 6), float)

    #  set INITIAL Periodic Space and Displacement Metric
    xy = 0.0  # set box tilt factor to zero to begin (unsheared box)
    _, shift_fn = space.periodic_general(
        jnp.array([[lx, ly * xy, lz * 0.0], [0.0, ly, lz * 0.0], [0.0, 0.0, lz]]),
        fractional_coordinates=False,
    )
    box = jnp.array([[lx, ly * xy, lz * 0.0], [0.0, ly, lz * 0.0], [0.0, 0.0, lz]])
    # set external applied forces/torques (no pair-interactions, will be added later)
    external_forces = jnp.zeros(3 * num_particles, float)
    external_torques = jnp.zeros(3 * num_particles, float)
    if buoyancy_flag == 1:  # apply buoyancy forces (in z direction)
        external_forces = external_forces.at[2::3].add(-1.0)
    if np.count_nonzero(constant_applied_forces) > 0:  # apply external forces
        external_forces += jnp.ravel(constant_applied_forces)
    if np.count_nonzero(constant_applied_torques) > 0:  # apply external torques
        external_torques += jnp.ravel(constant_applied_torques)

    # compute the Real Space cutoff for the Ewald Summation in the Far-Field computation (used in general to build a neighborlist)
    ewald_cut = jnp.sqrt(-jnp.log(error_tolerance)) / ewald_xi
    
    # compute neighbor lists 
    unique_pairs, nl_ff, _, _, nl_safety_margin = utils.cpu_nlist(positions, np.array([lx,ly,lz]), ewald_cut, 0., 0., xy)
    
    n_iter_Lanczos_ff = 2  # set initial Lanczos iterations, for thermal fluctuation calculation
    if boundary_flag == 0:
        (
            quadW,
            prefac,
            expfac,
            gaussPd2,
            gridk,
            gridh,
            gaussian_grid_spacing,
            key_ffwave,
            ewaldC1,
            m_self,
            grid_x,
            grid_y,
            grid_z,
            gauss_support,
            ewald_n,
            ewald_dr,
            eta,
            xisq,
            wave_bro_ind,
            wave_bro_nyind,
        ) = utils.init_periodic_box(
            error_tolerance, ewald_xi, lx, ly, lz, ewald_cut, max_strain, xy, positions, num_particles, temperature, seed_ffwave
        )
    elif boundary_flag == 1:
        nl_open = utils.compute_distinct_pairs(num_particles)

    # create RNG states for real space thermal fluctuations
    key_ffreal = random.PRNGKey(seed_ffreal)

    # check if particles overlap
    overlaps = utils.find_overlaps(positions, 2.0, num_particles, unique_pairs, box)
    if overlaps > 0:
        print("Warning: initial overlaps are ", (overlaps))
    print("Starting: compiling the code... This should not take more than 1-2 minutes.")

    for step in tqdm(range(num_steps), mininterval=0.5):
        # initialize generalized velocity (6*num_particlesarray, for linear and angular components)
        # this array stores the velocity for Brownian Dynamics, or the Brownian drift otherwise
        general_velocity = jnp.zeros(6 * num_particles, float)

        # define rhs array of the linear system Ax=b (with A the saddle point matrix)
        saddle_b = jnp.zeros(17 * num_particles, float)

        if boundary_flag == 0:
            # precompute quantities for periodic boundaries hydrodynamic calculation
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
            ) = utils.precompute_rpy(
                positions,
                gaussian_grid_spacing,
                nl_ff,
                xy,
                num_particles,
                lx,
                ly,
                lz,
                grid_x,
                grid_y,
                grid_z,
                prefac,
                expfac,
                quadW,
                int(gauss_support),
                gaussPd2,
                ewald_n,
                ewald_dr,
                ewald_cut,
                ewaldC1
            )
        elif boundary_flag == 1:
            (r, indices_i, indices_j, mobil_scalar) = utils.precompute_rpy_open(
                positions, nl_open
            )
        # compute shear-rate for current timestep: simple(shear_frequency=0) or oscillatory(shear_frequency>0)
        shear_rate = shear.update_shear_rate(time_step, step, shear_rate_0, shear_frequency, phase=0)

        # add applied forces and conservative (pair potential) forces to right-hand side of system
        saddle_b = applied_forces.sum_applied_forces(
            num_particles,
            external_forces,
            external_torques,
            saddle_b,
            interaction_strength,
            indices_i,
            indices_j,
            positions,
            interaction_cutoff,
            1,
            time_step,
            box
        )

        # compute Thermal Fluctuations only if temperature is not zero
        if temperature > 0:
            # compute far-field (real space contribution) slip velocity and set in rhs of linear system
            key_ffreal, random_array_real = utils.generate_random_array(key_ffreal, (11 * num_particles))
            if boundary_flag == 0:
                key_ffwave, random_array_wave = utils.generate_random_array(
                    key_ffwave, (3 * 2 * len(wave_bro_ind[:, 0, 0]) + 3 * len(wave_bro_nyind[:, 0]))
                )

                # compute far-field (wave space contribution) slip velocity and set in rhs of linear system
                ws_linvel, ws_angvel_strain = thermal.compute_wave_space_slipvelocity(
                    num_particles,
                    int(grid_x),
                    int(grid_y),
                    int(grid_z),
                    int(gauss_support),
                    temperature,
                    time_step,
                    gridh,
                    wave_bro_ind[:, 0, 0],
                    wave_bro_ind[:, 0, 1],
                    wave_bro_ind[:, 0, 2],
                    wave_bro_ind[:, 1, 0],
                    wave_bro_ind[:, 1, 1],
                    wave_bro_ind[:, 1, 2],
                    wave_bro_nyind[:, 0],
                    wave_bro_nyind[:, 1],
                    wave_bro_nyind[:, 2],
                    gridk,
                    random_array_wave,
                    all_indices_x,
                    all_indices_y,
                    all_indices_z,
                    gaussian_grid_spacing1,
                    gaussian_grid_spacing2,
                )
                rs_linvel, rs_angvel_strain, stepnormff, diag_ff = (
                    thermal.compute_real_space_slipvelocity(
                        num_particles,
                        m_self,
                        temperature,
                        time_step,
                        int(n_iter_Lanczos_ff),
                        random_array_real,
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
                    )
                )
                while (stepnormff > 1e-3) and (n_iter_Lanczos_ff < 150):
                    n_iter_Lanczos_ff += 20
                    rs_linvel, rs_angvel_strain, stepnormff, diag_ff = (
                        thermal.compute_real_space_slipvelocity(
                            num_particles,
                            m_self,
                            temperature,
                            time_step,
                            int(n_iter_Lanczos_ff),
                            random_array_real,
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
                        )
                    )
                # combine real- and wave-space part of thermal fluctuation
                saddle_b = saddle_b.at[: 11 * num_particles].add(
                    thermal.convert_to_generalized(
                        num_particles, ws_linvel, rs_linvel, ws_angvel_strain, rs_angvel_strain
                    )
                )

            elif boundary_flag == 1:
                rs_linvel, rs_angvel_strain, stepnormff, diag_ff = (
                    thermal.compute_real_space_slipvelocity_open(
                        num_particles,
                        temperature,
                        time_step,
                        int(n_iter_Lanczos_ff),
                        random_array_real,
                        r,
                        indices_i,
                        indices_j,
                        mobil_scalar,
                    )
                )
                while (stepnormff > 0.003) and (n_iter_Lanczos_ff < 150):
                    n_iter_Lanczos_ff += 20
                    rs_linvel, rs_angvel_strain, stepnormff, diag_ff = (
                        thermal.compute_real_space_slipvelocity_open(
                            num_particles,
                            temperature,
                            time_step,
                            int(n_iter_Lanczos_ff),
                            random_array_real,
                            r,
                            indices_i,
                            indices_j,
                            mobil_scalar,
                        )
                    )
                # convert real-space part of thermal fluctuation into a generalized velocity
                saddle_b = saddle_b.at[: 11 * num_particles].add(
                    thermal.convert_to_generalized(
                        num_particles,
                        jnp.zeros_like(rs_linvel),
                        rs_linvel,
                        jnp.zeros_like(rs_angvel_strain),
                        rs_angvel_strain,
                    )
                )

            # check that thermal fluctuation calculation went well
            if (not math.isfinite(stepnormff)) or (
                (n_iter_Lanczos_ff > 150) and (stepnormff > 0.003)
            ):
                check_overlap(positions, num_particles, unique_pairs, box)
                raise ValueError(
                    f"Far-field Lanczos did not converge! Stepnorm is {stepnormff}, iterations are {n_iter_Lanczos_ff}. Eigenvalues of tridiagonal matrix are {diag_ff}. Abort!"
                )

        # add random velocity to total velocity in RPY
        general_velocity += saddle_b[: 6 * num_particles]
        # add potential force contribution to total velocity in RPY
        if boundary_flag == 0:
            general_velocity += mobility.mobility_periodic(
                num_particles,
                grid_x,
                grid_y,
                grid_z,
                gauss_support,
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
                -saddle_b[11 * num_particles :],
            )
        elif boundary_flag == 1:
            general_velocity += mobility.mobility_open(
                num_particles,
                r,
                positions[nl_open[1,:]] - positions[nl_open[0,:]],
                indices_i,
                indices_j,
                -saddle_b[11 * num_particles :],
                mobil_scalar,
            )
        # update positions and neighborlists
        positions = update_positions(
            shear_rate, positions, general_velocity, time_step
        )
        
        # Update neighborlists
        nl_ff, _, _, nl_list_bound = utils.update_neighborlist(num_particles, positions, ewald_cut, 0., 0., unique_pairs, box)
        if not nl_list_bound: # Re-allocate list if number of neighbors exceeded list size.     
            unique_pairs, nl_ff, _, _, nl_safety_margin = utils.cpu_nlist(positions, np.array([lx,ly,lz]), ewald_cut, 0., 0., xy, initial_safety_margin=nl_safety_margin)
        if boundary_flag == 0:
            gaussian_grid_spacing = utils.precompute_grid_distancing(
                gauss_support, gridh[0], xy, positions, num_particles, grid_x, grid_y, grid_z, lx, ly, lz
            )
            
        # if system is sheared, strain wave-vectors grid and update box tilt factor
        if shear_rate_0 != 0:
            xy = shear.update_box_tilt_factor(time_step, shear_rate_0, xy, step, shear_frequency)
            box = jnp.array([[lx, ly * xy, lz * 0.0], [0.0, ly, lz * 0.0], [0.0, 0.0, lz]])
            _, shift_fn = space.periodic_general(box, fractional_coordinates=False,
            )
            if boundary_flag == 0:
                gridk = shear.compute_sheared_grid(
                    int(grid_x), int(grid_y), int(grid_z), xy, lx, ly, lz, eta, xisq
                )

        # reset Lanczos number of iteration once in a while (avoid to do too many iterations if not needed, can be tuned for better performance)
        elif (step % 100) == 0:
            n_iter_Lanczos_ff = 5

        # write trajectory to file
        if (step % writing_period) == 0:
            # check that the position to save do not contain 'nan' or 'inf'
            if (jnp.isnan(positions)).any() or (jnp.isinf(positions)).any():
                raise ValueError("Invalid particles positions. Abort!")

            # check that current configuration does not have overlapping particles
            check_overlap(positions, num_particles, unique_pairs, box)

            # save trajectory to file
            trajectory[int(step / writing_period), :, :] = positions
            if output is not None:
                np.save(output / "trajectory.npy", trajectory)

            # store velocity (linear and angular) to file
            if store_velocity > 0:
                velocities[int(step / writing_period), :, :] = jnp.reshape(general_velocity, (num_particles, 6))
                if output is not None:
                    np.save(output / "velocities.npy", velocities)

            # store orientation
            # TODO
    return trajectory, velocities


def wrap_bd(
    num_steps: int,
    writing_period: int,
    time_step: float,
    lx: float,
    ly: float,
    lz: float,
    num_particles: int,
    temperature: float,
    interaction_strength: float,
    buoyancy_flag: int,
    interaction_cutoff: float,
    positions: ArrayLike,
    seed: int,
    shear_rate_0: float,
    shear_frequency: float,
    output: str,
    store_velocity: bool,
    store_orientation: bool,
    constant_applied_forces: ArrayLike,
    constant_applied_torques: ArrayLike,
) -> tuple[Array, Array]:
    """Wrap all functions needed to integrate the particles equation of motions forward in time, using Brownian Dynamics method.

    Parameters
    ----------
    num_steps: (int)
        Number of timesteps
    writing_period: (int)
        Period for writing to file
    time_step: (float)
        Timestep
    lx: (float)
        Box size (x-direction)
    ly: (float)
        Box size (y-direction)
    lz: (float)
        Box size (z-direction)
    num_particles: (int)
        Number of particles
    temperature: (float)
        Thermal energy
    interaction_strength: (float)
        Interaction strength
    buoyancy_flag: (int)
        Set to 1 to have gravity acting on colloids
    interaction_cutoff: (float)
        Distance cutoff for interacting particles
    positions: (float)
        Array of particles initial positions (num_particles,3)
    seed: (int)
        Seed for random forces
    shear_rate_0: (float)
        Axisymmetric shear rate amplitude
    shear_frequency: (float)
        Frequency of shear, set to zero to have simple shear
    output: (str)
        File name for output
    store_stresslet: (int)
        To have stresslet in the output
    store_velocity: (int)
        To have velocities in the output/var/log/nvidia-installer.log
    store_orientation: (int)
        To have particle orientations in the output
    constant_applied_forces: (float)
        Array of external forces (num_particles,3)
    constant_applied_torques: (float)
        Array of external torques (num_particles,3)

    Returns
    -------
        trajectory, , velocities

    """

    @jit
    def update_positions(
        shear_rate: float,
        positions: ArrayLike,
        net_vel: ArrayLike,
        time_step: float,
    ) -> tuple[Array, Array]:
        """Update particle positions and neighbor lists

        Parameters
        ----------
        shear_rate: (float)
            Shear rate at current time step
        positions: (float)
            Array (num_particles,3) of particles positions
        net_vel: (float)
            Array (6*num_particles) of linear/angular velocities relative to the background flow
        time_step: (float)
            Timestep used to advance positions

        Returns
        -------
        positions (in-place update)

        """
        # Define array of displacement r(t+time_step)-r(t)
        dR = jnp.zeros((num_particles, 3), float)
        # Compute actual displacement due to velocities (relative to background flow)
        dR = dR.at[:, 0].set(time_step * net_vel[(0)::6])
        dR = dR.at[:, 1].set(time_step * net_vel[(1)::6])
        dR = dR.at[:, 2].set(time_step * net_vel[(2)::6])
        # Apply displacement and compute wrapped shift (Lees Edwards boundary conditions)
        positions = (
            shift_fn(positions + jnp.array([lx, ly, lz]) / 2, dR) - jnp.array([lx, ly, lz]) * 0.5
        )

        # Define array of displacement r(t+time_step)-r(t) (this time for displacement given by background flow)
        dR = jnp.zeros((num_particles, 3), float)
        dR = dR.at[:, 0].set(
            time_step * shear_rate * positions[:, 1]
        )  # Assuming y:gradient direction, x:background flow direction
        positions = (
            shift_fn(positions + jnp.array([lx, ly, lz]) / 2, dR) - jnp.array([lx, ly, lz]) * 0.5
        )  # Apply shift
        return positions

    if output is not None:
        output = Path(output)
        output.mkdir(exist_ok=True, parents=True)

    # set array for output trajectory, velocities and stresslet in time
    trajectory = np.zeros((int(num_steps / writing_period), num_particles, 3), float)
    velocities = np.zeros((int(num_steps / writing_period), num_particles, 6), float)

    #  set INITIAL Periodic Space and Displacement Metric
    xy = 0.0  # set box tilt factor to zero to begin (unsheared box)
    _, shift_fn = space.periodic_general(
        jnp.array([[lx, ly * xy, lz * 0.0], [0.0, ly, lz * 0.0], [0.0, 0.0, lz]]),
        fractional_coordinates=False,
    )
    box = jnp.array([[lx, ly * xy, lz * 0.0], [0.0, ly, lz * 0.0], [0.0, 0.0, lz]])
    # compute neighbor lists 
    unique_pairs, nl, _, _, nl_safety_margin = utils.cpu_nlist(positions, np.array([lx,ly,lz]), interaction_cutoff, 0., 0., xy)

    # set external applied forces/torques (no pair-interactions, will be added later)
    external_forces = jnp.zeros(3 * num_particles, float)
    external_torques = jnp.zeros(3 * num_particles, float)
    if buoyancy_flag == 1:  # apply buoyancy forces (in z direction)
        external_forces = external_forces.at[2::3].add(-1.0)
    if np.count_nonzero(constant_applied_forces) > 0:  # apply external forces
        external_forces += jnp.ravel(constant_applied_forces)
    if np.count_nonzero(constant_applied_torques) > 0:  # apply external torques
        external_torques += jnp.ravel(constant_applied_torques)

    # create RNG states
    key = random.PRNGKey(seed)

    # check if particles overlap
    overlaps = utils.find_overlaps(positions, 2.0, num_particles, unique_pairs, box)
    if overlaps > 0:
        print("Warning: initial overlaps are ", (overlaps))
    print("Starting: compiling the code... This should not take more than 1-2 minutes.")

    for step in tqdm(range(num_steps), mininterval=0.5):
        # initialize generalized velocity (6*num_particlesarray, for linear and angular components)
        # this array stores the velocity for Brownian Dynamics, or the Brownian drift otherwise
        general_velocity = jnp.zeros(6 * num_particles, float)

        # define rhs array of the linear system Ax=b (with A the saddle point matrix)
        saddle_b = jnp.zeros(17 * num_particles, float)

        # precompute quantities for Brownian dynamics calculation
        (r, indices_i, indices_j) = utils.precompute_bd(
            positions, nl, lx, ly, lz, xy
        )

        # compute shear-rate for current timestep: simple(shear_frequency=0) or oscillatory(shear_frequency>0)
        shear_rate = shear.update_shear_rate(time_step, step, shear_rate_0, shear_frequency, phase=0)

        # add applied forces and conservative (pair potential) forces to right-hand side of system
        saddle_b = applied_forces.sum_applied_forces(
            num_particles,
            external_forces,
            external_torques,
            saddle_b,
            interaction_strength,
            indices_i,
            indices_j,
            positions,
            interaction_cutoff,
            0,
            time_step,
            box
        )

        # compute Thermal Fluctuations only if temperature is not zero
        if temperature > 0:
            # generate random numbers for the various contributions of thermal noise
            key, random_array = utils.generate_random_array(key, (6 * num_particles))

            # compute random force for Brownian Dynamics
            random_velocity = thermal.compute_bd_randomforce(num_particles, temperature, time_step, random_array)
            general_velocity += random_velocity

        # add potential force contribution to total velocity (thermal contribution is already included)
        general_velocity += -saddle_b[11 * num_particles :]
        # update positions
        positions = update_positions(
            shear_rate, positions, general_velocity, time_step
        )
        # Update neighborlists
        nl, _, _, nl_list_bound = utils.update_neighborlist(num_particles, positions, interaction_cutoff, 0., 0., unique_pairs, box)
        if not nl_list_bound: # Re-allocate list if number of neighbors exceeded list size.     
            unique_pairs, nl, _, _, nl_safety_margin = utils.cpu_nlist(positions, np.array([lx,ly,lz]), interaction_cutoff, 0., 0., xy, initial_safety_margin=nl_safety_margin)
        
        # if system is sheared, strain wave-vectors grid and update box tilt factor
        if shear_rate_0 != 0:
            xy = shear.update_box_tilt_factor(time_step, shear_rate_0, xy, step, shear_frequency)
            box = jnp.array([[lx, ly * xy, lz * 0.0], [0.0, ly, lz * 0.0], [0.0, 0.0, lz]])
            _, shift_fn = space.periodic_general(box, fractional_coordinates=False
            )

        # write trajectory to file
        if (step % writing_period) == 0:
            # check that the position to save do not contain 'nan' or 'inf'
            if (jnp.isnan(positions)).any() or (jnp.isinf(positions)).any():
                raise ValueError("Invalid particles positions. Abort!")

            # check that current configuration does not have overlapping particles
            check_overlap(positions, num_particles, unique_pairs, box)

            # save trajectory to file
            trajectory[int(step / writing_period), :, :] = positions
            if output is not None:
                np.save(output / "trajectory.npy", trajectory)

            # store velocity (linear and angular) to file
            if store_velocity > 0:
                velocities[int(step / writing_period), :, :] = jnp.reshape(general_velocity, (num_particles, 6))
                if output is not None:
                    np.save(output / "velocities.npy", velocities)

            # store orientation (TODO)
    return trajectory, velocities
