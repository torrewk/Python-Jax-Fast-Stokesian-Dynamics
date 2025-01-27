import math
import time
from functools import partial
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np
from jax import Array, dtypes, jit, random
from jax import random as jrandom
from jax.typing import ArrayLike

from jfsd import ewald_tables, shear, thermal
from jfsd import jaxmd_partition as partition
from jfsd import jaxmd_space as space

# Define types for the functions
DisplacementFn = Callable[[Any, Any], Any]


@jit
def chol_fac(matrix: ArrayLike) -> Array:
    """Perform a Cholesky factorization of the input matrix.

    Parameters
    ----------
    A: (float)
        Array (6N,6N) containing lubrication resistance matrix to factorize

    Returns
    -------
    Lower triangle Cholesky factor of input matrix A

    """
    return jnp.linalg.cholesky(matrix)


def check_ewald_cutoff(ewald_cut: float, box_x: float, box_y: float, box_z: float, error: float):
    """Check that Ewald cutoff is small enough to avoid interaction with the particles images during real-space part of calculation.

    Parameters
    ----------
    ewald_cut: (float)
        Ewald space cut-off for real-space far-field hydrodynamic interactions
    box_x: (float)
        Box size in x direction
    box_y: (float)
        Box size in y direction
    box_z: (float)
        Box size in z direction
    error: (float)
        Tolerance error

    Returns
    -------

    """
    if (ewald_cut > box_x / 2) or (ewald_cut > box_y / 2) or (ewald_cut > box_z / 2):
        max_cut = max([box_x, box_y, box_z]) / 2.0
        new_xi = np.sqrt(-np.log(error)) / max_cut
        raise ValueError(f"Ewald cutoff radius is too large. Try with xi = {new_xi}")
    else:
        print("Ewald Cutoff is ", ewald_cut)
    return


def check_max_shear(
    gridh: ArrayLike, xisq: float, grid_nx: int, grid_ny: int, grid_nz: int, max_strain: float, error: float
) -> tuple[float, float]:
    """Check maximum eigenvalue of A'*A to scale support, P, for spreading on deformed grids (Fiore and Swan, J. Chem. Phys., 2018).

    Parameters
    ----------
    gridh: (float)
        Array (,3) containing wave space grid discrete spacing
    xisq: (float)
        Squared Ewald split parameter
    grid_nx: (int)
        Number of grid points in x direction
    grid_ny: (int)
        Number of grid points in y direction
    grid_nz: (int)
        Number of grid points in z direction
    max_strain: (float)
        Max strain applied to the box
    error: (float)
        Tolerance error

    Returns
    -------
    eta, gauss_support

    """
    gamma = max_strain
    lambdaa = 1 + gamma * gamma / 2 + gamma * np.sqrt(1 + gamma * gamma / 4)

    # Parameters for the Spectral Ewald Method (Lindbo and Tornberg, J. Comp. Phys., 2011)
    gaussm = 1.0
    while math.erfc(gaussm / np.sqrt(2 * lambdaa)) > error:
        gaussm += 0.01
    gauss_support = int(gaussm * gaussm / jnp.pi) + 1
    w = gauss_support * gridh[0] / 2.0  # Gaussian width in simulation units
    eta = (2.0 * w / gaussm) * (2.0 * w / gaussm) * (xisq)  # Gaussian splitting parameter

    # Check that the support size isn't somehow larger than the grid
    if gauss_support > min(grid_nx, min(grid_ny, grid_nz)):
        raise ValueError(
            f"Quadrature Support Exceeds Available Grid. \n (Mx, My, Mz) = ({grid_nx}), ({grid_ny}), ({grid_nz}). Support Size, P = {gauss_support}"
        )
    return eta, gauss_support


def compute_k_gridpoint_number(
    kmax: float, box_x: float, box_y: float, box_z: float
) -> tuple[int, int, int]:
    """Perform a Cholesky factorization of the input matrix.

    Parameters
    ----------
    kmax: (int)
        Max wave number
    box_x: (float)
        Box size in x direction
    box_y: (float)
        Box size in y direction
    box_z: (float)
        Box size in z direction

    Returns
    -------
    grid_nx, grid_ny, grid_nz

    """
    # Set initial number of grid points in wave space
    grid_nx = int(kmax * box_x / (2.0 * jnp.pi) * 2.0) + 1
    grid_ny = int(kmax * box_y / (2.0 * jnp.pi) * 2.0) + 1
    grid_nz = int(kmax * box_z / (2.0 * jnp.pi) * 2.0) + 1

    # Get list of int values between 8 and 512 that can be written as (2^a)*(3^b)*(5^c)
    # Then sort list from low to high and figure out how many entries there are
    mlist = []
    for ii in range(0, 10):
        pow2 = int(1)
        for i in range(0, ii):
            pow2 *= 2
        for jj in range(0, 6):
            pow3 = int(1)
            for j in range(0, jj):
                pow3 *= 3
            for kk in range(0, 4):
                pow5 = 1
                for k in range(0, kk):
                    pow5 *= 5
                mcurr = int(pow2 * pow3 * pow5)
                if (mcurr >= 8) and (mcurr <= 512):
                    mlist.append(mcurr)
    mlist = np.array(mlist)
    mlist = np.sort(mlist)
    nmult = len(mlist)

    # Compute the number of grid points in each direction (should be a power of 2,3,5 for most efficient FFTs)
    for ii in range(0, nmult):
        if grid_nx <= mlist[ii]:
            grid_nx = mlist[ii]
            break
    for ii in range(0, nmult):
        if grid_ny <= mlist[ii]:
            grid_ny = mlist[ii]
            break
    for ii in range(0, nmult):
        if grid_nz <= mlist[ii]:
            grid_nz = mlist[ii]
            break

    # Maximum number of FFT nodes is limited by available memory = 512 * 512 * 512 = 134217728
    if grid_nx * grid_ny * grid_nz > 134217728:
        raise ValueError("Requested Number of Fourier Nodes Exceeds Max Dimension of 512^3")
    return grid_nx, grid_ny, grid_nz


def precompute_grid_distancing(
    gauss_support: int,
    gridh: ArrayLike,
    tilt_factor: float,
    positions: ArrayLike,
    num_particles: int,
    grid_nx: int,
    grid_ny: int,
    grid_nz: int,
    box_x: float,
    box_y: float,
    box_z: float,
) -> Array:
    """Given a support size for Gaussian spread, compute distances in a (gauss_support x gauss_support x gauss_support) grid with gauss P shifted by 1 unit if it is odd or even.

    Parameters
    ----------
    gauss_support: (int)
        Gaussian support size for wave space calculation
    gridh: (float)
        Array (,3) containing wave space grid discrete spacing
    tilt_factor: (float)
        Current box tilt factor
    positions: (float)
        Array (num_particles,3) of current particles positions
    num_particles: (int)
        Number of particles
    grid_nx: (int)
        Number of grid points in x direction
    grid_ny: (int)
        Number of grid points in y direction
    grid_nz: (int)
        Number of grid points in z direction
    box_x: (float)
        Box size in x direction
    box_y: (float)
        Box size in y direction
    box_z: (float)
        Box size in z direction

    Returns
    -------
    grid

    """
    grid = np.zeros((gauss_support, gauss_support, gauss_support, num_particles))
    # center_offsets = (jnp.array(positions)+jnp.array([box_x,box_y,box_z])/2)*jnp.array([grid_nx,grid_ny,grid_nz])/jnp.array([box_x,box_y,box_z])
    center_offsets = np.array(positions) + np.array([box_x, box_y, box_z]) / 2
    center_offsets[:, 0] += -tilt_factor * center_offsets[:, 1]
    # center_offsets = center_offsets.at[:,0].add(-tilt_factor*center_offsets.at[:,1].get())
    center_offsets = center_offsets / np.array([box_x, box_y, box_z]) * np.array([grid_nx, grid_ny, grid_nz])
    # in grid units, not particle radius units
    center_offsets -= np.array(center_offsets, dtype=int)
    center_offsets = np.where(center_offsets > 0.5, -(1 - center_offsets), center_offsets)
    for i in range(gauss_support):
        for j in range(gauss_support):
            for k in range(gauss_support):
                grid[i, j, k, :] = (
                    gridh
                    * gridh
                    * (
                        (
                            i
                            - int(gauss_support / 2)
                            - center_offsets[:, 0]
                            + tilt_factor * (j - int(gauss_support / 2) - center_offsets[:, 1])
                        )
                        ** 2
                        + (j - int(gauss_support / 2) - center_offsets[:, 1]) ** 2
                        + (k - int(gauss_support / 2) - center_offsets[:, 2]) ** 2
                    )
                )
    return grid


def initialize_single_neighborlist(
    space_cut: float, box_x: float, box_y: float, box_z: float, displacement: DisplacementFn
) -> partition.NeighborListFns:
    """Initialize a single neighborlists, given a box and a distance cutoff.

    Note that creation of neighborlists is perfomed using code from jax_md.
    At the moment, the code does not use cell list, as it produces artifact.
    In future, the creation of neighborlists will be handled entirely by JFSD, leveraging cell lists.

    Parameters
    ----------
    space_cut: (float)
        Cutoff (max) distance for particles to be considered neighbors
    box_x: (float)
        Box size in x direction
    box_y: (float)
        Box size in y direction
    box_z: (float)
        Box size in z direction
    displacement: (DisplacementFn)
        Displacement metric

    Returns
    -------
    neighbor_fn

    """
    # For Lubrication Hydrodynamic Forces Calculation
    neighbor_fn = partition.neighbor_list(
        displacement,
        jnp.array([box_x, box_y, box_z]),
        r_cutoff=space_cut,  # Spatial cutoff for 2 particles to be neighbor
        # dr_threshold=0.1,  # displacement of particles threshold to recompute neighbor list
        capacity_multiplier=1.1,
        format=partition.NeighborListFormat.OrderedSparse,
        disable_cell_list=True,
    )
    return neighbor_fn


def allocate_nlist(positions: ArrayLike, nbrs: partition.NeighborList) -> partition.NeighborList:
    """Allocate particle neighbor list

    Parameters
    ----------
    positions:
        Array (num_particles,3) of particles positions
    nbrs:
        Input neighbor lists

    Returns
    -------
    nbrs

    """
    return nbrs.allocate(positions)


@jit
def update_nlist(positions: ArrayLike, nbrs: partition.NeighborList) -> partition.NeighborList:
    """Update particle neighbor list

    Parameters
    ----------
    positions:
        Array (num_particles,3) of particles positions
    nbrs:
        Input neighbor lists

    Returns
    -------
    nbrs

    """
    # Update neighbor list
    nbrs = nbrs.update(positions)
    return nbrs


def create_hardsphere_configuration(L: float, num_particles: int, seed: int, temperature: float) -> Array:
    """Create an (at equilibrium, or close) configuration of Brownian hard spheres at a given temperature temperature.

    First initialize a random configuration of num_particles ideal particles.
    Then, thermalize the system while applying a soft repulsive potential,
    until no overlaps are present in the system.

    Parameters
    ----------
    L: (float)
        Box size requested
    num_particles: (int)
        Number of particles requested
    seed: (int)
        Seed for random number generator
    temperature: (float)
        Thermal energy of the system

    Returns
    -------
    positions

    """

    @jit
    def compute_hardsphere(net_vel, displacements, indices_i, indices_j):
        fp = jnp.zeros((num_particles, num_particles, 3))

        # Reset particle velocity
        net_vel *= 0.0

        # Compute velocity from hard-sphere repulsion
        dist_mod = jnp.sqrt(
            displacements[:, :, 0] * displacements[:, :, 0]
            + displacements[:, :, 1] * displacements[:, :, 1]
            + displacements[:, :, 2] * displacements[:, :, 2]
        )

        fp_mod = jnp.where(
            indices_i != indices_j, k * (1 - sigma / dist_mod[indices_i, indices_j]), 0.0
        )
        fp_mod = jnp.where((dist_mod[indices_i, indices_j]) < sigma, fp_mod, 0.0)

        # get forces in components
        fp = fp.at[indices_i, indices_j, 0].add(fp_mod * displacements[indices_i, indices_j, 0])
        fp = fp.at[indices_i, indices_j, 1].add(fp_mod * displacements[indices_i, indices_j, 1])
        fp = fp.at[indices_i, indices_j, 2].add(fp_mod * displacements[indices_i, indices_j, 2])
        fp = fp.at[indices_j, indices_i, 0].add(fp_mod * displacements[indices_j, indices_i, 0])
        fp = fp.at[indices_j, indices_i, 1].add(fp_mod * displacements[indices_j, indices_i, 1])
        fp = fp.at[indices_j, indices_i, 2].add(fp_mod * displacements[indices_j, indices_i, 2])

        # sum all forces in each particle
        fp = jnp.sum(fp, 1)

        net_vel = net_vel.at[(0)::3].add(fp.at[:, 0].get())
        net_vel = net_vel.at[(1)::3].add(fp.at[:, 1].get())
        net_vel = net_vel.at[(2)::3].add(fp.at[:, 2].get())

        return net_vel

    @jit
    def update_pos(positions, displacements, net_vel, nbrs):
        # Define array of displacement r(t+dt)-r(t)
        dr = jnp.zeros((num_particles, 3), float)
        # Compute actual displacement due to velocities
        dr = dr.at[:, 0].set(dt * net_vel.at[(0)::3].get())
        dr = dr.at[:, 1].set(dt * net_vel.at[(1)::3].get())
        dr = dr.at[:, 2].set(dt * net_vel.at[(2)::3].get())
        # Apply displacement and compute wrapped shift
        # shift system origin to (0,0,0)
        positions = shift(positions + jnp.array([L, L, L]) / 2, dr)
        # re-shift origin back to box center
        positions = positions - jnp.array([L, L, L]) * 0.5

        # Compute new relative displacements between particles
        displacements = (space.map_product(displacement))(positions, positions)
        # Update neighbor list
        nbrs = update_nlist(positions, nbrs)
        # nl = np.array(nbrs.idx)  # extract lists in sparse format
        return positions, nbrs, displacements, net_vel

    @jit
    def add_thermal_noise(net_vel, brow):
        net_vel = net_vel.at[0::3].add(brow[0::6])
        net_vel = net_vel.at[1::3].add(brow[1::6])
        net_vel = net_vel.at[2::3].add(brow[2::6])
        return net_vel

    displacement, shift = space.periodic_general(
        jnp.array([[L, 0.0, 0.0], [0.0, L, 0.0], [0.0, 0.0, L]]), fractional_coordinates=False
    )

    net_vel = jnp.zeros(3 * num_particles)
    key = jrandom.PRNGKey(seed)
    temperature = 0.001
    sigma = 2.15  # 2.15 #2.05 #particle diameter
    phi_eff = num_particles / (L * L * L) * (sigma * sigma * sigma) * np.pi / 6
    phi_actual = ((np.pi / 6) * (2 * 2 * 2) * num_particles) / (L * L * L)
    brow_time = sigma * sigma / (4 * temperature)

    dt = brow_time / (1e5)

    Nsteps = int(brow_time / dt)
    key, random_coord = generate_random_array(key, (num_particles * 3))
    random_coord = (random_coord - 1 / 2) * L
    positions = jnp.zeros((num_particles, 3))
    positions = positions.at[:, 0].set(random_coord[0::3])
    positions = positions.at[:, 1].set(random_coord[1::3])
    positions = positions.at[:, 2].set(random_coord[2::3])

    displacements = (space.map_product(displacement))(positions, positions)

    neighbor_fn = initialize_single_neighborlist(2.2, L, L, L, displacement)
    nbrs = allocate_nlist(positions + jnp.array([L, L, L]) / 2, neighbor_fn)
    nl = np.array(nbrs.idx)

    overlaps, _ = find_overlaps(displacements, 2.002, num_particles)
    k = 30 * np.sqrt(6 * temperature / (sigma * sigma * dt))  # spring constant

    if phi_eff > 0.6754803226762013:
        print("System Volume Fraction is ", phi_actual, phi_eff)
        raise ValueError(
            "Attempted to create particles configuration too dense. Use imported coordinates instead. Abort!"
        )
    print(
        "Creating initial configuration with volume fraction ",
        phi_actual,
        ". This could take several minutes in dense systems.",
    )
    start_time = time.time()
    while overlaps > 0:
        for i_step in range(Nsteps):
            # Compute Velocity (Brownian + hard-sphere)

            # Compute distance vectors and neighbor list indices
            (r, indices_i, indices_j) = precompute_bd(positions, nl, displacements, num_particles, L, L, L)

            # compute forces for each pair
            net_vel = compute_hardsphere(net_vel, displacements, indices_i, indices_j)

            # Compute and add Brownian velocity
            key, random_noise = generate_random_array(key, (6 * num_particles))
            brow = thermal.compute_bd_randomforce(num_particles, temperature, dt, random_noise)
            net_vel = add_thermal_noise(net_vel, brow)

            # Update positions
            new_positions, nbrs, new_displacements, net_vel = update_pos(
                positions, displacements, net_vel, nbrs
            )

            # If the allocated neighbor list is too small in size for the new particle configuration, re-allocate it, otherwise just update it
            if nbrs.did_buffer_overflow:
                nbrs = allocate_nlist(positions, neighbor_fn)
                nl = np.array(nbrs.idx)
                (r, indices_i, indices_j) = precompute_bd(positions, nl, displacements, num_particles, L, L, L)
                net_vel = compute_hardsphere(net_vel, displacements, indices_i, indices_j)
                net_vel = add_thermal_noise(net_vel, brow)
                new_positions, nbrs, new_displacements, net_vel = update_pos(
                    positions, displacements, net_vel, nbrs
                )
                positions = new_positions
                nl = np.array(nbrs.idx)
                displacements = new_displacements
            else:
                positions = new_positions
                nl = np.array(nbrs.idx)
                displacements = new_displacements

        overlaps, _ = find_overlaps(displacements, 2.002, num_particles)
        if (time.time() - start_time) > 1800:  # interrupt if too computationally expensive
            raise ValueError("Creation of initial configuration failed. Abort!")
    print("Initial configuration created. Volume fraction is ", phi_actual)
    return positions


@partial(jit, static_argnums=[2])
def find_overlaps(dist: ArrayLike, sigma: float, num_particles: int) -> tuple[int, float]:
    """Check overlaps between particles and returns number of overlaps + number of particles.

    The radius of a particle is set to 1.
    Note that this function should be used only for debugging,
    as the memory cost scales quadratically with the number of particles num_particles.

    Parameters
    ----------
    dist: (float)
        Array (num_particles*num_particles,3) of distance vectors between particles in neighbor list
    sigma: (float)
        Particle diameter
    num_particles: (int)
        Particle number

    Returns
    -------
    output:
        Number of overlaps + Number of particles

    """
    sigma_sq = sigma * sigma
    dist_sq = (
        dist[:, :, 0] * dist[:, :, 0]
        + dist[:, :, 1] * dist[:, :, 1]
        + dist[:, :, 2] * dist[:, :, 2]
    )
    mask = jnp.where(dist_sq < sigma_sq, 1.0, 0.0)  # build mask from distances < cutoff
    mask = mask - np.eye(len(mask))  # remove diagonal mask (self-overlaps)
    total_overlaps = jnp.sum(mask) / 2  # divide by 2 to avoid overcounting
    indices = jnp.nonzero(mask, size=num_particles, fill_value=num_particles)
    return total_overlaps, indices


@partial(jit, static_argnums=[1])
def generate_random_array(key: dtypes.prng_key, size: int) -> tuple[dtypes.prng_key, Array]:
    """Generate array of random number using JAX.

    Parameters
    ----------
    key: (prng_key)
        Current key of random number generator
    size: (int)
        Size of random array to generate

    Returns
    -------
    subkey,  (jrandom.uniform(subkey, (size,)))

    """
    # advance RNG state (otherwise will get same random numbers)
    key, subkey = jrandom.split(key)
    return subkey, (jrandom.uniform(subkey, (size,)))


@partial(jit, static_argnums=[6, 16])
def precompute(
    positions: ArrayLike,
    gaussian_grid_spacing: ArrayLike,
    nl_ff: ArrayLike,
    nl_lub: ArrayLike,
    displacements_vector_matrix: ArrayLike,
    tilt_factor: float,
    num_particles: int,
    box_x: float,
    box_y: float,
    box_z: float,
    grid_nx: int,
    grid_ny: int,
    grid_nz: int,
    prefac: float,
    expfac: float,
    quadw: ArrayLike,
    gauss_support: int,
    gauss_support_d2: int,
    ewald_n: int,
    ewald_dr: float,
    ewald_cut: float,
    ewaldc: ArrayLike,
    res_table_min: float,
    res_table_dr: float,
    res_table_dist: ArrayLike,
    res_table_vals: ArrayLike,
    alpha: float,
    h0: float,
) -> tuple[
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    tuple[
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
    ],
]:
    """Compute all the necessary quantities needed to update the particle position at a given timestep.

    Parameters
    ----------
    positions: (float)
        Array (num_particles,3) of current particles positions
    gaussian_grid_spacing: (float)
        Array (,gauss_support*gauss_support*gauss_support) containing distances from support center to each gridpoint in the gaussian support
    nl_ff: (int)
        Array (2,n_pairs_ff) containing far-field neighborlist indices
    nl_lub: (int)
        Array (2,n_pairs_nf) containing near-field neighborlist indices
    displacements_vector_matrix: (float)
        Array (num_particles,num_particles,3) of current displacements between particles, with each element a vector
    tilt_factor: (float)
        Current box tilt factor
    num_particles: (int)
        Number of particles
    box_x: (float)
        Box size in x direction
    box_y: (float)
        Box size in y direction
    box_z: (float)
        Box size in z direction
    grid_nx: (int)
        Number of grid points in x direction
    grid_ny: (int)
        Number of grid points in y direction
    grid_nz: (int)
        Number of grid points in z direction
    prefac: (float)
        Prefactor needed for FFT
    expfac: (float)
        Exponential factor needed for FFT
    quadw: (float)
        Product of wave grid discretization parameter in each direction (grid_dx*grid_dy*grid_dz)
    gauss_support: (int)
        Gaussian support size for wave space calculation
    gauss_support_d2: (int)
        Integer part of Gaussian support size divide by 2
    ewald_n: (int)
        Number of entries in Ewald table, for each mobility function
    ewald_dr: (float)
        Ewald table discretization parameter
    ewald_cut: (float)
        Ewald space cut-off for real-space far-field hydrodynamic interactions
    ewaldc: (float)
        Array (,ewald_n*7) containing tabulated mobility scalar functions values
    res_table_min: (float)
        Minimum distance resolved for lubrication interactions
    res_table_dr: (float)
        Resistance table discretization parameter
    res_table_dist: (float)
        Array (,1000) containing tabulated distances for resistance functions
    res_table_vals: (float)
        Array (,1000*22) containing tabulated resistance scalar functions values
    alpha_friction: (float)
        strength of hydrodynamic friction
    h0_friction: (float)
        range of hydrodynamic friction

    Returns
    -------
    all_indices_x, all_indices_y, all_indices_z, gaussian_grid_spacing1, gaussian_grid_spacing2,
     r_unit, indices_i, indices_j, f1, f2, g1, g2, h1, h2, h3,
     r_lub_unit, indices_i_lub, indices_j_lub,res_func

    """
    # Wave Space calculation quantities

    # Compute fractional coordinates
    pos = positions + jnp.array([box_x, box_y, box_z]) / 2
    pos = pos.at[:, 0].add(-tilt_factor * pos.at[:, 1].get())
    pos = pos / jnp.array([box_x, box_y, box_z]) * jnp.array([grid_nx, grid_ny, grid_nz])
    # convert positions in the box in indices in the grid
    # pos = (positions+np.array([box_x, box_y, box_z])/2)/np.array([box_x, box_y, box_z]) * np.array([grid_nx, grid_ny, grid_nz])
    intpos = jnp.array(pos, int)  # integer positions
    intpos = jnp.where(pos - intpos > 0.5, intpos + 1, intpos)

    # actual values to put in the grid
    gaussian_grid_spacing1 = prefac * jnp.exp(-expfac * gaussian_grid_spacing)
    gaussian_grid_spacing2 = jnp.swapaxes(
        jnp.swapaxes(jnp.swapaxes(gaussian_grid_spacing1 * quadw, 3, 2), 2, 1), 1, 0
    )

    # Starting indices on the grid from particle positions
    start_index_x = (intpos.at[:, 0].get() - (gauss_support_d2)) % grid_nx
    start_index_y = (intpos.at[:, 1].get() - (gauss_support_d2)) % grid_ny
    start_index_z = (intpos.at[:, 2].get() - (gauss_support_d2)) % grid_nz
    # All indices on the grids from particle positions, ordered such that x is the 'slowest' index to change and z the 'fastest'
    all_indices_x = jnp.repeat(
        jnp.repeat(jnp.repeat(start_index_x, gauss_support), gauss_support), gauss_support
    ) + jnp.resize(
        jnp.repeat(jnp.repeat(jnp.arange(gauss_support), gauss_support), gauss_support), gauss_support * gauss_support * gauss_support * num_particles
    )
    all_indices_y = jnp.repeat(
        jnp.repeat(jnp.repeat(start_index_y, gauss_support), gauss_support), gauss_support
    ) + jnp.resize(
        jnp.resize(jnp.repeat(jnp.arange(gauss_support), gauss_support), gauss_support * gauss_support * gauss_support),
        gauss_support * gauss_support * gauss_support * num_particles,
    )
    all_indices_z = jnp.repeat(
        jnp.repeat(jnp.repeat(start_index_z, gauss_support), gauss_support), gauss_support
    ) + jnp.resize(
        jnp.resize(jnp.resize(jnp.arange(gauss_support), gauss_support * gauss_support), gauss_support * gauss_support * gauss_support),
        gauss_support * gauss_support * gauss_support * num_particles,
    )
    all_indices_x = all_indices_x % grid_nx
    all_indices_y = all_indices_y % grid_ny
    all_indices_z = all_indices_z % grid_nz

    ###################################################################################################################
    # Real Space (far-field) calculation quantities
    indices_i = nl_ff[0, :]  # Pair indices (i<j always)
    indices_j = nl_ff[1, :]
    # array of vectors from particles i to j (size = npairs)
    r = displacements_vector_matrix.at[indices_i, indices_j].get()
    dist = space.distance(r)  # distances between particles i and j
    r_unit = -r / dist.at[:, None].get()  # unit vector from particle j to i

    # Interpolate scalar mobility functions values from ewald table
    r_ind = ewald_n * (dist - ewald_dr) / (ewald_cut - ewald_dr)  # index in ewald table
    r_ind = r_ind.astype(int)  # truncate decimal part
    offset1 = 2 * r_ind  # even indices
    offset2 = 2 * r_ind + 1  # odd indices

    tewaldc1m = ewaldc.at[offset1].get()  # UF and UC
    tewaldC1p = ewaldc.at[offset1 + 2].get()
    tewaldc2m = ewaldc.at[offset2].get()  # DC
    tewaldC2p = ewaldc.at[offset2 + 2].get()

    fac_ff = dist / ewald_dr - r_ind - 1.0  # interpolation factor

    f1 = tewaldc1m.at[:, 0].get() + (tewaldC1p.at[:, 0].get() - tewaldc1m.at[:, 0].get()) * fac_ff
    f2 = tewaldc1m.at[:, 1].get() + (tewaldC1p.at[:, 1].get() - tewaldc1m.at[:, 1].get()) * fac_ff

    g1 = tewaldc1m.at[:, 2].get() + (tewaldC1p.at[:, 2].get() - tewaldc1m.at[:, 2].get()) * fac_ff
    g2 = tewaldc1m.at[:, 3].get() + (tewaldC1p.at[:, 3].get() - tewaldc1m.at[:, 3].get()) * fac_ff

    h1 = tewaldc2m.at[:, 0].get() + (tewaldC2p.at[:, 0].get() - tewaldc2m.at[:, 0].get()) * fac_ff
    h2 = tewaldc2m.at[:, 1].get() + (tewaldC2p.at[:, 1].get() - tewaldc2m.at[:, 1].get()) * fac_ff
    h3 = tewaldc2m.at[:, 2].get() + (tewaldC2p.at[:, 2].get() - tewaldc2m.at[:, 2].get()) * fac_ff

    ###################################################################################################################
    # Lubrication calculation quantities
    indices_i_lub = nl_lub[0, :]  # Pair indices (i<j always)
    indices_j_lub = nl_lub[1, :]
    # array of vectors from particle i to j (size = npairs)
    r_lub = displacements_vector_matrix.at[nl_lub[0, :], nl_lub[1, :]].get()
    dist_lub = space.distance(r_lub)  # distance between particle i and j
    # unit vector from particle j to i
    r_lub_unit = r_lub / dist_lub.at[:, None].get()

    # # Indices in resistance table
    ind = jnp.log10((dist_lub - 2.0) / res_table_min) / res_table_dr
    ind = ind.astype(int)
    dist_lub_lower = res_table_dist.at[ind].get()
    dist_lub_upper = res_table_dist.at[ind + 1].get()
    # # Linear interpolation of the Table values
    fac_lub = jnp.where(
        dist_lub_upper - dist_lub_lower > 0.0,
        (dist_lub - dist_lub_lower) / (dist_lub_upper - dist_lub_lower),
        0.0,
    )

    XA11 = jnp.where(
        dist_lub >= 2 + res_table_min,
        (
            res_table_vals[22 * (ind) + 0]
            + (res_table_vals[22 * (ind + 1) + 0] - res_table_vals[22 * (ind) + 0]) * fac_lub
        ),
        res_table_vals[0],
    )

    XA12 = jnp.where(
        dist_lub >= 2 + res_table_min,
        (
            res_table_vals[22 * (ind) + 1]
            + (res_table_vals[22 * (ind + 1) + 1] - res_table_vals[22 * (ind) + 1]) * fac_lub
        ),
        res_table_vals[1],
    )

    YA11 = jnp.where(
        dist_lub >= 2 + res_table_min,
        (
            res_table_vals[22 * (ind) + 2]
            + (res_table_vals[22 * (ind + 1) + 2] - res_table_vals[22 * (ind) + 2]) * fac_lub
        ),
        res_table_vals[2],
    )

    YA12 = jnp.where(
        dist_lub >= 2 + res_table_min,
        (
            res_table_vals[22 * (ind) + 3]
            + (res_table_vals[22 * (ind + 1) + 3] - res_table_vals[22 * (ind) + 3]) * fac_lub
        ),
        res_table_vals[3],
    )

    YB11 = jnp.where(
        dist_lub >= 2 + res_table_min,
        (
            res_table_vals[22 * (ind) + 4]
            + (res_table_vals[22 * (ind + 1) + 4] - res_table_vals[22 * (ind) + 4]) * fac_lub
        ),
        res_table_vals[4],
    )

    YB12 = jnp.where(
        dist_lub >= 2 + res_table_min,
        (
            res_table_vals[22 * (ind) + 5]
            + (res_table_vals[22 * (ind + 1) + 5] - res_table_vals[22 * (ind) + 5]) * fac_lub
        ),
        res_table_vals[5],
    )

    XC11 = jnp.where(
        dist_lub >= 2 + res_table_min,
        (
            res_table_vals[22 * (ind) + 6]
            + (res_table_vals[22 * (ind + 1) + 6] - res_table_vals[22 * (ind) + 6]) * fac_lub
        ),
        res_table_vals[6],
    )

    XC12 = jnp.where(
        dist_lub >= 2 + res_table_min,
        (
            res_table_vals[22 * (ind) + 7]
            + (res_table_vals[22 * (ind + 1) + 7] - res_table_vals[22 * (ind) + 7]) * fac_lub
        ),
        res_table_vals[7],
    )

    YC11 = jnp.where(
        dist_lub >= 2 + res_table_min,
        (
            res_table_vals[22 * (ind) + 8]
            + (res_table_vals[22 * (ind + 1) + 8] - res_table_vals[22 * (ind) + 8]) * fac_lub
        ),
        res_table_vals[8],
    )

    YC12 = jnp.where(
        dist_lub >= 2 + res_table_min,
        (
            res_table_vals[22 * (ind) + 9]
            + (res_table_vals[22 * (ind + 1) + 9] - res_table_vals[22 * (ind) + 9]) * fac_lub
        ),
        res_table_vals[9],
    )

    YB21 = -YB12  # symmetry condition

    XG11 = jnp.where(
        dist_lub >= 2 + res_table_min,
        (
            res_table_vals[22 * (ind) + 10]
            + (res_table_vals[22 * (ind + 1) + 10] - res_table_vals[22 * (ind) + 10]) * fac_lub
        ),
        res_table_vals[10],
    )

    XG12 = jnp.where(
        dist_lub >= 2 + res_table_min,
        (
            res_table_vals[22 * (ind) + 11]
            + (res_table_vals[22 * (ind + 1) + 11] - res_table_vals[22 * (ind) + 11]) * fac_lub
        ),
        res_table_vals[11],
    )

    YG11 = jnp.where(
        dist_lub >= 2 + res_table_min,
        (
            res_table_vals[22 * (ind) + 12]
            + (res_table_vals[22 * (ind + 1) + 12] - res_table_vals[22 * (ind) + 12]) * fac_lub
        ),
        res_table_vals[12],
    )

    YG12 = jnp.where(
        dist_lub >= 2 + res_table_min,
        (
            res_table_vals[22 * (ind) + 13]
            + (res_table_vals[22 * (ind + 1) + 13] - res_table_vals[22 * (ind) + 13]) * fac_lub
        ),
        res_table_vals[13],
    )

    YH11 = jnp.where(
        dist_lub >= 2 + res_table_min,
        (
            res_table_vals[22 * (ind) + 14]
            + (res_table_vals[22 * (ind + 1) + 14] - res_table_vals[22 * (ind) + 14]) * fac_lub
        ),
        res_table_vals[14],
    )

    YH12 = jnp.where(
        dist_lub >= 2 + res_table_min,
        (
            res_table_vals[22 * (ind) + 15]
            + (res_table_vals[22 * (ind + 1) + 15] - res_table_vals[22 * (ind) + 15]) * fac_lub
        ),
        res_table_vals[15],
    )

    XM11 = jnp.where(
        dist_lub >= 2 + res_table_min,
        (
            res_table_vals[22 * (ind) + 16]
            + (res_table_vals[22 * (ind + 1) + 16] - res_table_vals[22 * (ind) + 16]) * fac_lub
        ),
        res_table_vals[16],
    )

    XM12 = jnp.where(
        dist_lub >= 2 + res_table_min,
        (
            res_table_vals[22 * (ind) + 17]
            + (res_table_vals[22 * (ind + 1) + 17] - res_table_vals[22 * (ind) + 17]) * fac_lub
        ),
        res_table_vals[17],
    )

    YM11 = jnp.where(
        dist_lub >= 2 + res_table_min,
        (
            res_table_vals[22 * (ind) + 18]
            + (res_table_vals[22 * (ind + 1) + 18] - res_table_vals[22 * (ind) + 18]) * fac_lub
        ),
        res_table_vals[18],
    )

    YM12 = jnp.where(
        dist_lub >= 2 + res_table_min,
        (
            res_table_vals[22 * (ind) + 19]
            + (res_table_vals[22 * (ind + 1) + 19] - res_table_vals[22 * (ind) + 19]) * fac_lub
        ),
        res_table_vals[19],
    )

    ZM11 = jnp.where(
        dist_lub >= 2 + res_table_min,
        (
            res_table_vals[22 * (ind) + 20]
            + (res_table_vals[22 * (ind + 1) + 20] - res_table_vals[22 * (ind) + 20]) * fac_lub
        ),
        res_table_vals[20],
    )

    ZM12 = jnp.where(
        dist_lub >= 2 + res_table_min,
        (
            res_table_vals[22 * (ind) + 21]
            + (res_table_vals[22 * (ind + 1) + 21] - res_table_vals[22 * (ind) + 21]) * fac_lub
        ),
        res_table_vals[21],
    )

    res_func = jnp.array(
        [
            XA11,
            XA12,
            YA11,
            YA12,
            YB11,
            YB12,
            XC11,
            XC12,
            YC11,
            YC12,
            YB21,
            XG11,
            XG12,
            YG11,
            YG12,
            YH11,
            YH12,
            XM11,
            XM12,
            YM11,
            YM12,
            ZM11,
            ZM12,
        ]
    )

    # If particles are not smooth, add o(1/surf_dist) terms to tangential modes of lubrication
    # This mimics the presence of asperities at the particle surface.
    # See https://arxiv.org/pdf/2203.06300 for more detail
    alpha = jnp.array(
        [alpha, alpha, alpha, alpha, alpha]
    )  # set the friction coeff. for the 6 modes augmented
    surf_dist = dist_lub - 2.0
    surf_dist_sqr = surf_dist * surf_dist
    h02 = h0 * h0
    h03 = h0 * h02
    buffer = jnp.where(
        surf_dist <= h0, 2.0 / h03 * surf_dist_sqr - 3.0 / h02 * surf_dist + 1.0 / surf_dist, 0.0
    )

    res_func = res_func.at[2, :].add(alpha[0] * buffer)
    res_func = res_func.at[3, :].add(-alpha[1] * buffer)
    res_func = res_func.at[4, :].add(-alpha[2] * buffer)
    res_func = res_func.at[5, :].add(alpha[3] * buffer)
    res_func = res_func.at[8, :].add(alpha[4] * buffer)
    res_func = res_func.at[9, :].add(alpha[5] * buffer)

    return (
        (all_indices_x),
        (all_indices_y),
        (all_indices_z),
        gaussian_grid_spacing1,
        gaussian_grid_spacing2,
        r_unit,
        indices_i,
        indices_j,
        f1,
        f2,
        g1,
        g2,
        h1,
        h2,
        h3,
        r_lub_unit,
        indices_i_lub,
        indices_j_lub,
        res_func,
    )


@jit
def precompute_open(
    positions: ArrayLike,
    nl_ff: ArrayLike,
    nl_lub: ArrayLike,
    displacements_vector_matrix: ArrayLike,
    res_table_min: float,
    res_table_dr: float,
    res_table_dist: ArrayLike,
    res_table_vals: ArrayLike,
    alpha: float,
    h0: float,
) -> tuple[
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    tuple[
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
    ],
]:
    """Compute all the necessary quantities needed to update the particle position at a given timestep.

    Parameters
    ----------
    positions: (float)
        Array (num_particles,3) of current particles positions
    gaussian_grid_spacing: (float)
        Array (,gauss_support*gauss_support*gauss_support) containing distances from support center to each gridpoint in the gaussian support
    nl_ff: (int)
        Array (2,n_pairs_ff) containing far-field neighborlist indices
    nl_lub: (int)
        Array (2,n_pairs_nf) containing near-field neighborlist indices
    displacements_vector_matrix: (float)
        Array (num_particles,num_particles,3) of current displacements between particles, with each element a vector
    res_table_min: (float)
        Minimum distance resolved for lubrication interactions
    res_table_dr: (float)
        Resistance table discretization parameter
    res_table_dist: (float)
        Array (,1000) containing tabulated distances for resistance functions
    res_table_vals: (float)
        Array (,1000*22) containing tabulated resistance scalar functions values
    alpha_friction: (float)
        strength of hydrodynamic friction
    h0_friction: (float)
        range of hydrodynamic friction

    Returns
    -------
    all_indices_x, all_indices_y, all_indices_z, gaussian_grid_spacing1, gaussian_grid_spacing2,
     r_unit, indices_i, indices_j, f1, f2, g1, g2, h1, h2, h3,
     r_lub_unit, indices_i_lub, indices_j_lub,res_func,mobil_scalar

    """
    # Real Space (far-field) calculation quantities
    indices_i = nl_ff[0, :]  # Pair indices (i<j always)
    indices_j = nl_ff[1, :]
    # array of vectors from particles i to j (size = npairs)
    r = displacements_vector_matrix.at[indices_i, indices_j].get()
    dist = space.distance(r)  # distances between particles i and j
    r_unit = -r / dist.at[:, None].get()  # unit vector from particle j to i

    # compute mobility scalar functions in open boundaries (this should be moved to a separate module, and performed in double precision)
    xa12 = 3 / (2 * dist) - 1 / (dist * dist * dist)
    ya12 = 3 / (4 * dist) + 1 / (2 * dist * dist * dist)
    yb12 = -3 / (4 * dist * dist)
    xc12 = 3 / (4 * dist * dist * dist)
    yc12 = -3 / (8 * dist * dist * dist)
    xm12 = -9 / (2 * dist * dist * dist) + 54 / (5 * dist * dist * dist * dist * dist)
    ym12 = 9 / (4 * dist * dist * dist) - 36 / (5 * dist * dist * dist * dist * dist)
    zm12 = 9 / (5 * dist * dist * dist * dist * dist)
    xg12 = 9 / (4 * dist * dist) - 18 / (5 * dist * dist * dist * dist)
    yg12 = 6 / (5 * dist * dist * dist * dist)
    yh12 = -9 / (8 * dist * dist * dist)
    mobil_scalar = [xa12, ya12, yb12, xc12, yc12, xm12, ym12, zm12, xg12, yg12, yh12]

    # Lubrication calculation quantities
    indices_i_lub = nl_lub[0, :]  # Pair indices (i<j always)
    indices_j_lub = nl_lub[1, :]
    # array of vectors from particle i to j (size = npairs)
    r_lub = displacements_vector_matrix.at[nl_lub[0, :], nl_lub[1, :]].get()
    dist_lub = space.distance(r_lub)  # distance between particle i and j
    # unit vector from particle j to i
    r_lub_unit = r_lub / dist_lub.at[:, None].get()

    # # Indices in resistance table
    ind = jnp.log10((dist_lub - 2.0) / res_table_min) / res_table_dr
    ind = ind.astype(int)
    dist_lub_lower = res_table_dist.at[ind].get()
    dist_lub_upper = res_table_dist.at[ind + 1].get()
    # # Linear interpolation of the Table values
    fac_lub = jnp.where(
        dist_lub_upper - dist_lub_lower > 0.0,
        (dist_lub - dist_lub_lower) / (dist_lub_upper - dist_lub_lower),
        0.0,
    )

    XA11 = jnp.where(
        dist_lub >= 2 + res_table_min,
        (
            res_table_vals[22 * (ind) + 0]
            + (res_table_vals[22 * (ind + 1) + 0] - res_table_vals[22 * (ind) + 0]) * fac_lub
        ),
        res_table_vals[0],
    )

    XA12 = jnp.where(
        dist_lub >= 2 + res_table_min,
        (
            res_table_vals[22 * (ind) + 1]
            + (res_table_vals[22 * (ind + 1) + 1] - res_table_vals[22 * (ind) + 1]) * fac_lub
        ),
        res_table_vals[1],
    )

    YA11 = jnp.where(
        dist_lub >= 2 + res_table_min,
        (
            res_table_vals[22 * (ind) + 2]
            + (res_table_vals[22 * (ind + 1) + 2] - res_table_vals[22 * (ind) + 2]) * fac_lub
        ),
        res_table_vals[2],
    )

    YA12 = jnp.where(
        dist_lub >= 2 + res_table_min,
        (
            res_table_vals[22 * (ind) + 3]
            + (res_table_vals[22 * (ind + 1) + 3] - res_table_vals[22 * (ind) + 3]) * fac_lub
        ),
        res_table_vals[3],
    )

    YB11 = jnp.where(
        dist_lub >= 2 + res_table_min,
        (
            res_table_vals[22 * (ind) + 4]
            + (res_table_vals[22 * (ind + 1) + 4] - res_table_vals[22 * (ind) + 4]) * fac_lub
        ),
        res_table_vals[4],
    )

    YB12 = jnp.where(
        dist_lub >= 2 + res_table_min,
        (
            res_table_vals[22 * (ind) + 5]
            + (res_table_vals[22 * (ind + 1) + 5] - res_table_vals[22 * (ind) + 5]) * fac_lub
        ),
        res_table_vals[5],
    )

    XC11 = jnp.where(
        dist_lub >= 2 + res_table_min,
        (
            res_table_vals[22 * (ind) + 6]
            + (res_table_vals[22 * (ind + 1) + 6] - res_table_vals[22 * (ind) + 6]) * fac_lub
        ),
        res_table_vals[6],
    )

    XC12 = jnp.where(
        dist_lub >= 2 + res_table_min,
        (
            res_table_vals[22 * (ind) + 7]
            + (res_table_vals[22 * (ind + 1) + 7] - res_table_vals[22 * (ind) + 7]) * fac_lub
        ),
        res_table_vals[7],
    )

    YC11 = jnp.where(
        dist_lub >= 2 + res_table_min,
        (
            res_table_vals[22 * (ind) + 8]
            + (res_table_vals[22 * (ind + 1) + 8] - res_table_vals[22 * (ind) + 8]) * fac_lub
        ),
        res_table_vals[8],
    )

    YC12 = jnp.where(
        dist_lub >= 2 + res_table_min,
        (
            res_table_vals[22 * (ind) + 9]
            + (res_table_vals[22 * (ind + 1) + 9] - res_table_vals[22 * (ind) + 9]) * fac_lub
        ),
        res_table_vals[9],
    )

    YB21 = -YB12  # symmetry condition

    XG11 = jnp.where(
        dist_lub >= 2 + res_table_min,
        (
            res_table_vals[22 * (ind) + 10]
            + (res_table_vals[22 * (ind + 1) + 10] - res_table_vals[22 * (ind) + 10]) * fac_lub
        ),
        res_table_vals[10],
    )

    XG12 = jnp.where(
        dist_lub >= 2 + res_table_min,
        (
            res_table_vals[22 * (ind) + 11]
            + (res_table_vals[22 * (ind + 1) + 11] - res_table_vals[22 * (ind) + 11]) * fac_lub
        ),
        res_table_vals[11],
    )

    YG11 = jnp.where(
        dist_lub >= 2 + res_table_min,
        (
            res_table_vals[22 * (ind) + 12]
            + (res_table_vals[22 * (ind + 1) + 12] - res_table_vals[22 * (ind) + 12]) * fac_lub
        ),
        res_table_vals[12],
    )

    YG12 = jnp.where(
        dist_lub >= 2 + res_table_min,
        (
            res_table_vals[22 * (ind) + 13]
            + (res_table_vals[22 * (ind + 1) + 13] - res_table_vals[22 * (ind) + 13]) * fac_lub
        ),
        res_table_vals[13],
    )

    YH11 = jnp.where(
        dist_lub >= 2 + res_table_min,
        (
            res_table_vals[22 * (ind) + 14]
            + (res_table_vals[22 * (ind + 1) + 14] - res_table_vals[22 * (ind) + 14]) * fac_lub
        ),
        res_table_vals[14],
    )

    YH12 = jnp.where(
        dist_lub >= 2 + res_table_min,
        (
            res_table_vals[22 * (ind) + 15]
            + (res_table_vals[22 * (ind + 1) + 15] - res_table_vals[22 * (ind) + 15]) * fac_lub
        ),
        res_table_vals[15],
    )

    XM11 = jnp.where(
        dist_lub >= 2 + res_table_min,
        (
            res_table_vals[22 * (ind) + 16]
            + (res_table_vals[22 * (ind + 1) + 16] - res_table_vals[22 * (ind) + 16]) * fac_lub
        ),
        res_table_vals[16],
    )

    XM12 = jnp.where(
        dist_lub >= 2 + res_table_min,
        (
            res_table_vals[22 * (ind) + 17]
            + (res_table_vals[22 * (ind + 1) + 17] - res_table_vals[22 * (ind) + 17]) * fac_lub
        ),
        res_table_vals[17],
    )

    YM11 = jnp.where(
        dist_lub >= 2 + res_table_min,
        (
            res_table_vals[22 * (ind) + 18]
            + (res_table_vals[22 * (ind + 1) + 18] - res_table_vals[22 * (ind) + 18]) * fac_lub
        ),
        res_table_vals[18],
    )

    YM12 = jnp.where(
        dist_lub >= 2 + res_table_min,
        (
            res_table_vals[22 * (ind) + 19]
            + (res_table_vals[22 * (ind + 1) + 19] - res_table_vals[22 * (ind) + 19]) * fac_lub
        ),
        res_table_vals[19],
    )

    ZM11 = jnp.where(
        dist_lub >= 2 + res_table_min,
        (
            res_table_vals[22 * (ind) + 20]
            + (res_table_vals[22 * (ind + 1) + 20] - res_table_vals[22 * (ind) + 20]) * fac_lub
        ),
        res_table_vals[20],
    )

    ZM12 = jnp.where(
        dist_lub >= 2 + res_table_min,
        (
            res_table_vals[22 * (ind) + 21]
            + (res_table_vals[22 * (ind + 1) + 21] - res_table_vals[22 * (ind) + 21]) * fac_lub
        ),
        res_table_vals[21],
    )

    res_func = jnp.array(
        [
            XA11,
            XA12,
            YA11,
            YA12,
            YB11,
            YB12,
            XC11,
            XC12,
            YC11,
            YC12,
            YB21,
            XG11,
            XG12,
            YG11,
            YG12,
            YH11,
            YH12,
            XM11,
            XM12,
            YM11,
            YM12,
            ZM11,
            ZM12,
        ]
    )

    # If particles are not smooth, add o(1/surf_dist) terms to tangential modes of lubrication
    # This mimics the presence of asperities at the particle surface.
    # See https://arxiv.org/pdf/2203.06300 for more detail
    alpha = jnp.array(
        [alpha, alpha, alpha, alpha, alpha]
    )  # set the friction coeff. for the 6 modes augmented
    surf_dist = dist_lub - 2.0
    surf_dist_sqr = surf_dist * surf_dist
    h02 = h0 * h0
    h03 = h0 * h02
    buffer = jnp.where(
        surf_dist <= h0, 2.0 / h03 * surf_dist_sqr - 3.0 / h02 * surf_dist + 1.0 / surf_dist, 0.0
    )

    res_func = res_func.at[2, :].add(alpha[0] * buffer)
    res_func = res_func.at[3, :].add(-alpha[1] * buffer)
    res_func = res_func.at[4, :].add(-alpha[2] * buffer)
    res_func = res_func.at[5, :].add(alpha[3] * buffer)
    res_func = res_func.at[8, :].add(alpha[4] * buffer)
    res_func = res_func.at[9, :].add(alpha[5] * buffer)

    return (r_unit, indices_i, indices_j, r_lub_unit, indices_i_lub, indices_j_lub, res_func, mobil_scalar)


@partial(jit, static_argnums=[5, 15])
def precompute_rpy(
    positions: ArrayLike,
    gaussian_grid_spacing: ArrayLike,
    nl_ff: ArrayLike,
    displacements_vector_matrix: ArrayLike,
    tilt_factor: float,
    num_particles: int,
    box_x: float,
    box_y: float,
    box_z: float,
    grid_nx: int,
    grid_ny: int,
    grid_nz: int,
    prefac: float,
    expfac: float,
    quadw: ArrayLike,
    gauss_support: int,
    gauss_support_d2: int,
    ewald_n: int,
    ewald_dr: float,
    ewald_cut: float,
    ewaldc: ArrayLike,
) -> tuple[
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
]:
    """Compute all the necessary quantities needed to update the particle position at a given timestep.

    Parameters
    ----------
    positions: (float)
        Array (num_particles,3) of current particles positions
    gaussian_grid_spacing: (float)
        Array (,gauss_support*gauss_support*gauss_support) containing distances from support center to each gridpoint in the gaussian support
    nl_ff: (int)
        Array (2,n_pairs_ff) containing far-field neighborlist indices
    displacements_vector_matrix: (float)
        Array (num_particles,num_particles,3) of current displacements between particles, with each element a vector
    tilt_factor: (float)
        Current box tilt factor
    num_particles: (int)
        Number of particles
    box_x: (float)
        Box size in x direction
    box_y: (float)
        Box size in y direction
    box_z: (float)
        Box size in z direction
    grid_nx: (int)
        Number of grid points in x direction
    grid_ny: (int)
        Number of grid points in y direction
    grid_nz: (int)
        Number of grid points in z direction
    prefac: (float)
        Prefactor needed for FFT
    expfac: (float)
        Exponential factor needed for FFT
    quadw: (float)
        Product of wave grid discretization parameter in each direction (grid_dx*grid_dy*grid_dz)
    gauss_support: (int)
        Gaussian support size for wave space calculation
    gauss_support_d2: (int)
        Integer part of Gaussian support size divide by 2
    ewald_n: (int)
        Number of entries in Ewald table, for each mobility function
    ewald_dr: (float)
        Ewald table discretization parameter
    ewald_cut: (float)
        Ewald space cut-off for real-space far-field hydrodynamic interactions
    ewaldc: (float)
        Array (,ewald_n*7) containing tabulated mobility scalar functions values

    Returns
    -------
    all_indices_x, all_indices_y, all_indices_z, gaussian_grid_spacing1, gaussian_grid_spacing2,
     r_unit, indices_i, indices_j, f1, f2, g1, g2, h1, h2, h3

    """
    # Wave Space calculation quantities

    # Compute fractional coordinates
    pos = positions + jnp.array([box_x, box_y, box_z]) / 2
    pos = pos.at[:, 0].add(-tilt_factor * pos.at[:, 1].get())
    pos = pos / jnp.array([box_x, box_y, box_z]) * jnp.array([grid_nx, grid_ny, grid_nz])
    # convert positions in the box in indices in the grid
    # pos = (positions+np.array([box_x, box_y, box_z])/2)/np.array([box_x, box_y, box_z]) * np.array([grid_nx, grid_ny, grid_nz])
    intpos = jnp.array(pos, int)  # integer positions
    intpos = jnp.where(pos - intpos > 0.5, intpos + 1, intpos)

    # actual values to put in the grid
    gaussian_grid_spacing1 = prefac * jnp.exp(-expfac * gaussian_grid_spacing)
    gaussian_grid_spacing2 = jnp.swapaxes(
        jnp.swapaxes(jnp.swapaxes(gaussian_grid_spacing1 * quadw, 3, 2), 2, 1), 1, 0
    )

    # Starting indices on the grid from particle positions
    start_index_x = (intpos.at[:, 0].get() - (gauss_support_d2)) % grid_nx
    start_index_y = (intpos.at[:, 1].get() - (gauss_support_d2)) % grid_ny
    start_index_z = (intpos.at[:, 2].get() - (gauss_support_d2)) % grid_nz
    # All indices on the grids from particle positions, ordered such that x is the 'slowest' index to change and z the 'fastest'
    all_indices_x = jnp.repeat(
        jnp.repeat(jnp.repeat(start_index_x, gauss_support), gauss_support), gauss_support
    ) + jnp.resize(
        jnp.repeat(jnp.repeat(jnp.arange(gauss_support), gauss_support), gauss_support), gauss_support * gauss_support * gauss_support * num_particles
    )
    all_indices_y = jnp.repeat(
        jnp.repeat(jnp.repeat(start_index_y, gauss_support), gauss_support), gauss_support
    ) + jnp.resize(
        jnp.resize(jnp.repeat(jnp.arange(gauss_support), gauss_support), gauss_support * gauss_support * gauss_support),
        gauss_support * gauss_support * gauss_support * num_particles,
    )
    all_indices_z = jnp.repeat(
        jnp.repeat(jnp.repeat(start_index_z, gauss_support), gauss_support), gauss_support
    ) + jnp.resize(
        jnp.resize(jnp.resize(jnp.arange(gauss_support), gauss_support * gauss_support), gauss_support * gauss_support * gauss_support),
        gauss_support * gauss_support * gauss_support * num_particles,
    )
    all_indices_x = all_indices_x % grid_nx
    all_indices_y = all_indices_y % grid_ny
    all_indices_z = all_indices_z % grid_nz

    ###################################################################################################################
    # Real Space (far-field) calculation quantities
    indices_i = nl_ff[0, :]  # Pair indices (i<j always)
    indices_j = nl_ff[1, :]
    # array of vectors from particles i to j (size = npairs)
    r = displacements_vector_matrix.at[indices_i, indices_j].get()
    dist = space.distance(r)  # distances between particles i and j
    r_unit = -r / dist.at[:, None].get()  # unit vector from particle j to i

    # Interpolate scalar mobility functions values from ewald table
    r_ind = ewald_n * (dist - ewald_dr) / (ewald_cut - ewald_dr)  # index in ewald table
    r_ind = r_ind.astype(int)  # truncate decimal part
    offset1 = 2 * r_ind  # even indices
    offset2 = 2 * r_ind + 1  # odd indices

    tewaldc1m = ewaldc.at[offset1].get()  # UF and UC
    tewaldC1p = ewaldc.at[offset1 + 2].get()
    tewaldc2m = ewaldc.at[offset2].get()  # DC
    tewaldC2p = ewaldc.at[offset2 + 2].get()

    fac_ff = dist / ewald_dr - r_ind - 1.0  # interpolation factor

    f1 = tewaldc1m.at[:, 0].get() + (tewaldC1p.at[:, 0].get() - tewaldc1m.at[:, 0].get()) * fac_ff
    f2 = tewaldc1m.at[:, 1].get() + (tewaldC1p.at[:, 1].get() - tewaldc1m.at[:, 1].get()) * fac_ff

    g1 = tewaldc1m.at[:, 2].get() + (tewaldC1p.at[:, 2].get() - tewaldc1m.at[:, 2].get()) * fac_ff
    g2 = tewaldc1m.at[:, 3].get() + (tewaldC1p.at[:, 3].get() - tewaldc1m.at[:, 3].get()) * fac_ff

    h1 = tewaldc2m.at[:, 0].get() + (tewaldC2p.at[:, 0].get() - tewaldc2m.at[:, 0].get()) * fac_ff
    h2 = tewaldc2m.at[:, 1].get() + (tewaldC2p.at[:, 1].get() - tewaldc2m.at[:, 1].get()) * fac_ff
    h3 = tewaldc2m.at[:, 2].get() + (tewaldC2p.at[:, 2].get() - tewaldc2m.at[:, 2].get()) * fac_ff

    return (
        (all_indices_x),
        (all_indices_y),
        (all_indices_z),
        gaussian_grid_spacing1,
        gaussian_grid_spacing2,
        r_unit,
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


@jit
def precompute_rpy_open(
    positions: ArrayLike,
    nl_ff: ArrayLike,
    displacements_vector_matrix: ArrayLike,
) -> tuple[Array, Array, Array]:
    """Compute all the necessary quantities needed to update the particle position at a given timestep.

    Parameters
    ----------
    positions: (float)
        Array (num_particles,3) of current particles positions
    nl_ff: (int)
        Array (2,n_pairs_ff) containing far-field neighborlist indices
    displacements_vector_matrix: (float)
        Array (num_particles,num_particles,3) of current displacements between particles, with each element a vector

    Returns
    -------
    r_unit, indices_i, indices_j,mobil_scalar

    """
    indices_i = nl_ff[0, :]  # Pair indices (i<j always)
    indices_j = nl_ff[1, :]
    # array of vectors from particles i to j (size = npairs)
    r = displacements_vector_matrix.at[indices_i, indices_j].get()
    dist = space.distance(r)  # distances between particles i and j
    r_unit = -r / dist.at[:, None].get()  # unit vector from particle j to i

    # compute mobility scalar functions in open boundaries (this should be moved to a separate module, and performed in double precision)
    xa12 = 3 / (2 * dist) - 1 / (dist * dist * dist)
    ya12 = 3 / (4 * dist) + 1 / (2 * dist * dist * dist)
    yb12 = -3 / (4 * dist * dist)
    xc12 = 3 / (4 * dist * dist * dist)
    yc12 = -3 / (8 * dist * dist * dist)
    xm12 = -9 / (2 * dist * dist * dist) + 54 / (5 * dist * dist * dist * dist * dist)
    ym12 = 9 / (4 * dist * dist * dist) - 36 / (5 * dist * dist * dist * dist * dist)
    zm12 = 9 / (5 * dist * dist * dist * dist * dist)
    xg12 = 9 / (4 * dist * dist) - 18 / (5 * dist * dist * dist * dist)
    yg12 = 6 / (5 * dist * dist * dist * dist)
    yh12 = -9 / (8 * dist * dist * dist)
    mobil_scalar = [xa12, ya12, yb12, xc12, yc12, xm12, ym12, zm12, xg12, yg12, yh12]

    return (r_unit, indices_i, indices_j, mobil_scalar)


@partial(jit, static_argnums=[3])
def precompute_bd(
    positions: ArrayLike,
    nl: ArrayLike,
    displacements_vector_matrix: ArrayLike,
    num_particles: int,
    box_x: float,
    box_y: float,
    box_z: float,
) -> tuple[Array, Array, Array]:
    """Compute all the necessary quantities needed to update the particle position at a given timestep, for Brownian Dynamics.

    Parameters
    ----------
    positions: (float)
        Array (num_particles,3) of current particles positions
    nl: (int)
        Neighborlist indices
    displacements_vector_matrix: (float)
        Array (num_particles,num_particles,3) of current displacements between particles, with each element a vector
    num_particles: (int)
        Number of particles
    box_x: (float)
        Box size in x direction
    box_y: (float)
        Box size in y direction
    box_z: (float)
        Box size in z direction

    Returns
    -------
     r_unit, indices_i, indices_j

    """
    # Brownian Dynamics calculation quantities
    indices_i = nl[0, :]  # Pair indices (i<j always)
    indices_j = nl[1, :]
    # array of vectors from particle i to j (size = npairs)
    r = displacements_vector_matrix.at[nl[0, :], nl[1, :]].get()
    dist_lub = space.distance(r)  # distance between particle i and j
    # unit vector from particle j to i
    r_unit = r / dist_lub.at[:, None].get()

    return (r_unit, indices_i, indices_j)


@partial(jit, static_argnums=[0])
def compute_distinct_pairs(num_particles: int) -> ArrayLike:
    """Generate list of distinct pairs of particles, and order it as a neighbor list.

    Parameters
    ----------
    num_particles: (int)
        Number of particles in the system.

    Returns
    -------
    nl_ff

    """
    return jnp.stack(jnp.triu_indices(num_particles, 1), axis=1).T  # creates indices


def init_periodic_box(
    error, xi, box_x, box_y, box_z, ewald_cut, max_strain, xy, positions, num_particles, temperature, seed_ffwave
):
    """Initialize quantities needed to perform hydrodynamic calculations with periodic boundary conditions

    Parameters
    ----------
    xi: (float)
        Ewald split parameter
    box_x,box_y,box_z: (float)
        Box size
    ewald_cut: (float)
        Cutoff for real-space hydrodynamics
    max_strain: (float)
        Max strain allowed by simulation
    xy: (float)
        Box strain
    positions: (float)
        Positions of particles
    num_particles: (float)
        Number of particles
    temperature: (float)
        Temperature
    seed_ffwave: (float)
        Seed for thermal fluctuation in wave space

    Returns
    -------
    nl_ff

    """
    # parameter needed to make Chol. decomposition of R_FU converge (initially to 1)
    kmax = int(2.0 * jnp.sqrt(-jnp.log(error)) * xi) + 1  # Max wave number
    xisq = xi * xi
    # compute number of grid points in k space
    grid_nx, grid_ny, grid_nz = compute_k_gridpoint_number(kmax, box_x, box_y, box_z)
    gridh = jnp.array([box_x, box_y, box_z]) / jnp.array([grid_nx, grid_ny, grid_nz])  # Set Grid Spacing
    quadw = gridh[0] * gridh[1] * gridh[2]
    # check that ewald_cut is small enough to avoid interaction with periodic images)
    check_ewald_cutoff(ewald_cut, box_x, box_y, box_z, error)
    # check maximum eigenvalue of A'*A to scale support, P, for spreading on deformed grids
    eta, gauss_support = check_max_shear(gridh, xisq, grid_nx, grid_ny, grid_nz, max_strain, error)
    prefac = (2.0 * xisq / jnp.pi / eta) * jnp.sqrt(2.0 * xisq / jnp.pi / eta)
    expfac = 2.0 * xisq / eta
    gauss_support_d2 = jnp.array(gauss_support / 2, int)
    # get list of reciprocal space vectors, and scaling factor for the wave space calculation
    gridk = shear.compute_sheared_grid(int(grid_nx), int(grid_ny), int(grid_nz), xy, box_x, box_y, box_z, eta, xisq)

    # store the coefficients for the real space part of Ewald summation
    # will precompute scaling factors for real space component of summation for a given discretization to speed up GPU calculations
    # NOTE: Potential sensitivity of real space functions at small xi, tabulation computed in double prec., then truncated to single
    ewald_dr = 0.001  # Distance resolution
    ewald_n = ewald_cut / ewald_dr - 1  # Number of entries in tabulation
    # factors needed to compute mobility self contribution
    pi = jnp.pi  # pi
    pi12 = jnp.sqrt(pi)  # square root of pi
    a = 1.0  # radius
    axi = a * xi  # a * xi
    # compute mobility self contribution
    m_self = jnp.zeros(2, float)
    m_self = m_self.at[0].set(
        (1 + 4 * pi12 * axi * math.erfc(2.0 * axi) - math.exp(-4.0 * (axi * axi)))
        / (4.0 * pi12 * axi * a)
    )
    m_self = m_self.at[1].set(
        (-3.0 * math.erfc(2.0 * a * xi) * math.pow(a, -3.0)) / 10.0
        - (3.0 * math.pow(a, -6.0) * math.pow(pi, -0.5) * math.pow(xi, -3.0)) / 80.0
        - (9.0 * math.pow(a, -4) * math.pow(pi, -0.5) * math.pow(xi, -1)) / 40
        + (
            3.0
            * math.exp(-4 * math.pow(a, 2) * math.pow(xi, 2))
            * math.pow(a, -6)
            * math.pow(pi, -0.5)
            * math.pow(xi, -3)
            * (1 + 10 * math.pow(a, 2) * math.pow(xi, 2))
        )
        / 80
    )
    # create real space Ewald table
    ewald_entries = int(ewald_n + 1)  # number of entries in ewald table
    ewaldc1 = ewald_tables.compute_real_space_ewald_table(
        ewald_entries, a, xi
    )  # this version uses numpy long float
    ewaldc1 = jnp.array(ewaldc1)  # convert to single precision (32-bit)

    # precompute grid distances for FFT (same for each gaussian support)
    gaussian_grid_spacing = precompute_grid_distancing(
        gauss_support, gridh[0], xy, positions, num_particles, grid_nx, grid_ny, grid_nz, box_x, box_y, box_z
    )

    # create indices list for spreading random forces on the wave space grid (far field brownian velocity, wave space calculation)
    wave_bro_ind = wave_bro_nyind = 0.0
    if temperature > 0:
        (
            normal_indices_x,
            normal_indices_y,
            normal_indices_z,
            normal_conj_indices_x,
            normal_conj_indices_y,
            normal_conj_indices_z,
            nyquist_indices_x,
            nyquist_indices_y,
            nyquist_indices_z,
        ) = thermal.random_force_on_grid_indexing(grid_nx, grid_ny, grid_nz)

        # regroup indices in a unique array (this is done only once)
        wave_bro_ind = np.zeros((len(normal_indices_x), 2, 3), int)
        wave_bro_ind[:, 0, 0] = normal_indices_x
        wave_bro_ind[:, 0, 1] = normal_indices_y
        wave_bro_ind[:, 0, 2] = normal_indices_z
        wave_bro_ind[:, 1, 0] = normal_conj_indices_x
        wave_bro_ind[:, 1, 1] = normal_conj_indices_y
        wave_bro_ind[:, 1, 2] = normal_conj_indices_z
        wave_bro_nyind = np.zeros((len(nyquist_indices_x), 3), int)
        wave_bro_nyind[:, 0] = nyquist_indices_x
        wave_bro_nyind[:, 1] = nyquist_indices_y
        wave_bro_nyind[:, 2] = nyquist_indices_z

    # create RNG state for wave-space thermal fluctuations
    key_ffwave = random.PRNGKey(seed_ffwave)

    return (
        quadw,
        prefac,
        expfac,
        gauss_support_d2,
        gridk,
        gridh,
        gaussian_grid_spacing,
        key_ffwave,
        ewaldc1,
        m_self,
        grid_nx,
        grid_ny,
        grid_nz,
        gauss_support,
        ewald_n,
        ewald_dr,
        eta,
        xisq,
        wave_bro_ind,
        wave_bro_nyind
        )
