import math
import time
from functools import partial
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np
from jax import Array, dtypes, jit, random
from jax import random as jrandom
from jax.typing import ArrayLike

from jfsd import ewaldTables, shear, thermal
from jfsd import jaxmd_partition as partition
from jfsd import jaxmd_space as space

# Define types for the functions
DisplacementFn = Callable[[Any, Any], Any]


@jit
def chol_fac(A: ArrayLike) -> Array:
    """Perform a Cholesky factorization of the input matrix.

    Parameters
    ----------
    A: (float)
        Array (6N,6N) containing lubrication resistance matrix to factorize

    Returns
    -------
    Lower triangle Cholesky factor of input matrix A

    """
    return jnp.linalg.cholesky(A)


def Check_ewald_cut(ewald_cut: float, Lx: float, Ly: float, Lz: float, error: float):
    """Check that Ewald cutoff is small enough to avoid interaction with the particles images during real-space part of calculation.

    Parameters
    ----------
    ewald_cut: (float)
        Ewald space cut-off for real-space far-field hydrodynamic interactions
    Lx: (float)
        Box size in x direction
    Ly: (float)
        Box size in y direction
    Lz: (float)
        Box size in z direction
    error: (float)
        Tolerance error

    Returns
    -------

    """
    if (ewald_cut > Lx / 2) or (ewald_cut > Ly / 2) or (ewald_cut > Lz / 2):
        max_cut = max([Lx, Ly, Lz]) / 2.0
        new_xi = np.sqrt(-np.log(error)) / max_cut
        raise ValueError(f"Ewald cutoff radius is too large. Try with xi = {new_xi}")
    else:
        print("Ewald Cutoff is ", ewald_cut)
    return


def Check_max_shear(
    gridh: ArrayLike, xisq: float, Nx: int, Ny: int, Nz: int, max_strain: float, error: float
) -> tuple[float, float]:
    """Check maximum eigenvalue of A'*A to scale support, P, for spreading on deformed grids (Fiore and Swan, J. Chem. Phys., 2018).

    Parameters
    ----------
    gridh: (float)
        Array (,3) containing wave space grid discrete spacing
    xisq: (float)
        Squared Ewald split parameter
    Nx: (int)
        Number of grid points in x direction
    Ny: (int)
        Number of grid points in y direction
    Nz: (int)
        Number of grid points in z direction
    max_strain: (float)
        Max strain applied to the box
    error: (float)
        Tolerance error

    Returns
    -------
    eta, gaussP

    """
    gamma = max_strain
    lambdaa = 1 + gamma * gamma / 2 + gamma * np.sqrt(1 + gamma * gamma / 4)

    # Parameters for the Spectral Ewald Method (Lindbo and Tornberg, J. Comp. Phys., 2011)
    gaussm = 1.0
    while math.erfc(gaussm / np.sqrt(2 * lambdaa)) > error:
        gaussm += 0.01
    gaussP = int(gaussm * gaussm / jnp.pi) + 1
    w = gaussP * gridh[0] / 2.0  # Gaussian width in simulation units
    eta = (2.0 * w / gaussm) * (2.0 * w / gaussm) * (xisq)  # Gaussian splitting parameter

    # Check that the support size isn't somehow larger than the grid
    if gaussP > min(Nx, min(Ny, Nz)):
        raise ValueError(
            f"Quadrature Support Exceeds Available Grid. \n (Mx, My, Mz) = ({Nx}), ({Ny}), ({Nz}). Support Size, P = {gaussP}"
        )
    return eta, gaussP


def Compute_k_gridpoint_number(
    kmax: float, Lx: float, Ly: float, Lz: float
) -> tuple[int, int, int]:
    """Perform a Cholesky factorization of the input matrix.

    Parameters
    ----------
    kmax: (int)
        Max wave number
    Lx: (float)
        Box size in x direction
    Ly: (float)
        Box size in y direction
    Lz: (float)
        Box size in z direction

    Returns
    -------
    Nx, Ny, Nz

    """
    # Set initial number of grid points in wave space
    Nx = int(kmax * Lx / (2.0 * jnp.pi) * 2.0) + 1
    Ny = int(kmax * Ly / (2.0 * jnp.pi) * 2.0) + 1
    Nz = int(kmax * Lz / (2.0 * jnp.pi) * 2.0) + 1

    # Get list of int values between 8 and 512 that can be written as (2^a)*(3^b)*(5^c)
    # Then sort list from low to high and figure out how many entries there are
    Mlist = []
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
                Mcurr = int(pow2 * pow3 * pow5)
                if (Mcurr >= 8) and (Mcurr <= 512):
                    Mlist.append(Mcurr)
    Mlist = np.array(Mlist)
    Mlist = np.sort(Mlist)
    nmult = len(Mlist)

    # Compute the number of grid points in each direction (should be a power of 2,3,5 for most efficient FFTs)
    for ii in range(0, nmult):
        if Nx <= Mlist[ii]:
            Nx = Mlist[ii]
            break
    for ii in range(0, nmult):
        if Ny <= Mlist[ii]:
            Ny = Mlist[ii]
            break
    for ii in range(0, nmult):
        if Nz <= Mlist[ii]:
            Nz = Mlist[ii]
            break

    # Maximum number of FFT nodes is limited by available memory = 512 * 512 * 512 = 134217728
    if Nx * Ny * Nz > 134217728:
        raise ValueError("Requested Number of Fourier Nodes Exceeds Max Dimension of 512^3")
    return Nx, Ny, Nz


def Precompute_grid_distancing(
    gaussP: int,
    gridh: ArrayLike,
    tilt_factor: float,
    positions: ArrayLike,
    N: int,
    Nx: int,
    Ny: int,
    Nz: int,
    Lx: float,
    Ly: float,
    Lz: float,
) -> Array:
    """Given a support size for Gaussian spread, compute distances in a (gaussP x gaussP x gaussP) grid with gauss P shifted by 1 unit if it is odd or even.

    Parameters
    ----------
    gaussP: (int)
        Gaussian support size for wave space calculation
    gridh: (float)
        Array (,3) containing wave space grid discrete spacing
    tilt_factor: (float)
        Current box tilt factor
    positions: (float)
        Array (N,3) of current particles positions
    N: (int)
        Number of particles
    Nx: (int)
        Number of grid points in x direction
    Ny: (int)
        Number of grid points in y direction
    Nz: (int)
        Number of grid points in z direction
    Lx: (float)
        Box size in x direction
    Ly: (float)
        Box size in y direction
    Lz: (float)
        Box size in z direction

    Returns
    -------
    grid

    """
    grid = np.zeros((gaussP, gaussP, gaussP, N))
    # center_offsets = (jnp.array(positions)+jnp.array([Lx,Ly,Lz])/2)*jnp.array([Nx,Ny,Nz])/jnp.array([Lx,Ly,Lz])
    center_offsets = np.array(positions) + np.array([Lx, Ly, Lz]) / 2
    center_offsets[:, 0] += -tilt_factor * center_offsets[:, 1]
    # center_offsets = center_offsets.at[:,0].add(-tilt_factor*center_offsets.at[:,1].get())
    center_offsets = center_offsets / np.array([Lx, Ly, Lz]) * np.array([Nx, Ny, Nz])
    # in grid units, not particle radius units
    center_offsets -= np.array(center_offsets, dtype=int)
    center_offsets = np.where(center_offsets > 0.5, -(1 - center_offsets), center_offsets)
    for i in range(gaussP):
        for j in range(gaussP):
            for k in range(gaussP):
                grid[i, j, k, :] = (
                    gridh
                    * gridh
                    * (
                        (
                            i
                            - int(gaussP / 2)
                            - center_offsets[:, 0]
                            + tilt_factor * (j - int(gaussP / 2) - center_offsets[:, 1])
                        )
                        ** 2
                        + (j - int(gaussP / 2) - center_offsets[:, 1]) ** 2
                        + (k - int(gaussP / 2) - center_offsets[:, 2]) ** 2
                    )
                )
    return grid


def initialize_single_neighborlist(
    space_cut: float, Lx: float, Ly: float, Lz: float, displacement: DisplacementFn
) -> partition.NeighborListFns:
    """Initialize a single neighborlists, given a box and a distance cutoff.

    Note that creation of neighborlists is perfomed using code from jax_md.
    At the moment, the code does not use cell list, as it produces artifact.
    In future, the creation of neighborlists will be handled entirely by JFSD, leveraging cell lists.

    Parameters
    ----------
    space_cut: (float)
        Cutoff (max) distance for particles to be considered neighbors
    Lx: (float)
        Box size in x direction
    Ly: (float)
        Box size in y direction
    Lz: (float)
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
        jnp.array([Lx, Ly, Lz]),
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
        Array (N,3) of particles positions
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
        Array (N,3) of particles positions
    nbrs:
        Input neighbor lists

    Returns
    -------
    nbrs

    """
    # Update neighbor list
    nbrs = nbrs.update(positions)
    return nbrs


def create_hardsphere_configuration(L: float, N: int, seed: int, T: float) -> Array:
    """Create an (at equilibrium, or close) configuration of Brownian hard spheres at a given temperature T.

    First initialize a random configuration of N ideal particles.
    Then, thermalize the system while applying a soft repulsive potential,
    until no overlaps are present in the system.

    Parameters
    ----------
    L: (float)
        Box size requested
    N: (int)
        Number of particles requested
    seed: (int)
        Seed for random number generator
    T: (float)
        Thermal energy of the system

    Returns
    -------
    positions

    """

    @jit
    def compute_hardsphere(net_vel, displacements, indices_i, indices_j):
        Fp = jnp.zeros((N, N, 3))

        # Reset particle velocity
        net_vel *= 0.0

        # Compute velocity from hard-sphere repulsion
        dist_mod = jnp.sqrt(
            displacements[:, :, 0] * displacements[:, :, 0]
            + displacements[:, :, 1] * displacements[:, :, 1]
            + displacements[:, :, 2] * displacements[:, :, 2]
        )

        Fp_mod = jnp.where(
            indices_i != indices_j, k * (1 - sigma / dist_mod[indices_i, indices_j]), 0.0
        )
        Fp_mod = jnp.where((dist_mod[indices_i, indices_j]) < sigma, Fp_mod, 0.0)

        # get forces in components
        Fp = Fp.at[indices_i, indices_j, 0].add(Fp_mod * displacements[indices_i, indices_j, 0])
        Fp = Fp.at[indices_i, indices_j, 1].add(Fp_mod * displacements[indices_i, indices_j, 1])
        Fp = Fp.at[indices_i, indices_j, 2].add(Fp_mod * displacements[indices_i, indices_j, 2])
        Fp = Fp.at[indices_j, indices_i, 0].add(Fp_mod * displacements[indices_j, indices_i, 0])
        Fp = Fp.at[indices_j, indices_i, 1].add(Fp_mod * displacements[indices_j, indices_i, 1])
        Fp = Fp.at[indices_j, indices_i, 2].add(Fp_mod * displacements[indices_j, indices_i, 2])

        # sum all forces in each particle
        Fp = jnp.sum(Fp, 1)

        net_vel = net_vel.at[(0)::3].add(Fp.at[:, 0].get())
        net_vel = net_vel.at[(1)::3].add(Fp.at[:, 1].get())
        net_vel = net_vel.at[(2)::3].add(Fp.at[:, 2].get())

        return net_vel

    @jit
    def update_pos(positions, displacements, net_vel, nbrs):
        # Define array of displacement r(t+dt)-r(t)
        dR = jnp.zeros((N, 3), float)
        # Compute actual displacement due to velocities
        dR = dR.at[:, 0].set(dt * net_vel.at[(0)::3].get())
        dR = dR.at[:, 1].set(dt * net_vel.at[(1)::3].get())
        dR = dR.at[:, 2].set(dt * net_vel.at[(2)::3].get())
        # Apply displacement and compute wrapped shift
        # shift system origin to (0,0,0)
        positions = shift(positions + jnp.array([L, L, L]) / 2, dR)
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

    net_vel = jnp.zeros(3 * N)
    key = jrandom.PRNGKey(seed)
    T = 0.001
    sigma = 2.15  # 2.15 #2.05 #particle diameter
    phi_eff = N / (L * L * L) * (sigma * sigma * sigma) * np.pi / 6
    phi_actual = ((np.pi / 6) * (2 * 2 * 2) * N) / (L * L * L)
    brow_time = sigma * sigma / (4 * T)

    dt = brow_time / (1e5)

    Nsteps = int(brow_time / dt)
    key, random_coord = generate_random_array(key, (N * 3))
    random_coord = (random_coord - 1 / 2) * L
    positions = jnp.zeros((N, 3))
    positions = positions.at[:, 0].set(random_coord[0::3])
    positions = positions.at[:, 1].set(random_coord[1::3])
    positions = positions.at[:, 2].set(random_coord[2::3])

    displacements = (space.map_product(displacement))(positions, positions)

    neighbor_fn = initialize_single_neighborlist(2.2, L, L, L, displacement)
    nbrs = allocate_nlist(positions + jnp.array([L, L, L]) / 2, neighbor_fn)
    nl = np.array(nbrs.idx)

    overlaps, _ = find_overlaps(displacements, 2.002, N)
    k = 30 * np.sqrt(6 * T / (sigma * sigma * dt))  # spring constant

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
            (r, indices_i, indices_j) = precomputeBD(positions, nl, displacements, N, L, L, L)

            # compute forces for each pair
            net_vel = compute_hardsphere(net_vel, displacements, indices_i, indices_j)

            # Compute and add Brownian velocity
            key, random_noise = generate_random_array(key, (6 * N))
            brow = thermal.compute_BD_randomforce(N, T, dt, random_noise)
            net_vel = add_thermal_noise(net_vel, brow)

            # Update positions
            new_positions, nbrs, new_displacements, net_vel = update_pos(
                positions, displacements, net_vel, nbrs
            )

            # If the allocated neighbor list is too small in size for the new particle configuration, re-allocate it, otherwise just update it
            if nbrs.did_buffer_overflow:
                nbrs = allocate_nlist(positions, neighbor_fn)
                nl = np.array(nbrs.idx)
                (r, indices_i, indices_j) = precomputeBD(positions, nl, displacements, N, L, L, L)
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

        overlaps, _ = find_overlaps(displacements, 2.002, N)
        if (time.time() - start_time) > 1800:  # interrupt if too computationally expensive
            raise ValueError("Creation of initial configuration failed. Abort!")
    print("Initial configuration created. Volume fraction is ", phi_actual)
    return positions


@partial(jit, static_argnums=[2])
def find_overlaps(dist: ArrayLike, sigma: float, N: int) -> tuple[int, float]:
    """Check overlaps between particles and returns number of overlaps + number of particles.

    The radius of a particle is set to 1.
    Note that this function should be used only for debugging,
    as the memory cost scales quadratically with the number of particles N.

    Parameters
    ----------
    dist: (float)
        Array (N*N,3) of distance vectors between particles in neighbor list
    sigma: (float)
        Particle diameter
    N: (int)
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
    indices = jnp.nonzero(mask, size=N, fill_value=N)
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
    N: int,
    Lx: float,
    Ly: float,
    Lz: float,
    Nx: int,
    Ny: int,
    Nz: int,
    prefac: float,
    expfac: float,
    quadW: ArrayLike,
    gaussP: int,
    gaussPd2: int,
    ewald_n: int,
    ewald_dr: float,
    ewald_cut: float,
    ewaldC: ArrayLike,
    ResTable_min: float,
    ResTable_dr: float,
    ResTable_dist: ArrayLike,
    ResTable_vals: ArrayLike,
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
        Array (N,3) of current particles positions
    gaussian_grid_spacing: (float)
        Array (,gaussP*gaussP*gaussP) containing distances from support center to each gridpoint in the gaussian support
    nl_ff: (int)
        Array (2,n_pairs_ff) containing far-field neighborlist indices
    nl_lub: (int)
        Array (2,n_pairs_nf) containing near-field neighborlist indices
    displacements_vector_matrix: (float)
        Array (N,N,3) of current displacements between particles, with each element a vector
    tilt_factor: (float)
        Current box tilt factor
    N: (int)
        Number of particles
    Lx: (float)
        Box size in x direction
    Ly: (float)
        Box size in y direction
    Lz: (float)
        Box size in z direction
    Nx: (int)
        Number of grid points in x direction
    Ny: (int)
        Number of grid points in y direction
    Nz: (int)
        Number of grid points in z direction
    prefac: (float)
        Prefactor needed for FFT
    expfac: (float)
        Exponential factor needed for FFT
    quadW: (float)
        Product of wave grid discretization parameter in each direction (grid_dx*grid_dy*grid_dz)
    gaussP: (int)
        Gaussian support size for wave space calculation
    gaussPd2: (int)
        Integer part of Gaussian support size divide by 2
    ewald_n: (int)
        Number of entries in Ewald table, for each mobility function
    ewald_dr: (float)
        Ewald table discretization parameter
    ewald_cut: (float)
        Ewald space cut-off for real-space far-field hydrodynamic interactions
    ewaldC: (float)
        Array (,ewald_n*7) containing tabulated mobility scalar functions values
    ResTable_min: (float)
        Minimum distance resolved for lubrication interactions
    ResTable_dr: (float)
        Resistance table discretization parameter
    ResTable_dist: (float)
        Array (,1000) containing tabulated distances for resistance functions
    ResTable_vals: (float)
        Array (,1000*22) containing tabulated resistance scalar functions values
    alpha_friction: (float)
        strength of hydrodynamic friction
    h0_friction: (float)
        range of hydrodynamic friction

    Returns
    -------
    all_indices_x, all_indices_y, all_indices_z, gaussian_grid_spacing1, gaussian_grid_spacing2,
     r, indices_i, indices_j, f1, f2, g1, g2, h1, h2, h3,
     r_lub, indices_i_lub, indices_j_lub,ResFunc

    """
    ###Wave Space calculation quantities

    # Compute fractional coordinates
    pos = positions + jnp.array([Lx, Ly, Lz]) / 2
    pos = pos.at[:, 0].add(-tilt_factor * pos.at[:, 1].get())
    pos = pos / jnp.array([Lx, Ly, Lz]) * jnp.array([Nx, Ny, Nz])
    ###convert positions in the box in indices in the grid
    # pos = (positions+np.array([Lx, Ly, Lz])/2)/np.array([Lx, Ly, Lz]) * np.array([Nx, Ny, Nz])
    intpos = jnp.array(pos, int)  # integer positions
    intpos = jnp.where(pos - intpos > 0.5, intpos + 1, intpos)

    # actual values to put in the grid
    gaussian_grid_spacing1 = prefac * jnp.exp(-expfac * gaussian_grid_spacing)
    gaussian_grid_spacing2 = jnp.swapaxes(
        jnp.swapaxes(jnp.swapaxes(gaussian_grid_spacing1 * quadW, 3, 2), 2, 1), 1, 0
    )

    # Starting indices on the grid from particle positions
    start_index_x = (intpos.at[:, 0].get() - (gaussPd2)) % Nx
    start_index_y = (intpos.at[:, 1].get() - (gaussPd2)) % Ny
    start_index_z = (intpos.at[:, 2].get() - (gaussPd2)) % Nz
    # All indices on the grids from particle positions, ordered such that x is the 'slowest' index to change and z the 'fastest'
    all_indices_x = jnp.repeat(
        jnp.repeat(jnp.repeat(start_index_x, gaussP), gaussP), gaussP
    ) + jnp.resize(
        jnp.repeat(jnp.repeat(jnp.arange(gaussP), gaussP), gaussP), gaussP * gaussP * gaussP * N
    )
    all_indices_y = jnp.repeat(
        jnp.repeat(jnp.repeat(start_index_y, gaussP), gaussP), gaussP
    ) + jnp.resize(
        jnp.resize(jnp.repeat(jnp.arange(gaussP), gaussP), gaussP * gaussP * gaussP),
        gaussP * gaussP * gaussP * N,
    )
    all_indices_z = jnp.repeat(
        jnp.repeat(jnp.repeat(start_index_z, gaussP), gaussP), gaussP
    ) + jnp.resize(
        jnp.resize(jnp.resize(jnp.arange(gaussP), gaussP * gaussP), gaussP * gaussP * gaussP),
        gaussP * gaussP * gaussP * N,
    )
    all_indices_x = all_indices_x % Nx
    all_indices_y = all_indices_y % Ny
    all_indices_z = all_indices_z % Nz

    ###################################################################################################################
    # Real Space (far-field) calculation quantities
    indices_i = nl_ff[0, :]  # Pair indices (i<j always)
    indices_j = nl_ff[1, :]
    # array of vectors from particles i to j (size = npairs)
    R = displacements_vector_matrix.at[indices_i, indices_j].get()
    dist = space.distance(R)  # distances between particles i and j
    r = -R / dist.at[:, None].get()  # unit vector from particle j to i

    # Interpolate scalar mobility functions values from ewald table
    r_ind = ewald_n * (dist - ewald_dr) / (ewald_cut - ewald_dr)  # index in ewald table
    r_ind = r_ind.astype(int)  # truncate decimal part
    offset1 = 2 * r_ind  # even indices
    offset2 = 2 * r_ind + 1  # odd indices

    tewaldC1m = ewaldC.at[offset1].get()  # UF and UC
    tewaldC1p = ewaldC.at[offset1 + 2].get()
    tewaldC2m = ewaldC.at[offset2].get()  # DC
    tewaldC2p = ewaldC.at[offset2 + 2].get()

    fac_ff = dist / ewald_dr - r_ind - 1.0  # interpolation factor

    f1 = tewaldC1m.at[:, 0].get() + (tewaldC1p.at[:, 0].get() - tewaldC1m.at[:, 0].get()) * fac_ff
    f2 = tewaldC1m.at[:, 1].get() + (tewaldC1p.at[:, 1].get() - tewaldC1m.at[:, 1].get()) * fac_ff

    g1 = tewaldC1m.at[:, 2].get() + (tewaldC1p.at[:, 2].get() - tewaldC1m.at[:, 2].get()) * fac_ff
    g2 = tewaldC1m.at[:, 3].get() + (tewaldC1p.at[:, 3].get() - tewaldC1m.at[:, 3].get()) * fac_ff

    h1 = tewaldC2m.at[:, 0].get() + (tewaldC2p.at[:, 0].get() - tewaldC2m.at[:, 0].get()) * fac_ff
    h2 = tewaldC2m.at[:, 1].get() + (tewaldC2p.at[:, 1].get() - tewaldC2m.at[:, 1].get()) * fac_ff
    h3 = tewaldC2m.at[:, 2].get() + (tewaldC2p.at[:, 2].get() - tewaldC2m.at[:, 2].get()) * fac_ff

    ###################################################################################################################
    # Lubrication calculation quantities
    indices_i_lub = nl_lub[0, :]  # Pair indices (i<j always)
    indices_j_lub = nl_lub[1, :]
    # array of vectors from particle i to j (size = npairs)
    R_lub = displacements_vector_matrix.at[nl_lub[0, :], nl_lub[1, :]].get()
    dist_lub = space.distance(R_lub)  # distance between particle i and j
    # unit vector from particle j to i
    r_lub = R_lub / dist_lub.at[:, None].get()

    # # Indices in resistance table
    ind = jnp.log10((dist_lub - 2.0) / ResTable_min) / ResTable_dr
    ind = ind.astype(int)
    dist_lub_lower = ResTable_dist.at[ind].get()
    dist_lub_upper = ResTable_dist.at[ind + 1].get()
    # # Linear interpolation of the Table values
    fac_lub = jnp.where(
        dist_lub_upper - dist_lub_lower > 0.0,
        (dist_lub - dist_lub_lower) / (dist_lub_upper - dist_lub_lower),
        0.0,
    )

    XA11 = jnp.where(
        dist_lub >= 2 + ResTable_min,
        (
            ResTable_vals[22 * (ind) + 0]
            + (ResTable_vals[22 * (ind + 1) + 0] - ResTable_vals[22 * (ind) + 0]) * fac_lub
        ),
        ResTable_vals[0],
    )

    XA12 = jnp.where(
        dist_lub >= 2 + ResTable_min,
        (
            ResTable_vals[22 * (ind) + 1]
            + (ResTable_vals[22 * (ind + 1) + 1] - ResTable_vals[22 * (ind) + 1]) * fac_lub
        ),
        ResTable_vals[1],
    )

    YA11 = jnp.where(
        dist_lub >= 2 + ResTable_min,
        (
            ResTable_vals[22 * (ind) + 2]
            + (ResTable_vals[22 * (ind + 1) + 2] - ResTable_vals[22 * (ind) + 2]) * fac_lub
        ),
        ResTable_vals[2],
    )

    YA12 = jnp.where(
        dist_lub >= 2 + ResTable_min,
        (
            ResTable_vals[22 * (ind) + 3]
            + (ResTable_vals[22 * (ind + 1) + 3] - ResTable_vals[22 * (ind) + 3]) * fac_lub
        ),
        ResTable_vals[3],
    )

    YB11 = jnp.where(
        dist_lub >= 2 + ResTable_min,
        (
            ResTable_vals[22 * (ind) + 4]
            + (ResTable_vals[22 * (ind + 1) + 4] - ResTable_vals[22 * (ind) + 4]) * fac_lub
        ),
        ResTable_vals[4],
    )

    YB12 = jnp.where(
        dist_lub >= 2 + ResTable_min,
        (
            ResTable_vals[22 * (ind) + 5]
            + (ResTable_vals[22 * (ind + 1) + 5] - ResTable_vals[22 * (ind) + 5]) * fac_lub
        ),
        ResTable_vals[5],
    )

    XC11 = jnp.where(
        dist_lub >= 2 + ResTable_min,
        (
            ResTable_vals[22 * (ind) + 6]
            + (ResTable_vals[22 * (ind + 1) + 6] - ResTable_vals[22 * (ind) + 6]) * fac_lub
        ),
        ResTable_vals[6],
    )

    XC12 = jnp.where(
        dist_lub >= 2 + ResTable_min,
        (
            ResTable_vals[22 * (ind) + 7]
            + (ResTable_vals[22 * (ind + 1) + 7] - ResTable_vals[22 * (ind) + 7]) * fac_lub
        ),
        ResTable_vals[7],
    )

    YC11 = jnp.where(
        dist_lub >= 2 + ResTable_min,
        (
            ResTable_vals[22 * (ind) + 8]
            + (ResTable_vals[22 * (ind + 1) + 8] - ResTable_vals[22 * (ind) + 8]) * fac_lub
        ),
        ResTable_vals[8],
    )

    YC12 = jnp.where(
        dist_lub >= 2 + ResTable_min,
        (
            ResTable_vals[22 * (ind) + 9]
            + (ResTable_vals[22 * (ind + 1) + 9] - ResTable_vals[22 * (ind) + 9]) * fac_lub
        ),
        ResTable_vals[9],
    )

    YB21 = -YB12  # symmetry condition

    XG11 = jnp.where(
        dist_lub >= 2 + ResTable_min,
        (
            ResTable_vals[22 * (ind) + 10]
            + (ResTable_vals[22 * (ind + 1) + 10] - ResTable_vals[22 * (ind) + 10]) * fac_lub
        ),
        ResTable_vals[10],
    )

    XG12 = jnp.where(
        dist_lub >= 2 + ResTable_min,
        (
            ResTable_vals[22 * (ind) + 11]
            + (ResTable_vals[22 * (ind + 1) + 11] - ResTable_vals[22 * (ind) + 11]) * fac_lub
        ),
        ResTable_vals[11],
    )

    YG11 = jnp.where(
        dist_lub >= 2 + ResTable_min,
        (
            ResTable_vals[22 * (ind) + 12]
            + (ResTable_vals[22 * (ind + 1) + 12] - ResTable_vals[22 * (ind) + 12]) * fac_lub
        ),
        ResTable_vals[12],
    )

    YG12 = jnp.where(
        dist_lub >= 2 + ResTable_min,
        (
            ResTable_vals[22 * (ind) + 13]
            + (ResTable_vals[22 * (ind + 1) + 13] - ResTable_vals[22 * (ind) + 13]) * fac_lub
        ),
        ResTable_vals[13],
    )

    YH11 = jnp.where(
        dist_lub >= 2 + ResTable_min,
        (
            ResTable_vals[22 * (ind) + 14]
            + (ResTable_vals[22 * (ind + 1) + 14] - ResTable_vals[22 * (ind) + 14]) * fac_lub
        ),
        ResTable_vals[14],
    )

    YH12 = jnp.where(
        dist_lub >= 2 + ResTable_min,
        (
            ResTable_vals[22 * (ind) + 15]
            + (ResTable_vals[22 * (ind + 1) + 15] - ResTable_vals[22 * (ind) + 15]) * fac_lub
        ),
        ResTable_vals[15],
    )

    XM11 = jnp.where(
        dist_lub >= 2 + ResTable_min,
        (
            ResTable_vals[22 * (ind) + 16]
            + (ResTable_vals[22 * (ind + 1) + 16] - ResTable_vals[22 * (ind) + 16]) * fac_lub
        ),
        ResTable_vals[16],
    )

    XM12 = jnp.where(
        dist_lub >= 2 + ResTable_min,
        (
            ResTable_vals[22 * (ind) + 17]
            + (ResTable_vals[22 * (ind + 1) + 17] - ResTable_vals[22 * (ind) + 17]) * fac_lub
        ),
        ResTable_vals[17],
    )

    YM11 = jnp.where(
        dist_lub >= 2 + ResTable_min,
        (
            ResTable_vals[22 * (ind) + 18]
            + (ResTable_vals[22 * (ind + 1) + 18] - ResTable_vals[22 * (ind) + 18]) * fac_lub
        ),
        ResTable_vals[18],
    )

    YM12 = jnp.where(
        dist_lub >= 2 + ResTable_min,
        (
            ResTable_vals[22 * (ind) + 19]
            + (ResTable_vals[22 * (ind + 1) + 19] - ResTable_vals[22 * (ind) + 19]) * fac_lub
        ),
        ResTable_vals[19],
    )

    ZM11 = jnp.where(
        dist_lub >= 2 + ResTable_min,
        (
            ResTable_vals[22 * (ind) + 20]
            + (ResTable_vals[22 * (ind + 1) + 20] - ResTable_vals[22 * (ind) + 20]) * fac_lub
        ),
        ResTable_vals[20],
    )

    ZM12 = jnp.where(
        dist_lub >= 2 + ResTable_min,
        (
            ResTable_vals[22 * (ind) + 21]
            + (ResTable_vals[22 * (ind + 1) + 21] - ResTable_vals[22 * (ind) + 21]) * fac_lub
        ),
        ResTable_vals[21],
    )

    ResFunc = jnp.array(
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

    ResFunc = ResFunc.at[2, :].add(alpha[0] * buffer)
    ResFunc = ResFunc.at[3, :].add(-alpha[1] * buffer)
    ResFunc = ResFunc.at[4, :].add(-alpha[2] * buffer)
    ResFunc = ResFunc.at[5, :].add(alpha[3] * buffer)
    ResFunc = ResFunc.at[8, :].add(alpha[4] * buffer)
    ResFunc = ResFunc.at[9, :].add(alpha[5] * buffer)

    return (
        (all_indices_x),
        (all_indices_y),
        (all_indices_z),
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
        ResFunc,
    )


@jit
def precompute_open(
    positions: ArrayLike,
    nl_ff: ArrayLike,
    nl_lub: ArrayLike,
    displacements_vector_matrix: ArrayLike,
    ResTable_min: float,
    ResTable_dr: float,
    ResTable_dist: ArrayLike,
    ResTable_vals: ArrayLike,
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
        Array (N,3) of current particles positions
    gaussian_grid_spacing: (float)
        Array (,gaussP*gaussP*gaussP) containing distances from support center to each gridpoint in the gaussian support
    nl_ff: (int)
        Array (2,n_pairs_ff) containing far-field neighborlist indices
    nl_lub: (int)
        Array (2,n_pairs_nf) containing near-field neighborlist indices
    displacements_vector_matrix: (float)
        Array (N,N,3) of current displacements between particles, with each element a vector
    ResTable_min: (float)
        Minimum distance resolved for lubrication interactions
    ResTable_dr: (float)
        Resistance table discretization parameter
    ResTable_dist: (float)
        Array (,1000) containing tabulated distances for resistance functions
    ResTable_vals: (float)
        Array (,1000*22) containing tabulated resistance scalar functions values
    alpha_friction: (float)
        strength of hydrodynamic friction
    h0_friction: (float)
        range of hydrodynamic friction

    Returns
    -------
    all_indices_x, all_indices_y, all_indices_z, gaussian_grid_spacing1, gaussian_grid_spacing2,
     r, indices_i, indices_j, f1, f2, g1, g2, h1, h2, h3,
     r_lub, indices_i_lub, indices_j_lub,ResFunc,mobil_scalar

    """
    # Real Space (far-field) calculation quantities
    indices_i = nl_ff[0, :]  # Pair indices (i<j always)
    indices_j = nl_ff[1, :]
    # array of vectors from particles i to j (size = npairs)
    R = displacements_vector_matrix.at[indices_i, indices_j].get()
    dist = space.distance(R)  # distances between particles i and j
    r = -R / dist.at[:, None].get()  # unit vector from particle j to i

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
    R_lub = displacements_vector_matrix.at[nl_lub[0, :], nl_lub[1, :]].get()
    dist_lub = space.distance(R_lub)  # distance between particle i and j
    # unit vector from particle j to i
    r_lub = R_lub / dist_lub.at[:, None].get()

    # # Indices in resistance table
    ind = jnp.log10((dist_lub - 2.0) / ResTable_min) / ResTable_dr
    ind = ind.astype(int)
    dist_lub_lower = ResTable_dist.at[ind].get()
    dist_lub_upper = ResTable_dist.at[ind + 1].get()
    # # Linear interpolation of the Table values
    fac_lub = jnp.where(
        dist_lub_upper - dist_lub_lower > 0.0,
        (dist_lub - dist_lub_lower) / (dist_lub_upper - dist_lub_lower),
        0.0,
    )

    XA11 = jnp.where(
        dist_lub >= 2 + ResTable_min,
        (
            ResTable_vals[22 * (ind) + 0]
            + (ResTable_vals[22 * (ind + 1) + 0] - ResTable_vals[22 * (ind) + 0]) * fac_lub
        ),
        ResTable_vals[0],
    )

    XA12 = jnp.where(
        dist_lub >= 2 + ResTable_min,
        (
            ResTable_vals[22 * (ind) + 1]
            + (ResTable_vals[22 * (ind + 1) + 1] - ResTable_vals[22 * (ind) + 1]) * fac_lub
        ),
        ResTable_vals[1],
    )

    YA11 = jnp.where(
        dist_lub >= 2 + ResTable_min,
        (
            ResTable_vals[22 * (ind) + 2]
            + (ResTable_vals[22 * (ind + 1) + 2] - ResTable_vals[22 * (ind) + 2]) * fac_lub
        ),
        ResTable_vals[2],
    )

    YA12 = jnp.where(
        dist_lub >= 2 + ResTable_min,
        (
            ResTable_vals[22 * (ind) + 3]
            + (ResTable_vals[22 * (ind + 1) + 3] - ResTable_vals[22 * (ind) + 3]) * fac_lub
        ),
        ResTable_vals[3],
    )

    YB11 = jnp.where(
        dist_lub >= 2 + ResTable_min,
        (
            ResTable_vals[22 * (ind) + 4]
            + (ResTable_vals[22 * (ind + 1) + 4] - ResTable_vals[22 * (ind) + 4]) * fac_lub
        ),
        ResTable_vals[4],
    )

    YB12 = jnp.where(
        dist_lub >= 2 + ResTable_min,
        (
            ResTable_vals[22 * (ind) + 5]
            + (ResTable_vals[22 * (ind + 1) + 5] - ResTable_vals[22 * (ind) + 5]) * fac_lub
        ),
        ResTable_vals[5],
    )

    XC11 = jnp.where(
        dist_lub >= 2 + ResTable_min,
        (
            ResTable_vals[22 * (ind) + 6]
            + (ResTable_vals[22 * (ind + 1) + 6] - ResTable_vals[22 * (ind) + 6]) * fac_lub
        ),
        ResTable_vals[6],
    )

    XC12 = jnp.where(
        dist_lub >= 2 + ResTable_min,
        (
            ResTable_vals[22 * (ind) + 7]
            + (ResTable_vals[22 * (ind + 1) + 7] - ResTable_vals[22 * (ind) + 7]) * fac_lub
        ),
        ResTable_vals[7],
    )

    YC11 = jnp.where(
        dist_lub >= 2 + ResTable_min,
        (
            ResTable_vals[22 * (ind) + 8]
            + (ResTable_vals[22 * (ind + 1) + 8] - ResTable_vals[22 * (ind) + 8]) * fac_lub
        ),
        ResTable_vals[8],
    )

    YC12 = jnp.where(
        dist_lub >= 2 + ResTable_min,
        (
            ResTable_vals[22 * (ind) + 9]
            + (ResTable_vals[22 * (ind + 1) + 9] - ResTable_vals[22 * (ind) + 9]) * fac_lub
        ),
        ResTable_vals[9],
    )

    YB21 = -YB12  # symmetry condition

    XG11 = jnp.where(
        dist_lub >= 2 + ResTable_min,
        (
            ResTable_vals[22 * (ind) + 10]
            + (ResTable_vals[22 * (ind + 1) + 10] - ResTable_vals[22 * (ind) + 10]) * fac_lub
        ),
        ResTable_vals[10],
    )

    XG12 = jnp.where(
        dist_lub >= 2 + ResTable_min,
        (
            ResTable_vals[22 * (ind) + 11]
            + (ResTable_vals[22 * (ind + 1) + 11] - ResTable_vals[22 * (ind) + 11]) * fac_lub
        ),
        ResTable_vals[11],
    )

    YG11 = jnp.where(
        dist_lub >= 2 + ResTable_min,
        (
            ResTable_vals[22 * (ind) + 12]
            + (ResTable_vals[22 * (ind + 1) + 12] - ResTable_vals[22 * (ind) + 12]) * fac_lub
        ),
        ResTable_vals[12],
    )

    YG12 = jnp.where(
        dist_lub >= 2 + ResTable_min,
        (
            ResTable_vals[22 * (ind) + 13]
            + (ResTable_vals[22 * (ind + 1) + 13] - ResTable_vals[22 * (ind) + 13]) * fac_lub
        ),
        ResTable_vals[13],
    )

    YH11 = jnp.where(
        dist_lub >= 2 + ResTable_min,
        (
            ResTable_vals[22 * (ind) + 14]
            + (ResTable_vals[22 * (ind + 1) + 14] - ResTable_vals[22 * (ind) + 14]) * fac_lub
        ),
        ResTable_vals[14],
    )

    YH12 = jnp.where(
        dist_lub >= 2 + ResTable_min,
        (
            ResTable_vals[22 * (ind) + 15]
            + (ResTable_vals[22 * (ind + 1) + 15] - ResTable_vals[22 * (ind) + 15]) * fac_lub
        ),
        ResTable_vals[15],
    )

    XM11 = jnp.where(
        dist_lub >= 2 + ResTable_min,
        (
            ResTable_vals[22 * (ind) + 16]
            + (ResTable_vals[22 * (ind + 1) + 16] - ResTable_vals[22 * (ind) + 16]) * fac_lub
        ),
        ResTable_vals[16],
    )

    XM12 = jnp.where(
        dist_lub >= 2 + ResTable_min,
        (
            ResTable_vals[22 * (ind) + 17]
            + (ResTable_vals[22 * (ind + 1) + 17] - ResTable_vals[22 * (ind) + 17]) * fac_lub
        ),
        ResTable_vals[17],
    )

    YM11 = jnp.where(
        dist_lub >= 2 + ResTable_min,
        (
            ResTable_vals[22 * (ind) + 18]
            + (ResTable_vals[22 * (ind + 1) + 18] - ResTable_vals[22 * (ind) + 18]) * fac_lub
        ),
        ResTable_vals[18],
    )

    YM12 = jnp.where(
        dist_lub >= 2 + ResTable_min,
        (
            ResTable_vals[22 * (ind) + 19]
            + (ResTable_vals[22 * (ind + 1) + 19] - ResTable_vals[22 * (ind) + 19]) * fac_lub
        ),
        ResTable_vals[19],
    )

    ZM11 = jnp.where(
        dist_lub >= 2 + ResTable_min,
        (
            ResTable_vals[22 * (ind) + 20]
            + (ResTable_vals[22 * (ind + 1) + 20] - ResTable_vals[22 * (ind) + 20]) * fac_lub
        ),
        ResTable_vals[20],
    )

    ZM12 = jnp.where(
        dist_lub >= 2 + ResTable_min,
        (
            ResTable_vals[22 * (ind) + 21]
            + (ResTable_vals[22 * (ind + 1) + 21] - ResTable_vals[22 * (ind) + 21]) * fac_lub
        ),
        ResTable_vals[21],
    )

    ResFunc = jnp.array(
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

    ResFunc = ResFunc.at[2, :].add(alpha[0] * buffer)
    ResFunc = ResFunc.at[3, :].add(-alpha[1] * buffer)
    ResFunc = ResFunc.at[4, :].add(-alpha[2] * buffer)
    ResFunc = ResFunc.at[5, :].add(alpha[3] * buffer)
    ResFunc = ResFunc.at[8, :].add(alpha[4] * buffer)
    ResFunc = ResFunc.at[9, :].add(alpha[5] * buffer)

    return (r, indices_i, indices_j, r_lub, indices_i_lub, indices_j_lub, ResFunc, mobil_scalar)


@partial(jit, static_argnums=[5, 15])
def precomputeRPY(
    positions: ArrayLike,
    gaussian_grid_spacing: ArrayLike,
    nl_ff: ArrayLike,
    displacements_vector_matrix: ArrayLike,
    tilt_factor: float,
    N: int,
    Lx: float,
    Ly: float,
    Lz: float,
    Nx: int,
    Ny: int,
    Nz: int,
    prefac: float,
    expfac: float,
    quadW: ArrayLike,
    gaussP: int,
    gaussPd2: int,
    ewald_n: int,
    ewald_dr: float,
    ewald_cut: float,
    ewaldC: ArrayLike,
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
        Array (N,3) of current particles positions
    gaussian_grid_spacing: (float)
        Array (,gaussP*gaussP*gaussP) containing distances from support center to each gridpoint in the gaussian support
    nl_ff: (int)
        Array (2,n_pairs_ff) containing far-field neighborlist indices
    displacements_vector_matrix: (float)
        Array (N,N,3) of current displacements between particles, with each element a vector
    tilt_factor: (float)
        Current box tilt factor
    N: (int)
        Number of particles
    Lx: (float)
        Box size in x direction
    Ly: (float)
        Box size in y direction
    Lz: (float)
        Box size in z direction
    Nx: (int)
        Number of grid points in x direction
    Ny: (int)
        Number of grid points in y direction
    Nz: (int)
        Number of grid points in z direction
    prefac: (float)
        Prefactor needed for FFT
    expfac: (float)
        Exponential factor needed for FFT
    quadW: (float)
        Product of wave grid discretization parameter in each direction (grid_dx*grid_dy*grid_dz)
    gaussP: (int)
        Gaussian support size for wave space calculation
    gaussPd2: (int)
        Integer part of Gaussian support size divide by 2
    ewald_n: (int)
        Number of entries in Ewald table, for each mobility function
    ewald_dr: (float)
        Ewald table discretization parameter
    ewald_cut: (float)
        Ewald space cut-off for real-space far-field hydrodynamic interactions
    ewaldC: (float)
        Array (,ewald_n*7) containing tabulated mobility scalar functions values

    Returns
    -------
    all_indices_x, all_indices_y, all_indices_z, gaussian_grid_spacing1, gaussian_grid_spacing2,
     r, indices_i, indices_j, f1, f2, g1, g2, h1, h2, h3

    """
    ###Wave Space calculation quantities

    # Compute fractional coordinates
    pos = positions + jnp.array([Lx, Ly, Lz]) / 2
    pos = pos.at[:, 0].add(-tilt_factor * pos.at[:, 1].get())
    pos = pos / jnp.array([Lx, Ly, Lz]) * jnp.array([Nx, Ny, Nz])
    ###convert positions in the box in indices in the grid
    # pos = (positions+np.array([Lx, Ly, Lz])/2)/np.array([Lx, Ly, Lz]) * np.array([Nx, Ny, Nz])
    intpos = jnp.array(pos, int)  # integer positions
    intpos = jnp.where(pos - intpos > 0.5, intpos + 1, intpos)

    # actual values to put in the grid
    gaussian_grid_spacing1 = prefac * jnp.exp(-expfac * gaussian_grid_spacing)
    gaussian_grid_spacing2 = jnp.swapaxes(
        jnp.swapaxes(jnp.swapaxes(gaussian_grid_spacing1 * quadW, 3, 2), 2, 1), 1, 0
    )

    # Starting indices on the grid from particle positions
    start_index_x = (intpos.at[:, 0].get() - (gaussPd2)) % Nx
    start_index_y = (intpos.at[:, 1].get() - (gaussPd2)) % Ny
    start_index_z = (intpos.at[:, 2].get() - (gaussPd2)) % Nz
    # All indices on the grids from particle positions, ordered such that x is the 'slowest' index to change and z the 'fastest'
    all_indices_x = jnp.repeat(
        jnp.repeat(jnp.repeat(start_index_x, gaussP), gaussP), gaussP
    ) + jnp.resize(
        jnp.repeat(jnp.repeat(jnp.arange(gaussP), gaussP), gaussP), gaussP * gaussP * gaussP * N
    )
    all_indices_y = jnp.repeat(
        jnp.repeat(jnp.repeat(start_index_y, gaussP), gaussP), gaussP
    ) + jnp.resize(
        jnp.resize(jnp.repeat(jnp.arange(gaussP), gaussP), gaussP * gaussP * gaussP),
        gaussP * gaussP * gaussP * N,
    )
    all_indices_z = jnp.repeat(
        jnp.repeat(jnp.repeat(start_index_z, gaussP), gaussP), gaussP
    ) + jnp.resize(
        jnp.resize(jnp.resize(jnp.arange(gaussP), gaussP * gaussP), gaussP * gaussP * gaussP),
        gaussP * gaussP * gaussP * N,
    )
    all_indices_x = all_indices_x % Nx
    all_indices_y = all_indices_y % Ny
    all_indices_z = all_indices_z % Nz

    ###################################################################################################################
    # Real Space (far-field) calculation quantities
    indices_i = nl_ff[0, :]  # Pair indices (i<j always)
    indices_j = nl_ff[1, :]
    # array of vectors from particles i to j (size = npairs)
    R = displacements_vector_matrix.at[indices_i, indices_j].get()
    dist = space.distance(R)  # distances between particles i and j
    r = -R / dist.at[:, None].get()  # unit vector from particle j to i

    # Interpolate scalar mobility functions values from ewald table
    r_ind = ewald_n * (dist - ewald_dr) / (ewald_cut - ewald_dr)  # index in ewald table
    r_ind = r_ind.astype(int)  # truncate decimal part
    offset1 = 2 * r_ind  # even indices
    offset2 = 2 * r_ind + 1  # odd indices

    tewaldC1m = ewaldC.at[offset1].get()  # UF and UC
    tewaldC1p = ewaldC.at[offset1 + 2].get()
    tewaldC2m = ewaldC.at[offset2].get()  # DC
    tewaldC2p = ewaldC.at[offset2 + 2].get()

    fac_ff = dist / ewald_dr - r_ind - 1.0  # interpolation factor

    f1 = tewaldC1m.at[:, 0].get() + (tewaldC1p.at[:, 0].get() - tewaldC1m.at[:, 0].get()) * fac_ff
    f2 = tewaldC1m.at[:, 1].get() + (tewaldC1p.at[:, 1].get() - tewaldC1m.at[:, 1].get()) * fac_ff

    g1 = tewaldC1m.at[:, 2].get() + (tewaldC1p.at[:, 2].get() - tewaldC1m.at[:, 2].get()) * fac_ff
    g2 = tewaldC1m.at[:, 3].get() + (tewaldC1p.at[:, 3].get() - tewaldC1m.at[:, 3].get()) * fac_ff

    h1 = tewaldC2m.at[:, 0].get() + (tewaldC2p.at[:, 0].get() - tewaldC2m.at[:, 0].get()) * fac_ff
    h2 = tewaldC2m.at[:, 1].get() + (tewaldC2p.at[:, 1].get() - tewaldC2m.at[:, 1].get()) * fac_ff
    h3 = tewaldC2m.at[:, 2].get() + (tewaldC2p.at[:, 2].get() - tewaldC2m.at[:, 2].get()) * fac_ff

    return (
        (all_indices_x),
        (all_indices_y),
        (all_indices_z),
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
    )


@jit
def precomputeRPY_open(
    positions: ArrayLike,
    nl_ff: ArrayLike,
    displacements_vector_matrix: ArrayLike,
) -> tuple[Array, Array, Array]:
    """Compute all the necessary quantities needed to update the particle position at a given timestep.

    Parameters
    ----------
    positions: (float)
        Array (N,3) of current particles positions
    nl_ff: (int)
        Array (2,n_pairs_ff) containing far-field neighborlist indices
    displacements_vector_matrix: (float)
        Array (N,N,3) of current displacements between particles, with each element a vector

    Returns
    -------
    r, indices_i, indices_j,mobil_scalar

    """
    indices_i = nl_ff[0, :]  # Pair indices (i<j always)
    indices_j = nl_ff[1, :]
    # array of vectors from particles i to j (size = npairs)
    R = displacements_vector_matrix.at[indices_i, indices_j].get()
    dist = space.distance(R)  # distances between particles i and j
    r = -R / dist.at[:, None].get()  # unit vector from particle j to i

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

    return (r, indices_i, indices_j, mobil_scalar)


@partial(jit, static_argnums=[3])
def precomputeBD(
    positions: ArrayLike,
    nl: ArrayLike,
    displacements_vector_matrix: ArrayLike,
    N: int,
    Lx: float,
    Ly: float,
    Lz: float,
) -> tuple[Array, Array, Array]:
    """Compute all the necessary quantities needed to update the particle position at a given timestep, for Brownian Dynamics.

    Parameters
    ----------
    positions: (float)
        Array (N,3) of current particles positions
    nl: (int)
        Neighborlist indices
    displacements_vector_matrix: (float)
        Array (N,N,3) of current displacements between particles, with each element a vector
    N: (int)
        Number of particles
    Lx: (float)
        Box size in x direction
    Ly: (float)
        Box size in y direction
    Lz: (float)
        Box size in z direction

    Returns
    -------
     r, indices_i, indices_j

    """
    # Brownian Dynamics calculation quantities
    indices_i = nl[0, :]  # Pair indices (i<j always)
    indices_j = nl[1, :]
    # array of vectors from particle i to j (size = npairs)
    R = displacements_vector_matrix.at[nl[0, :], nl[1, :]].get()
    dist_lub = space.distance(R)  # distance between particle i and j
    # unit vector from particle j to i
    r = R / dist_lub.at[:, None].get()

    return (r, indices_i, indices_j)


@partial(jit, static_argnums=[0])
def compute_distinct_pairs(N: int) -> ArrayLike:
    """Generate list of distinct pairs of particles, and order it as a neighbor list.

    Parameters
    ----------
    N: (int)
        Number of particles in the system.

    Returns
    -------
    nl_ff

    """
    return jnp.stack(jnp.triu_indices(N, 1), axis=1).T  # creates indices


def init_periodic_box(
    error, xi, Lx, Ly, Lz, ewald_cut, max_strain, xy, positions, N, T, seed_ffwave
):
    """Initialize quantities needed to perform hydrodynamic calculations with periodic boundary conditions

    Parameters
    ----------
    xi: (float)
        Ewald split parameter
    Lx,Ly,Lz: (float)
        Box size
    ewald_cut: (float)
        Cutoff for real-space hydrodynamics
    max_strain: (float)
        Max strain allowed by simulation
    xy: (float)
        Box strain
    positions: (float)
        Positions of particles
    N: (float)
        Number of particles
    T: (float)
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
    Nx, Ny, Nz = Compute_k_gridpoint_number(kmax, Lx, Ly, Lz)
    gridh = jnp.array([Lx, Ly, Lz]) / jnp.array([Nx, Ny, Nz])  # Set Grid Spacing
    quadW = gridh[0] * gridh[1] * gridh[2]
    # check that ewald_cut is small enough to avoid interaction with periodic images)
    Check_ewald_cut(ewald_cut, Lx, Ly, Lz, error)
    # check maximum eigenvalue of A'*A to scale support, P, for spreading on deformed grids
    eta, gaussP = Check_max_shear(gridh, xisq, Nx, Ny, Nz, max_strain, error)
    prefac = (2.0 * xisq / jnp.pi / eta) * jnp.sqrt(2.0 * xisq / jnp.pi / eta)
    expfac = 2.0 * xisq / eta
    gaussPd2 = jnp.array(gaussP / 2, int)
    # get list of reciprocal space vectors, and scaling factor for the wave space calculation
    gridk = shear.compute_sheared_grid(int(Nx), int(Ny), int(Nz), xy, Lx, Ly, Lz, eta, xisq)

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
    nR = int(ewald_n + 1)  # number of entries in ewald table
    ewaldC1 = ewaldTables.Compute_real_space_ewald_table(
        nR, a, xi
    )  # this version uses numpy long float
    ewaldC1 = jnp.array(ewaldC1)  # convert to single precision (32-bit)

    # precompute grid distances for FFT (same for each gaussian support)
    gaussian_grid_spacing = Precompute_grid_distancing(
        gaussP, gridh[0], xy, positions, N, Nx, Ny, Nz, Lx, Ly, Lz
    )

    # create indices list for spreading random forces on the wave space grid (far field brownian velocity, wave space calculation)
    wave_bro_ind = wave_bro_nyind = 0.0
    if T > 0:
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
        ) = thermal.Random_force_on_grid_indexing(Nx, Ny, Nz)

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
        Nx,
        Ny,
        Nz,
        gaussP,
        ewald_n,
        ewald_dr,
        eta,
        xisq,
        wave_bro_ind,
        wave_bro_nyind,
    )
