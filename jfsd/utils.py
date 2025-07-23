import math
import time
from functools import partial
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np
from jax import Array, dtypes, jit, random, ops
from jax import random as jrandom
from jax.typing import ArrayLike

from jfsd import ewald_tables, shear, thermal
from jfsd import jaxmd_space as space

from tqdm import tqdm

# Define types for the functions
DisplacementFn = Callable[[Any, Any], Any]

def preprocess_sparse_triangular(m_sparse, num_particles, max_nonzero_per_row):
    """
    Efficiently precomputes sparse lookup structures for forward substitution.
    
    Parameters
    ----------
    m_sparse : ndarray
        Sparse representation of precondition matrix.
    num_particles: int 
        Number of particles.
    max_nonzero_per_row: int
        Max number of non-zero entries per row that can be stored.
    
    Returns
    -------
    row_indices : ndarray
        (num_particles, max_nonzero_per_row)  indices of nonzero entries
    row_values: ndarray
        (num_particles, max_nonzero_per_row) corresponding values
    """
    
    # Convert scipy sparse structure to NumPy for fast processing
    values = np.array(m_sparse.data)
    indices = np.array(m_sparse.nonzero()).T

    # **Step 1: Sort by row index for efficient processing**
    sorted_idx = np.argsort(indices[:, 0], kind='stable')
    indices = indices[sorted_idx]
    values = values[sorted_idx]

    # **Step 2: Compute row start/end positions using `searchsorted`**
    row_starts = np.searchsorted(indices[:, 0], np.arange(num_particles))
    row_ends = np.append(row_starts[1:], len(indices))

    # **Step 3: Preallocate arrays with correct shape**
    row_indices = np.full((num_particles, max_nonzero_per_row), -1)
    row_values = np.zeros((num_particles, max_nonzero_per_row))

    # **Step 4: Fill row indices & values using NumPy slicing**
    for i in range(num_particles):
        start, end = row_starts[i], row_ends[i]
        row_data = indices[start:end, 1]
        row_vals = values[start:end]

        row_indices[i, :len(row_data)] = row_data
        row_values[i, :len(row_vals)] = row_vals

    return row_indices, row_values

def displacement_fn(ra, rb, box):

    def _get_free_indices(n: int) -> str:
        return "".join([chr(ord("a") + i) for i in range(n)])    

    def transform(box, r):
        free_indices = _get_free_indices(r.ndim - 1)
        left_indices = free_indices + "j"
        right_indices = free_indices + "i"
        return jnp.einsum(f"ij,{left_indices}->{right_indices}", box, r)
        
    inv_box = jnp.linalg.inv(box)
    ra = transform(inv_box, ra)
    rb = transform(inv_box, rb)
    
    dr = jnp.mod((rb-ra) + 1 * (0.5), 1.) - (0.5) * 1
    
    dr = transform(box, dr)

    return dr

def cpu_nlist(positions, l, nl_cutoff, second_cutoff, third_cutoff, xy, initial_safety_margin=1.):
    """
    Compute a sparse neighbor list with an adaptive safety layer (R + dR) using a cell list.
    
    If overflow is detected (i.e., if padding slots are used), the safety margin is returned.
    
    Parameters
    ----------
    positions : (N, 3) ndarray
        Particle positions.
    l : (3,) ndarray
        Box dimensions.
    nl_cutoff : float
        Original neighbor cutoff distance.
    second_cutoff : float
        Secondary cutoff distance for filtering.
    third_cutoff : float
        Tertiary cutoff distance for filtering.
    xy : float
        Shear parameter for Lees-Edwards boundary conditions.
    initial_safety_margin : float, optional
        Initial additional margin (default: 1.0).
    
    Returns
    -------
    unique_pairs : jnp.array
        (2, num_pairs) array of unique neighbor pairs (with the smaller index first).
    masked_pairs : jnp.array
        Neighbor list where pairs with distance >= nl_cutoff are replaced with num_particles.
    smaller_list_1 : jnp.array
        Neighbor list where pairs with distance >= second_cutoff are replaced with num_particles.
    smaller_list_2 : jnp.array
        Neighbor list where pairs with distance >= third_cutoff are replaced with num_particles,
        sorted by the first index and truncated to a fixed maximum number of columns.
    safety_margin_used : float
        The final safety margin used.
    """
    
    num_particles = positions.shape[0]
    positions = np.array(positions)
    if num_particles == 1:
        empty_arr = jnp.empty((2, 0), dtype=int)
        return empty_arr, empty_arr, empty_arr, empty_arr, initial_safety_margin

    safety_margin = initial_safety_margin
    max_neigh_prec = num_particles * 15
    extended_cutoff = nl_cutoff + safety_margin

    # --- Cell List Construction ---
    # Choose cell size (here we use extended_cutoff*2 or the smallest box dimension)
    cell_size = min(extended_cutoff * 2, np.min(l))
    # Shift positions so that the box goes from 0 to l (assumes positions originally centered at 0)
    shifted_positions = positions + l * 0.5
    cell_indices = np.floor(shifted_positions / cell_size).astype(int)
    
    # Instead of using the full grid dims from the box, we use the occupied dimensions:
    occupied_dims = cell_indices.max(axis=0) + 1

    # Build a dictionary mapping a cell (as a tuple) to the list of particle indices in that cell.
    cell_dict = {}
    for i, cell in enumerate(cell_indices):
        key = tuple(cell)
        cell_dict.setdefault(key, []).append(i)
    
    # Precompute all neighbor offsets (27 total: each combination of -1, 0, 1 in x, y, z)
    offsets = np.array([[dx, dy, dz] 
                        for dx in (-1, 0, 1) 
                        for dy in (-1, 0, 1) 
                        for dz in (-1, 0, 1)])
    
    pairs_i = []
    pairs_j = []
    dists = []
    
    def process_pair_list(list1, list2, same_cell=False):
        pts1 = positions[list1]  # shape (n1, 3)
        pts2 = positions[list2]  # shape (n2, 3)
        # Compute vector differences (broadcasted)
        diff = pts1[:, np.newaxis, :] - pts2[np.newaxis, :, :]
        # Apply Lees–Edwards shear: adjust x by subtracting xy * y
        diff[:, :, 0] -= xy * diff[:, :, 1]
        # Apply periodic boundary conditions using the minimum image convention:
        diff = diff - np.round(diff / l) * l
        dist_sq = np.sum(diff**2, axis=-1)
        valid = (dist_sq > 0) & (dist_sq < extended_cutoff**2)
        inds1, inds2 = np.where(valid)
        for a, b in zip(inds1, inds2):
            # For particles in the same cell, enforce i < j to avoid duplicates.
            if same_cell and list1[a] >= list2[b]:
                continue
            pairs_i.append(list1[a])
            pairs_j.append(list2[b])
            dists.append(np.sqrt(dist_sq[a, b]))
    
    # Loop over each occupied cell.
    for key, part_list in cell_dict.items():
        cell_coord = np.array(key)
        for off in offsets:
            # Use occupied_dims for wrapping over the range of cells that actually contain particles.
            neighbor_coord = (cell_coord + off) % occupied_dims
            neighbor_key = tuple(neighbor_coord)
            if neighbor_key not in cell_dict:
                continue
            neighbor_list = cell_dict[neighbor_key]
            if neighbor_key == key:
                process_pair_list(part_list, neighbor_list, same_cell=True)
            else:
                # To avoid double counting, only process if key < neighbor_key (lexicographically)
                if key < neighbor_key:
                    process_pair_list(part_list, neighbor_list, same_cell=False)
    
    pairs_i = np.array(pairs_i, dtype=int)
    pairs_j = np.array(pairs_j, dtype=int)
    dists = np.array(dists)
    
    # If no neighbor pairs were found, return empty arrays.
    if pairs_i.size == 0:
        empty_arr = jnp.empty((2, 0), dtype=int)
        print('Warning: neighborlists are empty. Increase cell list cutoff, or the simulation might run at a slower speed.')
        return empty_arr, empty_arr, empty_arr, empty_arr, safety_margin
    
    # Stack into a (2, num_pairs) array and remove duplicate pairs.
    pairs = np.stack((pairs_i, pairs_j), axis=0)
    unique_pairs, unique_indices = np.unique(pairs, axis=1, return_index=True)
    dists = dists[unique_indices]
    
    # --- Build the Masked Neighbor Lists ---
    # nlist1: For pairs with distance >= nl_cutoff, set both indices to num_particles.
    nlist1 = unique_pairs.copy()
    nlist1[:, dists >= nl_cutoff] = num_particles
    # nlist2: For pairs with distance >= second_cutoff, set both indices to num_particles.
    nlist2 = unique_pairs.copy()
    nlist2[:, dists >= second_cutoff] = num_particles
    # nlist3: For pairs with distance >= third_cutoff, set both indices to num_particles,
    # then sort by the first index and truncate to max_neigh_prec columns.
    nlist_buffer = unique_pairs.copy()
    nlist_buffer[:, dists >= third_cutoff] = num_particles
    order = np.argsort(nlist_buffer[0, :])
    nlist_buffer = nlist_buffer[:, order]
    nlist3 = nlist_buffer[:, :max_neigh_prec]
    
    # --- Return results ---
    return (jnp.array(unique_pairs, int),
            jnp.array(nlist1, int),
            jnp.array(nlist2, int),
            jnp.array(nlist3, int),
            safety_margin)
    
def debug_nlist(positions, cutoff, box):
    """
    Build a neighbor list by computing all pairwise displacements.
    
    Parameters
    ----------
    positions : ndarray of shape (N, 3)
        Array of particle positions.
    cutoff : float
        Distance cutoff for neighbors.
    
    Returns
    -------
    i_vals : ndarray
        Array of first indices for neighbor pairs.
    j_vals : ndarray
        Array of second indices for neighbor pairs.
    distances : ndarray
        Array of distances corresponding to the neighbor pairs.
    """    
    num_particles = positions.shape[0]
    nlist = np.array(compute_distinct_pairs(num_particles))
    
    disp = displacement_fn(positions[nlist[0,:]], positions[nlist[1,:]], box)    
    
    # Compute the pairwise distances (N, N) by taking the norm along the last axis.
    dist = np.array(space.distance(disp))
    
    # Identify all pairs where the distance is below the cutoff.
    mask = (np.where(dist < cutoff))[0]
    
    distances = dist[mask]
    nlist = nlist[:,mask]
    
    # Sort the neighbor list by the first index (i_vals)
    order = np.argsort(nlist[0,:])
    nlist = nlist[:,order]
    nlist = nlist[:,order]
    distances = distances[order]
        
    return nlist[0,:], nlist[1,:], distances

@partial(jit, static_argnums=[0])
def update_neighborlist(num_particles, positions, nl_cutoff, second_cutoff, third_cutoff, unique_pairs, box):
    """
    Update the neighbor list by reactivating previously masked pairs.

    Returns
    -------
    nlist1 : jnp.array
        Neighbor list where pairs with distance >= nl_cutoff are replaced with num_particles.
    nlist2 : jnp.array
        Neighbor list where pairs with distance >= second_cutoff are replaced with num_particles.
    nlist3 : jnp.array
        Neighbor list where pairs with distance >= third_cutoff are replaced with num_particles,
        sorted by the first index and truncated to a fixed maximum number of columns.
    jnp.any(jnp.equal(nlist1, num_particles)) : bool
        Boolean flag to indicate when lists need to be re-allocated.
    
    """
    delta = displacement_fn(positions[unique_pairs[0],:], positions[unique_pairs[1],:], box)
    dist_sq = jnp.sum(delta ** 2, axis=-1)
    nlist1 = jnp.where(dist_sq > nl_cutoff*nl_cutoff, num_particles, unique_pairs)
    nlist2 = jnp.where(dist_sq > second_cutoff*second_cutoff, num_particles, unique_pairs)
    nlist3 = jnp.where(dist_sq > third_cutoff*third_cutoff, num_particles, unique_pairs)
    delta = jnp.argsort(nlist3[0,:])
    nlist3 = nlist3[:,delta]
    
    return nlist1, nlist2, nlist3[:, : num_particles*15], jnp.any(jnp.equal(nlist1, num_particles))


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

def create_hardsphere_configuration(l: float, num_particles: int, seed: int, temperature: float) -> Array:
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
    def precompute_hs(positions, nl, box):
        # Brownian Dynamics calculation quantities
        indices_i = nl[0, :]  # Pair indices (i<j always)
        indices_j = nl[1, :]
        # array of vectors from particle i to j (size = npairs)
        dist = displacement_fn(positions[indices_i,:], positions[indices_j,:], box)
        dist_mod = space.distance(dist)
        # unit vector from particle j to i
        return dist, dist_mod, indices_i, indices_j

    @jit
    def update_pos(positions, net_vel):
        # Define array of displacement r(t+dt)-r(t)
        dr = jnp.zeros((num_particles, 3), float)
        # Compute actual displacement due to velocities
        dr = dr.at[:, 0].set(dt * net_vel.at[(0)::3].get())
        dr = dr.at[:, 1].set(dt * net_vel.at[(1)::3].get())
        dr = dr.at[:, 2].set(dt * net_vel.at[(2)::3].get())
        # Apply displacement and compute wrapped shift
        # shift system origin to (0,0,0)
        positions = shift(positions + jnp.array([l, l, l]) / 2, dr)
        # re-shift origin back to box center
        positions = positions - jnp.array([l, l, l]) * 0.5
        # Compute new relative displacements between particles
        return positions

    @jit
    def add_thermal_noise(net_vel, brow):
        net_vel = net_vel.at[0::3].add(brow[0::6])
        net_vel = net_vel.at[1::3].add(brow[1::6])
        net_vel = net_vel.at[2::3].add(brow[2::6])
        return net_vel
    
    @jit
    def compute_hs_forces(dist: ArrayLike, dist_mod: ArrayLike,
        indices_i: ArrayLike, indices_j: ArrayLike, dt: float
    ) -> Array:

        fp = jnp.zeros((num_particles, 3))
        # particle diameter (shifted of 1% to help numerical stability)
        k = 1/dt * 0.1
        fp_mod = jnp.where(
            indices_i != indices_j, k * (1 - sigma / dist_mod), 0.0
        )
        fp_mod = jnp.where((dist_mod) < sigma, fp_mod, 0.0)
        fp += ops.segment_sum(fp_mod[:,None] * dist, indices_i, num_particles)  # Add contributions from i
        fp -= ops.segment_sum(fp_mod[:,None] * dist, indices_j, num_particles)  # Subtract contributions from j
        return jnp.ravel(fp)

    _, shift = space.periodic_general(
        jnp.array([[l, 0.0, 0.0], [0.0, l, 0.0], [0.0, 0.0, l]]), fractional_coordinates=False
    )

    net_vel = jnp.zeros(3 * num_particles)
    key = jrandom.PRNGKey(seed)
    temperature = 0.0005
    sigma = 2.15  # 2.15 #2.05 #particle diameter
    phi_eff = num_particles / (l * l * l) * (sigma * sigma * sigma) * np.pi / 6
    phi_actual = ((np.pi / 6) * (2 * 2 * 2) * num_particles) / (l * l * l)
    brow_time = sigma * sigma / (4 * temperature)

    dt = brow_time / (1e5)

    num_steps = int(brow_time / dt)
    key, random_coord = generate_random_array(key, (num_particles * 3))
    random_coord = (random_coord - 1 / 2) * l
    positions = jnp.zeros((num_particles, 3))
    positions = positions.at[:, 0].set(random_coord[0::3])
    positions = positions.at[:, 1].set(random_coord[1::3])
    positions = positions.at[:, 2].set(random_coord[2::3])
    unique_pairs, nl, _, _, _ = cpu_nlist(positions, np.array([l,l,l]), min(6,l/2), 0., 0., 0., initial_safety_margin=3.)
    overlaps = find_overlaps(positions, 2.002, num_particles, unique_pairs, np.array([[l,0.,0.],[0.,l,0.],[0.,0.,l]]))
    box = jnp.array([[l, 0., 0.], [0., l, 0.], [0., 0., l]])
    print('Initial overlaps in creating HS conf are', overlaps)
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
        for i_step in tqdm(range(num_steps), mininterval=0.5):
            # Compute Velocity (Brownian + hard-sphere)

            # Compute distance vectors and neighbor list indices
            (dist, dist_mod, indices_i, indices_j) = precompute_hs(positions, nl, box)

            # compute forces for each pair
            net_vel = compute_hs_forces(dist, dist_mod, indices_i, indices_j, dt)

            # Compute and add Brownian velocity
            key, random_noise = generate_random_array(key, (6 * num_particles))
            brow = thermal.compute_bd_randomforce(num_particles, temperature, dt, random_noise)
            net_vel = add_thermal_noise(net_vel, brow)

            # Update positions
            positions = update_pos(positions, net_vel)
            # Update neighborlists
            nl, _, _, nl_list_bound = update_neighborlist(num_particles, positions, 3., 0., 0., unique_pairs, box)
            if not nl_list_bound: # Re-allocate list if number of neighbors exceeded list size.   
                unique_pairs, nl, _, _, _ = cpu_nlist(positions, np.array([l,l,l]),  3., 0., 0., 0., initial_safety_margin=3.)

        overlaps = find_overlaps(positions, 2.002, num_particles, unique_pairs, np.array([[l,0.,0.],[0.,l,0.],[0.,0.,l]]))
        print('Thermalized for 1 Brownian time, found ', overlaps, 'overlaps')
        if (time.time() - start_time) > 1800:  # interrupt if too computationally expensive
            raise ValueError("Creation of initial configuration failed. Abort!")
    print("Initial configuration created. Volume fraction is ", phi_actual)
    return positions


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


@partial(jit, static_argnums=[2])
def find_overlaps(positions: ArrayLike, sigma: float, num_particles: int,
                  nlist: ArrayLike, box: ArrayLike) -> tuple[int, float]:
    """Check overlaps between particles and returns number of overlaps + number of particles.

    The radius of a particle is set to 1.
    Note that this function should be used only for debugging,
    as the memory cost scales quadratically with the number of particles num_particles.

    Parameters
    ----------
    positions: (float)
        Array (num_particles,3) of particle positions
    sigma: (float)
        Particle diameter
    num_particles: (int)
        Particle number
    nlist: (int)
        Array (2, num_pairs) of particle indices in neighbor list
    
    Returns
    -------
    output:
        Number of overlaps

    """
    dr = displacement_fn(positions[nlist[0,:],:], positions[nlist[1,:],:], box)
    dist_sq = jnp.sum(dr*dr,axis=1)    
    sigma_sq = sigma * sigma
    mask = jnp.where(dist_sq < sigma_sq, 1.0, 0.0)  # build mask from distances < cutoff    
    return jnp.sum(mask)


@partial(jit, static_argnums=[5, 15])
def precompute(
    positions: ArrayLike,
    gaussian_grid_spacing: ArrayLike,
    nl_ff: ArrayLike,
    nl_lub: ArrayLike,
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
    box = jnp.array([[box_x, box_y * tilt_factor, box_z * 0.0], [0.0, box_y, box_z * 0.0], [0.0, 0.0, box_z]])
    
    ###################################################################################################################
    # Wave Space calculation quantities
    pos = positions + jnp.array([box_x, box_y, box_z]) / 2 # Compute fractional coordinates
    pos = pos.at[:, 0].add(-tilt_factor * pos.at[:, 1].get()) # Include box deformation
    pos = pos / jnp.array([box_x, box_y, box_z]) * jnp.array([grid_nx, grid_ny, grid_nz]) # Periodic box
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
    r = displacement_fn(positions[indices_i,:], positions[indices_j,:], box)
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
    # Array of vectors from particle i to j (size = npairs)
    r_lub = displacement_fn(positions[indices_i_lub,:], positions[indices_j_lub,:], box)
    dist_lub = space.distance(r_lub)  # distance between particle i and j
    # unit vector from particle j to i
    r_lub_unit = r_lub / dist_lub.at[:, None].get()

    # Indices in resistance table
    ind = jnp.log10((dist_lub - 2.0) / res_table_min) / res_table_dr
    ind = ind.astype(int)
    dist_lub_lower = res_table_dist.at[ind].get()
    dist_lub_upper = res_table_dist.at[ind + 1].get()
    # Linear interpolation of the Table values
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
    h02 = jnp.where(h0>0, h0*h0, 1.)
    h03 = jnp.where(h0>0, h0*h02, 1.)
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
    r = positions[indices_j,:] - positions[indices_i,:]
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
    r_lub = positions[indices_j_lub,:] - positions[indices_i_lub,:]
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


@partial(jit, static_argnums=[4, 14])
def precompute_rpy(
    positions: ArrayLike,
    gaussian_grid_spacing: ArrayLike,
    nl_ff: ArrayLike,
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
    box = jnp.array([[box_x, box_y * tilt_factor, box_z * 0.0], [0.0, box_y, box_z * 0.0], [0.0, 0.0, box_z]])
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
    r = displacement_fn(positions[indices_i,:], positions[indices_j,:], box)
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
) -> tuple[Array, Array, Array]:
    """Compute all the necessary quantities needed to update the particle position at a given timestep.

    Parameters
    ----------
    positions: (float)
        Array (num_particles,3) of current particles positions
    nl_ff: (int)
        Array (2,n_pairs_ff) containing far-field neighborlist indices
    
    Returns
    -------
    r_unit, indices_i, indices_j,mobil_scalar

    """
    indices_i = nl_ff[0, :]  # Pair indices (i<j always)
    indices_j = nl_ff[1, :]
    # array of vectors from particles i to j (size = npairs)
    r = positions[indices_j,:] - positions[indices_i,:]
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


@jit
def precompute_bd(
    positions: ArrayLike,
    nl: ArrayLike,
    box_x: float,
    box_y: float,
    box_z: float,
    tilt_factor: float,
) -> tuple[Array, Array, Array]:
    """Compute all the necessary quantities needed to update the particle position at a given timestep, for Brownian Dynamics.

    Parameters
    ----------
    positions: (float)
        Array (num_particles,3) of current particles positions
    nl: (int)
        Neighborlist indices
    num_particles: (int)
        Number of particles
    box_x: (float)
        Box size in x direction
    box_y: (float)
        Box size in y direction
    box_z: (float)
        Box size in z direction
    tilt_factor: (float)
        Current box tilt factor
        
    Returns
    -------
     r_unit, indices_i, indices_j

    """
    box = jnp.array([[box_x, box_y * tilt_factor, box_z * 0.0], [0.0, box_y, box_z * 0.0], [0.0, 0.0, box_z]])
    # Brownian Dynamics calculation quantities
    indices_i = nl[0, :]  # Pair indices (i<j always)
    indices_j = nl[1, :]
    # array of vectors from particle i to j (size = npairs)
    r = displacement_fn(positions[indices_i,:], positions[indices_j,:], box)
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
