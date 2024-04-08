import numpy as np
import math
import random
from functools import partial
from jax_md import partition
import jax.numpy as jnp
from jax import random as jrandom
from jax import jit
from jax_md import space
np.set_printoptions(precision=8, suppress=True)

# Perform cholesky factorization and obtain lower triangle cholesky factor of input matrix


@jit
def chol_fac(A):
    return jnp.linalg.cholesky(A)

# Check that ewald_cut is small enough to avoid interaction with the particles images (in real space part of calculation)


def Check_ewald_cut(ewald_cut, Lx, Ly, Lz, error):
    if ((ewald_cut > Lx/2) or (ewald_cut > Ly/2) or (ewald_cut > Lz/2)):
        max_cut = max([Lx, Ly, Lz]) / 2.0
        new_xi = np.sqrt(-np.log(error)) / max_cut
        raise ValueError(
            f"Ewald cutoff radius is too large. Try with xi = {new_xi}")
    else:
        print('Ewald Cutoff is ', ewald_cut)
    return

# Maximum eigenvalue of A'*A to scale support, P, for spreading on deformed grids (Fiore and Swan, J. Chem. Phys., 2018)


def Check_max_shear(gridh, xisq, Nx, Ny, Nz, max_strain, error):

    gamma = max_strain
    lambdaa = 1 + gamma*gamma / 2 + gamma * np.sqrt(1+gamma*gamma/4)

    # Parameters for the Spectral Ewald Method (Lindbo and Tornberg, J. Comp. Phys., 2011)
    gaussm = 1.0
    while (math.erfc(gaussm / np.sqrt(2*lambdaa)) > error):
        gaussm += 0.01
    gaussP = int(gaussm*gaussm / jnp.pi) + 1
    w = gaussP*gridh[0] / 2.0  # Gaussian width in simulation units
    eta = (2.0*w/gaussm)*(2.0*w/gaussm) * \
        (xisq)  # Gaussian splitting parameter

    # Check that the support size isn't somehow larger than the grid
    if (gaussP > min(Nx, min(Ny, Nz))):
        raise ValueError(
            f"Quadrature Support Exceeds Available Grid. \n (Mx, My, Mz) = ({Nx}), ({Ny}), ({Nz}). Support Size, P = {gaussP}")
    return eta, gaussP


def Compute_k_gridpoint_number(kmax, Lx, Ly, Lz):
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
                if ((Mcurr >= 8) and (Mcurr <= 512)):
                    Mlist.append(Mcurr)
    Mlist = np.array(Mlist)
    Mlist = np.sort(Mlist)
    nmult = len(Mlist)

    # Compute the number of grid points in each direction (should be a power of 2,3,5 for most efficient FFTs)
    for ii in range(0, nmult):
        if(Nx <= Mlist[ii]):
            Nx = Mlist[ii]
            break
    for ii in range(0, nmult):
        if(Ny <= Mlist[ii]):
            Ny = Mlist[ii]
            break
    for ii in range(0, nmult):
        if(Nz <= Mlist[ii]):
            Nz = Mlist[ii]
            break

    # Maximum number of FFT nodes is limited by available memory = 512 * 512 * 512 = 134217728
    if (Nx*Ny*Nz > 134217728):
        raise ValueError(
            "Requested Number of Fourier Nodes Exceeds Max Dimension of 512^3")
    return Nx, Ny, Nz

# given a support size (for gaussian spread) -> compute distances in a gaussP x gaussP x gaussP grid (where gauss P might be corrected in it is odd or even...)


def Precompute_grid_distancing(gauss_P, gridh, tilt_factor, positions, N, Nx, Ny, Nz, Lx, Ly, Lz):
    # SEE Molbility.cu (line 265) for tips on how to change the distances when we have shear
    grid = np.zeros((gauss_P, gauss_P, gauss_P, N))
    # center_offsets = (jnp.array(positions)+jnp.array([Lx,Ly,Lz])/2)*jnp.array([Nx,Ny,Nz])/jnp.array([Lx,Ly,Lz])
    center_offsets = (np.array(positions)+np.array([Lx, Ly, Lz])/2)
    center_offsets[:, 0] += (-tilt_factor*center_offsets[:, 1])
    # center_offsets = center_offsets.at[:,0].add(-tilt_factor*center_offsets.at[:,1].get())
    center_offsets = center_offsets / \
        np.array([Lx, Ly, Lz]) * np.array([Nx, Ny, Nz])
    # in grid units, not particle radius units
    center_offsets -= np.array(center_offsets, dtype=int)
    center_offsets = np.where(
        center_offsets > 0.5, -(1-center_offsets), center_offsets)
    for i in range(gauss_P):
        for j in range(gauss_P):
            for k in range(gauss_P):
                grid[i, j, k, :] = gridh*gridh * ((i - int(gauss_P/2) - center_offsets[:, 0] + tilt_factor*(j - int(gauss_P/2) - center_offsets[:, 1]))**2
                                                  + (j - int(gauss_P/2) -
                                                     center_offsets[:, 1])**2
                                                  + (k - int(gauss_P/2) - center_offsets[:, 2])**2)
    return grid


def CreateRandomConfiguration(L, N, seed):

    def distance_periodic(p1, p2, L):
        d = np.abs(p1 - p2)
        d = np.where(d > L / 2, L - d, d)
        return (np.sum(d*d))

    positions = np.zeros((N, 3), float)
    max_attempts = 100000
    attempts = 0
    n = 0
    random.seed(seed)

    while n < N:
        attempts += 1
        x = random.uniform(-L / 2, L / 2)
        y = random.uniform(-L / 2, L / 2)
        z = random.uniform(-L / 2, L / 2)
        overlap = False

        for i in range(n):
            d = distance_periodic(positions[i, :], np.array([x, y, z]), L)
            if d < 4.5:  # The distance should be compared with the sum of the radii
                overlap = True
                break

        if not overlap:
            positions[n, :] = np.array([x, y, z])
            n += 1
            # print(n)

        if attempts > max_attempts:
            ValueError("Computation too long, abort.")
            break

    return jnp.array(positions)


# Set various Neighborlist
def initialize_neighborlist(U_cut, Lx, Ly, Lz, displacement, ewald_cut):

    # For Lubrication Hydrodynamic Forces Calculation
    lub_neighbor_fn = partition.neighbor_list(displacement,  # Displacement metric
                                              # Box size
                                              jnp.array([Lx, Ly, Lz]),
                                              # r_cutoff=0.,  # Spatial cutoff for 2 particles to be neighbor
                                              r_cutoff=4.,  # Spatial cutoff for 2 particles to be neighbor
                                              dr_threshold=0.1,  # displacement of particles threshold to recompute neighbor list
                                              capacity_multiplier=1,
                                              format=partition.NeighborListFormat.OrderedSparse)
    # For Precondition of Lubrication Hydrodynamic Forces Calculation
    prec_lub_neighbor_fn = partition.neighbor_list(displacement,  # Displacement metric
                                                   # Box size
                                                   jnp.array([Lx, Ly, Lz]),
                                                   r_cutoff=2.1,  # Spatial cutoff for 2 particles to be neighbor
                                                   # r_cutoff=0.,  # Spatial cutoff for 2 particles to be neighbor
                                                   dr_threshold=0.1,  # displacement of particles threshold to recompute neighbor list
                                                   capacity_multiplier=1,
                                                   format=partition.NeighborListFormat.OrderedSparse)

    # For Far-Field Real Space Hydrodynamic Forces Calculation
    ff_neighbor_fn = partition.neighbor_list(displacement,  # Displacement metric
                                             # Box size
                                             jnp.array([Lx, Ly, Lz]),
                                             r_cutoff=ewald_cut,  # Spatial cutoff for 2 particles to be neighbor
                                             dr_threshold=0.1,  # displacement of particles threshold to recompute neighbor list
                                             capacity_multiplier=1,
                                             format=partition.NeighborListFormat.OrderedSparse)
    # For Interparticle Potential Forces Calculation
    pot_neighbor_fn = partition.neighbor_list(displacement,  # Displacement metric
                                              # Box size
                                              jnp.array([Lx, Ly, Lz]),
                                              r_cutoff=U_cut,  # Spatial cutoff for 2 particles to be neighbor
                                              dr_threshold=0.1,  # displacement of particles threshold to recompute neighbor list
                                              capacity_multiplier=1,
                                              format=partition.NeighborListFormat.OrderedSparse)
    return lub_neighbor_fn, prec_lub_neighbor_fn, ff_neighbor_fn, pot_neighbor_fn


# Check overlaps between particles and returns (number of overlaps + number of particles) (radius of a particle is set to 1)
@jit
def check_overlap(dist):
    dist_sq = (dist[:, :, 0]*dist[:, :, 0]+dist[:, :, 1]
               * dist[:, :, 1]+dist[:, :, 2]*dist[:, :, 2])
    exitcode = jnp.where(dist_sq < 3.996, 1., 0.)
    exitcode2 = jnp.where(dist_sq < 3.996, dist_sq, 0.)
    exitcode2 = jnp.where(exitcode2 > 0, exitcode2, 0.)

    return exitcode, jnp.sqrt(exitcode2)  # return 0 if


@partial(jit, static_argnums=[1])
def generate_random_array(key, size):
    # advance RNG state (otherwise will get same random numbers)
    key, subkey = jrandom.split(key)
    return subkey,  (jrandom.uniform(subkey, (size,)))


@partial(jit, static_argnums=[6, 16])
def precompute(positions, gaussian_grid_spacing, nl_ff, nl_lub, displacements_vector_matrix, tilt_factor,
               N, Lx, Ly, Lz, Nx, Ny, Nz,
               prefac, expfac, quadW,
               gaussP, gaussPd2,
               ewald_n, ewald_dr, ewald_cut, ewaldC1,
               ResTable_min, ResTable_dr, ResTable_dist, ResTable_vals):

    ###Wave Space calculation quantities

    #Compute fractional coordinates
    pos = positions + jnp.array([Lx, Ly, Lz])/2
    pos = pos.at[:, 0].add(-tilt_factor*pos.at[:, 1].get())
    pos = pos / jnp.array([Lx, Ly, Lz]) * jnp.array([Nx, Ny, Nz])
    ###convert positions in the box in indices in the grid
    # pos = (positions+np.array([Lx, Ly, Lz])/2)/np.array([Lx, Ly, Lz]) * np.array([Nx, Ny, Nz])
    intpos = (jnp.array(pos, int))  # integer positions
    intpos = jnp.where(pos-intpos > 0.5, intpos+1, intpos)

    # actual values to put in the grid
    gaussian_grid_spacing1 = prefac * jnp.exp(-expfac * gaussian_grid_spacing)
    gaussian_grid_spacing2 = jnp.swapaxes(jnp.swapaxes(
        jnp.swapaxes(gaussian_grid_spacing1*quadW, 3, 2), 2, 1), 1, 0)

    #Starting indices on the grid from particle positions
    start_index_x = (intpos.at[:, 0].get()-(gaussPd2)) % Nx
    start_index_y = (intpos.at[:, 1].get()-(gaussPd2)) % Ny
    start_index_z = (intpos.at[:, 2].get()-(gaussPd2)) % Nz
    #All indices on the grids from particle positions, ordered such that x is the 'slowest' index to change and z the 'fastest'
    all_indices_x = jnp.repeat(jnp.repeat(jnp.repeat(start_index_x, gaussP), gaussP), gaussP) + jnp.resize(
        jnp.repeat(jnp.repeat(jnp.arange(gaussP), gaussP), gaussP), gaussP*gaussP*gaussP*N)
    all_indices_y = jnp.repeat(jnp.repeat(jnp.repeat(start_index_y, gaussP), gaussP), gaussP) + jnp.resize(
        jnp.resize(jnp.repeat(jnp.arange(gaussP), gaussP), gaussP*gaussP*gaussP), gaussP*gaussP*gaussP*N)
    all_indices_z = jnp.repeat(jnp.repeat(jnp.repeat(start_index_z, gaussP), gaussP), gaussP) + jnp.resize(
        jnp.resize(jnp.resize(jnp.arange(gaussP), gaussP*gaussP), gaussP*gaussP*gaussP), gaussP*gaussP*gaussP*N)
    all_indices_x = all_indices_x % Nx
    all_indices_y = all_indices_y % Ny
    all_indices_z = all_indices_z % Nz

    ###################################################################################################################
    #Real Space (far-field) calculation quantities
    indices_i = nl_ff[0, :]  # Pair indices (i<j always)
    indices_j = nl_ff[1, :]
    # array of vectors from particles i to j (size = npairs)
    R = displacements_vector_matrix.at[indices_i, indices_j].get()
    dist = space.distance(R)  # distances between particles i and j
    r = -R / dist.at[:, None].get()  # unit vector from particle j to i

    # Interpolate scalar mobility functions values from ewald table
    r_ind = (ewald_n * (dist - ewald_dr) /
             (ewald_cut - ewald_dr))  # index in ewald table
    r_ind = r_ind.astype(int)  # truncate decimal part
    offset1 = 2 * r_ind  # even indices
    offset2 = 2 * r_ind + 1  # odd indices

    tewaldC1m = ewaldC1.at[offset1].get()  # UF and UC
    tewaldC1p = ewaldC1.at[offset1+2].get()
    tewaldC2m = ewaldC1.at[offset2].get()  # DC
    tewaldC2p = ewaldC1.at[offset2+2].get()

    fac_ff = dist / ewald_dr - r_ind - 1.0  # interpolation factor

    f1 = tewaldC1m.at[:, 0].get() + (tewaldC1p.at[:, 0].get() -
                                     tewaldC1m.at[:, 0].get()) * fac_ff
    f2 = tewaldC1m.at[:, 1].get() + (tewaldC1p.at[:, 1].get() -
                                     tewaldC1m.at[:, 1].get()) * fac_ff

    g1 = tewaldC1m.at[:, 2].get() + (tewaldC1p.at[:, 2].get() -
                                     tewaldC1m.at[:, 2].get()) * fac_ff
    g2 = tewaldC1m.at[:, 3].get() + (tewaldC1p.at[:, 3].get() -
                                     tewaldC1m.at[:, 3].get()) * fac_ff

    h1 = tewaldC2m.at[:, 0].get() + (tewaldC2p.at[:, 0].get() -
                                     tewaldC2m.at[:, 0].get()) * fac_ff
    h2 = tewaldC2m.at[:, 1].get() + (tewaldC2p.at[:, 1].get() -
                                     tewaldC2m.at[:, 1].get()) * fac_ff
    h3 = tewaldC2m.at[:, 2].get() + (tewaldC2p.at[:, 2].get() -
                                     tewaldC2m.at[:, 2].get()) * fac_ff

    ###################################################################################################################
    #Lubrication calculation quantities
    indices_i_lub = nl_lub[0, :]  # Pair indices (i<j always)
    indices_j_lub = nl_lub[1, :]
    # array of vectors from particle i to j (size = npairs)
    R_lub = displacements_vector_matrix.at[nl_lub[0, :], nl_lub[1, :]].get()
    dist_lub = space.distance(R_lub)  # distance between particle i and j
    # unit vector from particle j to i
    r_lub = R_lub / dist_lub.at[:, None].get()

    # # Indices in resistance table
    ind = (jnp.log10((dist_lub - 2.) / ResTable_min) / ResTable_dr)
    ind = ind.astype(int)
    dist_lub_lower = ResTable_dist.at[ind].get()
    dist_lub_upper = ResTable_dist.at[ind+1].get()
    # # Linear interpolation of the Table values
    fac_lub = jnp.where(dist_lub_upper - dist_lub_lower > 0.,
                        (dist_lub - dist_lub_lower) / (dist_lub_upper - dist_lub_lower), 0.)

    XA11 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+0] + (
        ResTable_vals[22*(ind+1)+0]-ResTable_vals[22*(ind)+0]) * fac_lub), ResTable_vals[0])

    XA12 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+1] + (
        ResTable_vals[22*(ind+1)+1]-ResTable_vals[22*(ind)+1]) * fac_lub), ResTable_vals[1])

    YA11 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+2] + (
        ResTable_vals[22*(ind+1)+2]-ResTable_vals[22*(ind)+2]) * fac_lub), ResTable_vals[2])

    YA12 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+3] + (
        ResTable_vals[22*(ind+1)+3]-ResTable_vals[22*(ind)+3]) * fac_lub), ResTable_vals[3])

    YB11 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+4] + (
        ResTable_vals[22*(ind+1)+4]-ResTable_vals[22*(ind)+4]) * fac_lub), ResTable_vals[4])

    YB12 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+5] + (
        ResTable_vals[22*(ind+1)+5]-ResTable_vals[22*(ind)+5]) * fac_lub), ResTable_vals[5])

    XC11 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+6] + (
        ResTable_vals[22*(ind+1)+6]-ResTable_vals[22*(ind)+6]) * fac_lub), ResTable_vals[6])

    XC12 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+7] + (
        ResTable_vals[22*(ind+1)+7]-ResTable_vals[22*(ind)+7]) * fac_lub), ResTable_vals[7])

    YC11 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+8] + (
        ResTable_vals[22*(ind+1)+8]-ResTable_vals[22*(ind)+8]) * fac_lub), ResTable_vals[8])

    YC12 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+9] + (
        ResTable_vals[22*(ind+1)+9]-ResTable_vals[22*(ind)+9]) * fac_lub), ResTable_vals[9])

    YB21 = -YB12  # symmetry condition

    XG11 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+10] + (
        ResTable_vals[22*(ind+1)+10]-ResTable_vals[22*(ind)+10]) * fac_lub), ResTable_vals[10])

    XG12 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+11] + (
        ResTable_vals[22*(ind+1)+11]-ResTable_vals[22*(ind)+11]) * fac_lub), ResTable_vals[11])

    YG11 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+12] + (
        ResTable_vals[22*(ind+1)+12]-ResTable_vals[22*(ind)+12]) * fac_lub), ResTable_vals[12])

    YG12 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+13] + (
        ResTable_vals[22*(ind+1)+13]-ResTable_vals[22*(ind)+13]) * fac_lub), ResTable_vals[13])

    YH11 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+14] + (
        ResTable_vals[22*(ind+1)+14]-ResTable_vals[22*(ind)+14]) * fac_lub), ResTable_vals[14])

    YH12 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+15] + (
        ResTable_vals[22*(ind+1)+15]-ResTable_vals[22*(ind)+15]) * fac_lub), ResTable_vals[15])

    XM11 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+16] + (
        ResTable_vals[22*(ind+1)+16]-ResTable_vals[22*(ind)+16]) * fac_lub), ResTable_vals[16])

    XM12 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+17] + (
        ResTable_vals[22*(ind+1)+17]-ResTable_vals[22*(ind)+17]) * fac_lub), ResTable_vals[17])

    YM11 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+18] + (
        ResTable_vals[22*(ind+1)+18]-ResTable_vals[22*(ind)+18]) * fac_lub), ResTable_vals[18])

    YM12 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+19] + (
        ResTable_vals[22*(ind+1)+19]-ResTable_vals[22*(ind)+19]) * fac_lub), ResTable_vals[19])

    ZM11 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+20] + (
        ResTable_vals[22*(ind+1)+20]-ResTable_vals[22*(ind)+20]) * fac_lub), ResTable_vals[20])

    ZM12 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+21] + (
        ResTable_vals[22*(ind+1)+21]-ResTable_vals[22*(ind)+21]) * fac_lub), ResTable_vals[21])

    ResFunc = jnp.array([XA11, XA12, YA11, YA12, YB11, YB12, XC11, XC12, YC11, YC12, YB21,
                         XG11, XG12, YG11, YG12, YH11, YH12, XM11, XM12, YM11, YM12, ZM11, ZM12])

    return ((all_indices_x), (all_indices_y), (all_indices_z), gaussian_grid_spacing1, gaussian_grid_spacing2,
            r, indices_i, indices_j, f1, f2, g1, g2, h1, h2, h3,
            r_lub, indices_i_lub, indices_j_lub,ResFunc)