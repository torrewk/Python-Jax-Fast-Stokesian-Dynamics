from jax.config import config
import jaxmd_space as space
import jax.numpy as jnp
from jax import jit
from functools import partial
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
config.update("jax_enable_x64", False)


@partial(jit, static_argnums=[2, 3])
def RFU_Precondition(
        ichol_relaxer,
        R,
        N,
        n_pairs_lub_prec,
        nl_lub_prec,
):

    # Load resistance table
    ResTable_dist = jnp.load('files/ResTableDist.npy')
    ResTable_vals = jnp.load('files/ResTableVals.npy')
    # Smallest surface-to-surface distance (lower cut-off)
    ResTable_min = 0.0001
    # Table discretization (log space), i.e. ind * dr.set(log10( h[ind] / min )
    ResTable_dr = 0.004305

    #Define empty matrix
    R_fu_precondition = jnp.zeros((6*N, 6*N), float)

    # By definition, R points from particle 1 to particle 2 (i to j), otherwise the handedness and symmetry of the lubrication functions is lost
    dist = space.distance(R)  # distance between particle i and j
    r = R / dist[:, None]  # unit vector from particle j to i

    # # Indices in resistance table
    ind = (jnp.log10((dist - 2.0) / ResTable_min) / ResTable_dr)
    ind = ind.astype(int)

    dist_lower = ResTable_dist[ind]
    dist_upper = ResTable_dist[ind+1]

    # # Linear interpolation of the Table values
    fac = jnp.where(dist_upper - dist_lower > 0.,
                    (dist - dist_lower) / (dist_upper - dist_lower), 0.)

    XA11 = jnp.where(dist >= 2+ResTable_min, (ResTable_vals[22*(ind)+0] + (
        ResTable_vals[22*(ind+1)+0]-ResTable_vals[22*(ind)+0]) * fac), ResTable_vals[0])

    XA12 = jnp.where(dist >= 2+ResTable_min, (ResTable_vals[22*(ind)+1] + (
        ResTable_vals[22*(ind+1)+1]-ResTable_vals[22*(ind)+1]) * fac), ResTable_vals[1])

    YA11 = jnp.where(dist >= 2+ResTable_min, (ResTable_vals[22*(ind)+2] + (
        ResTable_vals[22*(ind+1)+2]-ResTable_vals[22*(ind)+2]) * fac), ResTable_vals[2])

    YA12 = jnp.where(dist >= 2+ResTable_min, (ResTable_vals[22*(ind)+3] + (
        ResTable_vals[22*(ind+1)+3]-ResTable_vals[22*(ind)+3]) * fac), ResTable_vals[3])

    YB11 = jnp.where(dist >= 2+ResTable_min, (ResTable_vals[22*(ind)+4] + (
        ResTable_vals[22*(ind+1)+4]-ResTable_vals[22*(ind)+4]) * fac), ResTable_vals[4])

    YB12 = jnp.where(dist >= 2+ResTable_min, (ResTable_vals[22*(ind)+5] + (
        ResTable_vals[22*(ind+1)+5]-ResTable_vals[22*(ind)+5]) * fac), ResTable_vals[5])

    XC11 = jnp.where(dist >= 2+ResTable_min, (ResTable_vals[22*(ind)+6] + (
        ResTable_vals[22*(ind+1)+6]-ResTable_vals[22*(ind)+6]) * fac), ResTable_vals[6])

    XC12 = jnp.where(dist >= 2+ResTable_min, (ResTable_vals[22*(ind)+7] + (
        ResTable_vals[22*(ind+1)+7]-ResTable_vals[22*(ind)+7]) * fac), ResTable_vals[7])

    YC11 = jnp.where(dist >= 2+ResTable_min, (ResTable_vals[22*(ind)+8] + (
        ResTable_vals[22*(ind+1)+8]-ResTable_vals[22*(ind)+8]) * fac), ResTable_vals[8])

    YC12 = jnp.where(dist >= 2+ResTable_min, (ResTable_vals[22*(ind)+9] + (
        ResTable_vals[22*(ind+1)+9]-ResTable_vals[22*(ind)+9]) * fac), ResTable_vals[9])

    epsr = jnp.array(
        [
            [jnp.zeros(n_pairs_lub_prec), r[:, 2], -r[:, 1]],
            [-r[:, 2], jnp.zeros(n_pairs_lub_prec), r[:, 0]],
            [r[:, 1], -r[:, 0], jnp.zeros(n_pairs_lub_prec)]
        ])
    Imrr = jnp.array(
        [[jnp.ones(n_pairs_lub_prec), jnp.zeros(n_pairs_lub_prec), jnp.zeros(n_pairs_lub_prec)],
         [jnp.zeros(n_pairs_lub_prec), jnp.ones(
             n_pairs_lub_prec), jnp.zeros(n_pairs_lub_prec)],
         [jnp.zeros(n_pairs_lub_prec), jnp.zeros(n_pairs_lub_prec), jnp.ones(n_pairs_lub_prec)]])

    rr = jnp.array([
        [r[:, 0]*r[:, 0], r[:, 0]*r[:, 1], r[:, 0]*r[:, 2]],
        [r[:, 1]*r[:, 0], r[:, 1]*r[:, 1], r[:, 1]*r[:, 2]],
        [r[:, 2]*r[:, 0], r[:, 2]*r[:, 1], r[:, 2]*r[:, 2]]])
    Imrr = Imrr - rr

    A_neigh = XA12 * (rr) + YA12 * (Imrr)
    A_self = XA11 * (rr) + YA11 * (Imrr)
    B_neigh = YB12 * (epsr)
    B_self = YB11 * (epsr)
    C_neigh = XC12 * (rr) + YC12 * (Imrr)
    C_self = XC11 * (rr) + YC11 * (Imrr)

    # Fill in matrix (pair contributions)
    # # this is for all the A12 blocks
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :], 6*nl_lub_prec[1, :]].add(A_neigh[0, 0, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :], 6*nl_lub_prec[1, :]+1].add(A_neigh[0, 1, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :], 6*nl_lub_prec[1, :]+2].add(A_neigh[0, 2, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :]+1, 6*nl_lub_prec[1, :]].add(A_neigh[1, 0, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :]+1, 6*nl_lub_prec[1, :]+1].add(A_neigh[1, 1, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :]+1, 6*nl_lub_prec[1, :]+2].add(A_neigh[1, 2, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :]+2, 6*nl_lub_prec[1, :]].add(A_neigh[2, 0, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :]+2, 6*nl_lub_prec[1, :]+1].add(A_neigh[2, 1, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :]+2, 6*nl_lub_prec[1, :]+2].add(A_neigh[2, 2, :])

    # # this is for all the C12 blocks
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :]+3, 6*nl_lub_prec[1, :]+3].add(C_neigh[0, 0, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :]+3, 6*nl_lub_prec[1, :]+4].add(C_neigh[0, 1, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :]+3, 6*nl_lub_prec[1, :]+5].add(C_neigh[0, 2, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :]+4, 6*nl_lub_prec[1, :]+3].add(C_neigh[1, 0, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :]+4, 6*nl_lub_prec[1, :]+4].add(C_neigh[1, 1, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :]+4, 6*nl_lub_prec[1, :]+5].add(C_neigh[1, 2, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :]+5, 6*nl_lub_prec[1, :]+3].add(C_neigh[2, 0, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :]+5, 6*nl_lub_prec[1, :]+4].add(C_neigh[2, 1, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :]+5, 6*nl_lub_prec[1, :]+5].add(C_neigh[2, 2, :])

    # # this is for all the Bt12 blocks (Bt12 = B12)
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :], 6*nl_lub_prec[1, :]+3].add(B_neigh[0, 0, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :], 6*nl_lub_prec[1, :]+4].add(B_neigh[0, 1, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :], 6*nl_lub_prec[1, :]+5].add(B_neigh[0, 2, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :]+1, 6*nl_lub_prec[1, :]+3].add(B_neigh[1, 0, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :]+1, 6*nl_lub_prec[1, :]+4].add(B_neigh[1, 1, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :]+1, 6*nl_lub_prec[1, :]+5].add(B_neigh[1, 2, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :]+2, 6*nl_lub_prec[1, :]+3].add(B_neigh[2, 0, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :]+2, 6*nl_lub_prec[1, :]+4].add(B_neigh[2, 1, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :]+2, 6*nl_lub_prec[1, :]+5].add(B_neigh[2, 2, :])

    # # this is for all the B12 blocks (Bt12 = B12)
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :]+3, 6*nl_lub_prec[1, :]].add(B_neigh[0, 0, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :]+3, 6*nl_lub_prec[1, :]+1].add(B_neigh[0, 1, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :]+3, 6*nl_lub_prec[1, :]+2].add(B_neigh[0, 2, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :]+4, 6*nl_lub_prec[1, :]].add(B_neigh[1, 0, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :]+4, 6*nl_lub_prec[1, :]+1].add(B_neigh[1, 1, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :]+4, 6*nl_lub_prec[1, :]+2].add(B_neigh[1, 2, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :]+5, 6*nl_lub_prec[1, :]].add(B_neigh[2, 0, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :]+5, 6*nl_lub_prec[1, :]+1].add(B_neigh[2, 1, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :]+5, 6*nl_lub_prec[1, :]+2].add(B_neigh[2, 2, :])

    # Fill in matrix (self contributions) (these are a sum of contributions from each pairs:
    # self contribution of particle 'i' will be a sum over all neighboring particles)

    #A11 Block
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :], 6*nl_lub_prec[0, :]].add(A_self[0, 0, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :], 6*nl_lub_prec[0, :]+1].add(A_self[0, 1, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :], 6*nl_lub_prec[0, :]+2].add(A_self[0, 2, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :]+1, 6*nl_lub_prec[0, :]+1].add(A_self[1, 1, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :]+1, 6*nl_lub_prec[0, :]+2].add(A_self[1, 2, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :]+2, 6*nl_lub_prec[0, :]+2].add(A_self[2, 2, :])

    #A22 Block
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,
                                                           :], 6*nl_lub_prec[1, :]].add(A_self[0, 0, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,
                                                           :], 6*nl_lub_prec[1, :]+1].add(A_self[0, 1, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,
                                                           :], 6*nl_lub_prec[1, :]+2].add(A_self[0, 2, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,
                                                           :]+1, 6*nl_lub_prec[1, :]+1].add(A_self[1, 1, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,
                                                           :]+1, 6*nl_lub_prec[1, :]+2].add(A_self[1, 2, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,
                                                           :]+2, 6*nl_lub_prec[1, :]+2].add(A_self[2, 2, :])

    #C11 Block
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :]+3, 6*nl_lub_prec[0, :]+3].add(C_self[0, 0, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :]+3, 6*nl_lub_prec[0, :]+4].add(C_self[0, 1, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :]+3, 6*nl_lub_prec[0, :]+5].add(C_self[0, 2, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :]+4, 6*nl_lub_prec[0, :]+4].add(C_self[1, 1, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :]+4, 6*nl_lub_prec[0, :]+5].add(C_self[1, 2, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :]+5, 6*nl_lub_prec[0, :]+5].add(C_self[2, 2, :])

    #C22 Block
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,
                                                           :]+3, 6*nl_lub_prec[1, :]+3].add(C_self[0, 0, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,
                                                           :]+3, 6*nl_lub_prec[1, :]+4].add(C_self[0, 1, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,
                                                           :]+3, 6*nl_lub_prec[1, :]+5].add(C_self[0, 2, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,
                                                           :]+4, 6*nl_lub_prec[1, :]+4].add(C_self[1, 1, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,
                                                           :]+4, 6*nl_lub_prec[1, :]+5].add(C_self[1, 2, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,
                                                           :]+5, 6*nl_lub_prec[1, :]+5].add(C_self[2, 2, :])

    #Bt11 Block
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :], 6*nl_lub_prec[0, :]+3].add(-B_self[0, 0, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :], 6*nl_lub_prec[0, :]+4].add(-B_self[0, 1, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :], 6*nl_lub_prec[0, :]+5].add(-B_self[0, 2, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :]+1, 6*nl_lub_prec[0, :]+3].add(-B_self[1, 0, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :]+1, 6*nl_lub_prec[0, :]+4].add(-B_self[1, 1, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :]+1, 6*nl_lub_prec[0, :]+5].add(-B_self[1, 2, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :]+2, 6*nl_lub_prec[0, :]+3].add(-B_self[2, 0, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :]+2, 6*nl_lub_prec[0, :]+4].add(-B_self[2, 1, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,
                                                           :]+2, 6*nl_lub_prec[0, :]+5].add(-B_self[2, 2, :])

    #Bt22 Block
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,
                                                           :], 6*nl_lub_prec[1, :]+3].add(B_self[0, 0, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,
                                                           :], 6*nl_lub_prec[1, :]+4].add(B_self[0, 1, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,
                                                           :], 6*nl_lub_prec[1, :]+5].add(B_self[0, 2, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,
                                                           :]+1, 6*nl_lub_prec[1, :]+3].add(B_self[1, 0, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,
                                                           :]+1, 6*nl_lub_prec[1, :]+4].add(B_self[1, 1, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,
                                                           :]+1, 6*nl_lub_prec[1, :]+5].add(B_self[1, 2, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,
                                                           :]+2, 6*nl_lub_prec[1, :]+3].add(B_self[2, 0, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,
                                                           :]+2, 6*nl_lub_prec[1, :]+4].add(B_self[2, 1, :])
    R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,
                                                           :]+2, 6*nl_lub_prec[1, :]+5].add(B_self[2, 2, :])

    # Symmetrize R_fu_nf
    diagonal_elements = R_fu_precondition.diagonal()  # extract the diagonal
    R_fu_precondition += jnp.transpose(R_fu_precondition -
                                       jnp.diag(diagonal_elements))
    # now we have a symmetrix matrix

    # Compress diagonal values (needed later for brownian calculations, to perform Lanczos decomposition and square root)
    diagonal_elements_for_brownian = jnp.where((diagonal_elements >= 1) | (
        diagonal_elements == 0), 1, jnp.sqrt(1/diagonal_elements))  # compress diagonal elements

    # Add identity for far field contribution and scale it properly
    # Because all values are made dimensionless on 6*pi*eta*a, the diagonal elements for FU (forces - velocities) are 1, but those for LW are 4/3(torques - angular velocities)
    diagonal_elements = jnp.where((jnp.arange(6*N)-6*(jnp.repeat(jnp.arange(N), 6))) <
                                  3, ichol_relaxer, ichol_relaxer*1.33333333333)

    # sum the diagonal to the rest of the precondition matrix (add far-field precondition...)
    R_fu_precondition += jnp.diag(diagonal_elements)

    return R_fu_precondition, diagonal_elements_for_brownian


@partial(jit, static_argnums=[5])
def ComputeLubricationFU(velocities,
                         indices_i_lub,
                         indices_j_lub,
                         ResFunctions,
                         r_lub,
                         N):

    XA11 = ResFunctions[0]
    YA11 = ResFunctions[2]
    XA12 = ResFunctions[1]
    YA12 = ResFunctions[3]
    YB11 = ResFunctions[4]
    YB12 = ResFunctions[5]
    XC11 = ResFunctions[6]
    YC11 = ResFunctions[8]
    XC12 = ResFunctions[7]
    YC12 = ResFunctions[9]
    YB21 = ResFunctions[10]

    vel_i = (jnp.reshape(velocities, (N, 6))).at[indices_i_lub].get()
    vel_j = (jnp.reshape(velocities, (N, 6))).at[indices_j_lub].get()

    # Dot product of r and U, i.e. axisymmetric projection (minus sign of rj is taken into account at the end of calculation)
    rdui = r_lub.at[:, 0].get()*vel_i.at[:, 0].get()+r_lub.at[:, 1].get() * \
        vel_i.at[:, 1].get()+r_lub.at[:, 2].get()*vel_i.at[:, 2].get()
    rduj = r_lub.at[:, 0].get()*vel_j.at[:, 0].get()+r_lub.at[:, 1].get() * \
        vel_j.at[:, 1].get()+r_lub.at[:, 2].get()*vel_j.at[:, 2].get()
    rdwi = r_lub.at[:, 0].get()*vel_i.at[:, 3].get()+r_lub.at[:, 1].get() * \
        vel_i.at[:, 4].get()+r_lub.at[:, 2].get()*vel_i.at[:, 5].get()
    rdwj = r_lub.at[:, 0].get()*vel_j.at[:, 3].get()+r_lub.at[:, 1].get() * \
        vel_j.at[:, 4].get()+r_lub.at[:, 2].get()*vel_j.at[:, 5].get()

    # Cross product of U and r, i.e. eps_ijk*r_k*U_j = Px dot U, (eps_ijk is the Levi-Civita symbol)
    epsrdui = jnp.array([r_lub.at[:, 2].get() * vel_i.at[:, 1].get() - r_lub.at[:, 1].get() * vel_i.at[:, 2].get(),
                        -r_lub.at[:, 2].get() * vel_i.at[:, 0].get() +
                         r_lub.at[:, 0].get() * vel_i.at[:, 2].get(),
                        r_lub.at[:, 1].get() * vel_i.at[:, 0].get() - r_lub.at[:, 0].get() * vel_i.at[:, 1].get()])

    epsrdwi = jnp.array([r_lub.at[:, 2].get() * vel_i.at[:, 4].get() - r_lub.at[:, 1].get() * vel_i.at[:, 5].get(),
                        -r_lub.at[:, 2].get() * vel_i.at[:, 3].get() +
                         r_lub.at[:, 0].get() * vel_i.at[:, 5].get(),
                        r_lub.at[:, 1].get() * vel_i.at[:, 3].get() - r_lub.at[:, 0].get() * vel_i.at[:, 4].get()])

    epsrduj = jnp.array([r_lub.at[:, 2].get() * vel_j.at[:, 1].get() - r_lub.at[:, 1].get() * vel_j.at[:, 2].get(),
                        -r_lub.at[:, 2].get() * vel_j.at[:, 0].get() +
                         r_lub.at[:, 0].get() * vel_j.at[:, 2].get(),
                        r_lub.at[:, 1].get() * vel_j.at[:, 0].get() - r_lub.at[:, 0].get() * vel_j.at[:, 1].get()])

    epsrdwj = jnp.array([r_lub.at[:, 2].get() * vel_j.at[:, 4].get() - r_lub.at[:, 1].get() * vel_j.at[:, 5].get(),
                        -r_lub.at[:, 2].get() * vel_j.at[:, 3].get() +
                         r_lub.at[:, 0].get() * vel_j.at[:, 5].get(),
                        r_lub.at[:, 1].get() * vel_j.at[:, 3].get() - r_lub.at[:, 0].get() * vel_j.at[:, 4].get()])

    forces = jnp.zeros((N, 6), float)

    # Compute the contributions to the force for particles i (Fi = A11*Ui + A12*Uj + BT11*Wi + BT12*Wj)
    f = ((XA11 - YA11).at[:, None].get() * rdui.at[:, None].get() * r_lub + YA11.at[:, None].get() * vel_i.at[:, :3].get()
         + (XA12 - YA12).at[:, None].get() * rduj.at[:, None].get() * r_lub + YA12.at[:, None].get() * vel_j.at[:, :3].get() + YB11.at[:, None].get() * (-epsrdwi.T) + YB21.at[:, None].get() * (-epsrdwj.T))
    forces = forces.at[indices_i_lub, :3].add(f)
    # Compute the contributions to the force for particles j (Fj = A11*Uj + A12*Ui + BT11*Wj + BT12*Wi)
    f = ((XA11 - YA11).at[:, None].get() * rduj.at[:, None].get() * r_lub + YA11.at[:, None].get() * vel_j.at[:, :3].get()
         + (XA12 - YA12).at[:, None].get() * rdui.at[:, None].get() * r_lub + YA12.at[:, None].get() * vel_i.at[:, :3].get() + YB11.at[:, None].get() * (epsrdwj.T) + YB21.at[:, None].get() * (epsrdwi.T))
    forces = forces.at[indices_j_lub, :3].add(f)
    # Compute the contributions to the torque for particles i (Li = B11*Ui + B12*Uj + C11*Wi + C12*Wj)
    l = (YB11.at[:, None].get() * epsrdui.T + YB12.at[:, None].get() * epsrduj.T
         + (XC11 - YC11).at[:, None].get() * rdwi.at[:, None].get() *
         r_lub + YC11.at[:, None].get() * vel_i.at[:, 3:].get()
         + (XC12 - YC12).at[:, None].get() * rdwj.at[:, None].get() * r_lub + YC12.at[:, None].get() * vel_j.at[:, 3:].get())
    forces = forces.at[indices_i_lub, 3:].add(l)
    # Compute the contributions to the torque for particles j (Lj = B11*Uj + B12*Ui + C11*Wj + C12*Wi)
    l = (-YB11.at[:, None].get() * epsrduj.T - YB12.at[:, None].get() * epsrdui.T
         + (XC11 - YC11).at[:, None].get() * rdwj.at[:, None].get() *
         r_lub + YC11.at[:, None].get() * vel_j.at[:, 3:].get()
         + (XC12 - YC12).at[:, None].get() * rdwi.at[:, None].get() * r_lub + YC12.at[:, None].get() * vel_i.at[:, 3:].get())
    forces = forces.at[indices_j_lub, 3:].add(l)

    return jnp.ravel(forces)


@partial(jit, static_argnums=[0])
def compute_RFE(N, shear_rate, r_lub, indices_i_lub, indices_j_lub, XG11, XG12, YG11, YG12, YH11, YH12, XG21, YG21, YH21):
    #These simulations are constructed so that, if there is strain,
	# x is the flow direction
	# y is the gradient direction
	# z is the vorticity direction
    # therefore,
	# Einf = [ 0 g/2 0 ]
	#	     [ g/2 0 0 ]
    #		 [ 0 0 0 ]

    #symmetry conditions
    # XG21 = -XG12
    # YG21 = -YG12
    # YH21 = YH12

    #define single particle strain (each particle experience same shear)
    E = jnp.zeros((3, 3), float)
    E = E.at[0, 1].add(shear_rate/2)
    E = E.at[1, 0].add(shear_rate/2)

    Edri = jnp.array([shear_rate/2 * r_lub.at[:, 1].get(),
                      shear_rate/2 * r_lub.at[:, 0].get(), r_lub.at[:, 0].get() * 0.])
    Edrj = Edri
    rdEdri = r_lub.at[:, 0].get() * Edri.at[0].get() + \
        r_lub.at[:, 1].get() * Edri.at[1].get()
    rdEdrj = rdEdri
    epsrdEdri = jnp.array([r_lub.at[:, 2].get() * Edri[1],
                           -r_lub.at[:, 2].get() * Edri[0],
                           r_lub.at[:, 1].get() * Edri[0] - r_lub.at[:, 0].get() * Edri[1]])

    forces = jnp.zeros((N, 6), float)

    f = ((XG11 - 2.0*YG11).at[:, None].get() * (-rdEdri.at[:, None].get()) * r_lub
         + 2.0 * YG11.at[:, None].get() * (-Edri.T)
         + (XG21 - 2.0*YG21).at[:, None].get() *
         (-rdEdri.at[:, None].get()) * r_lub
         + 2.0 * YG21.at[:, None].get() * (-Edrj.T))
    l = (YH11.at[:, None].get() * (2.0 * epsrdEdri.T)
         + YH21.at[:, None].get() * (2.0 * epsrdEdri.T))

    forces = forces.at[indices_i_lub, :3].add(f)
    forces = forces.at[indices_i_lub, 3:].add(l)

    f = ((XG11 - 2.0*YG11).at[:, None].get() * (rdEdri.at[:, None].get()) * r_lub
         + 2.0 * YG11.at[:, None].get() * (Edrj.T)
         + (XG21 - 2.0*YG21).at[:, None].get() *
         (rdEdri.at[:, None].get()) * r_lub
         + 2.0 * YG21.at[:, None].get() * (Edri.T))
    l = (YH11.at[:, None].get() * (2.0 * epsrdEdri.T)
         + YH21.at[:, None].get() * (2.0 * epsrdEdri.T))

    forces = forces.at[indices_j_lub, :3].add(f)
    forces = forces.at[indices_j_lub, 3:].add(l)

    return jnp.ravel(forces)


@partial(jit, static_argnums=[0])
def compute_RSE(N, shear_rate, r_lub, indices_i_lub, indices_j_lub, XM11, XM12, YM11, YM12, ZM11, ZM12, stresslet):
    #These simulations are constructed so that, if there is strain,
	# x is the flow direction
	# y is the gradient direction
	# z is the vorticity direction
    # therefore,
	# Einf = [ 0 g/2 0 ]
	#	     [ g/2 0 0 ]
    #		 [ 0 0 0 ]

    #symmetry conditions
    # XG21 = -XG12
    # YG21 = -YG12
    # YH21 = YH12

    #define single particle strain (each particle experience same shear)
    E = jnp.zeros((3, 3), float)
    E = E.at[0, 1].add(shear_rate/2)
    E = E.at[1, 0].add(shear_rate/2)

    Edri = jnp.array([shear_rate/2 * r_lub.at[:, 1].get(),
                      shear_rate/2 * r_lub.at[:, 0].get(), r_lub.at[:, 0].get() * 0.])
    Edrj = Edri
    rdEdri = r_lub.at[:, 0].get() * Edri.at[0].get() + \
        r_lub.at[:, 1].get() * Edri.at[1].get()
    rdEdrj = rdEdri
    epsrdEdri = jnp.array([r_lub.at[:, 2].get() * Edri[1],
                           -r_lub.at[:, 2].get() * Edri[0],
                           r_lub.at[:, 1].get() * Edri[0] - r_lub.at[:, 0].get() * Edri[1]])

    sgn = 1

    #compute stresslet for particles i
    stresslet = stresslet.at[indices_i_lub, 0].add(sgn*(
        1.5*XM11 * (r_lub[:, 0]*r_lub[:, 0] - 1./3.) * rdEdri
        + 1.5*XM12 * (r_lub[:, 0]*r_lub[:, 0] - 1./3.) * rdEdrj
        + 0.5*YM11 * (2.*r_lub[:, 0]*Edri[0] + 2.*r_lub[:, 0]
                      * Edri[0] - 4.*rdEdri*r_lub[:, 0]*r_lub[:, 0])
        + 0.5*YM12 * (2.*r_lub[:, 0]*Edrj[0] + 2.*r_lub[:, 0]
                      * Edrj[0] - 4.*rdEdrj*r_lub[:, 0]*r_lub[:, 0])
        + 0.5*ZM11 * (2.*E[0][0] + (1.0 + r_lub[:, 0]*r_lub[:, 0])
                      * rdEdri - 2.*r_lub[:, 0]*Edri[0] - 2.*r_lub[:, 0]*Edri[0])
        + 0.5*ZM12 * (2.*E[0][0] + (1.0 + r_lub[:, 0]*r_lub[:, 0])
                      * rdEdrj - 2.*r_lub[:, 0]*Edrj[0] - 2.*r_lub[:, 0]*Edrj[0]))
    )
    stresslet = stresslet.at[indices_i_lub, 1].add(sgn*(
        1.5*XM11 * (r_lub[:, 0]*r_lub[:, 1]) * rdEdri
        + 1.5*XM12 * (r_lub[:, 0]*r_lub[:, 1]) * rdEdrj
        + 0.5*YM11 * (2.*r_lub[:, 0]*Edri[1] + 2.*r_lub[:, 1]
                      * Edri[0] - 4.*rdEdri*r_lub[:, 0]*r_lub[:, 1])
        + 0.5*YM12 * (2.*r_lub[:, 0]*Edrj[1] + 2.*r_lub[:, 1]
                      * Edrj[0] - 4.*rdEdrj*r_lub[:, 0]*r_lub[:, 1])
        + 0.5*ZM11 * (2.*E[0][1] + (1.0 + r_lub[:, 0]*r_lub[:, 1])
                      * rdEdri - 2.*r_lub[:, 0]*Edri[1] - 2.*r_lub[:, 1]*Edri[0])
        + 0.5*ZM12 * (2.*E[0][1] + (1.0 + r_lub[:, 0]*r_lub[:, 1])
                      * rdEdrj - 2.*r_lub[:, 0]*Edrj[1] - 2.*r_lub[:, 1]*Edrj[0]))
    )
    stresslet = stresslet.at[indices_i_lub, 2].add(sgn*(
        1.5*XM11 * (r_lub[:, 0]*r_lub[:, 2]) * rdEdri
        + 1.5*XM12 * (r_lub[:, 0]*r_lub[:, 2]) * rdEdrj
        + 0.5*YM11 * (2.*r_lub[:, 0]*Edri[2] + 2.*r_lub[:, 2]
                      * Edri[0] - 4.*rdEdri*r_lub[:, 0]*r_lub[:, 2])
        + 0.5*YM12 * (2.*r_lub[:, 0]*Edrj[2] + 2.*r_lub[:, 2]
                      * Edrj[0] - 4.*rdEdrj*r_lub[:, 0]*r_lub[:, 2])
        + 0.5*ZM11 * (2.*E[0][2] + (1.0 + r_lub[:, 0]*r_lub[:, 2])
                      * rdEdri - 2.*r_lub[:, 0]*Edri[2] - 2.*r_lub[:, 2]*Edri[0])
        + 0.5*ZM12 * (2.*E[0][2] + (1.0 + r_lub[:, 0]*r_lub[:, 2])
                      * rdEdrj - 2.*r_lub[:, 0]*Edrj[2] - 2.*r_lub[:, 2]*Edrj[0]))
    )
    stresslet = stresslet.at[indices_i_lub, 3].add(sgn*(
        1.5*XM11 * (r_lub[:, 1]*r_lub[:, 2]) * rdEdri
        + 1.5*XM12 * (r_lub[:, 1]*r_lub[:, 2]) * rdEdrj
        + 0.5*YM11 * (2.*r_lub[:, 1]*Edri[2] + 2.*r_lub[:, 2]
                      * Edri[1] - 4.*rdEdri*r_lub[:, 1]*r_lub[:, 2])
        + 0.5*YM12 * (2.*r_lub[:, 1]*Edrj[2] + 2.*r_lub[:, 2]
                      * Edrj[1] - 4.*rdEdrj*r_lub[:, 1]*r_lub[:, 2])
        + 0.5*ZM11 * (2.*E[1][2] + (1.0 + r_lub[:, 1]*r_lub[:, 2])
                      * rdEdri - 2.*r_lub[:, 1]*Edri[2] - 2.*r_lub[:, 2]*Edri[1])
        + 0.5*ZM12 * (2.*E[1][2] + (1.0 + r_lub[:, 1]*r_lub[:, 2])
                      * rdEdrj - 2.*r_lub[:, 1]*Edrj[2] - 2.*r_lub[:, 2]*Edrj[1]))
    )
    stresslet = stresslet.at[indices_i_lub, 4].add(sgn*(
        1.5*XM11 * (r_lub[:, 1]*r_lub[:, 1]) * rdEdri
        + 1.5*XM12 * (r_lub[:, 1]*r_lub[:, 1]) * rdEdrj
        + 0.5*YM11 * (2.*r_lub[:, 1]*Edri[1] + 2.*r_lub[:, 1]
                      * Edri[1] - 4.*rdEdri*r_lub[:, 1]*r_lub[:, 1])
        + 0.5*YM12 * (2.*r_lub[:, 1]*Edrj[1] + 2.*r_lub[:, 1]
                      * Edrj[1] - 4.*rdEdrj*r_lub[:, 1]*r_lub[:, 1])
        + 0.5*ZM11 * (2.*E[1][1] + (1.0 + r_lub[:, 1]*r_lub[:, 1])
                      * rdEdri - 2.*r_lub[:, 1]*Edri[1] - 2.*r_lub[:, 1]*Edri[1])
        + 0.5*ZM12 * (2.*E[1][1] + (1.0 + r_lub[:, 1]*r_lub[:, 1])
                      * rdEdrj - 2.*r_lub[:, 1]*Edrj[1] - 2.*r_lub[:, 1]*Edrj[1]))
    )

    #compute stresslet for particles j
    stresslet = stresslet.at[indices_j_lub, 0].add(sgn*(
        1.5*XM11 * (r_lub[:, 0]*r_lub[:, 0] - 1./3.) * rdEdrj
        + 1.5*XM12 * (r_lub[:, 0]*r_lub[:, 0] - 1./3.) * rdEdri
        + 0.5*YM11 * (2.*r_lub[:, 0]*Edri[0] + 2.*r_lub[:, 0]
                      * Edri[0] - 4.*rdEdrj*r_lub[:, 0]*r_lub[:, 0])
        + 0.5*YM12 * (2.*r_lub[:, 0]*Edrj[0] + 2.*r_lub[:, 0]
                      * Edrj[0] - 4.*rdEdri*r_lub[:, 0]*r_lub[:, 0])
        + 0.5*ZM11 * (2.*E[0][0] + (1.0 + r_lub[:, 0]*r_lub[:, 0])
                      * rdEdrj - 2.*r_lub[:, 0]*Edri[0] - 2.*r_lub[:, 0]*Edri[0])
        + 0.5*ZM12 * (2.*E[0][0] + (1.0 + r_lub[:, 0]*r_lub[:, 0])
                      * rdEdri - 2.*r_lub[:, 0]*Edrj[0] - 2.*r_lub[:, 0]*Edrj[0]))
    )
    stresslet = stresslet.at[indices_j_lub, 1].add(sgn*(
        1.5*XM11 * (r_lub[:, 0]*r_lub[:, 1]) * rdEdrj
        + 1.5*XM12 * (r_lub[:, 0]*r_lub[:, 1]) * rdEdri
        + 0.5*YM11 * (2.*r_lub[:, 0]*Edri[1] + 2.*r_lub[:, 1]
                      * Edri[0] - 4.*rdEdrj*r_lub[:, 0]*r_lub[:, 1])
        + 0.5*YM12 * (2.*r_lub[:, 0]*Edrj[1] + 2.*r_lub[:, 1]
                      * Edrj[0] - 4.*rdEdri*r_lub[:, 0]*r_lub[:, 1])
        + 0.5*ZM11 * (2.*E[0][1] + (1.0 + r_lub[:, 0]*r_lub[:, 1])
                      * rdEdrj - 2.*r_lub[:, 0]*Edri[1] - 2.*r_lub[:, 1]*Edri[0])
        + 0.5*ZM12 * (2.*E[0][1] + (1.0 + r_lub[:, 0]*r_lub[:, 1])
                      * rdEdri - 2.*r_lub[:, 0]*Edrj[1] - 2.*r_lub[:, 1]*Edrj[0]))
    )
    stresslet = stresslet.at[indices_j_lub, 2].add(sgn*(
        1.5*XM11 * (r_lub[:, 0]*r_lub[:, 2]) * rdEdrj
        + 1.5*XM12 * (r_lub[:, 0]*r_lub[:, 2]) * rdEdri
        + 0.5*YM11 * (2.*r_lub[:, 0]*Edri[2] + 2.*r_lub[:, 2]
                      * Edri[0] - 4.*rdEdrj*r_lub[:, 0]*r_lub[:, 2])
        + 0.5*YM12 * (2.*r_lub[:, 0]*Edrj[2] + 2.*r_lub[:, 2]
                      * Edrj[0] - 4.*rdEdri*r_lub[:, 0]*r_lub[:, 2])
        + 0.5*ZM11 * (2.*E[0][2] + (1.0 + r_lub[:, 0]*r_lub[:, 2])
                      * rdEdrj - 2.*r_lub[:, 0]*Edri[2] - 2.*r_lub[:, 2]*Edri[0])
        + 0.5*ZM12 * (2.*E[0][2] + (1.0 + r_lub[:, 0]*r_lub[:, 2])
                      * rdEdri - 2.*r_lub[:, 0]*Edrj[2] - 2.*r_lub[:, 2]*Edrj[0]))
    )
    stresslet = stresslet.at[indices_j_lub, 3].add(sgn*(
        1.5*XM11 * (r_lub[:, 1]*r_lub[:, 2]) * rdEdrj
        + 1.5*XM12 * (r_lub[:, 1]*r_lub[:, 2]) * rdEdri
        + 0.5*YM11 * (2.*r_lub[:, 1]*Edri[2] + 2.*r_lub[:, 2]
                      * Edri[1] - 4.*rdEdrj*r_lub[:, 1]*r_lub[:, 2])
        + 0.5*YM12 * (2.*r_lub[:, 1]*Edrj[2] + 2.*r_lub[:, 2]
                      * Edrj[1] - 4.*rdEdri*r_lub[:, 1]*r_lub[:, 2])
        + 0.5*ZM11 * (2.*E[1][2] + (1.0 + r_lub[:, 1]*r_lub[:, 2])
                      * rdEdrj - 2.*r_lub[:, 1]*Edri[2] - 2.*r_lub[:, 2]*Edri[1])
        + 0.5*ZM12 * (2.*E[1][2] + (1.0 + r_lub[:, 1]*r_lub[:, 2])
                      * rdEdri - 2.*r_lub[:, 1]*Edrj[2] - 2.*r_lub[:, 2]*Edrj[1]))
    )
    stresslet = stresslet.at[indices_j_lub, 4].add(sgn*(
        1.5*XM11 * (r_lub[:, 1]*r_lub[:, 1]) * rdEdrj
        + 1.5*XM12 * (r_lub[:, 1]*r_lub[:, 1]) * rdEdri
        + 0.5*YM11 * (2.*r_lub[:, 1]*Edri[1] + 2.*r_lub[:, 1]
                      * Edri[1] - 4.*rdEdrj*r_lub[:, 1]*r_lub[:, 1])
        + 0.5*YM12 * (2.*r_lub[:, 1]*Edrj[1] + 2.*r_lub[:, 1]
                      * Edrj[1] - 4.*rdEdri*r_lub[:, 1]*r_lub[:, 1])
        + 0.5*ZM11 * (2.*E[1][1] + (1.0 + r_lub[:, 1]*r_lub[:, 1])
                      * rdEdrj - 2.*r_lub[:, 1]*Edri[1] - 2.*r_lub[:, 1]*Edri[1])
        + 0.5*ZM12 * (2.*E[1][1] + (1.0 + r_lub[:, 1]*r_lub[:, 1])
                      * rdEdri - 2.*r_lub[:, 1]*Edrj[1] - 2.*r_lub[:, 1]*Edrj[1]))
        
    )

    return stresslet


@partial(jit, static_argnums=[6])
def compute_RSU(stresslet,
                velocities,
                indices_i_lub,
                indices_j_lub,
                ResFunctions,
                r_lub,
                N):

    XG11 = ResFunctions[11]
    XG12 = ResFunctions[12]
    YG11 = ResFunctions[13]
    YG12 = ResFunctions[14]
    YH11 = ResFunctions[15]
    YH12 = ResFunctions[16]

    vel_i = (jnp.reshape(velocities, (N, 6))).at[indices_i_lub].get()
    vel_j = (jnp.reshape(velocities, (N, 6))).at[indices_j_lub].get()

    # Dot product of r and U, i.e. axisymmetric projection (minus sign of rj is taken into account at the end of calculation)
    rdui = r_lub.at[:, 0].get()*vel_i.at[:, 0].get()+r_lub.at[:, 1].get() * \
        vel_i.at[:, 1].get()+r_lub.at[:, 2].get()*vel_i.at[:, 2].get()
    rduj = r_lub.at[:, 0].get()*vel_j.at[:, 0].get()+r_lub.at[:, 1].get() * \
        vel_j.at[:, 1].get()+r_lub.at[:, 2].get()*vel_j.at[:, 2].get()

    sgn = -1
    #compute stresslet for particles i
    stresslet = stresslet.at[indices_i_lub, 0].add(sgn*(
        XG11 * (r_lub[:, 0]*r_lub[:, 0] - 1./3.) * rdui
        + XG12 * (r_lub[:, 0]*r_lub[:, 0] - 1./3.) * rduj
        + YG11 * (vel_i[:, 0] * r_lub[:, 0] + r_lub[:, 0] *
                  vel_i[:, 0] - 2. * r_lub[:, 0] * r_lub[:, 0] * rdui)
        + YG12 * (vel_j[:, 0] * r_lub[:, 0] + r_lub[:, 0] *
                  vel_j[:, 0] - 2. * r_lub[:, 0] * r_lub[:, 0] * rduj))
    )
    stresslet = stresslet.at[indices_i_lub, 1].add(sgn*(
        XG11 * (r_lub[:, 0]*r_lub[:, 1]) * rdui
        + XG12 * (r_lub[:, 0]*r_lub[:, 1]) * rduj
        + YG11 * (vel_i[:, 0] * r_lub[:, 1] + r_lub[:, 0] *
                  vel_i[:, 1] - 2. * r_lub[:, 0] * r_lub[:, 1] * rdui)
        + YG12 * (vel_j[:, 0] * r_lub[:, 1] + r_lub[:, 0] *
                  vel_j[:, 1] - 2. * r_lub[:, 0] * r_lub[:, 1] * rduj))
    )
    stresslet = stresslet.at[indices_i_lub, 2].add(sgn*(
        XG11 * (r_lub[:, 0]*r_lub[:, 2]) * rdui
        + XG12 * (r_lub[:, 0]*r_lub[:, 2]) * rduj
        + YG11 * (vel_i[:, 0] * r_lub[:, 2] + r_lub[:, 0] *
                  vel_i[:, 2] - 2. * r_lub[:, 0] * r_lub[:, 2] * rdui)
        + YG12 * (vel_j[:, 0] * r_lub[:, 2] + r_lub[:, 0] *
                  vel_j[:, 2] - 2. * r_lub[:, 0] * r_lub[:, 2] * rduj))
    )
    stresslet = stresslet.at[indices_i_lub, 3].add(sgn*(
        XG11 * (r_lub[:, 1]*r_lub[:, 2]) * rdui
        + XG12 * (r_lub[:, 1]*r_lub[:, 2]) * rduj
        + YG11 * (vel_i[:, 1] * r_lub[:, 2] + r_lub[:, 1] *
                  vel_i[:, 2] - 2. * r_lub[:, 1] * r_lub[:, 2] * rdui)
        + YG12 * (vel_j[:, 1] * r_lub[:, 2] + r_lub[:, 1] *
                  vel_j[:, 2] - 2. * r_lub[:, 1] * r_lub[:, 2] * rduj))
    )
    stresslet = stresslet.at[indices_i_lub, 4].add(sgn*(
        XG11 * (r_lub[:, 1]*r_lub[:, 1] - 1./3.) * rdui
        + XG12 * (r_lub[:, 1]*r_lub[:, 1] - 1./3.) * rduj
        + YG11 * (vel_i[:, 1] * r_lub[:, 1] + r_lub[:, 1] *
                  vel_i[:, 1] - 2. * r_lub[:, 1] * r_lub[:, 1] * rdui)
        + YG12 * (vel_j[:, 1] * r_lub[:, 1] + r_lub[:, 1] *
                  vel_j[:, 1] - 2. * r_lub[:, 1] * r_lub[:, 1] * rduj))
    )

    #compute stresslet for particles j
    stresslet = stresslet.at[indices_j_lub, 0].add(sgn*(
        XG11 * (r_lub[:, 0]*r_lub[:, 0] - 1./3.) * rduj
        + XG12 * (r_lub[:, 0]*r_lub[:, 0] - 1./3.) * rdui
        + YG11 * (-vel_j[:, 0] * r_lub[:, 0] - r_lub[:, 0] *
                  vel_j[:, 0] - 2. * r_lub[:, 0] * r_lub[:, 0] * rduj)
        + YG12 * (-vel_i[:, 0] * r_lub[:, 0] - r_lub[:, 0] *
                  vel_i[:, 0] - 2. * r_lub[:, 0] * r_lub[:, 0] * rdui))
    )
    stresslet = stresslet.at[indices_j_lub, 1].add(sgn*(
        XG11 * (r_lub[:, 0]*r_lub[:, 1]) * rduj
        + XG12 * (r_lub[:, 0]*r_lub[:, 1]) * rdui
        + YG11 * (-vel_j[:, 0] * r_lub[:, 1] - r_lub[:, 0] *
                  vel_j[:, 1] - 2. * r_lub[:, 0] * r_lub[:, 1] * rduj)
        + YG12 * (-vel_i[:, 0] * r_lub[:, 1] - r_lub[:, 0] *
                  vel_i[:, 1] - 2. * r_lub[:, 0] * r_lub[:, 1] * rdui))
    )
    stresslet = stresslet.at[indices_j_lub, 2].add(sgn*(
        XG11 * (r_lub[:, 0]*r_lub[:, 2]) * rduj
        + XG12 * (r_lub[:, 0]*r_lub[:, 2]) * rdui
        + YG11 * (-vel_j[:, 0] * r_lub[:, 2] - r_lub[:, 0] *
                  vel_j[:, 2] - 2. * r_lub[:, 0] * r_lub[:, 2] * rduj)
        + YG12 * (-vel_i[:, 0] * r_lub[:, 2] - r_lub[:, 0] *
                  vel_i[:, 2] - 2. * r_lub[:, 0] * r_lub[:, 2] * rdui))
    )
    stresslet = stresslet.at[indices_j_lub, 3].add(sgn*(
        XG11 * (r_lub[:, 1]*r_lub[:, 2]) * rduj
        + XG12 * (r_lub[:, 1]*r_lub[:, 2]) * rdui
        + YG11 * (-vel_j[:, 1] * r_lub[:, 2] - r_lub[:, 1] *
                  vel_j[:, 2] - 2. * r_lub[:, 1] * r_lub[:, 2] * rduj)
        + YG12 * (-vel_i[:, 1] * r_lub[:, 2] - r_lub[:, 1] *
                  vel_i[:, 2] - 2. * r_lub[:, 1] * r_lub[:, 2] * rdui))
    )
    stresslet = stresslet.at[indices_j_lub, 4].add(sgn*(
        XG11 * (r_lub[:, 1]*r_lub[:, 1] - 1./3.) * rduj
        + XG12 * (r_lub[:, 1]*r_lub[:, 1] - 1./3.) * rdui
        + YG11 * (-vel_j[:, 1] * r_lub[:, 1] - r_lub[:, 1] *
                  vel_j[:, 1] - 2. * r_lub[:, 1] * r_lub[:, 1] * rduj)
        + YG12 * (-vel_i[:, 1] * r_lub[:, 1] - r_lub[:, 1] *
                  vel_i[:, 1] - 2. * r_lub[:, 1] * r_lub[:, 1] * rdui))
    )

    epsrdwi = jnp.array([r_lub.at[:, 2].get() * vel_i.at[:, 4].get() - r_lub.at[:, 1].get() * vel_i.at[:, 5].get(),
                        -r_lub.at[:, 2].get() * vel_i.at[:, 3].get() +
                         r_lub.at[:, 0].get() * vel_i.at[:, 5].get(),
                        r_lub.at[:, 1].get() * vel_i.at[:, 3].get() - r_lub.at[:, 0].get() * vel_i.at[:, 4].get()])

    epsrdwj = jnp.array([r_lub.at[:, 2].get() * vel_j.at[:, 4].get() - r_lub.at[:, 1].get() * vel_j.at[:, 5].get(),
                        -r_lub.at[:, 2].get() * vel_j.at[:, 3].get() +
                         r_lub.at[:, 0].get() * vel_j.at[:, 5].get(),
                        r_lub.at[:, 1].get() * vel_j.at[:, 3].get() - r_lub.at[:, 0].get() * vel_j.at[:, 4].get()])


   
    #compute stresslet for particles i
    stresslet = stresslet.at[indices_i_lub, 0].add(sgn*(
        YH11 * (r_lub[:, 0] * epsrdwi[0] + epsrdwi[0] * r_lub[:, 0])
        + YH12 * (r_lub[:, 0] * epsrdwj[0] + epsrdwj[0] * r_lub[:, 0]))
    )
    stresslet = stresslet.at[indices_i_lub, 1].add(sgn*(
        YH11 * (r_lub[:, 0] * epsrdwi[1] + epsrdwi[0] * r_lub[:, 1])
        + YH12 * (r_lub[:, 0] * epsrdwj[1] + epsrdwj[0] * r_lub[:, 1]))
    )
    stresslet = stresslet.at[indices_i_lub, 2].add(sgn*(
        YH11 * (r_lub[:, 0] * epsrdwi[2] + epsrdwi[0] * r_lub[:, 2])
        + YH12 * (r_lub[:, 0] * epsrdwj[2] + epsrdwj[0] * r_lub[:, 2]))
    )
    stresslet = stresslet.at[indices_i_lub, 3].add(sgn*(
        YH11 * (r_lub[:, 1] * epsrdwi[2] + epsrdwi[1] * r_lub[:, 2])
        + YH12 * (r_lub[:, 1] * epsrdwj[2] + epsrdwj[1] * r_lub[:, 2]))
    )
    stresslet = stresslet.at[indices_i_lub, 4].add(sgn*(
        YH11 * (r_lub[:, 2] * epsrdwi[2] + epsrdwi[2] * r_lub[:, 2])
        + YH12 * (r_lub[:, 2] * epsrdwj[2] + epsrdwj[2] * r_lub[:, 2]))
    )

    #compute stresslet for particles j
    stresslet = stresslet.at[indices_j_lub, 0].add(sgn*(
        YH11 * (r_lub[:, 0] * epsrdwj[0] + epsrdwj[0] * r_lub[:, 0])
        + YH12 * (r_lub[:, 0] * epsrdwi[0] + epsrdwi[0] * r_lub[:, 0]))
    )
    stresslet = stresslet.at[indices_j_lub, 1].add(sgn*(
        YH11 * (r_lub[:, 0] * epsrdwj[1] + epsrdwj[0] * r_lub[:, 1])
        + YH12 * (r_lub[:, 0] * epsrdwi[1] + epsrdwi[0] * r_lub[:, 1]))
    )
    stresslet = stresslet.at[indices_j_lub, 2].add(sgn*(
        YH11 * (r_lub[:, 0] * epsrdwj[2] + epsrdwj[0] * r_lub[:, 2])
        + YH12 * (r_lub[:, 0] * epsrdwi[2] + epsrdwi[0] * r_lub[:, 2]))
    )
    stresslet = stresslet.at[indices_j_lub, 3].add(sgn*(
        YH11 * (r_lub[:, 1] * epsrdwj[2] + epsrdwj[1] * r_lub[:, 2])
        + YH12 * (r_lub[:, 1] * epsrdwi[2] + epsrdwi[1] * r_lub[:, 2]))
    )
    stresslet = stresslet.at[indices_j_lub, 4].add(sgn*(
        YH11 * (r_lub[:, 2] * epsrdwj[2] + epsrdwj[2] * r_lub[:, 2])
        + YH12 * (r_lub[:, 2] * epsrdwi[2] + epsrdwi[2] * r_lub[:, 2]))
    )

    return stresslet
