from functools import partial

import jax.numpy as jnp
from jax import Array, jit
from jax.typing import ArrayLike

from jfsd import jaxmd_space as space


@partial(jit, static_argnums=[2, 3])
def rfu_precondition(
    ichol_relaxer: float, r: ArrayLike, num_particles: int, n_pairs_lub_prec: int, nl_lub_prec: ArrayLike
) -> tuple[Array, Array]:
    """Construct the lubrication resistance matrix R_FU for particle pairs very close (d <= 2.1*radius).

    This is used as part of the full precondition matrix used in the saddle point problem.

    Parameters
    ----------
    ichol_relaxer: (float)
        Relaxation factor for Cholesky decomposition
    r: (float)
        Array (n_pair_nf_prec,3) of distance vectors between particles in neighbor list
    num_particles: (int)
        Number of particles
    n_pairs_lub_prec: (int)
        Number of particle pairs to include in the lubrication matrix
    nl_lub_prec: (int)
        Array (2,n_pairs_nf) containing near-field precondition neighborlist indices

    Returns
    -------
    rfu_precondition, diagonal_elements_for_brownian

    """
    # Load resistance table
    restable_dist = jnp.load("files/ResTableDist.npy")
    restable_vals = jnp.load("files/ResTableVals.npy")
    # Smallest surface-to-surface distance (lower cut-off)
    restable_min = 0.0001
    # Table discretization (log space), i.e. ind * dr.set(log10( h[ind] / min )
    restable_dr = 0.004305

    # Define empty matrix
    rfu_precondition = jnp.zeros((6 * num_particles, 6 * num_particles), float)

    # By definition, r points from particle 1 to particle 2 (i to j), otherwise the handedness and symmetry of the lubrication functions is lost
    dist = space.distance(r)  # distance between particle i and j
    r_unit = r / dist[:, None]  # unit vector from particle j to i

    # # Indices in resistance table
    ind = jnp.log10((dist - 2.0) / restable_min) / restable_dr
    ind = ind.astype(int)

    dist_lower = restable_dist[ind]
    dist_upper = restable_dist[ind + 1]

    # # Linear interpolation of the Table values
    fac = jnp.where(
        dist_upper - dist_lower > 0.0, (dist - dist_lower) / (dist_upper - dist_lower), 0.0
    )

    xa11 = jnp.where(
        dist >= 2 + restable_min,
        (
            restable_vals[22 * (ind) + 0]
            + (restable_vals[22 * (ind + 1) + 0] - restable_vals[22 * (ind) + 0]) * fac
        ),
        restable_vals[0],
    )

    xa12 = jnp.where(
        dist >= 2 + restable_min,
        (
            restable_vals[22 * (ind) + 1]
            + (restable_vals[22 * (ind + 1) + 1] - restable_vals[22 * (ind) + 1]) * fac
        ),
        restable_vals[1],
    )

    ya11 = jnp.where(
        dist >= 2 + restable_min,
        (
            restable_vals[22 * (ind) + 2]
            + (restable_vals[22 * (ind + 1) + 2] - restable_vals[22 * (ind) + 2]) * fac
        ),
        restable_vals[2],
    )

    ya12 = jnp.where(
        dist >= 2 + restable_min,
        (
            restable_vals[22 * (ind) + 3]
            + (restable_vals[22 * (ind + 1) + 3] - restable_vals[22 * (ind) + 3]) * fac
        ),
        restable_vals[3],
    )

    yb11 = jnp.where(
        dist >= 2 + restable_min,
        (
            restable_vals[22 * (ind) + 4]
            + (restable_vals[22 * (ind + 1) + 4] - restable_vals[22 * (ind) + 4]) * fac
        ),
        restable_vals[4],
    )

    yb12 = jnp.where(
        dist >= 2 + restable_min,
        (
            restable_vals[22 * (ind) + 5]
            + (restable_vals[22 * (ind + 1) + 5] - restable_vals[22 * (ind) + 5]) * fac
        ),
        restable_vals[5],
    )

    xc11 = jnp.where(
        dist >= 2 + restable_min,
        (
            restable_vals[22 * (ind) + 6]
            + (restable_vals[22 * (ind + 1) + 6] - restable_vals[22 * (ind) + 6]) * fac
        ),
        restable_vals[6],
    )

    xc12 = jnp.where(
        dist >= 2 + restable_min,
        (
            restable_vals[22 * (ind) + 7]
            + (restable_vals[22 * (ind + 1) + 7] - restable_vals[22 * (ind) + 7]) * fac
        ),
        restable_vals[7],
    )

    yc11 = jnp.where(
        dist >= 2 + restable_min,
        (
            restable_vals[22 * (ind) + 8]
            + (restable_vals[22 * (ind + 1) + 8] - restable_vals[22 * (ind) + 8]) * fac
        ),
        restable_vals[8],
    )

    yc12 = jnp.where(
        dist >= 2 + restable_min,
        (
            restable_vals[22 * (ind) + 9]
            + (restable_vals[22 * (ind + 1) + 9] - restable_vals[22 * (ind) + 9]) * fac
        ),
        restable_vals[9],
    )

    epsr = jnp.array(
        [
            [jnp.zeros(n_pairs_lub_prec), r_unit[:, 2], -r_unit[:, 1]],
            [-r_unit[:, 2], jnp.zeros(n_pairs_lub_prec), r_unit[:, 0]],
            [r_unit[:, 1], -r_unit[:, 0], jnp.zeros(n_pairs_lub_prec)],
        ]
    )
    identity_m_rr = jnp.array(
        [
            [jnp.ones(n_pairs_lub_prec), jnp.zeros(n_pairs_lub_prec), jnp.zeros(n_pairs_lub_prec)],
            [jnp.zeros(n_pairs_lub_prec), jnp.ones(n_pairs_lub_prec), jnp.zeros(n_pairs_lub_prec)],
            [jnp.zeros(n_pairs_lub_prec), jnp.zeros(n_pairs_lub_prec), jnp.ones(n_pairs_lub_prec)],
        ]
    )

    rr = jnp.array(
        [
            [r_unit[:, 0] * r_unit[:, 0], r_unit[:, 0] * r_unit[:, 1], r_unit[:, 0] * r_unit[:, 2]],
            [r_unit[:, 1] * r_unit[:, 0], r_unit[:, 1] * r_unit[:, 1], r_unit[:, 1] * r_unit[:, 2]],
            [r_unit[:, 2] * r_unit[:, 0], r_unit[:, 2] * r_unit[:, 1], r_unit[:, 2] * r_unit[:, 2]],
        ]
    )
    identity_m_rr = identity_m_rr - rr

    a_neigh = xa12 * (rr) + ya12 * (identity_m_rr)
    a_self = xa11 * (rr) + ya11 * (identity_m_rr)
    b_neigh = yb12 * (epsr)
    b_self = yb11 * (epsr)
    c_neigh = xc12 * (rr) + yc12 * (identity_m_rr)
    c_self = xc11 * (rr) + yc11 * (identity_m_rr)

    # Fill in matrix (pair contributions)
    # # this is for all the A12 blocks
    rfu_precondition = rfu_precondition.at[6 * nl_lub_prec[0, :], 6 * nl_lub_prec[1, :]].add(
        a_neigh[0, 0, :]
    )
    rfu_precondition = rfu_precondition.at[6 * nl_lub_prec[0, :], 6 * nl_lub_prec[1, :] + 1].add(
        a_neigh[0, 1, :]
    )
    rfu_precondition = rfu_precondition.at[6 * nl_lub_prec[0, :], 6 * nl_lub_prec[1, :] + 2].add(
        a_neigh[0, 2, :]
    )
    rfu_precondition = rfu_precondition.at[6 * nl_lub_prec[0, :] + 1, 6 * nl_lub_prec[1, :]].add(
        a_neigh[1, 0, :]
    )
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[0, :] + 1, 6 * nl_lub_prec[1, :] + 1
    ].add(a_neigh[1, 1, :])
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[0, :] + 1, 6 * nl_lub_prec[1, :] + 2
    ].add(a_neigh[1, 2, :])
    rfu_precondition = rfu_precondition.at[6 * nl_lub_prec[0, :] + 2, 6 * nl_lub_prec[1, :]].add(
        a_neigh[2, 0, :]
    )
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[0, :] + 2, 6 * nl_lub_prec[1, :] + 1
    ].add(a_neigh[2, 1, :])
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[0, :] + 2, 6 * nl_lub_prec[1, :] + 2
    ].add(a_neigh[2, 2, :])

    # # this is for all the C12 blocks
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[0, :] + 3, 6 * nl_lub_prec[1, :] + 3
    ].add(c_neigh[0, 0, :])
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[0, :] + 3, 6 * nl_lub_prec[1, :] + 4
    ].add(c_neigh[0, 1, :])
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[0, :] + 3, 6 * nl_lub_prec[1, :] + 5
    ].add(c_neigh[0, 2, :])
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[0, :] + 4, 6 * nl_lub_prec[1, :] + 3
    ].add(c_neigh[1, 0, :])
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[0, :] + 4, 6 * nl_lub_prec[1, :] + 4
    ].add(c_neigh[1, 1, :])
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[0, :] + 4, 6 * nl_lub_prec[1, :] + 5
    ].add(c_neigh[1, 2, :])
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[0, :] + 5, 6 * nl_lub_prec[1, :] + 3
    ].add(c_neigh[2, 0, :])
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[0, :] + 5, 6 * nl_lub_prec[1, :] + 4
    ].add(c_neigh[2, 1, :])
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[0, :] + 5, 6 * nl_lub_prec[1, :] + 5
    ].add(c_neigh[2, 2, :])

    # # this is for all the Bt12 blocks (Bt12 = B12)
    rfu_precondition = rfu_precondition.at[6 * nl_lub_prec[0, :], 6 * nl_lub_prec[1, :] + 3].add(
        b_neigh[0, 0, :]
    )
    rfu_precondition = rfu_precondition.at[6 * nl_lub_prec[0, :], 6 * nl_lub_prec[1, :] + 4].add(
        b_neigh[0, 1, :]
    )
    rfu_precondition = rfu_precondition.at[6 * nl_lub_prec[0, :], 6 * nl_lub_prec[1, :] + 5].add(
        b_neigh[0, 2, :]
    )
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[0, :] + 1, 6 * nl_lub_prec[1, :] + 3
    ].add(b_neigh[1, 0, :])
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[0, :] + 1, 6 * nl_lub_prec[1, :] + 4
    ].add(b_neigh[1, 1, :])
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[0, :] + 1, 6 * nl_lub_prec[1, :] + 5
    ].add(b_neigh[1, 2, :])
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[0, :] + 2, 6 * nl_lub_prec[1, :] + 3
    ].add(b_neigh[2, 0, :])
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[0, :] + 2, 6 * nl_lub_prec[1, :] + 4
    ].add(b_neigh[2, 1, :])
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[0, :] + 2, 6 * nl_lub_prec[1, :] + 5
    ].add(b_neigh[2, 2, :])

    # # this is for all the B12 blocks (Bt12 = B12)
    rfu_precondition = rfu_precondition.at[6 * nl_lub_prec[0, :] + 3, 6 * nl_lub_prec[1, :]].add(
        b_neigh[0, 0, :]
    )
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[0, :] + 3, 6 * nl_lub_prec[1, :] + 1
    ].add(b_neigh[0, 1, :])
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[0, :] + 3, 6 * nl_lub_prec[1, :] + 2
    ].add(b_neigh[0, 2, :])
    rfu_precondition = rfu_precondition.at[6 * nl_lub_prec[0, :] + 4, 6 * nl_lub_prec[1, :]].add(
        b_neigh[1, 0, :]
    )
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[0, :] + 4, 6 * nl_lub_prec[1, :] + 1
    ].add(b_neigh[1, 1, :])
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[0, :] + 4, 6 * nl_lub_prec[1, :] + 2
    ].add(b_neigh[1, 2, :])
    rfu_precondition = rfu_precondition.at[6 * nl_lub_prec[0, :] + 5, 6 * nl_lub_prec[1, :]].add(
        b_neigh[2, 0, :]
    )
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[0, :] + 5, 6 * nl_lub_prec[1, :] + 1
    ].add(b_neigh[2, 1, :])
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[0, :] + 5, 6 * nl_lub_prec[1, :] + 2
    ].add(b_neigh[2, 2, :])

    # Fill in matrix (self contributions) (these are a sum of contributions from each pairs:
    # self contribution of particle 'i' will be a sum over all neighboring particles)

    # A11 Block
    rfu_precondition = rfu_precondition.at[6 * nl_lub_prec[0, :], 6 * nl_lub_prec[0, :]].add(
        a_self[0, 0, :]
    )
    rfu_precondition = rfu_precondition.at[6 * nl_lub_prec[0, :], 6 * nl_lub_prec[0, :] + 1].add(
        a_self[0, 1, :]
    )
    rfu_precondition = rfu_precondition.at[6 * nl_lub_prec[0, :], 6 * nl_lub_prec[0, :] + 2].add(
        a_self[0, 2, :]
    )
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[0, :] + 1, 6 * nl_lub_prec[0, :] + 1
    ].add(a_self[1, 1, :])
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[0, :] + 1, 6 * nl_lub_prec[0, :] + 2
    ].add(a_self[1, 2, :])
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[0, :] + 2, 6 * nl_lub_prec[0, :] + 2
    ].add(a_self[2, 2, :])

    # A22 Block
    rfu_precondition = rfu_precondition.at[6 * nl_lub_prec[1, :], 6 * nl_lub_prec[1, :]].add(
        a_self[0, 0, :]
    )
    rfu_precondition = rfu_precondition.at[6 * nl_lub_prec[1, :], 6 * nl_lub_prec[1, :] + 1].add(
        a_self[0, 1, :]
    )
    rfu_precondition = rfu_precondition.at[6 * nl_lub_prec[1, :], 6 * nl_lub_prec[1, :] + 2].add(
        a_self[0, 2, :]
    )
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[1, :] + 1, 6 * nl_lub_prec[1, :] + 1
    ].add(a_self[1, 1, :])
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[1, :] + 1, 6 * nl_lub_prec[1, :] + 2
    ].add(a_self[1, 2, :])
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[1, :] + 2, 6 * nl_lub_prec[1, :] + 2
    ].add(a_self[2, 2, :])

    # C11 Block
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[0, :] + 3, 6 * nl_lub_prec[0, :] + 3
    ].add(c_self[0, 0, :])
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[0, :] + 3, 6 * nl_lub_prec[0, :] + 4
    ].add(c_self[0, 1, :])
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[0, :] + 3, 6 * nl_lub_prec[0, :] + 5
    ].add(c_self[0, 2, :])
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[0, :] + 4, 6 * nl_lub_prec[0, :] + 4
    ].add(c_self[1, 1, :])
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[0, :] + 4, 6 * nl_lub_prec[0, :] + 5
    ].add(c_self[1, 2, :])
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[0, :] + 5, 6 * nl_lub_prec[0, :] + 5
    ].add(c_self[2, 2, :])

    # C22 Block
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[1, :] + 3, 6 * nl_lub_prec[1, :] + 3
    ].add(c_self[0, 0, :])
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[1, :] + 3, 6 * nl_lub_prec[1, :] + 4
    ].add(c_self[0, 1, :])
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[1, :] + 3, 6 * nl_lub_prec[1, :] + 5
    ].add(c_self[0, 2, :])
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[1, :] + 4, 6 * nl_lub_prec[1, :] + 4
    ].add(c_self[1, 1, :])
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[1, :] + 4, 6 * nl_lub_prec[1, :] + 5
    ].add(c_self[1, 2, :])
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[1, :] + 5, 6 * nl_lub_prec[1, :] + 5
    ].add(c_self[2, 2, :])

    # Bt11 Block
    rfu_precondition = rfu_precondition.at[6 * nl_lub_prec[0, :], 6 * nl_lub_prec[0, :] + 3].add(
        -b_self[0, 0, :]
    )
    rfu_precondition = rfu_precondition.at[6 * nl_lub_prec[0, :], 6 * nl_lub_prec[0, :] + 4].add(
        -b_self[0, 1, :]
    )
    rfu_precondition = rfu_precondition.at[6 * nl_lub_prec[0, :], 6 * nl_lub_prec[0, :] + 5].add(
        -b_self[0, 2, :]
    )
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[0, :] + 1, 6 * nl_lub_prec[0, :] + 3
    ].add(-b_self[1, 0, :])
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[0, :] + 1, 6 * nl_lub_prec[0, :] + 4
    ].add(-b_self[1, 1, :])
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[0, :] + 1, 6 * nl_lub_prec[0, :] + 5
    ].add(-b_self[1, 2, :])
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[0, :] + 2, 6 * nl_lub_prec[0, :] + 3
    ].add(-b_self[2, 0, :])
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[0, :] + 2, 6 * nl_lub_prec[0, :] + 4
    ].add(-b_self[2, 1, :])
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[0, :] + 2, 6 * nl_lub_prec[0, :] + 5
    ].add(-b_self[2, 2, :])

    # Bt22 Block
    rfu_precondition = rfu_precondition.at[6 * nl_lub_prec[1, :], 6 * nl_lub_prec[1, :] + 3].add(
        b_self[0, 0, :]
    )
    rfu_precondition = rfu_precondition.at[6 * nl_lub_prec[1, :], 6 * nl_lub_prec[1, :] + 4].add(
        b_self[0, 1, :]
    )
    rfu_precondition = rfu_precondition.at[6 * nl_lub_prec[1, :], 6 * nl_lub_prec[1, :] + 5].add(
        b_self[0, 2, :]
    )
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[1, :] + 1, 6 * nl_lub_prec[1, :] + 3
    ].add(b_self[1, 0, :])
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[1, :] + 1, 6 * nl_lub_prec[1, :] + 4
    ].add(b_self[1, 1, :])
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[1, :] + 1, 6 * nl_lub_prec[1, :] + 5
    ].add(b_self[1, 2, :])
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[1, :] + 2, 6 * nl_lub_prec[1, :] + 3
    ].add(b_self[2, 0, :])
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[1, :] + 2, 6 * nl_lub_prec[1, :] + 4
    ].add(b_self[2, 1, :])
    rfu_precondition = rfu_precondition.at[
        6 * nl_lub_prec[1, :] + 2, 6 * nl_lub_prec[1, :] + 5
    ].add(b_self[2, 2, :])

    # Symmetrize R_fu_nf
    diagonal_elements = rfu_precondition.diagonal()  # extract the diagonal
    rfu_precondition += jnp.transpose(rfu_precondition - jnp.diag(diagonal_elements))
    # now we have a symmetrix matrix

    # Compress diagonal values (needed later for brownian calculations, to perform Lanczos decomposition and square root)
    diagonal_elements_for_brownian = jnp.where(
        (diagonal_elements >= 1) | (diagonal_elements == 0), 1, jnp.sqrt(1 / diagonal_elements)
    )  # compress diagonal elements

    # Add identity for far field contribution and scale it properly
    # Because all values are made dimensionless on 6*pi*eta*a, the diagonal elements for FU (forces - velocities) are 1, but those for LW are 4/3(torques - angular velocities)
    diagonal_elements = jnp.where(
        (jnp.arange(6 * num_particles) - 6 * (jnp.repeat(jnp.arange(num_particles), 6))) < 3,
        ichol_relaxer,
        ichol_relaxer * 1.33333333333,
    )

    # sum the diagonal to the rest of the precondition matrix (add far-field precondition...)
    rfu_precondition += jnp.diag(diagonal_elements)

    return rfu_precondition, diagonal_elements_for_brownian


@partial(jit, static_argnums=[5])
def compute_lubrication_fu(
    velocities: ArrayLike,
    indices_i_lub: ArrayLike,
    indices_j_lub: ArrayLike,
    res_functions: tuple[
        ArrayLike,
        ArrayLike,
        ArrayLike,
        ArrayLike,
        ArrayLike,
        ArrayLike,
        ArrayLike,
        ArrayLike,
        ArrayLike,
        ArrayLike,
        ArrayLike,
    ],
    r_lub: ArrayLike,
    num_particles: int,
) -> Array:
    """Compute matrix-vector product of lubrication R_FU resistance matrix with particle velocities.

    Parameters
    ----------
    velocities: (float)
        Array (,6*num_particles) containing input particle linear/angular velocities
    indices_i_lub: (int)
        Array (,n_pair_nf) containing indices of first particle in near-field neighbor list pairs
    indices_j_lub: (int)
        Array (,n_pair_nf) containing indices of second particle in near-field neighbor list pairs
    res_functions: (float)
        Array (11,n_pair_nf) containing resistance scalar functions evaluated for the current particle configuration
    r_lub: (float)
        Array (n_pair_nf,3) containing units vectors connecting each pair of particles in the near-field neighbor list
    num_particles: (int)
        Number of particles

    Returns
    -------
    jnp.ravel(forces)

    """
    xa11 = res_functions[0]
    ya11 = res_functions[2]
    xa12 = res_functions[1]
    ya12 = res_functions[3]
    yb11 = res_functions[4]
    yb12 = res_functions[5]
    xc11 = res_functions[6]
    yc11 = res_functions[8]
    xc12 = res_functions[7]
    yc12 = res_functions[9]
    yb21 = res_functions[10]

    vel_i = (jnp.reshape(velocities, (num_particles, 6))).at[indices_i_lub].get()
    vel_j = (jnp.reshape(velocities, (num_particles, 6))).at[indices_j_lub].get()

    # Dot product of r_unit and U, i.e. axisymmetric projection (minus sign of rj is taken into account at the end of calculation)
    rdui = (
        r_lub.at[:, 0].get() * vel_i.at[:, 0].get()
        + r_lub.at[:, 1].get() * vel_i.at[:, 1].get()
        + r_lub.at[:, 2].get() * vel_i.at[:, 2].get()
    )
    rduj = (
        r_lub.at[:, 0].get() * vel_j.at[:, 0].get()
        + r_lub.at[:, 1].get() * vel_j.at[:, 1].get()
        + r_lub.at[:, 2].get() * vel_j.at[:, 2].get()
    )
    rdwi = (
        r_lub.at[:, 0].get() * vel_i.at[:, 3].get()
        + r_lub.at[:, 1].get() * vel_i.at[:, 4].get()
        + r_lub.at[:, 2].get() * vel_i.at[:, 5].get()
    )
    rdwj = (
        r_lub.at[:, 0].get() * vel_j.at[:, 3].get()
        + r_lub.at[:, 1].get() * vel_j.at[:, 4].get()
        + r_lub.at[:, 2].get() * vel_j.at[:, 5].get()
    )

    # Cross product of U and r_unit, i.e. eps_ijk*r_k*U_j = Px dot U, (eps_ijk is the Levi-Civita symbol)
    epsrdui = jnp.array(
        [
            r_lub.at[:, 2].get() * vel_i.at[:, 1].get()
            - r_lub.at[:, 1].get() * vel_i.at[:, 2].get(),
            -r_lub.at[:, 2].get() * vel_i.at[:, 0].get()
            + r_lub.at[:, 0].get() * vel_i.at[:, 2].get(),
            r_lub.at[:, 1].get() * vel_i.at[:, 0].get()
            - r_lub.at[:, 0].get() * vel_i.at[:, 1].get(),
        ]
    )

    epsrdwi = jnp.array(
        [
            r_lub.at[:, 2].get() * vel_i.at[:, 4].get()
            - r_lub.at[:, 1].get() * vel_i.at[:, 5].get(),
            -r_lub.at[:, 2].get() * vel_i.at[:, 3].get()
            + r_lub.at[:, 0].get() * vel_i.at[:, 5].get(),
            r_lub.at[:, 1].get() * vel_i.at[:, 3].get()
            - r_lub.at[:, 0].get() * vel_i.at[:, 4].get(),
        ]
    )

    epsrduj = jnp.array(
        [
            r_lub.at[:, 2].get() * vel_j.at[:, 1].get()
            - r_lub.at[:, 1].get() * vel_j.at[:, 2].get(),
            -r_lub.at[:, 2].get() * vel_j.at[:, 0].get()
            + r_lub.at[:, 0].get() * vel_j.at[:, 2].get(),
            r_lub.at[:, 1].get() * vel_j.at[:, 0].get()
            - r_lub.at[:, 0].get() * vel_j.at[:, 1].get(),
        ]
    )

    epsrdwj = jnp.array(
        [
            r_lub.at[:, 2].get() * vel_j.at[:, 4].get()
            - r_lub.at[:, 1].get() * vel_j.at[:, 5].get(),
            -r_lub.at[:, 2].get() * vel_j.at[:, 3].get()
            + r_lub.at[:, 0].get() * vel_j.at[:, 5].get(),
            r_lub.at[:, 1].get() * vel_j.at[:, 3].get()
            - r_lub.at[:, 0].get() * vel_j.at[:, 4].get(),
        ]
    )

    forces = jnp.zeros((num_particles, 6), float)

    # Compute the contributions to the force for particles i (Fi = A11*Ui + A12*Uj + BT11*Wi + BT12*Wj)
    f = (
        (xa11 - ya11).at[:, None].get() * rdui.at[:, None].get() * r_lub
        + ya11.at[:, None].get() * vel_i.at[:, :3].get()
        + (xa12 - ya12).at[:, None].get() * rduj.at[:, None].get() * r_lub
        + ya12.at[:, None].get() * vel_j.at[:, :3].get()
        + yb11.at[:, None].get() * (-epsrdwi.T)
        + yb21.at[:, None].get() * (-epsrdwj.T)
    )
    forces = forces.at[indices_i_lub, :3].add(f)
    # Compute the contributions to the force for particles j (Fj = A11*Uj + A12*Ui + BT11*Wj + BT12*Wi)
    f = (
        (xa11 - ya11).at[:, None].get() * rduj.at[:, None].get() * r_lub
        + ya11.at[:, None].get() * vel_j.at[:, :3].get()
        + (xa12 - ya12).at[:, None].get() * rdui.at[:, None].get() * r_lub
        + ya12.at[:, None].get() * vel_i.at[:, :3].get()
        + yb11.at[:, None].get() * (epsrdwj.T)
        + yb21.at[:, None].get() * (epsrdwi.T)
    )
    forces = forces.at[indices_j_lub, :3].add(f)
    # Compute the contributions to the torque for particles i (Li = B11*Ui + B12*Uj + C11*Wi + C12*Wj)
    l = (
        yb11.at[:, None].get() * epsrdui.T
        + yb12.at[:, None].get() * epsrduj.T
        + (xc11 - yc11).at[:, None].get() * rdwi.at[:, None].get() * r_lub
        + yc11.at[:, None].get() * vel_i.at[:, 3:].get()
        + (xc12 - yc12).at[:, None].get() * rdwj.at[:, None].get() * r_lub
        + yc12.at[:, None].get() * vel_j.at[:, 3:].get()
    )
    forces = forces.at[indices_i_lub, 3:].add(l)
    # Compute the contributions to the torque for particles j (Lj = B11*Uj + B12*Ui + C11*Wj + C12*Wi)
    l = (
        -yb11.at[:, None].get() * epsrduj.T
        - yb12.at[:, None].get() * epsrdui.T
        + (xc11 - yc11).at[:, None].get() * rdwj.at[:, None].get() * r_lub
        + yc11.at[:, None].get() * vel_j.at[:, 3:].get()
        + (xc12 - yc12).at[:, None].get() * rdwi.at[:, None].get() * r_lub
        + yc12.at[:, None].get() * vel_i.at[:, 3:].get()
    )
    forces = forces.at[indices_j_lub, 3:].add(l)

    return jnp.ravel(forces)


@partial(jit, static_argnums=[0])
def compute_rfe(
    num_particles: int,
    shear_rate: float,
    r_lub: ArrayLike,
    indices_i_lub: ArrayLike,
    indices_j_lub: ArrayLike,
    xg11: ArrayLike,
    xg12: ArrayLike,
    yg11: ArrayLike,
    yg12: ArrayLike,
    yh11: ArrayLike,
    yh12: ArrayLike,
    xg21: ArrayLike,
    yg21: ArrayLike,
    yh21: ArrayLike,
) -> Array:
    """Compute matrix-vector product of lubrication R_FE resistance matrix with (minus) the ambient rate of strain.

    These simulations are constructed so that, if there is strain,
        x is the flow direction
        y is the gradient direction
        z is the vorticity direction
    therefore,
        Einf = [ 0  g/2 0 ]
                   [g/2  0  0 ]
               [ 0   0  0 ]

    Parameters
    ----------
    num_particles: (int)
        Number of particles
    shear_rate: (float)
        Shear rate at current step
    r_lub: (float)
        Array (n_pair_nf,3) containing units vectors connecting each pair of particles in the near-field neighbor list
    indices_i_lub: (int)
        Array (,n_pair_nf) containing indices of first particle in near-field neighbor list pairs
    indices_j_lub: (int)
        Array (,n_pair_nf) containing indices of second particle in near-field neighbor list pairs
    xg11: (float)
        Array (,n_pair_nf) containing resistance scalar functions evaluated for the current particle configuration
    xg12: (float)
        Array (,n_pair_nf) containing resistance scalar functions evaluated for the current particle configuration
    yg11: (float)
        Array (,n_pair_nf) containing resistance scalar functions evaluated for the current particle configuration
    yg12: (float)
        Array (,n_pair_nf) containing resistance scalar functions evaluated for the current particle configuration
    yh11: (float)
        Array (,n_pair_nf) containing resistance scalar functions evaluated for the current particle configuration
    yh12: (float)
        Array (,n_pair_nf) containing resistance scalar functions evaluated for the current particle configuration
    xg21: (float)
        Array (,n_pair_nf) containing resistance scalar functions evaluated for the current particle configuration
    yg21: (float)
        Array (,n_pair_nf) containing resistance scalar functions evaluated for the current particle configuration
    yh21: (float)
        Array (,n_pair_nf) containing resistance scalar functions evaluated for the current particle configuration

    Returns
    -------
    jnp.ravel(forces)

    """
    # symmetry conditions
    # xg21 = -xg12
    # yg21 = -yg12
    # yh21 = yh12

    # define single particle strain (each particle experience same shear)
    strain = jnp.zeros((3, 3), float)
    strain = strain.at[0, 1].add(shear_rate / 2)
    strain = strain.at[1, 0].add(shear_rate / 2)

    edri = jnp.array(
        [
            shear_rate / 2 * r_lub.at[:, 1].get(),
            shear_rate / 2 * r_lub.at[:, 0].get(),
            r_lub.at[:, 0].get() * 0.0,
        ]
    )
    edrj = edri
    rdedri = r_lub.at[:, 0].get() * edri.at[0].get() + r_lub.at[:, 1].get() * edri.at[1].get()
    # rdedrj = rdedri
    epsrdedri = jnp.array(
        [
            r_lub.at[:, 2].get() * edri[1],
            -r_lub.at[:, 2].get() * edri[0],
            r_lub.at[:, 1].get() * edri[0] - r_lub.at[:, 0].get() * edri[1],
        ]
    )
    forces = jnp.zeros((num_particles, 6), float)
    f = (
        (xg11 - 2.0 * yg11).at[:, None].get() * (-rdedri.at[:, None].get()) * r_lub
        + 2.0 * yg11.at[:, None].get() * (-edri.T)
        + (xg21 - 2.0 * yg21).at[:, None].get() * (-rdedri.at[:, None].get()) * r_lub
        + 2.0 * yg21.at[:, None].get() * (-edrj.T)
    )

    l = yh11.at[:, None].get() * (2.0 * epsrdedri.T) + yh21.at[:, None].get() * (2.0 * epsrdedri.T)

    forces = forces.at[indices_i_lub, :3].add(f)
    forces = forces.at[indices_i_lub, 3:].add(l)

    f = (
        (xg11 - 2.0 * yg11).at[:, None].get() * (rdedri.at[:, None].get()) * r_lub
        + 2.0 * yg11.at[:, None].get() * (edrj.T)
        + (xg21 - 2.0 * yg21).at[:, None].get() * (rdedri.at[:, None].get()) * r_lub
        + 2.0 * yg21.at[:, None].get() * (edri.T)
    )

    l = yh11.at[:, None].get() * (2.0 * epsrdedri.T) + yh21.at[:, None].get() * (2.0 * epsrdedri.T)

    forces = forces.at[indices_j_lub, :3].add(f)
    forces = forces.at[indices_j_lub, 3:].add(l)

    return jnp.ravel(forces)


@partial(jit, static_argnums=[0])
def compute_rse(
    num_particles: int,
    shear_rate: float,
    r_lub: ArrayLike,
    indices_i_lub: ArrayLike,
    indices_j_lub: ArrayLike,
    xm11: ArrayLike,
    xm12: ArrayLike,
    ym11: ArrayLike,
    ym12: ArrayLike,
    zm11: ArrayLike,
    zm12: ArrayLike,
    stresslet: ArrayLike,
) -> Array:
    """Compute matrix-vector product of lubrication R_SE resistance matrix with particle rate of strain.

    These simulations are constructed so that, if there is strain,
        x is the flow direction
        y is the gradient direction
        z is the vorticity direction
    therefore,
        Einf = [ 0  g/2 0 ]
                   [g/2  0  0 ]
               [ 0   0  0 ]

    Parameters
    ----------
    num_particles: (int)
        Number of particles
    shear_rate: (float)
        Shear rate at current step
    r_lub: (float)
        Array (n_pair_nf,3) containing units vectors connecting each pair of particles in the near-field neighbor list
    indices_i_lub: (int)
        Array (,n_pair_nf) containing indices of first particle in near-field neighbor list pairs
    indices_j_lub: (int)
        Array (,n_pair_nf) containing indices of second particle in near-field neighbor list pairs
    xm11: (float)
        Array (,n_pair_nf) containing resistance scalar functions evaluated for the current particle configuration
    xm12: (float)
        Array (,n_pair_nf) containing resistance scalar functions evaluated for the current particle configuration
    ym11: (float)
        Array (,n_pair_nf) containing resistance scalar functions evaluated for the current particle configuration
    ym12: (float)
        Array (,n_pair_nf) containing resistance scalar functions evaluated for the current particle configuration
    zm11: (float)
        Array (,n_pair_nf) containing resistance scalar functions evaluated for the current particle configuration
    zm12: (float)
        Array (,n_pair_nf) containing resistance scalar functions evaluated for the current particle configuration
    stresslet: (float)
        Array (,5*num_particles) containing stresslet

    Returns
    -------
    stresslet

    """
    # symmetry conditions
    # xg21 = -xg12
    # yg21 = -yg12
    # yh21 = yh12

    # define single particle strain (each particle experience same shear)
    strain = jnp.zeros((3, 3), float)
    strain = strain.at[0, 1].add(shear_rate / 2)
    strain = strain.at[1, 0].add(shear_rate / 2)

    edri = jnp.array(
        [
            shear_rate / 2 * r_lub.at[:, 1].get(),
            shear_rate / 2 * r_lub.at[:, 0].get(),
            r_lub.at[:, 0].get() * 0.0,
        ]
    )
    edrj = edri  # external shear is the same for each particle
    rdedri = r_lub.at[:, 0].get() * edri.at[0].get() + r_lub.at[:, 1].get() * edri.at[1].get()
    rdedrj = rdedri

    # compute stresslet for particles i
    stresslet = stresslet.at[indices_i_lub, 0].add(
        (
            1.5 * xm11 * (r_lub[:, 0] * r_lub[:, 0] - 1.0 / 3.0) * rdedri
            + 1.5 * xm12 * (r_lub[:, 0] * r_lub[:, 0] - 1.0 / 3.0) * rdedrj
            + 0.5
            * ym11
            * (
                2.0 * r_lub[:, 0] * edri[0]
                + 2.0 * r_lub[:, 0] * edri[0]
                - 4.0 * rdedri * r_lub[:, 0] * r_lub[:, 0]
            )
            + 0.5
            * ym12
            * (
                2.0 * r_lub[:, 0] * edrj[0]
                + 2.0 * r_lub[:, 0] * edrj[0]
                - 4.0 * rdedrj * r_lub[:, 0] * r_lub[:, 0]
            )
            + 0.5
            * zm11
            * (
                2.0 * strain[0][0]
                + (1.0 + r_lub[:, 0] * r_lub[:, 0]) * rdedri
                - 2.0 * r_lub[:, 0] * edri[0]
                - 2.0 * r_lub[:, 0] * edri[0]
            )
            + 0.5
            * zm12
            * (
                2.0 * strain[0][0]
                + (1.0 + r_lub[:, 0] * r_lub[:, 0]) * rdedrj
                - 2.0 * r_lub[:, 0] * edrj[0]
                - 2.0 * r_lub[:, 0] * edrj[0]
            )
        )
    )

    stresslet = stresslet.at[indices_i_lub, 1].add(
        (
            1.5 * xm11 * (r_lub[:, 0] * r_lub[:, 1]) * rdedri
            + 1.5 * xm12 * (r_lub[:, 0] * r_lub[:, 1]) * rdedrj
            + 0.5
            * ym11
            * (
                2.0 * r_lub[:, 0] * edri[1]
                + 2.0 * r_lub[:, 1] * edri[0]
                - 4.0 * rdedri * r_lub[:, 0] * r_lub[:, 1]
            )
            + 0.5
            * ym12
            * (
                2.0 * r_lub[:, 0] * edrj[1]
                + 2.0 * r_lub[:, 1] * edrj[0]
                - 4.0 * rdedrj * r_lub[:, 0] * r_lub[:, 1]
            )
            + 0.5
            * zm11
            * (
                2.0 * strain[0][1]
                + (r_lub[:, 0] * r_lub[:, 1]) * rdedri
                - 2.0 * r_lub[:, 0] * edri[1]
                - 2.0 * r_lub[:, 1] * edri[0]
            )
            + 0.5
            * zm12
            * (
                2.0 * strain[0][1]
                + (r_lub[:, 0] * r_lub[:, 1]) * rdedrj
                - 2.0 * r_lub[:, 0] * edrj[1]
                - 2.0 * r_lub[:, 1] * edrj[0]
            )
        )
    )

    stresslet = stresslet.at[indices_i_lub, 2].add(
        (
            1.5 * xm11 * (r_lub[:, 0] * r_lub[:, 2]) * rdedri
            + 1.5 * xm12 * (r_lub[:, 0] * r_lub[:, 2]) * rdedrj
            + 0.5
            * ym11
            * (
                2.0 * r_lub[:, 0] * edri[2]
                + 2.0 * r_lub[:, 2] * edri[0]
                - 4.0 * rdedri * r_lub[:, 0] * r_lub[:, 2]
            )
            + 0.5
            * ym12
            * (
                2.0 * r_lub[:, 0] * edrj[2]
                + 2.0 * r_lub[:, 2] * edrj[0]
                - 4.0 * rdedrj * r_lub[:, 0] * r_lub[:, 2]
            )
            + 0.5
            * zm11
            * (
                2.0 * strain[0][2]
                + (r_lub[:, 0] * r_lub[:, 2]) * rdedri
                - 2.0 * r_lub[:, 0] * edri[2]
                - 2.0 * r_lub[:, 2] * edri[0]
            )
            + 0.5
            * zm12
            * (
                2.0 * strain[0][2]
                + (r_lub[:, 0] * r_lub[:, 2]) * rdedrj
                - 2.0 * r_lub[:, 0] * edrj[2]
                - 2.0 * r_lub[:, 2] * edrj[0]
            )
        )
    )
    stresslet = stresslet.at[indices_i_lub, 3].add(
        (
            1.5 * xm11 * (r_lub[:, 1] * r_lub[:, 2]) * rdedri
            + 1.5 * xm12 * (r_lub[:, 1] * r_lub[:, 2]) * rdedrj
            + 0.5
            * ym11
            * (
                2.0 * r_lub[:, 1] * edri[2]
                + 2.0 * r_lub[:, 2] * edri[1]
                - 4.0 * rdedri * r_lub[:, 1] * r_lub[:, 2]
            )
            + 0.5
            * ym12
            * (
                2.0 * r_lub[:, 1] * edrj[2]
                + 2.0 * r_lub[:, 2] * edrj[1]
                - 4.0 * rdedrj * r_lub[:, 1] * r_lub[:, 2]
            )
            + 0.5
            * zm11
            * (
                2.0 * strain[1][2]
                + (r_lub[:, 1] * r_lub[:, 2]) * rdedri
                - 2.0 * r_lub[:, 1] * edri[2]
                - 2.0 * r_lub[:, 2] * edri[1]
            )
            + 0.5
            * zm12
            * (
                2.0 * strain[1][2]
                + (r_lub[:, 1] * r_lub[:, 2]) * rdedrj
                - 2.0 * r_lub[:, 1] * edrj[2]
                - 2.0 * r_lub[:, 2] * edrj[1]
            )
        )
    )
    stresslet = stresslet.at[indices_i_lub, 4].add(
        (
            1.5 * xm11 * (r_lub[:, 1] * r_lub[:, 1] - 1.0 / 3.0) * rdedri
            + 1.5 * xm12 * (r_lub[:, 1] * r_lub[:, 1] - 1.0 / 3.0) * rdedrj
            + 0.5
            * ym11
            * (
                2.0 * r_lub[:, 1] * edri[1]
                + 2.0 * r_lub[:, 1] * edri[1]
                - 4.0 * rdedri * r_lub[:, 1] * r_lub[:, 1]
            )
            + 0.5
            * ym12
            * (
                2.0 * r_lub[:, 1] * edrj[1]
                + 2.0 * r_lub[:, 1] * edrj[1]
                - 4.0 * rdedrj * r_lub[:, 1] * r_lub[:, 1]
            )
            + 0.5
            * zm11
            * (
                2.0 * strain[1][1]
                + (1.0 + r_lub[:, 1] * r_lub[:, 1]) * rdedri
                - 2.0 * r_lub[:, 1] * edri[1]
                - 2.0 * r_lub[:, 1] * edri[1]
            )
            + 0.5
            * zm12
            * (
                2.0 * strain[1][1]
                + (1.0 + r_lub[:, 1] * r_lub[:, 1]) * rdedrj
                - 2.0 * r_lub[:, 1] * edrj[1]
                - 2.0 * r_lub[:, 1] * edrj[1]
            )
        )
    )

    # compute stresslet for particles j
    stresslet = stresslet.at[indices_j_lub, 0].add(
        (
            1.5 * xm11 * (r_lub[:, 0] * r_lub[:, 0] - 1.0 / 3.0) * rdedrj
            + 1.5 * xm12 * (r_lub[:, 0] * r_lub[:, 0] - 1.0 / 3.0) * rdedri
            + 0.5
            * ym11
            * (
                2.0 * r_lub[:, 0] * edri[0]
                + 2.0 * r_lub[:, 0] * edri[0]
                - 4.0 * rdedrj * r_lub[:, 0] * r_lub[:, 0]
            )
            + 0.5
            * ym12
            * (
                2.0 * r_lub[:, 0] * edrj[0]
                + 2.0 * r_lub[:, 0] * edrj[0]
                - 4.0 * rdedri * r_lub[:, 0] * r_lub[:, 0]
            )
            + 0.5
            * zm11
            * (
                2.0 * strain[0][0]
                + (1.0 + r_lub[:, 0] * r_lub[:, 0]) * rdedrj
                - 2.0 * r_lub[:, 0] * edri[0]
                - 2.0 * r_lub[:, 0] * edri[0]
            )
            + 0.5
            * zm12
            * (
                2.0 * strain[0][0]
                + (1.0 + r_lub[:, 0] * r_lub[:, 0]) * rdedri
                - 2.0 * r_lub[:, 0] * edrj[0]
                - 2.0 * r_lub[:, 0] * edrj[0]
            )
        )
    )
    stresslet = stresslet.at[indices_j_lub, 1].add(
        (
            1.5 * xm11 * (r_lub[:, 0] * r_lub[:, 1]) * rdedrj
            + 1.5 * xm12 * (r_lub[:, 0] * r_lub[:, 1]) * rdedri
            + 0.5
            * ym11
            * (
                2.0 * r_lub[:, 0] * edri[1]
                + 2.0 * r_lub[:, 1] * edri[0]
                - 4.0 * rdedrj * r_lub[:, 0] * r_lub[:, 1]
            )
            + 0.5
            * ym12
            * (
                2.0 * r_lub[:, 0] * edrj[1]
                + 2.0 * r_lub[:, 1] * edrj[0]
                - 4.0 * rdedri * r_lub[:, 0] * r_lub[:, 1]
            )
            + 0.5
            * zm11
            * (
                2.0 * strain[0][1]
                + (r_lub[:, 0] * r_lub[:, 1]) * rdedrj
                - 2.0 * r_lub[:, 0] * edri[1]
                - 2.0 * r_lub[:, 1] * edri[0]
            )
            + 0.5
            * zm12
            * (
                2.0 * strain[0][1]
                + (r_lub[:, 0] * r_lub[:, 1]) * rdedri
                - 2.0 * r_lub[:, 0] * edrj[1]
                - 2.0 * r_lub[:, 1] * edrj[0]
            )
        )
    )
    stresslet = stresslet.at[indices_j_lub, 2].add(
        (
            1.5 * xm11 * (r_lub[:, 0] * r_lub[:, 2]) * rdedrj
            + 1.5 * xm12 * (r_lub[:, 0] * r_lub[:, 2]) * rdedri
            + 0.5
            * ym11
            * (
                2.0 * r_lub[:, 0] * edri[2]
                + 2.0 * r_lub[:, 2] * edri[0]
                - 4.0 * rdedrj * r_lub[:, 0] * r_lub[:, 2]
            )
            + 0.5
            * ym12
            * (
                2.0 * r_lub[:, 0] * edrj[2]
                + 2.0 * r_lub[:, 2] * edrj[0]
                - 4.0 * rdedri * r_lub[:, 0] * r_lub[:, 2]
            )
            + 0.5
            * zm11
            * (
                2.0 * strain[0][2]
                + (r_lub[:, 0] * r_lub[:, 2]) * rdedrj
                - 2.0 * r_lub[:, 0] * edri[2]
                - 2.0 * r_lub[:, 2] * edri[0]
            )
            + 0.5
            * zm12
            * (
                2.0 * strain[0][2]
                + (r_lub[:, 0] * r_lub[:, 2]) * rdedri
                - 2.0 * r_lub[:, 0] * edrj[2]
                - 2.0 * r_lub[:, 2] * edrj[0]
            )
        )
    )
    stresslet = stresslet.at[indices_j_lub, 3].add(
        (
            1.5 * xm11 * (r_lub[:, 1] * r_lub[:, 2]) * rdedrj
            + 1.5 * xm12 * (r_lub[:, 1] * r_lub[:, 2]) * rdedri
            + 0.5
            * ym11
            * (
                2.0 * r_lub[:, 1] * edri[2]
                + 2.0 * r_lub[:, 2] * edri[1]
                - 4.0 * rdedrj * r_lub[:, 1] * r_lub[:, 2]
            )
            + 0.5
            * ym12
            * (
                2.0 * r_lub[:, 1] * edrj[2]
                + 2.0 * r_lub[:, 2] * edrj[1]
                - 4.0 * rdedri * r_lub[:, 1] * r_lub[:, 2]
            )
            + 0.5
            * zm11
            * (
                2.0 * strain[1][2]
                + (r_lub[:, 1] * r_lub[:, 2]) * rdedrj
                - 2.0 * r_lub[:, 1] * edri[2]
                - 2.0 * r_lub[:, 2] * edri[1]
            )
            + 0.5
            * zm12
            * (
                2.0 * strain[1][2]
                + (r_lub[:, 1] * r_lub[:, 2]) * rdedri
                - 2.0 * r_lub[:, 1] * edrj[2]
                - 2.0 * r_lub[:, 2] * edrj[1]
            )
        )
    )
    stresslet = stresslet.at[indices_j_lub, 4].add(
        (
            1.5 * xm11 * (r_lub[:, 1] * r_lub[:, 1] - 1.0 / 3.0) * rdedrj
            + 1.5 * xm12 * (r_lub[:, 1] * r_lub[:, 1] - 1.0 / 3.0) * rdedri
            + 0.5
            * ym11
            * (
                2.0 * r_lub[:, 1] * edri[1]
                + 2.0 * r_lub[:, 1] * edri[1]
                - 4.0 * rdedrj * r_lub[:, 1] * r_lub[:, 1]
            )
            + 0.5
            * ym12
            * (
                2.0 * r_lub[:, 1] * edrj[1]
                + 2.0 * r_lub[:, 1] * edrj[1]
                - 4.0 * rdedri * r_lub[:, 1] * r_lub[:, 1]
            )
            + 0.5
            * zm11
            * (
                2.0 * strain[1][1]
                + (1.0 + r_lub[:, 1] * r_lub[:, 1]) * rdedrj
                - 2.0 * r_lub[:, 1] * edri[1]
                - 2.0 * r_lub[:, 1] * edri[1]
            )
            + 0.5
            * zm12
            * (
                2.0 * strain[1][1]
                + (1.0 + r_lub[:, 1] * r_lub[:, 1]) * rdedri
                - 2.0 * r_lub[:, 1] * edrj[1]
                - 2.0 * r_lub[:, 1] * edrj[1]
            )
        )
    )

    return stresslet


@partial(jit, static_argnums=[6])
def compute_rsu(
    stresslet: ArrayLike,
    velocities: ArrayLike,
    indices_i_lub: ArrayLike,
    indices_j_lub: ArrayLike,
    res_functions: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike],
    r_lub: ArrayLike,
    num_particles: int,
) -> Array:
    """Compute matrix-vector product of lubrication R_SU resistance matrix with particle velocities.

    Parameters
    ----------
    stresslet: (float)
        Array (,5*num_particles) containing particle stresslet
    velocities: (float)
        Array (,6*num_particles) containing particle linear/angular velocities
    indices_i_lub: (int)
        Array (,n_pair_nf) containing indices of first particle in near-field neighbor list pairs
    indices_j_lub: (int)
        Array (,n_pair_nf) containing indices of second particle in near-field neighbor list pairs
    res_functions: (float)
        Array (6,n_pair_nf) containing resistance scalar functions evaluated for the current particle configuration
    r_lub: (float)
        Array (n_pair_nf,3) containing units vectors connecting each pair of particles in the near-field neighbor list
    num_particles: (int)
        Number of particles

    Returns
    -------
    stresslet

    """
    xg11 = res_functions[11]
    xg12 = res_functions[12]
    yg11 = res_functions[13]
    yg12 = res_functions[14]
    yh11 = res_functions[15]
    yh12 = res_functions[16]

    vel_i = (jnp.reshape(velocities, (num_particles, 6))).at[indices_i_lub].get()
    vel_j = (jnp.reshape(velocities, (num_particles, 6))).at[indices_j_lub].get()

    # Dot product of r_unit and U, i.e. axisymmetric projection
    rdui = (
        r_lub.at[:, 0].get() * vel_i.at[:, 0].get()
        + r_lub.at[:, 1].get() * vel_i.at[:, 1].get()
        + r_lub.at[:, 2].get() * vel_i.at[:, 2].get()
    )
    rduj = (
        r_lub.at[:, 0].get() * vel_j.at[:, 0].get()
        + r_lub.at[:, 1].get() * vel_j.at[:, 1].get()
        + r_lub.at[:, 2].get() * vel_j.at[:, 2].get()
    )

    sgn = -1  # needed because we want to add the term (-R_SU*U) to the total stresslet
    # compute stresslet for particles i
    stresslet = stresslet.at[indices_i_lub, 0].add(
        sgn
        * (
            xg11 * (r_lub[:, 0] * r_lub[:, 0] - 1.0 / 3.0) * rdui
            + xg12 * (r_lub[:, 0] * r_lub[:, 0] - 1.0 / 3.0) * rduj
            + yg11
            * (
                vel_i[:, 0] * r_lub[:, 0]
                + r_lub[:, 0] * vel_i[:, 0]
                - 2.0 * r_lub[:, 0] * r_lub[:, 0] * rdui
            )
            + yg12
            * (
                vel_j[:, 0] * r_lub[:, 0]
                + r_lub[:, 0] * vel_j[:, 0]
                - 2.0 * r_lub[:, 0] * r_lub[:, 0] * rduj
            )
        )
    )
    stresslet = stresslet.at[indices_i_lub, 1].add(
        sgn
        * (
            xg11 * (r_lub[:, 0] * r_lub[:, 1]) * rdui
            + xg12 * (r_lub[:, 0] * r_lub[:, 1]) * rduj
            + yg11
            * (
                vel_i[:, 0] * r_lub[:, 1]
                + r_lub[:, 0] * vel_i[:, 1]
                - 2.0 * r_lub[:, 0] * r_lub[:, 1] * rdui
            )
            + yg12
            * (
                vel_j[:, 0] * r_lub[:, 1]
                + r_lub[:, 0] * vel_j[:, 1]
                - 2.0 * r_lub[:, 0] * r_lub[:, 1] * rduj
            )
        )
    )
    stresslet = stresslet.at[indices_i_lub, 2].add(
        sgn
        * (
            xg11 * (r_lub[:, 0] * r_lub[:, 2]) * rdui
            + xg12 * (r_lub[:, 0] * r_lub[:, 2]) * rduj
            + yg11
            * (
                vel_i[:, 0] * r_lub[:, 2]
                + r_lub[:, 0] * vel_i[:, 2]
                - 2.0 * r_lub[:, 0] * r_lub[:, 2] * rdui
            )
            + yg12
            * (
                vel_j[:, 0] * r_lub[:, 2]
                + r_lub[:, 0] * vel_j[:, 2]
                - 2.0 * r_lub[:, 0] * r_lub[:, 2] * rduj
            )
        )
    )
    stresslet = stresslet.at[indices_i_lub, 3].add(
        sgn
        * (
            xg11 * (r_lub[:, 1] * r_lub[:, 2]) * rdui
            + xg12 * (r_lub[:, 1] * r_lub[:, 2]) * rduj
            + yg11
            * (
                vel_i[:, 1] * r_lub[:, 2]
                + r_lub[:, 1] * vel_i[:, 2]
                - 2.0 * r_lub[:, 1] * r_lub[:, 2] * rdui
            )
            + yg12
            * (
                vel_j[:, 1] * r_lub[:, 2]
                + r_lub[:, 1] * vel_j[:, 2]
                - 2.0 * r_lub[:, 1] * r_lub[:, 2] * rduj
            )
        )
    )
    stresslet = stresslet.at[indices_i_lub, 4].add(
        sgn
        * (
            xg11 * (r_lub[:, 1] * r_lub[:, 1] - 1.0 / 3.0) * rdui
            + xg12 * (r_lub[:, 1] * r_lub[:, 1] - 1.0 / 3.0) * rduj
            + yg11
            * (
                vel_i[:, 1] * r_lub[:, 1]
                + r_lub[:, 1] * vel_i[:, 1]
                - 2.0 * r_lub[:, 1] * r_lub[:, 1] * rdui
            )
            + yg12
            * (
                vel_j[:, 1] * r_lub[:, 1]
                + r_lub[:, 1] * vel_j[:, 1]
                - 2.0 * r_lub[:, 1] * r_lub[:, 1] * rduj
            )
        )
    )

    # compute stresslet for particles j
    stresslet = stresslet.at[indices_j_lub, 0].add(
        sgn
        * (
            xg11 * (r_lub[:, 0] * r_lub[:, 0] - 1.0 / 3.0) * (-rduj)
            + xg12 * (r_lub[:, 0] * r_lub[:, 0] - 1.0 / 3.0) * (-rdui)
            + yg11
            * (
                -vel_j[:, 0] * r_lub[:, 0]
                - r_lub[:, 0] * vel_j[:, 0]
                + 2.0 * r_lub[:, 0] * r_lub[:, 0] * rduj
            )
            + yg12
            * (
                -vel_i[:, 0] * r_lub[:, 0]
                - r_lub[:, 0] * vel_i[:, 0]
                + 2.0 * r_lub[:, 0] * r_lub[:, 0] * rdui
            )
        )
    )
    stresslet = stresslet.at[indices_j_lub, 1].add(
        sgn
        * (
            xg11 * (r_lub[:, 0] * r_lub[:, 1]) * (-rduj)
            + xg12 * (r_lub[:, 0] * r_lub[:, 1]) * (-rdui)
            + yg11
            * (
                -vel_j[:, 0] * r_lub[:, 1]
                - r_lub[:, 0] * vel_j[:, 1]
                + 2.0 * r_lub[:, 0] * r_lub[:, 1] * rduj
            )
            + yg12
            * (
                -vel_i[:, 0] * r_lub[:, 1]
                - r_lub[:, 0] * vel_i[:, 1]
                + 2.0 * r_lub[:, 0] * r_lub[:, 1] * rdui
            )
        )
    )
    stresslet = stresslet.at[indices_j_lub, 2].add(
        sgn
        * (
            xg11 * (r_lub[:, 0] * r_lub[:, 2]) * (-rduj)
            + xg12 * (r_lub[:, 0] * r_lub[:, 2]) * (-rdui)
            + yg11
            * (
                -vel_j[:, 0] * r_lub[:, 2]
                - r_lub[:, 0] * vel_j[:, 2]
                + 2.0 * r_lub[:, 0] * r_lub[:, 2] * rduj
            )
            + yg12
            * (
                -vel_i[:, 0] * r_lub[:, 2]
                - r_lub[:, 0] * vel_i[:, 2]
                + 2.0 * r_lub[:, 0] * r_lub[:, 2] * rdui
            )
        )
    )
    stresslet = stresslet.at[indices_j_lub, 3].add(
        sgn
        * (
            xg11 * (r_lub[:, 1] * r_lub[:, 2]) * (-rduj)
            + xg12 * (r_lub[:, 1] * r_lub[:, 2]) * (-rdui)
            + yg11
            * (
                -vel_j[:, 1] * r_lub[:, 2]
                - r_lub[:, 1] * vel_j[:, 2]
                + 2.0 * r_lub[:, 1] * r_lub[:, 2] * rduj
            )
            + yg12
            * (
                -vel_i[:, 1] * r_lub[:, 2]
                - r_lub[:, 1] * vel_i[:, 2]
                + 2.0 * r_lub[:, 1] * r_lub[:, 2] * rdui
            )
        )
    )
    stresslet = stresslet.at[indices_j_lub, 4].add(
        sgn
        * (
            xg11 * (r_lub[:, 1] * r_lub[:, 1] - 1.0 / 3.0) * (-rduj)
            + xg12 * (r_lub[:, 1] * r_lub[:, 1] - 1.0 / 3.0) * (-rdui)
            + yg11
            * (
                -vel_j[:, 1] * r_lub[:, 1]
                - r_lub[:, 1] * vel_j[:, 1]
                + 2.0 * r_lub[:, 1] * r_lub[:, 1] * rduj
            )
            + yg12
            * (
                -vel_i[:, 1] * r_lub[:, 1]
                - r_lub[:, 1] * vel_i[:, 1]
                + 2.0 * r_lub[:, 1] * r_lub[:, 1] * rdui
            )
        )
    )

    epsrdwi = jnp.array(
        [
            r_lub.at[:, 2].get() * vel_i.at[:, 4].get()
            - r_lub.at[:, 1].get() * vel_i.at[:, 5].get(),
            -r_lub.at[:, 2].get() * vel_i.at[:, 3].get()
            + r_lub.at[:, 0].get() * vel_i.at[:, 5].get(),
            r_lub.at[:, 1].get() * vel_i.at[:, 3].get()
            - r_lub.at[:, 0].get() * vel_i.at[:, 4].get(),
        ]
    )

    epsrdwj = jnp.array(
        [
            r_lub.at[:, 2].get() * vel_j.at[:, 4].get()
            - r_lub.at[:, 1].get() * vel_j.at[:, 5].get(),
            -r_lub.at[:, 2].get() * vel_j.at[:, 3].get()
            + r_lub.at[:, 0].get() * vel_j.at[:, 5].get(),
            r_lub.at[:, 1].get() * vel_j.at[:, 3].get()
            - r_lub.at[:, 0].get() * vel_j.at[:, 4].get(),
        ]
    )

    # compute stresslet for particles i
    stresslet = stresslet.at[indices_i_lub, 0].add(
        sgn
        * (
            yh11 * (r_lub[:, 0] * epsrdwi[0] + epsrdwi[0] * r_lub[:, 0])
            + yh12 * (r_lub[:, 0] * epsrdwj[0] + epsrdwj[0] * r_lub[:, 0])
        )
    )
    stresslet = stresslet.at[indices_i_lub, 1].add(
        sgn
        * (
            yh11 * (r_lub[:, 0] * epsrdwi[1] + epsrdwi[0] * r_lub[:, 1])
            + yh12 * (r_lub[:, 0] * epsrdwj[1] + epsrdwj[0] * r_lub[:, 1])
        )
    )
    stresslet = stresslet.at[indices_i_lub, 2].add(
        sgn
        * (
            yh11 * (r_lub[:, 0] * epsrdwi[2] + epsrdwi[0] * r_lub[:, 2])
            + yh12 * (r_lub[:, 0] * epsrdwj[2] + epsrdwj[0] * r_lub[:, 2])
        )
    )
    stresslet = stresslet.at[indices_i_lub, 3].add(
        sgn
        * (
            yh11 * (r_lub[:, 1] * epsrdwi[2] + epsrdwi[1] * r_lub[:, 2])
            + yh12 * (r_lub[:, 1] * epsrdwj[2] + epsrdwj[1] * r_lub[:, 2])
        )
    )
    stresslet = stresslet.at[indices_i_lub, 4].add(
        sgn
        * (
            yh11 * (r_lub[:, 2] * epsrdwi[2] + epsrdwi[2] * r_lub[:, 2])
            + yh12 * (r_lub[:, 2] * epsrdwj[2] + epsrdwj[2] * r_lub[:, 2])
        )
    )

    # compute stresslet for particles j
    stresslet = stresslet.at[indices_j_lub, 0].add(
        sgn
        * (
            yh11 * (r_lub[:, 0] * epsrdwj[0] + epsrdwj[0] * r_lub[:, 0])
            + yh12 * (r_lub[:, 0] * epsrdwi[0] + epsrdwi[0] * r_lub[:, 0])
        )
    )
    stresslet = stresslet.at[indices_j_lub, 1].add(
        sgn
        * (
            yh11 * (r_lub[:, 0] * epsrdwj[1] + epsrdwj[0] * r_lub[:, 1])
            + yh12 * (r_lub[:, 0] * epsrdwi[1] + epsrdwi[0] * r_lub[:, 1])
        )
    )
    stresslet = stresslet.at[indices_j_lub, 2].add(
        sgn
        * (
            yh11 * (r_lub[:, 0] * epsrdwj[2] + epsrdwj[0] * r_lub[:, 2])
            + yh12 * (r_lub[:, 0] * epsrdwi[2] + epsrdwi[0] * r_lub[:, 2])
        )
    )
    stresslet = stresslet.at[indices_j_lub, 3].add(
        sgn
        * (
            yh11 * (r_lub[:, 1] * epsrdwj[2] + epsrdwj[1] * r_lub[:, 2])
            + yh12 * (r_lub[:, 1] * epsrdwi[2] + epsrdwi[1] * r_lub[:, 2])
        )
    )
    stresslet = stresslet.at[indices_j_lub, 4].add(
        sgn
        * (
            yh11 * (r_lub[:, 2] * epsrdwj[2] + epsrdwj[2] * r_lub[:, 2])
            + yh12 * (r_lub[:, 2] * epsrdwi[2] + epsrdwi[2] * r_lub[:, 2])
        )
    )

    return stresslet