import os
from functools import partial

import jax.numpy as jnp
from jax import jit
from jax.config import config

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
config.update("jax_enable_x64", False)


@partial(jit, static_argnums=[0])
def sumAppliedForces(
    N, AppliedForce, AppliedTorques, saddle_b, U, indices_i, indices_j, dist, Ucutoff
):

    def compute_LJhe_potentialforces(U, indices_i, indices_j, dist):
        Fp = jnp.zeros((N, N, 3))
        dist_mod = jnp.sqrt(
            dist[:, :, 0] * dist[:, :, 0]
            + dist[:, :, 1] * dist[:, :, 1]
            + dist[:, :, 2] * dist[:, :, 2]
        )
        sigma = 2.0  # particle diameter
        # 1% shift to avoid overlaps for the hydrodynamic integrator (and non-positive def operators)
        sigma += sigma * 0.01

        # compute sigma/delta_r for each pair
        sigmdr = sigma / jnp.where(
            indices_i != indices_j, dist_mod[indices_i, indices_j], 0.0
        )
        sigmdr = jnp.power(sigmdr, 48)

        # compute forces for each pair
        Fp_mod = (
            96
            * U
            / (dist_mod[indices_i, indices_j] * dist_mod[indices_i, indices_j])
            * sigmdr
            * (1 - sigmdr)
        )
        Fp_mod = jnp.where(
            (dist_mod[indices_i, indices_j]) > sigma * Ucutoff, 0.0, Fp_mod
        )

        # get forces in components
        Fp = Fp.at[indices_i, indices_j, 0].add(Fp_mod * dist[indices_i, indices_j, 0])
        Fp = Fp.at[indices_i, indices_j, 1].add(Fp_mod * dist[indices_i, indices_j, 1])
        Fp = Fp.at[indices_i, indices_j, 2].add(Fp_mod * dist[indices_i, indices_j, 2])
        Fp = Fp.at[indices_j, indices_i, 0].add(Fp_mod * dist[indices_j, indices_i, 0])
        Fp = Fp.at[indices_j, indices_i, 1].add(Fp_mod * dist[indices_j, indices_i, 1])
        Fp = Fp.at[indices_j, indices_i, 2].add(Fp_mod * dist[indices_j, indices_i, 2])

        # sum all forces in each particle
        Fp = jnp.sum(Fp, 1)

        return Fp

    def compute_AO_potentialforces(U, indices_i, indices_j, dist):
        Fp = jnp.zeros((N, N, 3))
        dist_sqr = (
            dist[:, :, 0] * dist[:, :, 0]
            + dist[:, :, 1] * dist[:, :, 1]
            + dist[:, :, 2] * dist[:, :, 2]
        )
        # sum of polymer and colloid radii (in unit of colloid radius)
        onepdelta = 1.01

        # compute forces for each pair
        Fp_mod = (
            U
            * (-3 * onepdelta * onepdelta / 2 + 3 / 8 * dist_sqr[indices_i, indices_j])
            / (2 + onepdelta * onepdelta * onepdelta - 3 * onepdelta * onepdelta + 1)
        )
        Fp_mod = jnp.where((dist_sqr[indices_i, indices_j]) <= 4.0, 0.0, Fp_mod)
        Fp_mod = jnp.where(
            (dist_sqr[indices_i, indices_j]) >= (4.0 * onepdelta * onepdelta),
            0.0,
            Fp_mod,
        )
        Fp_mod = Fp_mod / jnp.sqrt(dist_sqr[indices_i, indices_j])

        # get forces in components
        Fp = Fp.at[indices_i, indices_j, 0].add(Fp_mod * dist[indices_i, indices_j, 0])
        Fp = Fp.at[indices_i, indices_j, 1].add(Fp_mod * dist[indices_i, indices_j, 1])
        Fp = Fp.at[indices_i, indices_j, 2].add(Fp_mod * dist[indices_i, indices_j, 2])
        Fp = Fp.at[indices_j, indices_i, 0].add(Fp_mod * dist[indices_j, indices_i, 0])
        Fp = Fp.at[indices_j, indices_i, 1].add(Fp_mod * dist[indices_j, indices_i, 1])
        Fp = Fp.at[indices_j, indices_i, 2].add(Fp_mod * dist[indices_j, indices_i, 2])

        # sum all forces in each particle
        Fp = jnp.sum(Fp, 1)

        return Fp

    def compute_hs_forces(indices_i, indices_j, dist):
        Fp = jnp.zeros((N, N, 3))
        dist_mod = jnp.sqrt(
            dist[:, :, 0] * dist[:, :, 0]
            + dist[:, :, 1] * dist[:, :, 1]
            + dist[:, :, 2] * dist[:, :, 2]
        )
        # particle diameter (shifted of 1% to help numerical stability)
        sigma = 2.0 * (1.01)

        # compute forces for each pair
        # relax constant (If lubrication is not on, k should to be ~ o(1) )
        k = 1000.0
        Fp_mod = jnp.where(
            indices_i != indices_j,
            k * (1 - sigma / dist_mod[indices_i, indices_j]),
            0.0,
        )
        Fp_mod = jnp.where((dist_mod[indices_i, indices_j]) < sigma, Fp_mod, 0.0)

        # get forces in components
        Fp = Fp.at[indices_i, indices_j, 0].add(Fp_mod * dist[indices_i, indices_j, 0])
        Fp = Fp.at[indices_i, indices_j, 1].add(Fp_mod * dist[indices_i, indices_j, 1])
        Fp = Fp.at[indices_i, indices_j, 2].add(Fp_mod * dist[indices_i, indices_j, 2])
        Fp = Fp.at[indices_j, indices_i, 0].add(Fp_mod * dist[indices_j, indices_i, 0])
        Fp = Fp.at[indices_j, indices_i, 1].add(Fp_mod * dist[indices_j, indices_i, 1])
        Fp = Fp.at[indices_j, indices_i, 2].add(Fp_mod * dist[indices_j, indices_i, 2])

        # sum all forces in each particle
        Fp = jnp.sum(Fp, 1)

        return Fp

    hs_Force = compute_hs_forces(indices_i, indices_j, dist)
    AO_force = compute_AO_potentialforces(U, indices_i, indices_j, dist)

    saddle_b = saddle_b.at[(11 * N + 0) :: 6].add(
        -AppliedForce.at[0::3].get() - hs_Force.at[:, 0].get() - AO_force.at[:, 0].get()
    )  # Add imposed (-forces) to rhs of linear system
    saddle_b = saddle_b.at[(11 * N + 1) :: 6].add(
        -AppliedForce.at[1::3].get() - hs_Force.at[:, 1].get() - AO_force.at[:, 1].get()
    )
    saddle_b = saddle_b.at[(11 * N + 2) :: 6].add(
        -AppliedForce.at[2::3].get() - hs_Force.at[:, 2].get() - AO_force.at[:, 2].get()
    )

    # Add imposed (-torques) to rhs of linear system
    saddle_b = saddle_b.at[(11 * N + 3) :: 6].add(-AppliedTorques.at[0::3].get())
    saddle_b = saddle_b.at[(11 * N + 4) :: 6].add(-AppliedTorques.at[1::3].get())
    saddle_b = saddle_b.at[(11 * N + 5) :: 6].add(-AppliedTorques.at[2::3].get())

    return saddle_b
