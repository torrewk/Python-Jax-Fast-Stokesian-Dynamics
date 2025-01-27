from functools import partial

import jax.numpy as jnp
from jax import Array, jit
from jax.typing import ArrayLike


@partial(jit, static_argnums=[0])
def sum_applied_forces(
    num_particles: int,
    constant_forces: ArrayLike,
    constant_torques: ArrayLike,
    saddle_b: ArrayLike,
    interaction_strength: float,
    indices_i: ArrayLike,
    indices_j: ArrayLike,
    dist: ArrayLike,
    interaction_cutoff: float,
    hydrodynamics_flag: int,
    dt: float,
) -> Array:
    """Sum all applied forces/torques for each particle.

    Take into account:
        external forces/torques,
        hard-sphere repulsions,
        short-range attractions

    Parameters
    ----------
    num_particles: (int)
        Number of particles
    constant_forces: (float)
        Array (num_particles,3) of applied (external) forces, e.g. buoyancy
    constant_torques: (float)
        Array (num_particles,3) of applied (external) torques
    saddle_b: (float)
        Right-hand side vector (17*num_particles) of saddle point system Ax=b
    interaction_strength: (float)
        Energy of a single colloidal bond
    indices_i: (int)
        Array (n_pair) of indices of first particle in neighbor list pairs
    indices_j: (int)
        Array (n_pair) of indices of second particle in neighbor list pairs
    dist: (float)
        Array (n_pair,3) of distance vectors between particles in neighbor list
    interaction_cutoff: (float)
        Cutoff (max) distance for pair-interactions
    hydrodynamics_flag: (int)
        Flag used to set level of hydrodynamic interaction. 0 for BD, 1 for SD.
    dt: (float)
        Timestep. Needed to compute 'potential-free' hard sphere repulsion

    Returns
    -------
    saddle_b

    """

    def compute_lj_highexponent(
        interaction_strength: float, indices_i: ArrayLike, indices_j: ArrayLike, dist: ArrayLike
    ) -> Array:
        """Compute pair interactions using a "high exponent" Lennard-Jones potential (attractions+repulsions)

        Parameters
        ----------
        interaction_strength: (float)
            Energy of a single colloidal bond
        indices_i: (int)
            Array (n_pair) of indices of first particle in neighbor list pairs
        indices_j: (int)
            Array (n_pair) of indices of second particle in neighbor list pairs
        dist: (float)
            Array (n_pair,3) of distance vectors between particles in neighbor list

        Returns
        -------
        fp

        """
        fp = jnp.zeros((num_particles, num_particles, 3))
        dist_mod = jnp.sqrt(
            dist[:, :, 0] * dist[:, :, 0]
            + dist[:, :, 1] * dist[:, :, 1]
            + dist[:, :, 2] * dist[:, :, 2]
        )
        sigma = 2.0  # particle diameter
        # 1% shift to avoid overlaps for the hydrodynamic integrator (and non-positive def operators)
        sigma += sigma * 0.01

        # compute sigma/delta_r for each pair
        sigmdr = sigma / jnp.where(indices_i != indices_j, dist_mod[indices_i, indices_j], 0.0)
        sigmdr = jnp.power(sigmdr, 48)

        # compute forces for each pair
        fp_mod = (
            96
            * interaction_strength
            / (dist_mod[indices_i, indices_j] * dist_mod[indices_i, indices_j])
            * sigmdr
            * (1 - sigmdr)
        )
        fp_mod = jnp.where((dist_mod[indices_i, indices_j]) > sigma * interaction_cutoff, 0.0, fp_mod)

        # get forces in components
        fp = fp.at[indices_i, indices_j, 0].add(fp_mod * dist[indices_i, indices_j, 0])
        fp = fp.at[indices_i, indices_j, 1].add(fp_mod * dist[indices_i, indices_j, 1])
        fp = fp.at[indices_i, indices_j, 2].add(fp_mod * dist[indices_i, indices_j, 2])
        fp = fp.at[indices_j, indices_i, 0].add(fp_mod * dist[indices_j, indices_i, 0])
        fp = fp.at[indices_j, indices_i, 1].add(fp_mod * dist[indices_j, indices_i, 1])
        fp = fp.at[indices_j, indices_i, 2].add(fp_mod * dist[indices_j, indices_i, 2])

        # sum all forces in each particle
        fp = jnp.sum(fp, 1)

        return fp

    def compute_asakura_oosawa_vrij(
        interaction_strength: float, indices_i: ArrayLike, indices_j: ArrayLike, dist: ArrayLike
    ) -> Array:
        """Compute attractive pair interactions using an Asakura-Osawa potential.

        Parameters
        ----------
        interaction_strength: (float)
            Energy of a single colloidal bond
        indices_i: (int)
            Array (n_pair) of indices of first particle in neighbor list pairs
        indices_j: (int)
            Array (n_pair) of indices of second particle in neighbor list pairs
        dist: (float)
            Array (n_pair,3) of distance vectors between particles in neighbor list

        Returns
        -------
        fp

        """
        fp = jnp.zeros((num_particles, num_particles, 3))
        dist_sqr = (
            dist[:, :, 0] * dist[:, :, 0]
            + dist[:, :, 1] * dist[:, :, 1]
            + dist[:, :, 2] * dist[:, :, 2]
        )
        # sum of polymer and colloid radii (in unit of colloid radius)
        onepdelta = 1.1
        diameter = 2.002
        alpha = diameter * onepdelta
        # compute forces for each pair
        # fp_mod = interaction_strength * (-3*onepdelta*onepdelta/2 + 3 / 8 * dist_sqr[indices_i, indices_j]) / (
        #     2+onepdelta*onepdelta*onepdelta-3*onepdelta*onepdelta+1)

        fp_mod = (
            interaction_strength
            * 3
            * (-alpha * alpha + dist_sqr[indices_i, indices_j])
            / (
                2 * alpha * alpha * alpha
                - 3 * alpha * alpha * diameter
                + diameter * diameter * diameter
            )
        )

        fp_mod = jnp.where((dist_sqr[indices_i, indices_j]) <= (diameter * diameter), 0.0, fp_mod)
        fp_mod = jnp.where(
            (dist_sqr[indices_i, indices_j]) >= (4.0 * onepdelta * onepdelta), 0.0, fp_mod
        )
        fp_mod = -fp_mod / jnp.sqrt(dist_sqr[indices_i, indices_j])

        # get forces in components
        fp = fp.at[indices_i, indices_j, 0].add(fp_mod * dist[indices_i, indices_j, 0])
        fp = fp.at[indices_i, indices_j, 1].add(fp_mod * dist[indices_i, indices_j, 1])
        fp = fp.at[indices_i, indices_j, 2].add(fp_mod * dist[indices_i, indices_j, 2])
        fp = fp.at[indices_j, indices_i, 0].add(fp_mod * dist[indices_j, indices_i, 0])
        fp = fp.at[indices_j, indices_i, 1].add(fp_mod * dist[indices_j, indices_i, 1])
        fp = fp.at[indices_j, indices_i, 2].add(fp_mod * dist[indices_j, indices_i, 2])

        # sum all forces in each particle
        fp = jnp.sum(fp, 1)

        return fp

    def compute_hs_forces(
        indices_i: ArrayLike, indices_j: ArrayLike, dist: ArrayLike, dt: float
    ) -> Array:
        """Compute repulsive hard-sphere pair interactions using an asymmetric harmonic potential.

        Parameters
        ----------
        indices_i: (int)
            Array (n_pair) of indices of first particle in neighbor list pairs
        indices_j: (int)
            Array (n_pair) of indices of second particle in neighbor list pairs
        dist: (float)
            Array (n_pair,3) of distance vectors between particles in neighbor list

        Returns
        -------
        fp

        """
        fp = jnp.zeros((num_particles, num_particles, 3))
        dist_mod = jnp.sqrt(
            dist[:, :, 0] * dist[:, :, 0]
            + dist[:, :, 1] * dist[:, :, 1]
            + dist[:, :, 2] * dist[:, :, 2]
        )
        # particle diameter (shifted of 1% to help numerical stability)
        sigma = 2.0 * (1.001)

        # compute forces for each pair
        # spring constant must be calibrated to exactly remove the current overlap
        # with lubrication hydrodynamic this is ~ o(1000) because of divergent (at contact) effective drag coeff
        k = jnp.where(hydrodynamics_flag > 1, (2500.839791) / dt, 1 / dt)

        fp_mod = jnp.where(
            indices_i != indices_j, k * (1 - sigma / dist_mod[indices_i, indices_j]), 0.0
        )
        fp_mod = jnp.where((dist_mod[indices_i, indices_j]) < sigma, fp_mod, 0.0)

        # get forces in components
        fp = fp.at[indices_i, indices_j, 0].add(fp_mod * dist[indices_i, indices_j, 0])
        fp = fp.at[indices_i, indices_j, 1].add(fp_mod * dist[indices_i, indices_j, 1])
        fp = fp.at[indices_i, indices_j, 2].add(fp_mod * dist[indices_i, indices_j, 2])
        fp = fp.at[indices_j, indices_i, 0].add(fp_mod * dist[indices_j, indices_i, 0])
        fp = fp.at[indices_j, indices_i, 1].add(fp_mod * dist[indices_j, indices_i, 1])
        fp = fp.at[indices_j, indices_i, 2].add(fp_mod * dist[indices_j, indices_i, 2])

        # sum all forces in each particle
        fp = jnp.sum(fp, 1)

        return fp

    # compute hard sphere repulsion, and short-range attractions
    hs_Force = compute_hs_forces(indices_i, indices_j, dist, dt)
    aov_force = compute_asakura_oosawa_vrij(interaction_strength, indices_i, indices_j, dist)

    # add imposed (-forces) to rhs of linear system
    saddle_b = saddle_b.at[(11 * num_particles + 0) :: 6].add(
        -constant_forces.at[0::3].get() - hs_Force.at[:, 0].get() - aov_force.at[:, 0].get()
    )
    saddle_b = saddle_b.at[(11 * num_particles + 1) :: 6].add(
        -constant_forces.at[1::3].get() - hs_Force.at[:, 1].get() - aov_force.at[:, 1].get()
    )
    saddle_b = saddle_b.at[(11 * num_particles + 2) :: 6].add(
        -constant_forces.at[2::3].get() - hs_Force.at[:, 2].get() - aov_force.at[:, 2].get()
    )

    # add imposed (-torques) to rhs of linear system
    saddle_b = saddle_b.at[(11 * num_particles + 3) :: 6].add(-constant_torques.at[0::3].get())
    saddle_b = saddle_b.at[(11 * num_particles + 4) :: 6].add(-constant_torques.at[1::3].get())
    saddle_b = saddle_b.at[(11 * num_particles + 5) :: 6].add(-constant_torques.at[2::3].get())
    # if there are no HIs, divide torques by rotational drag coeff (not done for forces as the translational drag coeff is set to 1 in simulation units)
    saddle_b = saddle_b.at[(11 * num_particles + 3) :: 6].set(
        jnp.where(
            hydrodynamics_flag > 0,
            saddle_b.at[(11 * num_particles + 3) :: 6].get(),
            saddle_b.at[(11 * num_particles + 3) :: 6].get() * 3 / 4,
        )
    )
    saddle_b = saddle_b.at[(11 * num_particles + 4) :: 6].set(
        jnp.where(
            hydrodynamics_flag > 0,
            saddle_b.at[(11 * num_particles + 4) :: 6].get(),
            saddle_b.at[(11 * num_particles + 4) :: 6].get() * 3 / 4,
        )
    )
    saddle_b = saddle_b.at[(11 * num_particles + 5) :: 6].set(
        jnp.where(
            hydrodynamics_flag > 0,
            saddle_b.at[(11 * num_particles + 5) :: 6].get(),
            saddle_b.at[(11 * num_particles + 5) :: 6].get() * 3 / 4,
        )
    )

    return saddle_b