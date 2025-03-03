from functools import partial

import jax.numpy as jnp
from jax import Array, jit, ops
from jax.typing import ArrayLike
from jfsd import utils

@partial(jit, static_argnums=[0])
def sum_applied_forces(
    num_particles: int,
    constant_forces: ArrayLike,
    constant_torques: ArrayLike,
    saddle_b: ArrayLike,
    interaction_strength: float,
    indices_i: ArrayLike,
    indices_j: ArrayLike,
    positions: ArrayLike,
    interaction_cutoff: float,
    hydrodynamics_flag: int,
    dt: float,
    box: ArrayLike
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
    positions: (float)
        Array (num_particles,3) of particle positions
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
        interaction_strength: float, indices_i: ArrayLike, indices_j: ArrayLike
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
        
        Returns
        -------
        fp

        """
        fp = jnp.zeros((num_particles, 3))
        sigma = 2.0  # particle diameter
        # 1% shift to avoid overlaps for the hydrodynamic integrator (and non-positive def operators)
        sigma += sigma * 0.01
        # compute sigma/delta_r for each pair
        sigmdr = sigma / jnp.where(indices_i != indices_j, dist_mod, 0.0)
        sigmdr = jnp.power(sigmdr, 48)
        # compute forces for each pair
        fp_mod = (
            96
            * interaction_strength
            / (dist_mod * dist_mod)
            * sigmdr
            * (1 - sigmdr)
        )
        fp_mod = jnp.where((dist_mod) > sigma * interaction_cutoff, 0.0, fp_mod)
        fp += ops.segment_sum(fp_mod[:,None] * dist, indices_i, num_particles)  # Add contributions from i
        fp -= ops.segment_sum(fp_mod[:,None] * dist, indices_j, num_particles)  # Subtract contributions from j
        return fp

    def compute_asakura_oosawa_vrij(
        interaction_strength: float, indices_i: ArrayLike, indices_j: ArrayLike
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

        Returns
        -------
        fp

        """
        fp = jnp.zeros((num_particles, 3))
        # sum of polymer and colloid radii (in unit of colloid radius)
        onepdelta = 1.1
        diameter = 2.002
        alpha = diameter * onepdelta
        # compute forces for each pair
        fp_mod = (
            interaction_strength
            * 3
            * (-alpha * alpha + dist_sqr)
            / (
                2 * alpha * alpha * alpha
                - 3 * alpha * alpha * diameter
                + diameter * diameter * diameter
            )
        )
        fp_mod = jnp.where((dist_sqr) <= (diameter * diameter), 0.0, fp_mod)
        fp_mod = jnp.where(
            (dist_sqr) >= (4.0 * onepdelta * onepdelta), 0.0, fp_mod
        )
        fp_mod = -fp_mod / jnp.sqrt(dist_sqr)
        fp += ops.segment_sum(fp_mod[:,None] * dist, indices_i, num_particles)  # Add contributions from i
        fp -= ops.segment_sum(fp_mod[:,None] * dist, indices_j, num_particles)  # Subtract contributions from j
        return fp

    def compute_hs_forces(
        indices_i: ArrayLike, indices_j: ArrayLike, dt: float
    ) -> Array:
        """Compute repulsive hard-sphere pair interactions using an asymmetric harmonic potential.

        Parameters
        ----------
        indices_i: (int)
            Array (n_pair) of indices of first particle in neighbor list pairs
        indices_j: (int)
            Array (n_pair) of indices of second particle in neighbor list pairs
        
        Returns
        -------
        fp

        """
        fp = jnp.zeros((num_particles, 3))
        # particle diameter (shifted of 1% to help numerical stability)
        sigma = 2.0 * (1.001)

        # compute forces for each pair
        # spring constant must be calibrated to exactly remove the current overlap
        # with lubrication hydrodynamic this is ~ o(1000) because of divergent (at contact) effective drag coeff
        k = jnp.where(hydrodynamics_flag > 1, (2500.839791) / dt, 1 / dt)

        fp_mod = jnp.where(
            indices_i != indices_j, k * (1 - sigma / dist_mod), 0.0
        )
        fp_mod = jnp.where((dist_mod) < sigma, fp_mod, 0.0)
        fp += ops.segment_sum(fp_mod[:,None] * dist, indices_i, num_particles)  # Add contributions from i
        fp -= ops.segment_sum(fp_mod[:,None] * dist, indices_j, num_particles)  # Subtract contributions from j
        return fp
    
    dist = utils.displacement_fn(positions[indices_i,:], positions[indices_j,:], box)    
    dist_sqr = dist[:,0]*dist[:,0]+ dist[:,1]*dist[:,1]+ dist[:,2]*dist[:,2]
    dist_mod = jnp.sqrt(dist_sqr)
    # compute hard sphere repulsion, and short-range attractions
    hs_Force = compute_hs_forces(indices_i, indices_j, dt)
    aov_force = compute_asakura_oosawa_vrij(interaction_strength, indices_i, indices_j)

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
