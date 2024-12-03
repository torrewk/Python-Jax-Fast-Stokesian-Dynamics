from functools import partial
from jax.typing import ArrayLike
import jax.numpy as jnp
from jax import jit, Array


@partial(jit, static_argnums=[0])
def sumAppliedForces(
    N: int,
    AppliedForce: ArrayLike,
    AppliedTorques: ArrayLike,
    saddle_b: ArrayLike,
    U: float,
    indices_i: ArrayLike,
    indices_j: ArrayLike,
    dist: ArrayLike,
    Ucutoff: float,
    HIs_flag: int,
    dt: float,
) -> Array:
    """Sum all applied forces/torques for each particle.

    Take into account:
        external forces/torques,
        hard-sphere repulsions,
        short-range attractions

    Parameters
    ----------
    N: (int)
        Number of particles
    AppliedForce: (float)
        Array (N,3) of applied (external) forces, e.g. buoyancy
    AppliedTorques: (float)
        Array (N,3) of applied (external) torques
    saddle_b: (float)
        Right-hand side vector (17*N) of saddle point system Ax=b
    U: (float)
        Energy of a single colloidal bond
    indices_i: (int)
        Array (n_pair) of indices of first particle in neighbor list pairs
    indices_j: (int)
        Array (n_pair) of indices of second particle in neighbor list pairs
    dist: (float)
        Array (n_pair,3) of distance vectors between particles in neighbor list
    Ucutoff: (float)
        Cutoff (max) distance for pair-interactions
    HIs_flag: (int)
        Flag used to set level of hydrodynamic interaction. 0 for BD, 1 for SD.
    dt: (float)
        Timestep. Needed to compute 'potential-free' hard sphere repulsion

    Returns
    -------
    saddle_b

    """

    def compute_LJhe_potentialforces(
        U: float, indices_i: ArrayLike, indices_j: ArrayLike, dist: ArrayLike
    ) -> Array:
        """Compute pair interactions using a "high exponent" Lennard-Jones potential (attractions+repulsions)

        Parameters
        ----------
        U: (float)
            Energy of a single colloidal bond
        indices_i: (int)
            Array (n_pair) of indices of first particle in neighbor list pairs
        indices_j: (int)
            Array (n_pair) of indices of second particle in neighbor list pairs
        dist: (float)
            Array (n_pair,3) of distance vectors between particles in neighbor list

        Returns
        -------
        Fp

        """
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
        sigmdr = sigma / jnp.where(indices_i != indices_j, dist_mod[indices_i, indices_j], 0.0)
        sigmdr = jnp.power(sigmdr, 48)

        # compute forces for each pair
        Fp_mod = (
            96
            * U
            / (dist_mod[indices_i, indices_j] * dist_mod[indices_i, indices_j])
            * sigmdr
            * (1 - sigmdr)
        )
        Fp_mod = jnp.where((dist_mod[indices_i, indices_j]) > sigma * Ucutoff, 0.0, Fp_mod)

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

    def compute_AO_potentialforces(
        U: float, indices_i: ArrayLike, indices_j: ArrayLike, dist: ArrayLike
    ) -> Array:
        """Compute attractive pair interactions using an Asakura-Osawa potential.

        Parameters
        ----------
        U: (float)
            Energy of a single colloidal bond
        indices_i: (int)
            Array (n_pair) of indices of first particle in neighbor list pairs
        indices_j: (int)
            Array (n_pair) of indices of second particle in neighbor list pairs
        dist: (float)
            Array (n_pair,3) of distance vectors between particles in neighbor list

        Returns
        -------
        Fp

        """
        Fp = jnp.zeros((N, N, 3))
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
        # Fp_mod = U * (-3*onepdelta*onepdelta/2 + 3 / 8 * dist_sqr[indices_i, indices_j]) / (
        #     2+onepdelta*onepdelta*onepdelta-3*onepdelta*onepdelta+1)

        Fp_mod = (
            U
            * 3
            * (-alpha * alpha + dist_sqr[indices_i, indices_j])
            / (
                2 * alpha * alpha * alpha
                - 3 * alpha * alpha * diameter
                + diameter * diameter * diameter
            )
        )

        Fp_mod = jnp.where((dist_sqr[indices_i, indices_j]) <= (diameter * diameter), 0.0, Fp_mod)
        Fp_mod = jnp.where(
            (dist_sqr[indices_i, indices_j]) >= (4.0 * onepdelta * onepdelta), 0.0, Fp_mod
        )
        Fp_mod = -Fp_mod / jnp.sqrt(dist_sqr[indices_i, indices_j])

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
        Fp

        """
        Fp = jnp.zeros((N, N, 3))
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
        k = jnp.where(HIs_flag > 1, (2500.839791) / dt, 1 / dt)

        Fp_mod = jnp.where(
            indices_i != indices_j, k * (1 - sigma / dist_mod[indices_i, indices_j]), 0.0
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

    # compute hard sphere repulsion, and short-range attractions
    hs_Force = compute_hs_forces(indices_i, indices_j, dist, dt)
    AO_force = compute_AO_potentialforces(U, indices_i, indices_j, dist)

    # add imposed (-forces) to rhs of linear system
    saddle_b = saddle_b.at[(11 * N + 0) :: 6].add(
        -AppliedForce.at[0::3].get() - hs_Force.at[:, 0].get() - AO_force.at[:, 0].get()
    )
    saddle_b = saddle_b.at[(11 * N + 1) :: 6].add(
        -AppliedForce.at[1::3].get() - hs_Force.at[:, 1].get() - AO_force.at[:, 1].get()
    )
    saddle_b = saddle_b.at[(11 * N + 2) :: 6].add(
        -AppliedForce.at[2::3].get() - hs_Force.at[:, 2].get() - AO_force.at[:, 2].get()
    )

    # add imposed (-torques) to rhs of linear system
    saddle_b = saddle_b.at[(11 * N + 3) :: 6].add(-AppliedTorques.at[0::3].get())
    saddle_b = saddle_b.at[(11 * N + 4) :: 6].add(-AppliedTorques.at[1::3].get())
    saddle_b = saddle_b.at[(11 * N + 5) :: 6].add(-AppliedTorques.at[2::3].get())
    # if there are no HIs, divide torques by rotational drag coeff (not done for forces as the translational drag coeff is set to 1 in simulation units)
    saddle_b = saddle_b.at[(11 * N + 3) :: 6].set(
        jnp.where(
            HIs_flag > 0,
            saddle_b.at[(11 * N + 3) :: 6].get(),
            saddle_b.at[(11 * N + 3) :: 6].get() * 3 / 4,
        )
    )
    saddle_b = saddle_b.at[(11 * N + 4) :: 6].set(
        jnp.where(
            HIs_flag > 0,
            saddle_b.at[(11 * N + 4) :: 6].get(),
            saddle_b.at[(11 * N + 4) :: 6].get() * 3 / 4,
        )
    )
    saddle_b = saddle_b.at[(11 * N + 5) :: 6].set(
        jnp.where(
            HIs_flag > 0,
            saddle_b.at[(11 * N + 5) :: 6].get(),
            saddle_b.at[(11 * N + 5) :: 6].get() * 3 / 4,
        )
    )

    return saddle_b
