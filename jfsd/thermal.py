from functools import partial

import jax.numpy as jnp
import numpy as np
import scipy
from jax import Array, jit
from jax.typing import ArrayLike

from jfsd import lanczos


def random_force_on_grid_indexing(
    grid_nx: int, grid_ny: int, grid_nz: int
) -> tuple[int, int, int, int, int, int, int, int, int]:
    """Compute indexing for wave space grid.

    Relevant for wave space calculation of thermal fluctuations.

    Parameters
    ----------
    grid_nx: (int)
        Number of grid points in x direction
    grid_ny: (int)
        Number of grid points in y direction
    grid_nz: (int)
        Number of grid points in z direction

    Returns
    -------
    normal_indices_x,normal_indices_y,normal_indices_z, normal_conj_indices_x,normal_conj_indices_y,normal_conj_indices_z, nyquist_indices_x,nyquist_indices_y,nyquist_indices_z

    """
    normal_indices = []
    normal_conj_indices = []
    nyquist_indices = []
    for i in range(grid_nx):
        for j in range(grid_ny):
            for k in range(grid_nz):
                if (
                    not (2 * k >= grid_nz + 1)
                    and not ((k == 0) and (2 * j >= grid_ny + 1))
                    and not ((k == 0) and (j == 0) and (2 * i >= grid_nx + 1))
                    and not ((k == 0) and (j == 0) and (i == 0))
                ):
                    ii_nyquist = (i == int(grid_nx / 2)) and (int(grid_nx / 2) == int((grid_nx + 1) / 2))
                    jj_nyquist = (j == int(grid_ny / 2)) and (int(grid_ny / 2) == int((grid_ny + 1) / 2))
                    kk_nyquist = (k == int(grid_nz / 2)) and (int(grid_nz / 2) == int((grid_nz + 1) / 2))
                    if (
                        (i == 0 and jj_nyquist and k == 0)
                        or (ii_nyquist and j == 0 and k == 0)
                        or (ii_nyquist and jj_nyquist and k == 0)
                        or (i == 0 and j == 0 and kk_nyquist)
                        or (i == 0 and jj_nyquist and kk_nyquist)
                        or (ii_nyquist and j == 0 and kk_nyquist)
                        or (ii_nyquist and jj_nyquist and kk_nyquist)
                    ):
                        nyquist_indices.append([i, j, k])
                    else:
                        normal_indices.append([i, j, k])
                        if ii_nyquist or (i == 0):
                            i_conj = i
                        else:
                            i_conj = grid_nx - i
                        if jj_nyquist or (j == 0):
                            j_conj = j
                        else:
                            j_conj = grid_ny - j
                        if kk_nyquist or (k == 0):
                            k_conj = k
                        else:
                            k_conj = grid_nz - k
                        normal_conj_indices.append([i_conj, j_conj, k_conj])

    normal_indices = jnp.array(normal_indices)
    normal_conj_indices = jnp.array(normal_conj_indices)
    nyquist_indices = jnp.array(nyquist_indices)

    normal_indices_x = normal_indices[:, 0]
    normal_indices_y = normal_indices[:, 1]
    normal_indices_z = normal_indices[:, 2]
    normal_conj_indices_x = normal_conj_indices[:, 0]
    normal_conj_indices_y = normal_conj_indices[:, 1]
    normal_conj_indices_z = normal_conj_indices[:, 2]
    if len(nyquist_indices) > 0:
        nyquist_indices_x = nyquist_indices[:, 0]
        nyquist_indices_y = nyquist_indices[:, 1]
        nyquist_indices_z = nyquist_indices[:, 2]
    else:
        nyquist_indices_x = np.array([-1])
        nyquist_indices_y = np.array([-1])
        nyquist_indices_z = np.array([-1])
    return (
        normal_indices_x,
        normal_indices_y,
        normal_indices_z,
        normal_conj_indices_x,
        normal_conj_indices_y,
        normal_conj_indices_z,
        nyquist_indices_x,
        nyquist_indices_y,
        nyquist_indices_z,
    )

@partial(jit, static_argnums=[0])
def number_of_neigh_jit(num_particles: int, indices_i_lub, indices_j_lub):
    # Count occurrences of each particle in the neighbor list arrays
    counts_i = jnp.bincount(indices_i_lub, minlength=num_particles, length=num_particles)
    counts_j = jnp.bincount(indices_j_lub, minlength=num_particles, length=num_particles)
    brow_lub_precondition = counts_i + counts_j
    return jnp.repeat(brow_lub_precondition, 6)


@partial(jit, static_argnums=[0, 2, 3, 4])
def compute_real_space_slipvelocity(
    num_particles: int,
    m_self: ArrayLike,
    temperature: float,
    dt: float,
    n_iter_lanczos_ff: int,
    random_array_real: ArrayLike,
    r: ArrayLike,
    indices_i: ArrayLike,
    indices_j: ArrayLike,
    f1: ArrayLike,
    f2: ArrayLike,
    g1: ArrayLike,
    g2: ArrayLike,
    h1: ArrayLike,
    h2: ArrayLike,
    h3: ArrayLike,
) -> tuple[Array, float, Array]:
    """Compute real space far-field thermal fluctuation.

    Here, the square root of the mobility operator is performed using Lanczos decomposition.

    Parameters
    ----------
    num_particles: (int)
        Number of particles
    m_self: (float)
        Array (,2) containing mobility self contributions
    temperature: (float)
        Thermal energy
    dt: (float)
        Time step
    n_iter_lanczos_ff: (int)
        Number of Lanczos iteration to perform
    random_array_real: (float)
        Array (,11*num_particles) of random numbers with the proper variance
    r: (float)
        Array (n_pair_ff,3) containing units vectors connecting each pair of particles in the far-field neighbor list
    indices_i: (int)
        Array (n_pair_ff) of indices of first particle in neighbor list pairs
    indices_j: (int)
        Array (n_pair_ff) of indices of second particle in neighbor list pairs
    f1: (float)
        Array (,n_pair_ff) containing mobility scalar function evaluated for the current particle configuration
    f2: (float)
        Array (,n_pair_ff) containing mobility scalar function evaluated for the current particle configuration
    g1: (float)
        Array (,n_pair_ff) containing mobility scalar function evaluated for the current particle configuration
    g2: (float)
        Array (,n_pair_ff) containing mobility scalar function evaluated for the current particle configuration
    h1: (float)
        Array (,n_pair_ff) containing mobility scalar function evaluated for the current particle configuration
    h2: (float)
        Array (,n_pair_ff) containing mobility scalar function evaluated for the current particle configuration
    h3: (float)
        Array (,n_pair_ff) containing mobility scalar function evaluated for the current particle configuration

    Returns
    -------
    lin_vel, ang_vel_and_strain, stepnorm, trid

    """

    def helper_mpsi(random_array: ArrayLike) -> ArrayLike:
        """Compute matrix-vector product of real-space granmobility matrix with a generalized force array.

        Parameters
        ----------
        random_array:
            Input random generalized forces.

        Returns
        -------
        slip_velocity

        """
        # input is already in the format: [Forces, Torque+Stresslet] (not in the generalized format [Force+Torque,Stresslet] like in the saddle point solver)
        forces = random_array.at[: 3 * num_particles].get()

        couplets = jnp.zeros(8 * num_particles)
        couplets = couplets.at[::8].add(random_array.at[(3 * num_particles + 3) :: 8].get())  # C[0] = S[0]
        couplets = couplets.at[1::8].add(
            random_array.at[(3 * num_particles + 4) :: 8].get() + random_array.at[(3 * num_particles + 2) :: 8].get() * 0.5
        )  # C[1] = S[1] + L[2]/2
        couplets = couplets.at[2::8].add(
            random_array.at[(3 * num_particles + 5) :: 8].get() - random_array.at[(3 * num_particles + 1) :: 8].get() * 0.5
        )  # C[2] = S[2] - L[1]/2
        couplets = couplets.at[3::8].add(
            random_array.at[(3 * num_particles + 6) :: 8].get() + random_array.at[(3 * num_particles + 0) :: 8].get() * 0.5
        )  # C[3] = S[3] + L[0]/2
        couplets = couplets.at[4::8].add(random_array.at[(3 * num_particles + 7) :: 8].get())  # C[4] = S[4]
        couplets = couplets.at[5::8].add(
            random_array.at[(3 * num_particles + 4) :: 8].get() - random_array.at[(3 * num_particles + 2) :: 8].get() * 0.5
        )  # C[5] = S[1] - L[2]/2
        couplets = couplets.at[6::8].add(
            random_array.at[(3 * num_particles + 5) :: 8].get() + random_array.at[(3 * num_particles + 1) :: 8].get() * 0.5
        )  # C[6] = S[2] + L[1]/2
        couplets = couplets.at[7::8].add(
            random_array.at[(3 * num_particles + 6) :: 8].get() - random_array.at[(3 * num_particles + 0) :: 8].get() * 0.5
        )  # C[7] = S[3] - L[0]/2

        # Allocate arrays for Linear velocities and Velocity gradients (from which we can extract angular velocities and rate of strain)
        r_lin_velocities = jnp.zeros((num_particles, 3), float)
        r_velocity_gradient = jnp.zeros((num_particles, 8), float)

        # SELF CONTRIBUTIONS
        r_lin_velocities = r_lin_velocities.at[:, 0].add(m_self.at[0].get() * forces.at[0::3].get())
        r_lin_velocities = r_lin_velocities.at[:, 1].add(m_self.at[0].get() * forces.at[1::3].get())
        r_lin_velocities = r_lin_velocities.at[:, 2].add(m_self.at[0].get() * forces.at[2::3].get())

        r_velocity_gradient = r_velocity_gradient.at[:, 0].add(
            m_self.at[1].get() * (couplets.at[0::8].get() - 4 * couplets.at[0::8].get())
        )
        r_velocity_gradient = r_velocity_gradient.at[:, 5].add(
            m_self.at[1].get() * (couplets.at[1::8].get() - 4 * couplets.at[5::8].get())
        )
        r_velocity_gradient = r_velocity_gradient.at[:, 6].add(
            m_self.at[1].get() * (couplets.at[2::8].get() - 4 * couplets.at[6::8].get())
        )
        r_velocity_gradient = r_velocity_gradient.at[:, 7].add(
            m_self.at[1].get() * (couplets.at[3::8].get() - 4 * couplets.at[7::8].get())
        )

        r_velocity_gradient = r_velocity_gradient.at[:, 4].add(
            m_self.at[1].get() * (couplets.at[4::8].get() - 4 * couplets.at[4::8].get())
        )
        r_velocity_gradient = r_velocity_gradient.at[:, 1].add(
            m_self.at[1].get() * (couplets.at[5::8].get() - 4 * couplets.at[1::8].get())
        )
        r_velocity_gradient = r_velocity_gradient.at[:, 2].add(
            m_self.at[1].get() * (couplets.at[6::8].get() - 4 * couplets.at[2::8].get())
        )
        r_velocity_gradient = r_velocity_gradient.at[:, 3].add(
            m_self.at[1].get() * (couplets.at[7::8].get() - 4 * couplets.at[3::8].get())
        )

        # # Geometric quantities
        rdotf_j = (
            r.at[:, 0].get() * forces.at[3 * indices_j + 0].get()
            + r.at[:, 1].get() * forces.at[3 * indices_j + 1].get()
            + r.at[:, 2].get() * forces.at[3 * indices_j + 2].get()
        )
        mrdotf_i = -(
            r.at[:, 0].get() * forces.at[3 * indices_i + 0].get()
            + r.at[:, 1].get() * forces.at[3 * indices_i + 1].get()
            + r.at[:, 2].get() * forces.at[3 * indices_i + 2].get()
        )

        cj_dotr = jnp.array(
            [
                couplets.at[8 * indices_j + 0].get() * r.at[:, 0].get()
                + couplets.at[8 * indices_j + 1].get() * r.at[:, 1].get()
                + couplets.at[8 * indices_j + 2].get() * r.at[:, 2].get(),
                couplets.at[8 * indices_j + 5].get() * r.at[:, 0].get()
                + couplets.at[8 * indices_j + 4].get() * r.at[:, 1].get()
                + couplets.at[8 * indices_j + 3].get() * r.at[:, 2].get(),
                couplets.at[8 * indices_j + 6].get() * r.at[:, 0].get()
                + couplets.at[8 * indices_j + 7].get() * r.at[:, 1].get()
                - (couplets.at[8 * indices_j + 0].get() + couplets.at[8 * indices_j + 4].get())
                * r.at[:, 2].get(),
            ]
        )

        ci_dotmr = jnp.array(
            [
                -couplets.at[8 * indices_i + 0].get() * r.at[:, 0].get()
                - couplets.at[8 * indices_i + 1].get() * r.at[:, 1].get()
                - couplets.at[8 * indices_i + 2].get() * r.at[:, 2].get(),
                -couplets.at[8 * indices_i + 5].get() * r.at[:, 0].get()
                - couplets.at[8 * indices_i + 4].get() * r.at[:, 1].get()
                - couplets.at[8 * indices_i + 3].get() * r.at[:, 2].get(),
                -couplets.at[8 * indices_i + 6].get() * r.at[:, 0].get()
                - couplets.at[8 * indices_i + 7].get() * r.at[:, 1].get()
                + (couplets.at[8 * indices_i + 0].get() + couplets.at[8 * indices_i + 4].get())
                * r.at[:, 2].get(),
            ]
        )

        rdotc_j = jnp.array(
            [
                couplets.at[8 * indices_j + 0].get() * r.at[:, 0].get()
                + couplets.at[8 * indices_j + 5].get() * r.at[:, 1].get()
                + couplets.at[8 * indices_j + 6].get() * r.at[:, 2].get(),
                couplets.at[8 * indices_j + 1].get() * r.at[:, 0].get()
                + couplets.at[8 * indices_j + 4].get() * r.at[:, 1].get()
                + couplets.at[8 * indices_j + 7].get() * r.at[:, 2].get(),
                couplets.at[8 * indices_j + 2].get() * r.at[:, 0].get()
                + couplets.at[8 * indices_j + 3].get() * r.at[:, 1].get()
                - (couplets.at[8 * indices_j + 0].get() + couplets.at[8 * indices_j + 4].get())
                * r.at[:, 2].get(),
            ]
        )

        mrdotc_i = jnp.array(
            [
                -couplets.at[8 * indices_i + 0].get() * r.at[:, 0].get()
                - couplets.at[8 * indices_i + 5].get() * r.at[:, 1].get()
                - couplets.at[8 * indices_i + 6].get() * r.at[:, 2].get(),
                -couplets.at[8 * indices_i + 1].get() * r.at[:, 0].get()
                - couplets.at[8 * indices_i + 4].get() * r.at[:, 1].get()
                - couplets.at[8 * indices_i + 7].get() * r.at[:, 2].get(),
                -couplets.at[8 * indices_i + 2].get() * r.at[:, 0].get()
                - couplets.at[8 * indices_i + 3].get() * r.at[:, 1].get()
                + (couplets.at[8 * indices_i + 0].get() + couplets.at[8 * indices_i + 4].get())
                * r.at[:, 2].get(),
            ]
        )

        rdotc_jj_dotr = (
            r.at[:, 0].get() * cj_dotr.at[0, :].get()
            + r.at[:, 1].get() * cj_dotr.at[1, :].get()
            + r.at[:, 2].get() * cj_dotr.at[2, :].get()
        )
        mrdotc_ii_dotmr = -(
            r.at[:, 0].get() * ci_dotmr.at[0, :].get()
            + r.at[:, 1].get() * ci_dotmr.at[1, :].get()
            + r.at[:, 2].get() * ci_dotmr.at[2, :].get()
        )

        # Compute Velocity for particles i
        r_lin_velocities = r_lin_velocities.at[indices_i, 0].add(
            f1 * forces.at[3 * indices_j].get() + (f2 - f1) * rdotf_j * r.at[:, 0].get()
        )
        r_lin_velocities = r_lin_velocities.at[indices_i, 1].add(
            f1 * forces.at[3 * indices_j + 1].get() + (f2 - f1) * rdotf_j * r.at[:, 1].get()
        )
        r_lin_velocities = r_lin_velocities.at[indices_i, 2].add(
            f1 * forces.at[3 * indices_j + 2].get() + (f2 - f1) * rdotf_j * r.at[:, 2].get()
        )
        r_lin_velocities = r_lin_velocities.at[indices_i, 0].add(
            g1 * (cj_dotr.at[0, :].get() - rdotc_jj_dotr * r.at[:, 0].get())
            + g2 * (rdotc_j.at[0, :].get() - 4.0 * rdotc_jj_dotr * r.at[:, 0].get())
        )
        r_lin_velocities = r_lin_velocities.at[indices_i, 1].add(
            g1 * (cj_dotr.at[1, :].get() - rdotc_jj_dotr * r.at[:, 1].get())
            + g2 * (rdotc_j.at[1, :].get() - 4.0 * rdotc_jj_dotr * r.at[:, 1].get())
        )
        r_lin_velocities = r_lin_velocities.at[indices_i, 2].add(
            g1 * (cj_dotr.at[2, :].get() - rdotc_jj_dotr * r.at[:, 2].get())
            + g2 * (rdotc_j.at[2, :].get() - 4.0 * rdotc_jj_dotr * r.at[:, 2].get())
        )
        # Compute Velocity for particles j
        r_lin_velocities = r_lin_velocities.at[indices_j, 0].add(
            f1 * forces.at[3 * indices_i].get() - (f2 - f1) * mrdotf_i * r.at[:, 0].get()
        )
        r_lin_velocities = r_lin_velocities.at[indices_j, 1].add(
            f1 * forces.at[3 * indices_i + 1].get() - (f2 - f1) * mrdotf_i * r.at[:, 1].get()
        )
        r_lin_velocities = r_lin_velocities.at[indices_j, 2].add(
            f1 * forces.at[3 * indices_i + 2].get() - (f2 - f1) * mrdotf_i * r.at[:, 2].get()
        )
        r_lin_velocities = r_lin_velocities.at[indices_j, 0].add(
            g1 * (ci_dotmr.at[0, :].get() + mrdotc_ii_dotmr * r.at[:, 0].get())
            + g2 * (mrdotc_i.at[0, :].get() + 4.0 * mrdotc_ii_dotmr * r.at[:, 0].get())
        )
        r_lin_velocities = r_lin_velocities.at[indices_j, 1].add(
            g1 * (ci_dotmr.at[1, :].get() + mrdotc_ii_dotmr * r.at[:, 1].get())
            + g2 * (mrdotc_i.at[1, :].get() + 4.0 * mrdotc_ii_dotmr * r.at[:, 1].get())
        )
        r_lin_velocities = r_lin_velocities.at[indices_j, 2].add(
            g1 * (ci_dotmr.at[2, :].get() + mrdotc_ii_dotmr * r.at[:, 2].get())
            + g2 * (mrdotc_i.at[2, :].get() + 4.0 * mrdotc_ii_dotmr * r.at[:, 2].get())
        )

        # Compute Velocity Gradient for particles i and j
        r_velocity_gradient = r_velocity_gradient.at[indices_i, 0].add(
            (-1)
            * g1
            * (
                r.at[:, 0].get() * forces.at[3 * indices_j + 0].get()
                - rdotf_j * r.at[:, 0].get() * r.at[:, 0].get()
            )
            + (-1)
            * g2
            * (
                rdotf_j
                + forces.at[3 * indices_j + 0].get() * r.at[:, 0].get()
                - 4.0 * rdotf_j * r.at[:, 0].get() * r.at[:, 0].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_j, 0].add(
            (-1)
            * g1
            * (
                -r.at[:, 0].get() * forces.at[3 * indices_i + 0].get()
                - mrdotf_i * r.at[:, 0].get() * r.at[:, 0].get()
            )
            + (-1)
            * g2
            * (
                mrdotf_i
                - forces.at[3 * indices_i + 0].get() * r.at[:, 0].get()
                - 4.0 * mrdotf_i * r.at[:, 0].get() * r.at[:, 0].get()
            )
        )
        r_velocity_gradient = r_velocity_gradient.at[indices_i, 0].add(
            h1 * (couplets.at[8 * indices_j + 0].get() - 4.0 * couplets.at[8 * indices_j + 0].get())
            + h2
            * (
                r.at[:, 0].get() * cj_dotr.at[0, :].get()
                - rdotc_jj_dotr * r.at[:, 0].get() * r.at[:, 0].get()
            )
            + h3
            * (
                rdotc_jj_dotr
                + cj_dotr.at[0, :].get() * r.at[:, 0].get()
                + r.at[:, 0].get() * rdotc_j.at[0, :].get()
                + rdotc_j.at[0, :].get() * r.at[:, 0].get()
                - 6.0 * rdotc_jj_dotr * r.at[:, 0].get() * r.at[:, 0].get()
                - couplets.at[8 * indices_j + 0].get()
            )
        )
        r_velocity_gradient = r_velocity_gradient.at[indices_j, 0].add(
            h1 * (couplets.at[8 * indices_i + 0].get() - 4.0 * couplets.at[8 * indices_i + 0].get())
            + h2
            * (
                -r.at[:, 0].get() * ci_dotmr.at[0, :].get()
                - mrdotc_ii_dotmr * r.at[:, 0].get() * r.at[:, 0].get()
            )
            + h3
            * (
                mrdotc_ii_dotmr
                - ci_dotmr.at[0, :].get() * r.at[:, 0].get()
                - r.at[:, 0].get() * mrdotc_i.at[0, :].get()
                - mrdotc_i.at[0, :].get() * r.at[:, 0].get()
                - 6.0 * mrdotc_ii_dotmr * r.at[:, 0].get() * r.at[:, 0].get()
                - couplets.at[8 * indices_i + 0].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_i, 1].add(
            (-1)
            * g1
            * (
                r.at[:, 1].get() * forces.at[3 * indices_j + 0].get()
                - rdotf_j * r.at[:, 1].get() * r.at[:, 0].get()
            )
            + (-1)
            * g2
            * (
                forces.at[3 * indices_j + 1].get() * r.at[:, 0].get()
                - 4.0 * rdotf_j * r.at[:, 1].get() * r.at[:, 0].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_j, 1].add(
            (-1)
            * g1
            * (
                -r.at[:, 1].get() * forces.at[3 * indices_i + 0].get()
                - mrdotf_i * r.at[:, 1].get() * r.at[:, 0].get()
            )
            + (-1)
            * g2
            * (
                -forces.at[3 * indices_i + 1].get() * r.at[:, 0].get()
                - 4.0 * mrdotf_i * r.at[:, 1].get() * r.at[:, 0].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_i, 1].add(
            h1 * (couplets.at[8 * indices_j + 5].get() - 4.0 * couplets.at[8 * indices_j + 1].get())
            + h2
            * (
                r.at[:, 1].get() * cj_dotr.at[0, :].get()
                - rdotc_jj_dotr * r.at[:, 1].get() * r.at[:, 0].get()
            )
            + h3
            * (
                cj_dotr.at[1, :].get() * r.at[:, 0].get()
                + r.at[:, 1].get() * rdotc_j.at[0, :].get()
                + rdotc_j.at[1, :].get() * r.at[:, 0].get()
                - 6.0 * rdotc_jj_dotr * r.at[:, 1].get() * r.at[:, 0].get()
                - couplets.at[8 * indices_j + 1].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_j, 1].add(
            h1 * (couplets.at[8 * indices_i + 5].get() - 4.0 * couplets.at[8 * indices_i + 1].get())
            + h2
            * (
                -r.at[:, 1].get() * ci_dotmr.at[0, :].get()
                - mrdotc_ii_dotmr * r.at[:, 1].get() * r.at[:, 0].get()
            )
            + h3
            * (
                -ci_dotmr.at[1, :].get() * r.at[:, 0].get()
                - r.at[:, 1].get() * mrdotc_i.at[0, :].get()
                - mrdotc_i.at[1, :].get() * r.at[:, 0].get()
                - 6.0 * mrdotc_ii_dotmr * r.at[:, 1].get() * r.at[:, 0].get()
                - couplets.at[8 * indices_i + 1].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_i, 2].add(
            (-1)
            * g1
            * (
                r.at[:, 2].get() * forces.at[3 * indices_j + 0].get()
                - rdotf_j * r.at[:, 2].get() * r.at[:, 0].get()
            )
            + (-1)
            * g2
            * (
                forces.at[3 * indices_j + 2].get() * r.at[:, 0].get()
                - 4.0 * rdotf_j * r.at[:, 2].get() * r.at[:, 0].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_j, 2].add(
            (-1)
            * g1
            * (
                -r.at[:, 2].get() * forces.at[3 * indices_i + 0].get()
                - mrdotf_i * r.at[:, 2].get() * r.at[:, 0].get()
            )
            + (-1)
            * g2
            * (
                -forces.at[3 * indices_i + 2].get() * r.at[:, 0].get()
                - 4.0 * mrdotf_i * r.at[:, 2].get() * r.at[:, 0].get()
            )
        )
        r_velocity_gradient = r_velocity_gradient.at[indices_i, 2].add(
            h1 * (couplets.at[8 * indices_j + 6].get() - 4.0 * couplets.at[8 * indices_j + 2].get())
            + h2
            * (
                r.at[:, 2].get() * cj_dotr.at[0, :].get()
                - rdotc_jj_dotr * r.at[:, 2].get() * r.at[:, 0].get()
            )
            + h3
            * (
                cj_dotr.at[2, :].get() * r.at[:, 0].get()
                + r.at[:, 2].get() * rdotc_j.at[0, :].get()
                + rdotc_j.at[2, :].get() * r.at[:, 0].get()
                - 6.0 * rdotc_jj_dotr * r.at[:, 2].get() * r.at[:, 0].get()
                - couplets.at[8 * indices_j + 2].get()
            )
        )
        r_velocity_gradient = r_velocity_gradient.at[indices_j, 2].add(
            h1 * (couplets.at[8 * indices_i + 6].get() - 4.0 * couplets.at[8 * indices_i + 2].get())
            + h2
            * (
                r.at[:, 2].get() * ci_dotmr.at[0, :].get() * (-1)
                - mrdotc_ii_dotmr * r.at[:, 2].get() * r.at[:, 0].get()
            )
            + h3
            * (
                -ci_dotmr.at[2, :].get() * r.at[:, 0].get()
                - r.at[:, 2].get() * mrdotc_i.at[0, :].get()
                - mrdotc_i.at[2, :].get() * r.at[:, 0].get()
                - 6.0 * mrdotc_ii_dotmr * r.at[:, 2].get() * r.at[:, 0].get()
                - couplets.at[8 * indices_i + 2].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_i, 3].add(
            (-1)
            * g1
            * (
                r.at[:, 2].get() * forces.at[3 * indices_j + 1].get()
                - rdotf_j * r.at[:, 2].get() * r.at[:, 1].get()
            )
            + (-1)
            * g2
            * (
                forces.at[3 * indices_j + 2].get() * r.at[:, 1].get()
                - 4.0 * rdotf_j * r.at[:, 2].get() * r.at[:, 1].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_j, 3].add(
            (-1)
            * g1
            * (
                -r.at[:, 2].get() * forces.at[3 * indices_i + 1].get()
                - mrdotf_i * r.at[:, 2].get() * r.at[:, 1].get()
            )
            + (-1)
            * g2
            * (
                -forces.at[3 * indices_i + 2].get() * r.at[:, 1].get()
                - 4.0 * mrdotf_i * r.at[:, 2].get() * r.at[:, 1].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_i, 3].add(
            h1 * (couplets.at[8 * indices_j + 7].get() - 4.0 * couplets.at[8 * indices_j + 3].get())
            + h2
            * (
                r.at[:, 2].get() * cj_dotr.at[1, :].get()
                - rdotc_jj_dotr * r.at[:, 2].get() * r.at[:, 1].get()
            )
            + h3
            * (
                cj_dotr.at[2, :].get() * r.at[:, 1].get()
                + r.at[:, 2].get() * rdotc_j.at[1, :].get()
                + rdotc_j.at[2, :].get() * r.at[:, 1].get()
                - 6.0 * rdotc_jj_dotr * r.at[:, 2].get() * r.at[:, 1].get()
                - couplets.at[8 * indices_j + 3].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_j, 3].add(
            h1 * (couplets.at[8 * indices_i + 7].get() - 4.0 * couplets.at[8 * indices_i + 3].get())
            + h2
            * (
                -r.at[:, 2].get() * ci_dotmr.at[1, :].get()
                - mrdotc_ii_dotmr * r.at[:, 2].get() * r.at[:, 1].get()
            )
            + h3
            * (
                -ci_dotmr.at[2, :].get() * r.at[:, 1].get()
                - r.at[:, 2].get() * mrdotc_i.at[1, :].get()
                - mrdotc_i.at[2, :].get() * r.at[:, 1].get()
                - 6.0 * mrdotc_ii_dotmr * r.at[:, 2].get() * r.at[:, 1].get()
                - couplets.at[8 * indices_i + 3].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_i, 4].add(
            (-1)
            * g1
            * (
                r.at[:, 1].get() * forces.at[3 * indices_j + 1].get()
                - rdotf_j * r.at[:, 1].get() * r.at[:, 1].get()
            )
            + (-1)
            * g2
            * (
                rdotf_j
                + forces.at[3 * indices_j + 1].get() * r.at[:, 1].get()
                - 4.0 * rdotf_j * r.at[:, 1].get() * r.at[:, 1].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_j, 4].add(
            (-1)
            * g1
            * (
                -r.at[:, 1].get() * forces.at[3 * indices_i + 1].get()
                - mrdotf_i * r.at[:, 1].get() * r.at[:, 1].get()
            )
            + (-1)
            * g2
            * (
                mrdotf_i
                - forces.at[3 * indices_i + 1].get() * r.at[:, 1].get()
                - 4.0 * mrdotf_i * r.at[:, 1].get() * r.at[:, 1].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_i, 4].add(
            h1 * (couplets.at[8 * indices_j + 4].get() - 4.0 * couplets.at[8 * indices_j + 4].get())
            + h2
            * (
                r.at[:, 1].get() * cj_dotr.at[1, :].get()
                - rdotc_jj_dotr * r.at[:, 1].get() * r.at[:, 1].get()
            )
            + h3
            * (
                rdotc_jj_dotr
                + cj_dotr.at[1, :].get() * r.at[:, 1].get()
                + r.at[:, 1].get() * rdotc_j.at[1, :].get()
                + rdotc_j.at[1, :].get() * r.at[:, 1].get()
                - 6.0 * rdotc_jj_dotr * r.at[:, 1].get() * r.at[:, 1].get()
                - couplets.at[8 * indices_j + 4].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_j, 4].add(
            h1 * (couplets.at[8 * indices_i + 4].get() - 4.0 * couplets.at[8 * indices_i + 4].get())
            + h2
            * (
                -r.at[:, 1].get() * ci_dotmr.at[1, :].get()
                - mrdotc_ii_dotmr * r.at[:, 1].get() * r.at[:, 1].get()
            )
            + h3
            * (
                mrdotc_ii_dotmr
                - ci_dotmr.at[1, :].get() * r.at[:, 1].get()
                - r.at[:, 1].get() * mrdotc_i.at[1, :].get()
                - mrdotc_i.at[1, :].get() * r.at[:, 1].get()
                - 6.0 * mrdotc_ii_dotmr * r.at[:, 1].get() * r.at[:, 1].get()
                - couplets.at[8 * indices_i + 4].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_i, 5].add(
            (-1)
            * g1
            * (
                r.at[:, 0].get() * forces.at[3 * indices_j + 1].get()
                - rdotf_j * r.at[:, 0].get() * r.at[:, 1].get()
            )
            + (-1)
            * g2
            * (
                forces.at[3 * indices_j + 0].get() * r.at[:, 1].get()
                - 4.0 * rdotf_j * r.at[:, 0].get() * r.at[:, 1].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_j, 5].add(
            (-1)
            * g1
            * (
                -r.at[:, 0].get() * forces.at[3 * indices_i + 1].get()
                - mrdotf_i * r.at[:, 0].get() * r.at[:, 1].get()
            )
            + (-1)
            * g2
            * (
                -forces.at[3 * indices_i + 0].get() * r.at[:, 1].get()
                - 4.0 * mrdotf_i * r.at[:, 0].get() * r.at[:, 1].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_i, 5].add(
            h1 * (couplets.at[8 * indices_j + 1].get() - 4.0 * couplets.at[8 * indices_j + 5].get())
            + h2
            * (
                r.at[:, 0].get() * cj_dotr.at[1, :].get()
                - rdotc_jj_dotr * r.at[:, 0].get() * r.at[:, 1].get()
            )
            + h3
            * (
                cj_dotr.at[0, :].get() * r.at[:, 1].get()
                + r.at[:, 0].get() * rdotc_j.at[1, :].get()
                + rdotc_j.at[0, :].get() * r.at[:, 1].get()
                - 6.0 * rdotc_jj_dotr * r.at[:, 0].get() * r.at[:, 1].get()
                - couplets.at[8 * indices_j + 5].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_j, 5].add(
            h1 * (couplets.at[8 * indices_i + 1].get() - 4.0 * couplets.at[8 * indices_i + 5].get())
            + h2
            * (
                -r.at[:, 0].get() * ci_dotmr.at[1, :].get()
                - mrdotc_ii_dotmr * r.at[:, 0].get() * r.at[:, 1].get()
            )
            + h3
            * (
                -ci_dotmr.at[0, :].get() * r.at[:, 1].get()
                - r.at[:, 0].get() * mrdotc_i.at[1, :].get()
                - mrdotc_i.at[0, :].get() * r.at[:, 1].get()
                - 6.0 * mrdotc_ii_dotmr * r.at[:, 0].get() * r.at[:, 1].get()
                - couplets.at[8 * indices_i + 5].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_i, 6].add(
            (-1)
            * g1
            * (
                r.at[:, 0].get() * forces.at[3 * indices_j + 2].get()
                - rdotf_j * r.at[:, 0].get() * r.at[:, 2].get()
            )
            + (-1)
            * g2
            * (
                forces.at[3 * indices_j + 0].get() * r.at[:, 2].get()
                - 4.0 * rdotf_j * r.at[:, 0].get() * r.at[:, 2].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_j, 6].add(
            (-1)
            * g1
            * (
                -r.at[:, 0].get() * forces.at[3 * indices_i + 2].get()
                - mrdotf_i * r.at[:, 0].get() * r.at[:, 2].get()
            )
            + (-1)
            * g2
            * (
                -forces.at[3 * indices_i + 0].get() * r.at[:, 2].get()
                - 4.0 * mrdotf_i * r.at[:, 0].get() * r.at[:, 2].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_i, 6].add(
            h1 * (couplets.at[8 * indices_j + 2].get() - 4.0 * couplets.at[8 * indices_j + 6].get())
            + h2
            * (
                r.at[:, 0].get() * cj_dotr.at[2, :].get()
                - rdotc_jj_dotr * r.at[:, 0].get() * r.at[:, 2].get()
            )
            + h3
            * (
                cj_dotr.at[0, :].get() * r.at[:, 2].get()
                + r.at[:, 0].get() * rdotc_j.at[2, :].get()
                + rdotc_j.at[0, :].get() * r.at[:, 2].get()
                - 6.0 * rdotc_jj_dotr * r.at[:, 0].get() * r.at[:, 2].get()
                - couplets.at[8 * indices_j + 6].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_j, 6].add(
            h1 * (couplets.at[8 * indices_i + 2].get() - 4.0 * couplets.at[8 * indices_i + 6].get())
            + h2
            * (
                -r.at[:, 0].get() * ci_dotmr.at[2, :].get()
                - mrdotc_ii_dotmr * r.at[:, 0].get() * r.at[:, 2].get()
            )
            + h3
            * (
                -ci_dotmr.at[0, :].get() * r.at[:, 2].get()
                - r.at[:, 0].get() * mrdotc_i.at[2, :].get()
                - mrdotc_i.at[0, :].get() * r.at[:, 2].get()
                - 6.0 * mrdotc_ii_dotmr * r.at[:, 0].get() * r.at[:, 2].get()
                - couplets.at[8 * indices_i + 6].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_i, 7].add(
            (-1)
            * g1
            * (
                r.at[:, 1].get() * forces.at[3 * indices_j + 2].get()
                - rdotf_j * r.at[:, 1].get() * r.at[:, 2].get()
            )
            + (-1)
            * g2
            * (
                forces.at[3 * indices_j + 1].get() * r.at[:, 2].get()
                - 4.0 * rdotf_j * r.at[:, 1].get() * r.at[:, 2].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_j, 7].add(
            (-1)
            * g1
            * (
                -r.at[:, 1].get() * forces.at[3 * indices_i + 2].get()
                - mrdotf_i * r.at[:, 1].get() * r.at[:, 2].get()
            )
            + (-1)
            * g2
            * (
                -forces.at[3 * indices_i + 1].get() * r.at[:, 2].get()
                - 4.0 * mrdotf_i * r.at[:, 1].get() * r.at[:, 2].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_i, 7].add(
            h1 * (couplets.at[8 * indices_j + 3].get() - 4.0 * couplets.at[8 * indices_j + 7].get())
            + h2
            * (
                r.at[:, 1].get() * cj_dotr.at[2, :].get()
                - rdotc_jj_dotr * r.at[:, 1].get() * r.at[:, 2].get()
            )
            + h3
            * (
                cj_dotr.at[1, :].get() * r.at[:, 2].get()
                + r.at[:, 1].get() * rdotc_j.at[2, :].get()
                + rdotc_j.at[1, :].get() * r.at[:, 2].get()
                - 6.0 * rdotc_jj_dotr * r.at[:, 1].get() * r.at[:, 2].get()
                - couplets.at[8 * indices_j + 7].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_j, 7].add(
            h1 * (couplets.at[8 * indices_i + 3].get() - 4.0 * couplets.at[8 * indices_i + 7].get())
            + h2
            * (
                -r.at[:, 1].get() * ci_dotmr.at[2, :].get()
                - mrdotc_ii_dotmr * r.at[:, 1].get() * r.at[:, 2].get()
            )
            + h3
            * (
                -ci_dotmr.at[1, :].get() * r.at[:, 2].get()
                - r.at[:, 1].get() * mrdotc_i.at[2, :].get()
                - mrdotc_i.at[1, :].get() * r.at[:, 2].get()
                - 6.0 * mrdotc_ii_dotmr * r.at[:, 1].get() * r.at[:, 2].get()
                - couplets.at[8 * indices_i + 7].get()
            )
        )

        # # Convert to angular velocities and rate of strain
        r_ang_vel_and_strain = jnp.zeros((num_particles, 8))
        r_ang_vel_and_strain = r_ang_vel_and_strain.at[:, 0].add(
            (r_velocity_gradient.at[:, 3].get() - r_velocity_gradient.at[:, 7].get()) * 0.5
        )
        r_ang_vel_and_strain = r_ang_vel_and_strain.at[:, 1].add(
            (r_velocity_gradient.at[:, 6].get() - r_velocity_gradient.at[:, 2].get()) * 0.5
        )
        r_ang_vel_and_strain = r_ang_vel_and_strain.at[:, 2].add(
            (r_velocity_gradient.at[:, 1].get() - r_velocity_gradient.at[:, 5].get()) * 0.5
        )
        r_ang_vel_and_strain = r_ang_vel_and_strain.at[:, 3].add(
            2 * r_velocity_gradient.at[:, 0].get() + r_velocity_gradient.at[:, 4].get()
        )
        r_ang_vel_and_strain = r_ang_vel_and_strain.at[:, 4].add(
            r_velocity_gradient.at[:, 1].get() + r_velocity_gradient.at[:, 5].get()
        )
        r_ang_vel_and_strain = r_ang_vel_and_strain.at[:, 5].add(
            r_velocity_gradient.at[:, 2].get() + r_velocity_gradient.at[:, 6].get()
        )
        r_ang_vel_and_strain = r_ang_vel_and_strain.at[:, 6].add(
            r_velocity_gradient.at[:, 3].get() + r_velocity_gradient.at[:, 7].get()
        )
        r_ang_vel_and_strain = r_ang_vel_and_strain.at[:, 7].add(
            r_velocity_gradient.at[:, 0].get() + 2 * r_velocity_gradient.at[:, 4].get()
        )

        slip_velocity = jnp.zeros(11 * num_particles)
        slip_velocity = slip_velocity.at[: 3 * num_particles].add(jnp.reshape(r_lin_velocities, 3 * num_particles))
        slip_velocity = slip_velocity.at[3 * num_particles :].add(jnp.reshape(r_ang_vel_and_strain, 8 * num_particles))

        return slip_velocity

    def helper_compute_m12psi(
        n_iter_lanczos_ff: int, tridiagonal: ArrayLike, vectors: ArrayLike, norm: float
    ) -> ArrayLike:
        """Parameters
        ----------
        n_iter_lanczos_ff:
            Number of Lanczos iteration performed
        tridiagonal:
            Tridiagonal matrix obtained from Lanczos decomposition
        vectors:
            Vectors spanning Krylov subspace, obtained from Lanczos decomposition
        norm:
            Norm of initial random array

        Returns
        -------
        jnp.dot(vectors.T,jnp.dot(a,betae1)) * jnp.sqrt(2.0*temperature/dt)

        """
        betae1 = jnp.zeros(n_iter_lanczos_ff)
        betae1 = betae1.at[0].add(1 * norm)

        a, b = jnp.linalg.eigh(tridiagonal)
        a = jnp.where(a < 0, 0.0, a)
        a = jnp.dot((jnp.dot(b, jnp.diag(jnp.sqrt(a)))), b.T)
        return jnp.dot(vectors.T, jnp.dot(a, betae1))

    random_array_real = (2.0 * random_array_real - 1.0) * jnp.sqrt(3.0)
    trid, vectors = lanczos.lanczos_algorithm(helper_mpsi, 11 * num_particles, n_iter_lanczos_ff, random_array_real)

    psinorm = jnp.linalg.norm(random_array_real)
    m12_psi_old = helper_compute_m12psi(
        (n_iter_lanczos_ff - 1),
        trid[: (n_iter_lanczos_ff - 1), : (n_iter_lanczos_ff - 1)],
        vectors[: (n_iter_lanczos_ff - 1), :],
        psinorm,
    )
    m12_psi = helper_compute_m12psi(n_iter_lanczos_ff, trid, vectors, psinorm)
    buff = jnp.linalg.norm(m12_psi)
    stepnorm = jnp.linalg.norm(m12_psi - m12_psi_old)
    stepnorm = jnp.where(buff > 1.0, stepnorm / buff, stepnorm)
    m12_psi = m12_psi * jnp.sqrt(2.0 * temperature / dt)
    # combine w_lin_velocities, w_ang_vel_and_strain
    lin_vel = m12_psi.at[: 3 * num_particles].get()
    ang_vel_and_strain = m12_psi.at[3 * num_particles :].get()

    return lin_vel, ang_vel_and_strain, stepnorm, trid


@partial(jit, static_argnums=[0, 1, 2, 3])
def compute_real_space_slipvelocity_open(
    num_particles: int,
    temperature: float,
    dt: float,
    n_iter_lanczos_ff: int,
    random_array_real: ArrayLike,
    r: ArrayLike,
    indices_i: ArrayLike,
    indices_j: ArrayLike,
    mobil_scal: ArrayLike,
) -> tuple[Array, float, Array]:
    """Compute real space far-field thermal fluctuation in open boundary conditions.

    Here, the square root of the mobility operator is performed using Lanczos decomposition.

    Parameters
    ----------
    num_particles: (int)
        Number of particles
    temperature: (float)
        Thermal energy
    dt: (float)
        Time step
    n_iter_lanczos_ff: (int)
        Number of Lanczos iteration to perform
    random_array_real: (float)
        Array (,11*num_particles) of random numbers with the proper variance
    r: (float)
        Array (n_pair_ff,3) containing units vectors connecting each pair of particles in the far-field neighbor list
    indices_i: (int)
        Array (n_pair_ff) of indices of first particle in neighbor list pairs
    indices_j: (int)
        Array (n_pair_ff) of indices of second particle in neighbor list pairs
    mobil_scal: (float)
        Array (11,num_particles*(num_particles-1)/2)) containing mobility functions evaluated for the current particle configuration

    Returns
    -------
    lin_vel, ang_vel_and_strain, stepnorm, trid

    """

    def helper_mpsi(generalized_forces: ArrayLike) -> Array:
        """Thermal helper: compute matrix-vector product of the grandmobility matrix with a generalized force vector (and stresslet), in open boundary conditions.

        Parameters
        ----------
        generalized_forces: (float)
            Array (,11*num_particles) containing input generalized forces (force/torque/stresslet)

        Returns
        -------
        generalized_velocities (linear/angular velocities and rateOfStrain)

        """
        strain = jnp.zeros((num_particles, 5), float)
        velocities = jnp.zeros((num_particles, 6), float)

        forces_torques = generalized_forces[: 6 * num_particles]
        forces_torques = -forces_torques
        ft_i = (jnp.reshape(forces_torques, (num_particles, 6))).at[indices_i].get()
        ft_j = (jnp.reshape(forces_torques, (num_particles, 6))).at[indices_j].get()
        stresslets = generalized_forces[
            6 * num_particles :
        ]  # stresslet in vector form has the format [Sxx,Sxy,Sxz,Syz,Syy]
        s_i = (jnp.reshape(stresslets, (num_particles, 5))).at[indices_i].get()
        s_i = jnp.array(
            [
                [s_i[:, 0], s_i[:, 1], s_i[:, 2]],
                [s_i[:, 1], s_i[:, 4], s_i[:, 3]],
                [s_i[:, 2], s_i[:, 3], -s_i[:, 0] - s_i[:, 4]],
            ]
        )
        s_j = (jnp.reshape(stresslets, (num_particles, 5))).at[indices_j].get()
        s_j = jnp.array(
            [
                [s_j[:, 0], s_j[:, 1], s_j[:, 2]],
                [s_j[:, 1], s_j[:, 4], s_j[:, 3]],
                [s_j[:, 2], s_j[:, 3], -s_j[:, 0] - s_j[:, 4]],
            ]
        )

        # Dot product of levi-civita-symbol and r
        epsr = jnp.array(
            [
                [jnp.zeros(int(num_particles * (num_particles - 1) / 2)), r[:, 2], -r[:, 1]],
                [-r[:, 2], jnp.zeros(int(num_particles * (num_particles - 1) / 2)), r[:, 0]],
                [r[:, 1], -r[:, 0], jnp.zeros(int(num_particles * (num_particles - 1) / 2))],
            ]
        )

        # Dot product of r and U, i.e. axisymmetric projection
        rdfi = (
            r.at[:, 0].get() * ft_i.at[:, 0].get()
            + r.at[:, 1].get() * ft_i.at[:, 1].get()
            + r.at[:, 2].get() * ft_i.at[:, 2].get()
        )
        rdfj = (
            r.at[:, 0].get() * ft_j.at[:, 0].get()
            + r.at[:, 1].get() * ft_j.at[:, 1].get()
            + r.at[:, 2].get() * ft_j.at[:, 2].get()
        )
        rdti = (
            r.at[:, 0].get() * ft_i.at[:, 3].get()
            + r.at[:, 1].get() * ft_i.at[:, 4].get()
            + r.at[:, 2].get() * ft_i.at[:, 5].get()
        )
        rdtj = (
            r.at[:, 0].get() * ft_j.at[:, 3].get()
            + r.at[:, 1].get() * ft_j.at[:, 4].get()
            + r.at[:, 2].get() * ft_j.at[:, 5].get()
        )

        # Cross product of U and r, i.e. eps_ijk*r_k*U_j = Px dot U, (eps_ijk is the Levi-Civita symbol)
        epsrdfi = jnp.array(
            [
                r.at[:, 2].get() * ft_i.at[:, 1].get() - r.at[:, 1].get() * ft_i.at[:, 2].get(),
                -r.at[:, 2].get() * ft_i.at[:, 0].get() + r.at[:, 0].get() * ft_i.at[:, 2].get(),
                r.at[:, 1].get() * ft_i.at[:, 0].get() - r.at[:, 0].get() * ft_i.at[:, 1].get(),
            ]
        )

        epsrdti = jnp.array(
            [
                r.at[:, 2].get() * ft_i.at[:, 4].get() - r.at[:, 1].get() * ft_i.at[:, 5].get(),
                -r.at[:, 2].get() * ft_i.at[:, 3].get() + r.at[:, 0].get() * ft_i.at[:, 5].get(),
                r.at[:, 1].get() * ft_i.at[:, 3].get() - r.at[:, 0].get() * ft_i.at[:, 4].get(),
            ]
        )

        epsrdfj = jnp.array(
            [
                r.at[:, 2].get() * ft_j.at[:, 1].get() - r.at[:, 1].get() * ft_j.at[:, 2].get(),
                -r.at[:, 2].get() * ft_j.at[:, 0].get() + r.at[:, 0].get() * ft_j.at[:, 2].get(),
                r.at[:, 1].get() * ft_j.at[:, 0].get() - r.at[:, 0].get() * ft_j.at[:, 1].get(),
            ]
        )

        epsrdtj = jnp.array(
            [
                r.at[:, 2].get() * ft_j.at[:, 4].get() - r.at[:, 1].get() * ft_j.at[:, 5].get(),
                -r.at[:, 2].get() * ft_j.at[:, 3].get() + r.at[:, 0].get() * ft_j.at[:, 5].get(),
                r.at[:, 1].get() * ft_j.at[:, 3].get() - r.at[:, 0].get() * ft_j.at[:, 4].get(),
            ]
        )

        Sdri = jnp.array(
            [
                s_i.at[0, 0, :].get() * r.at[:, 0].get()
                + s_i.at[0, 1, :].get() * r.at[:, 1].get()
                + s_i.at[0, 2, :].get() * r.at[:, 2].get(),
                s_i.at[1, 0, :].get() * r.at[:, 0].get()
                + s_i.at[1, 1, :].get() * r.at[:, 1].get()
                + s_i.at[1, 2, :].get() * r.at[:, 2].get(),
                s_i.at[2, 0, :].get() * r.at[:, 0].get()
                + s_i.at[2, 1, :].get() * r.at[:, 1].get()
                + s_i.at[2, 2, :].get() * r.at[:, 2].get(),
            ]
        )

        Sdrj = jnp.array(
            [
                s_j.at[0, 0, :].get() * r.at[:, 0].get()
                + s_j.at[0, 1, :].get() * r.at[:, 1].get()
                + s_j.at[0, 2, :].get() * r.at[:, 2].get(),
                s_j.at[1, 0, :].get() * r.at[:, 0].get()
                + s_j.at[1, 1, :].get() * r.at[:, 1].get()
                + s_j.at[1, 2, :].get() * r.at[:, 2].get(),
                s_j.at[2, 0, :].get() * r.at[:, 0].get()
                + s_j.at[2, 1, :].get() * r.at[:, 1].get()
                + s_j.at[2, 2, :].get() * r.at[:, 2].get(),
            ]
        )

        rdSdri = (
            r.at[:, 0].get() * Sdri.at[0].get()
            + r.at[:, 1].get() * Sdri.at[1].get()
            + r.at[:, 2].get() * Sdri.at[2].get()
        )
        rdSdrj = (
            r.at[:, 0].get() * Sdrj.at[0].get()
            + r.at[:, 1].get() * Sdrj.at[1].get()
            + r.at[:, 2].get() * Sdrj.at[2].get()
        )

        epsrdSdri = jnp.array(
            [
                epsr[0, 0, :] * Sdri[0, :]
                + epsr[0, 1, :] * Sdri[1, :]
                + epsr[0, 2, :] * Sdri[2, :],
                epsr[1, 0, :] * Sdri[0, :]
                + epsr[1, 1, :] * Sdri[1, :]
                + epsr[1, 2, :] * Sdri[2, :],
                epsr[2, 0, :] * Sdri[0, :]
                + epsr[2, 1, :] * Sdri[1, :]
                + epsr[2, 2, :] * Sdri[2, :],
            ]
        )
        epsrdSdrj = jnp.array(
            [
                epsr[0, 0, :] * Sdrj[0, :]
                + epsr[0, 1, :] * Sdrj[1, :]
                + epsr[0, 2, :] * Sdrj[2, :],
                epsr[1, 0, :] * Sdrj[0, :]
                + epsr[1, 1, :] * Sdrj[1, :]
                + epsr[1, 2, :] * Sdrj[2, :],
                epsr[2, 0, :] * Sdrj[0, :]
                + epsr[2, 1, :] * Sdrj[1, :]
                + epsr[2, 2, :] * Sdrj[2, :],
            ]
        )

        xa12 = mobil_scal[0]
        ya12 = mobil_scal[1]
        yb12 = mobil_scal[2]
        xc12 = mobil_scal[3]
        yc12 = mobil_scal[4]
        xm12 = mobil_scal[5]
        ym12 = mobil_scal[6]
        zm12 = mobil_scal[7]
        xg12 = mobil_scal[8]
        yg12 = mobil_scal[9]
        yh12 = mobil_scal[10]
        n_pairs = int(num_particles * (num_particles - 1) / 2)

        # normalize self terms (avoid double counting)
        normaliz_factor = jnp.where(num_particles > 1, num_particles - 1, 1)

        # M_UF * F

        # Compute the contributions to the force for particles i (Fi = A11*Ui + A12*Uj + BT11*Wi + BT12*Wj)
        u = (
            (jnp.ones(n_pairs)).at[:, None].get() * (ft_i.at[:, :3].get() / normaliz_factor)
            + (xa12 - ya12).at[:, None].get() * rdfj.at[:, None].get() * r
            + ya12.at[:, None].get() * ft_j.at[:, :3].get()
            + (-yb12).at[:, None].get() * (-epsrdtj.T)
        )
        velocities = velocities.at[indices_i, :3].add(u)
        # Compute the contributions to the force for particles j (Fj = A11*Uj + A12*Ui + BT11*Wj + BT12*Wi)
        u = (
            (jnp.ones(n_pairs)).at[:, None].get() * (ft_j.at[:, :3].get() / normaliz_factor)
            + (xa12 - ya12).at[:, None].get() * rdfi.at[:, None].get() * r
            + ya12.at[:, None].get() * ft_i.at[:, :3].get()
            + (-yb12).at[:, None].get() * (epsrdti.T)
        )
        velocities = velocities.at[indices_j, :3].add(u)
        # Compute the contributions to the torque for particles i (Li = B11*Ui + B12*Uj + C11*Wi + C12*Wj)
        w = (
            (3 / 4 * jnp.ones(n_pairs)).at[:, None].get() * (ft_i.at[:, 3:].get() / normaliz_factor)
            + yb12.at[:, None].get() * epsrdfj.T
            + (xc12 - yc12).at[:, None].get() * rdtj.at[:, None].get() * r
            + yc12.at[:, None].get() * ft_j.at[:, 3:].get()
        )
        velocities = velocities.at[indices_i, 3:].add(w)
        # Compute the contributions to the torque for particles j (Lj = B11*Uj + B12*Ui + C11*Wj + C12*Wi)
        w = (
            (3 / 4 * jnp.ones(n_pairs)).at[:, None].get() * (ft_j.at[:, 3:].get() / normaliz_factor)
            - yb12.at[:, None].get() * epsrdfi.T
            + (xc12 - yc12).at[:, None].get() * rdti.at[:, None].get() * r
            + yc12.at[:, None].get() * ft_i.at[:, 3:].get()
        )
        velocities = velocities.at[indices_j, 3:].add(w)

        # M_US * S

        u = ((-xg12) - 2.0 * (-yg12)).at[:, None].get() * (rdSdrj.at[:, None].get()) * r + 2.0 * (
            -yg12
        ).at[:, None].get() * (Sdrj.T)
        w = yh12.at[:, None].get() * (2.0 * epsrdSdrj.T)

        velocities = velocities.at[indices_i, :3].add(u)
        velocities = velocities.at[indices_i, 3:].add(w)

        u = ((-xg12) - 2.0 * (-yg12)).at[:, None].get() * (rdSdri.at[:, None].get()) * (
            -r
        ) + 2.0 * (-yg12).at[:, None].get() * (-Sdri.T)
        w = yh12.at[:, None].get() * (2.0 * epsrdSdri.T)

        velocities = velocities.at[indices_j, :3].add(u)
        velocities = velocities.at[indices_j, 3:].add(w)

        # #M_EF * F
        # translational part

        # strain_xx component
        strain_xx_i = +xg12 * (r[:, 0] * r[:, 0] - 1.0 / 3.0) * rdfj + yg12 * (
            2 * ft_j[:, 0] * r[:, 0] - 2.0 * r[:, 0] * r[:, 0] * rdfj
        )
        # strain_xy component
        strain_xy_i = +xg12 * (r[:, 0] * r[:, 1]) * rdfj + yg12 * (
            ft_j[:, 0] * r[:, 1] + r[:, 0] * ft_j[:, 1] - 2.0 * r[:, 0] * r[:, 1] * rdfj
        )
        # strain_xz component
        strain_xz_i = +xg12 * (r[:, 0] * r[:, 2]) * rdfj + yg12 * (
            ft_j[:, 0] * r[:, 2] + r[:, 0] * ft_j[:, 2] - 2.0 * r[:, 0] * r[:, 2] * rdfj
        )
        # strain_yz component
        strain_yz_i = +xg12 * (r[:, 1] * r[:, 2]) * rdfj + yg12 * (
            ft_j[:, 1] * r[:, 2] + r[:, 1] * ft_j[:, 2] - 2.0 * r[:, 1] * r[:, 2] * rdfj
        )
        # strain_yy component
        strain_yy_i = +xg12 * (r[:, 1] * r[:, 1] - 1.0 / 3.0) * rdfj + yg12 * (
            2 * ft_j[:, 1] * r[:, 1] - 2.0 * r[:, 1] * r[:, 1] * rdfj
        )
        # compute strain for particles j
        # strain_xx component
        strain_xx_j = +xg12 * (r[:, 0] * r[:, 0] - 1.0 / 3.0) * (-rdfi) + yg12 * (
            2.0 * ft_i[:, 0] * (-r[:, 0]) - 2.0 * r[:, 0] * r[:, 0] * (-rdfi)
        )
        # strain_xy component
        strain_xy_j = +xg12 * (r[:, 0] * r[:, 1]) * (-rdfi) + yg12 * (
            ft_i[:, 0] * (-r[:, 1]) + (-r[:, 0]) * ft_i[:, 1] - 2.0 * r[:, 0] * r[:, 1] * (-rdfi)
        )
        # strain_xz component
        strain_xz_j = +xg12 * (r[:, 0] * r[:, 2]) * (-rdfi) + yg12 * (
            ft_i[:, 0] * (-r[:, 2]) + (-r[:, 0]) * ft_i[:, 2] - 2.0 * r[:, 0] * r[:, 2] * (-rdfi)
        )
        # strain_yz component
        strain_yz_j = +xg12 * (r[:, 1] * r[:, 2]) * (-rdfi) + yg12 * (
            ft_i[:, 1] * (-r[:, 2]) + (-r[:, 1]) * ft_i[:, 2] - 2.0 * r[:, 1] * r[:, 2] * (-rdfi)
        )
        # strain_yy component
        strain_yy_j = +xg12 * (r[:, 1] * r[:, 1] - 1.0 / 3.0) * (-rdfi) + yg12 * (
            2.0 * ft_i[:, 1] * (-r[:, 1]) - 2.0 * r[:, 1] * r[:, 1] * (-rdfi)
        )

        # rotational part

        # compute strain for particles i
        strain_xx_i += yh12 * (r[:, 0] * epsrdtj[0] + epsrdtj[0] * r[:, 0])
        strain_xy_i += yh12 * (r[:, 0] * epsrdtj[1] + epsrdtj[0] * r[:, 1])
        strain_xz_i += yh12 * (r[:, 0] * epsrdtj[2] + epsrdtj[0] * r[:, 2])
        strain_yz_i += yh12 * (r[:, 1] * epsrdtj[2] + epsrdtj[1] * r[:, 2])
        strain_yy_i += yh12 * (r[:, 1] * epsrdtj[1] + epsrdtj[1] * r[:, 1])

        # compute strain for particles j
        strain_xx_j += yh12 * (r[:, 0] * epsrdti[0] + epsrdti[0] * r[:, 0])
        strain_xy_j += yh12 * (r[:, 0] * epsrdti[1] + epsrdti[0] * r[:, 1])
        strain_xz_j += yh12 * (r[:, 0] * epsrdti[2] + epsrdti[0] * r[:, 2])
        strain_yz_j += yh12 * (r[:, 1] * epsrdti[2] + epsrdti[1] * r[:, 2])
        strain_yy_j += yh12 * (r[:, 1] * epsrdti[1] + epsrdti[1] * r[:, 1])

        strain = strain.at[indices_i, 0].add((2.0 * strain_xx_i + strain_yy_i))
        strain = strain.at[indices_i, 1].add(2.0 * strain_xy_i)
        strain = strain.at[indices_i, 2].add(2.0 * strain_xz_i)
        strain = strain.at[indices_i, 3].add(2.0 * strain_yz_i)
        strain = strain.at[indices_i, 4].add((strain_xx_i + 2.0 * strain_yy_i))

        strain = strain.at[indices_j, 0].add((2.0 * strain_xx_j + strain_yy_j))
        strain = strain.at[indices_j, 1].add(2.0 * strain_xy_j)
        strain = strain.at[indices_j, 2].add(2.0 * strain_xz_j)
        strain = strain.at[indices_j, 3].add(2.0 * strain_yz_j)
        strain = strain.at[indices_j, 4].add((strain_xx_j + 2.0 * strain_yy_j))

        # M_ES * S

        # compute strain for particles i
        # strain_xx component
        strain_xx_i = (
            9 / 10 * jnp.ones(n_pairs) * (s_i[0, 0, :]) / normaliz_factor
            + 1.5 * xm12 * (r[:, 0] * r[:, 0] - 1.0 / 3.0) * rdSdrj
            + 0.5 * ym12 * (4.0 * r[:, 0] * Sdrj[0] - 4.0 * rdSdrj * r[:, 0] * r[:, 0])
            + 0.5
            * zm12
            * (2.0 * s_j[0, 0, :] + (1.0 + r[:, 0] * r[:, 0]) * rdSdrj - 4.0 * r[:, 0] * Sdrj[0])
        )
        strain_xx_j = (
            9 / 10 * jnp.ones(n_pairs) * (s_j[0, 0, :]) / normaliz_factor
            + 1.5 * xm12 * (r[:, 0] * r[:, 0] - 1.0 / 3.0) * rdSdri
            + 0.5 * ym12 * (4.0 * r[:, 0] * Sdri[0] - 4.0 * rdSdri * r[:, 0] * r[:, 0])
            + 0.5
            * zm12
            * (2.0 * s_i[0, 0, :] + (1.0 + r[:, 0] * r[:, 0]) * rdSdri - 4.0 * r[:, 0] * Sdri[0])
        )

        # strain_xy component
        strain_xy_i = (
            9 / 10 * jnp.ones(n_pairs) * (s_i[0, 1, :]) / normaliz_factor
            + 1.5 * xm12 * (r[:, 0] * r[:, 1]) * rdSdrj
            + 0.5
            * ym12
            * (2.0 * r[:, 0] * Sdrj[1] + 2.0 * r[:, 1] * Sdrj[0] - 4.0 * rdSdrj * r[:, 0] * r[:, 1])
            + 0.5
            * zm12
            * (
                2.0 * s_j[0, 1, :]
                + (r[:, 0] * r[:, 1]) * rdSdrj
                - 2.0 * r[:, 0] * Sdrj[1]
                - 2.0 * r[:, 1] * Sdrj[0]
            )
        )
        strain_xy_j = (
            9 / 10 * jnp.ones(n_pairs) * (s_j[0, 1, :]) / normaliz_factor
            + 1.5 * xm12 * (r[:, 0] * r[:, 1]) * rdSdri
            + 0.5
            * ym12
            * (2.0 * r[:, 0] * Sdri[1] + 2.0 * r[:, 1] * Sdri[0] - 4.0 * rdSdri * r[:, 0] * r[:, 1])
            + 0.5
            * zm12
            * (
                2.0 * s_i[0, 1, :]
                + (r[:, 0] * r[:, 1]) * rdSdri
                - 2.0 * r[:, 0] * Sdri[1]
                - 2.0 * r[:, 1] * Sdri[0]
            )
        )

        # strain_xz component
        strain_xz_i = (
            9 / 10 * jnp.ones(n_pairs) * (s_i[0, 2, :]) / normaliz_factor
            + 1.5 * xm12 * (r[:, 0] * r[:, 2]) * rdSdrj
            + 0.5
            * ym12
            * (2.0 * r[:, 0] * Sdrj[2] + 2.0 * r[:, 2] * Sdrj[0] - 4.0 * rdSdrj * r[:, 0] * r[:, 2])
            + 0.5
            * zm12
            * (
                2.0 * s_j[0, 2, :]
                + (r[:, 0] * r[:, 2]) * rdSdrj
                - 2.0 * r[:, 0] * Sdrj[2]
                - 2.0 * r[:, 2] * Sdrj[0]
            )
        )
        strain_xz_j = (
            9 / 10 * jnp.ones(n_pairs) * (s_j[0, 2, :]) / normaliz_factor
            + 1.5 * xm12 * (r[:, 0] * r[:, 2]) * rdSdri
            + 0.5
            * ym12
            * (2.0 * r[:, 0] * Sdri[2] + 2.0 * r[:, 2] * Sdri[0] - 4.0 * rdSdri * r[:, 0] * r[:, 2])
            + 0.5
            * zm12
            * (
                2.0 * s_i[0, 2, :]
                + (r[:, 0] * r[:, 2]) * rdSdri
                - 2.0 * r[:, 0] * Sdri[2]
                - 2.0 * r[:, 2] * Sdri[0]
            )
        )

        # strain_yz component
        strain_yz_i = (
            9 / 10 * jnp.ones(n_pairs) * (s_i[1, 2, :]) / normaliz_factor
            + 1.5 * xm12 * (r[:, 1] * r[:, 2]) * rdSdrj
            + 0.5
            * ym12
            * (2.0 * r[:, 1] * Sdrj[2] + 2.0 * r[:, 2] * Sdrj[1] - 4.0 * rdSdrj * r[:, 1] * r[:, 2])
            + 0.5
            * zm12
            * (
                2.0 * s_j[1, 2, :]
                + (r[:, 1] * r[:, 2]) * rdSdrj
                - 2.0 * r[:, 1] * Sdrj[2]
                - 2.0 * r[:, 2] * Sdrj[1]
            )
        )
        strain_yz_j = (
            9 / 10 * jnp.ones(n_pairs) * (s_j[1, 2, :]) / normaliz_factor
            + 1.5 * xm12 * (r[:, 1] * r[:, 2]) * rdSdri
            + 0.5
            * ym12
            * (2.0 * r[:, 1] * Sdri[2] + 2.0 * r[:, 2] * Sdri[1] - 4.0 * rdSdri * r[:, 1] * r[:, 2])
            + 0.5
            * zm12
            * (
                2.0 * s_i[1, 2, :]
                + (r[:, 1] * r[:, 2]) * rdSdri
                - 2.0 * r[:, 1] * Sdri[2]
                - 2.0 * r[:, 2] * Sdri[1]
            )
        )

        # strain_yy component
        strain_yy_i = (
            9 / 10 * jnp.ones(n_pairs) * (s_i[1, 1, :]) / normaliz_factor
            + 1.5 * xm12 * (r[:, 1] * r[:, 1] - 1.0 / 3.0) * rdSdrj
            + 0.5 * ym12 * (4.0 * r[:, 1] * Sdrj[1] - 4.0 * rdSdrj * r[:, 1] * r[:, 1])
            + 0.5
            * zm12
            * (2.0 * s_j[1, 1, :] + (1.0 + r[:, 1] * r[:, 1]) * rdSdrj - 4.0 * r[:, 1] * Sdrj[1])
        )
        strain_yy_j = (
            9 / 10 * jnp.ones(n_pairs) * (s_j[1, 1, :]) / normaliz_factor
            + 1.5 * xm12 * (r[:, 1] * r[:, 1] - 1.0 / 3.0) * rdSdri
            + 0.5 * ym12 * (4.0 * r[:, 1] * Sdri[1] - 4.0 * rdSdri * r[:, 1] * r[:, 1])
            + 0.5
            * zm12
            * (2.0 * s_i[1, 1, :] + (1.0 + r[:, 1] * r[:, 1]) * rdSdri - 4.0 * r[:, 1] * Sdri[1])
        )

        strain = strain.at[indices_i, 0].add((2.0 * strain_xx_i + strain_yy_i))
        strain = strain.at[indices_i, 1].add(2.0 * strain_xy_i)
        strain = strain.at[indices_i, 2].add(2.0 * strain_xz_i)
        strain = strain.at[indices_i, 3].add(2.0 * strain_yz_i)
        strain = strain.at[indices_i, 4].add((strain_xx_i + 2.0 * strain_yy_i))

        strain = strain.at[indices_j, 0].add((2.0 * strain_xx_j + strain_yy_j))
        strain = strain.at[indices_j, 1].add(2.0 * strain_xy_j)
        strain = strain.at[indices_j, 2].add(2.0 * strain_xz_j)
        strain = strain.at[indices_j, 3].add(2.0 * strain_yz_j)
        strain = strain.at[indices_j, 4].add((strain_xx_j + 2.0 * strain_yy_j))

        velocities = jnp.ravel(velocities)
        strain = jnp.ravel(strain)
        gen_vel = jnp.zeros(11 * num_particles)
        # mobility is build to return the particle velocity (instead of -velocity)
        # this is because of the input 'r' and 'forces/torques' signs
        gen_vel = gen_vel.at[: 6 * num_particles].set(-velocities)
        gen_vel = gen_vel.at[6 * num_particles :].set(strain)

        return gen_vel

    def helper_compute_m12psi(
        n_iter_lanczos_ff: int, tridiagonal: ArrayLike, vectors: ArrayLike, norm: float
    ) -> ArrayLike:
        """Parameters
        ----------
        n_iter_lanczos_ff:
            Number of Lanczos iteration performed
        tridiagonal:
            Tridiagonal matrix obtained from Lanczos decomposition
        vectors:
            Vectors spanning Krylov subspace, obtained from Lanczos decomposition
        norm:
            Norm of initial random array

        Returns
        -------
        jnp.dot(vectors.T,jnp.dot(a,betae1)) * jnp.sqrt(2.0*temperature/dt)

        """
        betae1 = jnp.zeros(n_iter_lanczos_ff)
        betae1 = betae1.at[0].add(1 * norm)

        a, b = jnp.linalg.eigh(tridiagonal)
        a = jnp.where(a < 0, 0.0, a)
        a = jnp.dot((jnp.dot(b, jnp.diag(jnp.sqrt(a)))), b.T)
        return jnp.dot(vectors.T, jnp.dot(a, betae1))

    r = -r  # change sign of r go get correct output from helper_mpsi (do this only once)

    random_array_real = (2.0 * random_array_real - 1.0) * jnp.sqrt(3.0)
    trid, vectors = lanczos.lanczos_algorithm(helper_mpsi, 11 * num_particles, n_iter_lanczos_ff, random_array_real)

    psinorm = jnp.linalg.norm(random_array_real)
    m12_psi_old = helper_compute_m12psi(
        (n_iter_lanczos_ff - 1),
        trid[: (n_iter_lanczos_ff - 1), : (n_iter_lanczos_ff - 1)],
        vectors[: (n_iter_lanczos_ff - 1), :],
        psinorm,
    )
    m12_psi = helper_compute_m12psi(n_iter_lanczos_ff, trid, vectors, psinorm)
    buff = jnp.linalg.norm(m12_psi)
    stepnorm = jnp.linalg.norm(m12_psi - m12_psi_old)
    stepnorm = jnp.where(buff > 1.0, stepnorm / buff, stepnorm)
    m12_psi = m12_psi * jnp.sqrt(2.0 * temperature / dt)
    # combine w_lin_velocities, w_ang_vel_and_strain
    lin_vel = m12_psi.at[: 3 * num_particles].get()
    ang_vel_and_strain = m12_psi.at[3 * num_particles :].get()

    r = -r  # restore sign of r

    return lin_vel, ang_vel_and_strain, stepnorm, trid


@partial(jit, static_argnums=[0, 1, 2, 3, 4])
def compute_wave_space_slipvelocity(
    num_particles: int,
    grid_nx: int,
    grid_ny: int,
    grid_nz: int,
    gaussP: int,
    temperature: float,
    dt: float,
    gridh: ArrayLike,
    normal_indices_x: ArrayLike,
    normal_indices_y: ArrayLike,
    normal_indices_z: ArrayLike,
    normal_conj_indices_x: ArrayLike,
    normal_conj_indices_y: ArrayLike,
    normal_conj_indices_z: ArrayLike,
    nyquist_indices_x: ArrayLike,
    nyquist_indices_y: ArrayLike,
    nyquist_indices_z: ArrayLike,
    gridk: ArrayLike,
    random_array_wave: ArrayLike,
    all_indices_x: ArrayLike,
    all_indices_y: ArrayLike,
    all_indices_z: ArrayLike,
    gaussian_grid_spacing1: ArrayLike,
    gaussian_grid_spacing2: ArrayLike,
) -> tuple[Array, Array]:
    """Compute wave space far-field thermal fluctuation.

    Parameters
    ----------
    num_particles: (int)
        Number of particles
    grid_nx: (int)
        Number of grid points in x direction
    grid_ny: (int)
        Number of grid points in y direction
    grid_nz: (int)
        Number of grid points in z direction
    gaussP: (int)
        Gaussian support size for wave space calculation
    temperature: (float)
        Thermal energy
    dt: (float)
        Time step
    gridh: (float)
        Array (,3) containing wave space grid discrete spacing
    normal_indices_x: (int)
        Indexing for wave space grid, x component
    normal_indices_y: (int)
        Indexing for wave space grid, y component
    normal_indices_z: (int)
        Indexing for wave space grid, z component
    normal_conj_indices_x: (int)
        Indexing for wave space grid, x component, of conjugate points
    normal_conj_indices_y: (int)
        Indexing for wave space grid, y component, of conjugate points
    normal_conj_indices_z: (int)
        Indexing for wave space grid, z component, of conjugate points
    nyquist_indices_x: (int)
        Indexing for wave space grid, x component, of nyquist points
    nyquist_indices_y: (int)
        Indexing for wave space grid, y component, of nyquist points
    nyquist_indices_z: (int)
        Indexing for wave space grid, z component, of nyquist points
    gridk: (float)
        Array (grid_nx,grid_ny,grid_nz,4) containing wave vectors and scaling factors for far-field wavespace calculation
    random_array_wave: (float)
        Array (3 * 2 * len(normal_indices_x) + 3 * len(nyquist_indices_x))) of random numbers with the proper variance
    all_indices_x: (int)
        Array (,num_particles*gaussP*gaussP*gaussP) containing all the x-indices of wave grid points overlapping with each particle Gaussian support
    all_indices_y: (int)
        Array (,num_particles*gaussP*gaussP*gaussP) containing all the y-indices of wave grid points overlapping with each particle Gaussian support
    all_indices_z: (int)
        Array (,num_particles*gaussP*gaussP*gaussP) containing all the z-indices of wave grid points overlapping with each particle Gaussian support
    gaussian_grid_spacing1: (float)
        Array (,gaussP*gaussP*gaussP) containing scaled distances from support center to each gridpoint in the gaussian support (for FFT)
    gaussian_grid_spacing2: (float)
        Array (,gaussP*gaussP*gaussP) containing scaled distances from support center to each gridpoint in the gaussian support (for inverse FFT)

    Returns
    -------
    w_lin_vel, w_ang_vel_and_strain

    """
    grid_x = jnp.zeros((grid_nx, grid_ny, grid_nz))
    grid_y = jnp.zeros((grid_nx, grid_ny, grid_nz))
    grid_z = jnp.zeros((grid_nx, grid_ny, grid_nz))
    grid_xx = jnp.zeros((grid_nx, grid_ny, grid_nz))
    grid_xy = jnp.zeros((grid_nx, grid_ny, grid_nz))
    grid_xz = jnp.zeros((grid_nx, grid_ny, grid_nz))
    grid_yx = jnp.zeros((grid_nx, grid_ny, grid_nz))
    grid_yy = jnp.zeros((grid_nx, grid_ny, grid_nz))
    grid_yz = jnp.zeros((grid_nx, grid_ny, grid_nz))
    grid_zx = jnp.zeros((grid_nx, grid_ny, grid_nz))
    grid_zy = jnp.zeros((grid_nx, grid_ny, grid_nz))

    ### WAVE SPACE part
    grid_x = jnp.array(grid_x, complex)
    grid_y = jnp.array(grid_y, complex)
    grid_z = jnp.array(grid_z, complex)

    fac = jnp.sqrt(3.0 * temperature / dt / (gridh[0] * gridh[1] * gridh[2]))
    random_array_wave = (2.0 * random_array_wave - 1) * fac

    len_norm_indices = len(normal_indices_x)
    len_nyquist_indices = len(nyquist_indices_x)

    ###############################################################################################################################################
    grid_x = grid_x.at[normal_indices_x, normal_indices_y, normal_indices_z].add(
        random_array_wave.at[:len_norm_indices].get()
        + 1j * random_array_wave.at[len_norm_indices : 2 * len_norm_indices].get()
    )
    grid_x = grid_x.at[normal_conj_indices_x, normal_conj_indices_y, normal_conj_indices_z].add(
        random_array_wave.at[:len_norm_indices].get()
        - 1j * random_array_wave.at[len_norm_indices : 2 * len_norm_indices].get()
    )
    grid_x = grid_x.at[nyquist_indices_x, nyquist_indices_y, nyquist_indices_z].add(
        random_array_wave.at[
            6 * len_norm_indices : (6 * len_norm_indices + 1 * len_nyquist_indices)
        ].get()
        * 1.414213562373095
        + 0j
    )

    grid_y = grid_y.at[normal_indices_x, normal_indices_y, normal_indices_z].add(
        random_array_wave.at[(2 * len_norm_indices) : (3 * len_norm_indices)].get()
        + 1j * random_array_wave.at[(3 * len_norm_indices) : (4 * len_norm_indices)].get()
    )
    grid_y = grid_y.at[normal_conj_indices_x, normal_conj_indices_y, normal_conj_indices_z].add(
        random_array_wave.at[(2 * len_norm_indices) : (3 * len_norm_indices)].get()
        - 1j * random_array_wave.at[(3 * len_norm_indices) : (4 * len_norm_indices)].get()
    )
    grid_y = grid_y.at[nyquist_indices_x, nyquist_indices_y, nyquist_indices_z].add(
        random_array_wave.at[
            (6 * len_norm_indices + 1 * len_nyquist_indices) : (
                6 * len_norm_indices + 2 * len_nyquist_indices
            )
        ].get()
        * 1.414213562373095
        + 0j
    )

    grid_z = grid_z.at[normal_indices_x, normal_indices_y, normal_indices_z].add(
        random_array_wave.at[(4 * len_norm_indices) : (5 * len_norm_indices)].get()
        + 1j * random_array_wave.at[(5 * len_norm_indices) : (6 * len_norm_indices)].get()
    )
    grid_z = grid_z.at[normal_conj_indices_x, normal_conj_indices_y, normal_conj_indices_z].add(
        random_array_wave.at[(4 * len_norm_indices) : (5 * len_norm_indices)].get()
        - 1j * random_array_wave.at[(5 * len_norm_indices) : (6 * len_norm_indices)].get()
    )
    grid_z = grid_z.at[nyquist_indices_x, nyquist_indices_y, nyquist_indices_z].add(
        random_array_wave.at[
            (6 * len_norm_indices + 2 * len_nyquist_indices) : (
                6 * len_norm_indices + 3 * len_nyquist_indices
            )
        ].get()
        * 1.414213562373095
        + 0j
    )
    ###############################################################################################################################################

    # Compute k^2 and (|| k ||)
    gridk_sqr = (
        gridk.at[:, :, :, 0].get() * gridk.at[:, :, :, 0].get()
        + gridk.at[:, :, :, 1].get() * gridk.at[:, :, :, 1].get()
        + gridk.at[:, :, :, 2].get() * gridk.at[:, :, :, 2].get()
    )
    gridk_mod = jnp.sqrt(gridk_sqr)

    # Scaling factors
    b = jnp.where(gridk_mod > 0, jnp.sqrt(gridk.at[:, :, :, 3].get()), 0.0)
    su = jnp.where(gridk_mod > 0, jnp.sin(gridk_mod) / gridk_mod, 0.0)
    sd = jnp.where(
        gridk_mod > 0,
        3.0 * (jnp.sin(gridk_mod) - gridk_mod * jnp.cos(gridk_mod)) / (gridk_sqr * gridk_mod),
        0.0,
    )

    # Conjugate
    sd = -1.0 * sd

    # Square root of Green's function times dW
    kdf = jnp.where(
        gridk_mod > 0.0,
        (
            gridk.at[:, :, :, 0].get() * grid_x
            + gridk.at[:, :, :, 1].get() * grid_y
            + gridk.at[:, :, :, 2].get() * grid_z
        )
        / gridk_sqr,
        0,
    )

    bdwx = (grid_x - gridk.at[:, :, :, 0].get() * kdf) * b
    bdwy = (grid_y - gridk.at[:, :, :, 1].get() * kdf) * b
    bdwz = (grid_z - gridk.at[:, :, :, 2].get() * kdf) * b

    bdwkxx = bdwx * gridk.at[:, :, :, 0].get()
    bdwkxy = bdwx * gridk.at[:, :, :, 1].get()
    bdwkxz = bdwx * gridk.at[:, :, :, 2].get()
    bdwkyx = bdwy * gridk.at[:, :, :, 0].get()
    bdwkyy = bdwy * gridk.at[:, :, :, 1].get()
    bdwkyz = bdwy * gridk.at[:, :, :, 2].get()
    bdwkzx = bdwz * gridk.at[:, :, :, 0].get()
    bdwkzy = bdwz * gridk.at[:, :, :, 1].get()

    grid_x = su * bdwx
    grid_y = su * bdwy
    grid_z = su * bdwz
    grid_xx = sd * (-jnp.imag(bdwkxx) + 1j * jnp.real(bdwkxx))
    grid_xy = sd * (-jnp.imag(bdwkxy) + 1j * jnp.real(bdwkxy))
    grid_xz = sd * (-jnp.imag(bdwkxz) + 1j * jnp.real(bdwkxz))
    grid_yx = sd * (-jnp.imag(bdwkyx) + 1j * jnp.real(bdwkyx))
    grid_yy = sd * (-jnp.imag(bdwkyy) + 1j * jnp.real(bdwkyy))
    grid_yz = sd * (-jnp.imag(bdwkyz) + 1j * jnp.real(bdwkyz))
    grid_zx = sd * (-jnp.imag(bdwkzx) + 1j * jnp.real(bdwkzx))
    grid_zy = sd * (-jnp.imag(bdwkzy) + 1j * jnp.real(bdwkzy))

    # Return rescaled forces to real space (Inverse FFT)
    grid_x = jnp.real(jnp.fft.ifftn(grid_x, norm="forward"))
    grid_y = jnp.real(jnp.fft.ifftn(grid_y, norm="forward"))
    grid_z = jnp.real(jnp.fft.ifftn(grid_z, norm="forward"))
    grid_xx = jnp.real(jnp.fft.ifftn(grid_xx, norm="forward"))
    grid_xy = jnp.real(jnp.fft.ifftn(grid_xy, norm="forward"))
    grid_xz = jnp.real(jnp.fft.ifftn(grid_xz, norm="forward"))
    grid_yx = jnp.real(jnp.fft.ifftn(grid_yx, norm="forward"))
    grid_yy = jnp.real(jnp.fft.ifftn(grid_yy, norm="forward"))
    grid_yz = jnp.real(jnp.fft.ifftn(grid_yz, norm="forward"))
    grid_zx = jnp.real(jnp.fft.ifftn(grid_zx, norm="forward"))
    grid_zy = jnp.real(jnp.fft.ifftn(grid_zy, norm="forward"))

    # Compute Linear velocities and Velocity gradients (from which we can extract angular velocities and rate of strain)
    w_lin_velocities = jnp.zeros((num_particles, 3), float)
    w_velocity_gradient = jnp.zeros((num_particles, 8), float)

    w_lin_velocities = w_lin_velocities.at[:, 0].add(
        jnp.sum(
            gaussian_grid_spacing2
            * jnp.reshape(
                grid_x.at[all_indices_x, all_indices_y, all_indices_z].get(),
                (num_particles, gaussP, gaussP, gaussP),
            ),
            axis=(1, 2, 3),
        )
    )
    w_lin_velocities = w_lin_velocities.at[:, 1].add(
        jnp.sum(
            gaussian_grid_spacing2
            * jnp.reshape(
                grid_y.at[all_indices_x, all_indices_y, all_indices_z].get(),
                (num_particles, gaussP, gaussP, gaussP),
            ),
            axis=(1, 2, 3),
        )
    )
    w_lin_velocities = w_lin_velocities.at[:, 2].add(
        jnp.sum(
            gaussian_grid_spacing2
            * jnp.reshape(
                grid_z.at[all_indices_x, all_indices_y, all_indices_z].get(),
                (num_particles, gaussP, gaussP, gaussP),
            ),
            axis=(1, 2, 3),
        )
    )

    w_velocity_gradient = w_velocity_gradient.at[:, 0].add(
        jnp.sum(
            gaussian_grid_spacing2
            * jnp.reshape(
                grid_xx.at[all_indices_x, all_indices_y, all_indices_z].get(),
                (num_particles, gaussP, gaussP, gaussP),
            ),
            axis=(1, 2, 3),
        )
    )

    # index might be 1 instead if 5 (originally is 1, and so on for the ones below)
    w_velocity_gradient = w_velocity_gradient.at[:, 5].add(
        jnp.sum(
            gaussian_grid_spacing2
            * jnp.reshape(
                grid_xy.at[all_indices_x, all_indices_y, all_indices_z].get(),
                (num_particles, gaussP, gaussP, gaussP),
            ),
            axis=(1, 2, 3),
        )
    )
    # index might be 2 instead if 6
    w_velocity_gradient = w_velocity_gradient.at[:, 6].add(
        jnp.sum(
            gaussian_grid_spacing2
            * jnp.reshape(
                grid_xz.at[all_indices_x, all_indices_y, all_indices_z].get(),
                (num_particles, gaussP, gaussP, gaussP),
            ),
            axis=(1, 2, 3),
        )
    )
    # index might be 3 instead if 7
    w_velocity_gradient = w_velocity_gradient.at[:, 7].add(
        jnp.sum(
            gaussian_grid_spacing2
            * jnp.reshape(
                grid_yz.at[all_indices_x, all_indices_y, all_indices_z].get(),
                (num_particles, gaussP, gaussP, gaussP),
            ),
            axis=(1, 2, 3),
        )
    )

    w_velocity_gradient = w_velocity_gradient.at[:, 4].add(
        jnp.sum(
            gaussian_grid_spacing2
            * jnp.reshape(
                grid_yy.at[all_indices_x, all_indices_y, all_indices_z].get(),
                (num_particles, gaussP, gaussP, gaussP),
            ),
            axis=(1, 2, 3),
        )
    )

    # index might be 5 instead if 1
    w_velocity_gradient = w_velocity_gradient.at[:, 1].add(
        jnp.sum(
            gaussian_grid_spacing2
            * jnp.reshape(
                grid_yx.at[all_indices_x, all_indices_y, all_indices_z].get(),
                (num_particles, gaussP, gaussP, gaussP),
            ),
            axis=(1, 2, 3),
        )
    )
    # index might be 6 instead if 2
    w_velocity_gradient = w_velocity_gradient.at[:, 2].add(
        jnp.sum(
            gaussian_grid_spacing2
            * jnp.reshape(
                grid_zx.at[all_indices_x, all_indices_y, all_indices_z].get(),
                (num_particles, gaussP, gaussP, gaussP),
            ),
            axis=(1, 2, 3),
        )
    )
    # index might be 7 instead if 3
    w_velocity_gradient = w_velocity_gradient.at[:, 3].add(
        jnp.sum(
            gaussian_grid_spacing2
            * jnp.reshape(
                grid_zy.at[all_indices_x, all_indices_y, all_indices_z].get(),
                (num_particles, gaussP, gaussP, gaussP),
            ),
            axis=(1, 2, 3),
        )
    )

    # Convert to angular velocities and rate of strain
    w_ang_vel_and_strain = jnp.zeros((num_particles, 8))
    w_ang_vel_and_strain = w_ang_vel_and_strain.at[:, 0].add(
        (w_velocity_gradient.at[:, 3].get() - w_velocity_gradient.at[:, 7].get()) * 0.5
    )
    w_ang_vel_and_strain = w_ang_vel_and_strain.at[:, 1].add(
        (w_velocity_gradient.at[:, 6].get() - w_velocity_gradient.at[:, 2].get()) * 0.5
    )
    w_ang_vel_and_strain = w_ang_vel_and_strain.at[:, 2].add(
        (w_velocity_gradient.at[:, 1].get() - w_velocity_gradient.at[:, 5].get()) * 0.5
    )
    w_ang_vel_and_strain = w_ang_vel_and_strain.at[:, 3].add(
        2 * w_velocity_gradient.at[:, 0].get() + w_velocity_gradient.at[:, 4].get()
    )
    w_ang_vel_and_strain = w_ang_vel_and_strain.at[:, 4].add(
        w_velocity_gradient.at[:, 1].get() + w_velocity_gradient.at[:, 5].get()
    )
    w_ang_vel_and_strain = w_ang_vel_and_strain.at[:, 5].add(
        w_velocity_gradient.at[:, 2].get() + w_velocity_gradient.at[:, 6].get()
    )
    w_ang_vel_and_strain = w_ang_vel_and_strain.at[:, 6].add(
        w_velocity_gradient.at[:, 3].get() + w_velocity_gradient.at[:, 7].get()
    )
    w_ang_vel_and_strain = w_ang_vel_and_strain.at[:, 7].add(
        w_velocity_gradient.at[:, 0].get() + 2 * w_velocity_gradient.at[:, 4].get()
    )

    # combine w_lin_velocities, w_ang_vel_and_strain into generalized velocities
    w_lin_vel = jnp.reshape(w_lin_velocities, 3 * num_particles)
    w_ang_vel_and_strain = jnp.reshape(w_ang_vel_and_strain, 8 * num_particles)

    return w_lin_vel, w_ang_vel_and_strain


@partial(jit, static_argnums=[0, 19])
def compute_nearfield_brownianforce(
    num_particles: int,
    temperature: float,
    dt: float,
    random_array: ArrayLike,
    r_lub: ArrayLike,
    indices_i_lub: ArrayLike,
    indices_j_lub: ArrayLike,
    xa11: ArrayLike,
    xa12: ArrayLike,
    ya11: ArrayLike,
    ya12: ArrayLike,
    yb11: ArrayLike,
    yb12: ArrayLike,
    xc11: ArrayLike,
    xc12: ArrayLike,
    yc11: ArrayLike,
    yc12: ArrayLike,
    yb21: ArrayLike,
    # diagonal_elements_for_brownian: ArrayLike,
    # R_fu_prec_lower_triang: ArrayLike,
    diagonal_zeroes_for_brownian: ArrayLike,
    n_iter_lanczos_nf: int,
) -> tuple[Array, float, Array]:
    """Compute near-field thermal fluctuation.

    Here, the square root of the resistance lubrication operator is performed using Lanczos decomposition.

    Parameters
    ----------
    num_particles: (int)
        Number of particles
    temperature: (float)
        Thermal energy
    dt: (float)
        Time step
    random_array: (float)
        Array (,6*num_particles) of random numbers with the proper variance
    r_lub: (float)
        Array (n_pair_nf,3) containing units vectors connecting each pair of particles in the near-field neighbor list
    indices_i_lub: (int)
        Array (,n_pair_nf) containing indices of first particle in near-field neighbor list pairs
    indices_j_lub: (int)
        Array (,n_pair_nf) containing indices of second particle in near-field neighbor list pairs
    xa11: (float)
        Array (,n_pair_nf) containing resistance scalar function evaluated for the current particle configuration
    xa12: (float)
        Array (,n_pair_nf) containing resistance scalar function evaluated for the current particle configuration
    ya11: (float)
        Array (,n_pair_nf) containing resistance scalar function evaluated for the current particle configuration
    ya12: (float)
        Array (,n_pair_nf) containing resistance scalar function evaluated for the current particle configuration
    yb11: (float)
        Array (,n_pair_nf) containing resistance scalar function evaluated for the current particle configuration
    yb12: (float)
        Array (,n_pair_nf) containing resistance scalar function evaluated for the current particle configuration
    xc11: (float)
        Array (,n_pair_nf) containing resistance scalar function evaluated for the current particle configuration
    xc12: (float)
        Array (,n_pair_nf) containing resistance scalar function evaluated for the current particle configuration
    yc11: (float)
        Array (,n_pair_nf) containing resistance scalar function evaluated for the current particle configuration
    yc12: (float)
        Array (,n_pair_nf) containing resistance scalar function evaluated for the current particle configuration
    yb21: (float)
        Array (,n_pair_nf) containing resistance scalar function evaluated for the current particle configuration
    diagonal_elements_for_brownian: (float)
        Elements needed to precondition large eigenvalues of lubrication matrix
    R_fu_prec_lower_triang: (float)
        Lower triangular Cholesky factor of lubrication resistance matrix R_FU
    diagonal_zeroes_for_brownian: (int)
        Elements needed to apply near-field thermal noise only on particles in lubrication cutoff
    n_iter_lanczos_nf: (int)
        Number of Lanczos iteration to perform

    Returns
    -------
    r_fu12psi, stepnorm, trid

    """
    # def Precondition_DiagMult_kernel(
    #         x: float,
    #         diag: float,
    #         direction: int) -> tuple:

    #     """Apply precondition for small eigenvalues of lubrication resistance matrix.

    #     Parameters
    #     ----------
    #     x:
    #         Input matrix-vector product
    #     diag:
    #         Diagonal elements used to precondition
    #     direction:
    #         Time step

    #     Returns
    #     -------
    #     x*jnp.where(direction==1,diag, 1/diag)

    #     """

    #     return x*jnp.where(direction==1,diag, 1/diag)

    def precondition_projector_pair(  ###
        x: ArrayLike, diagonal_zeroes_for_brownian: ArrayLike
    ) -> ArrayLike:
        """Project out noise related to pair of particles not close enough.

        Parameters
        ----------
        x:
            Input matrix-vector product
        diagonal_zeroes_for_brownian:
            Number of neighbors for each particle, considering the lubrication cutoff

        Returns
        -------
        x

        """
        identity = jnp.where(
            (jnp.arange(6 * num_particles) - 6 * (jnp.repeat(jnp.arange(num_particles), 6))) < 3, 1, 1.333333333
        )
        x = jnp.where(diagonal_zeroes_for_brownian == 0.0, x * identity, 0.0)
        return x

    def precondition_projector_pair_undo(
        x: ArrayLike, diagonal_zeroes_for_brownian: ArrayLike
    ) -> ArrayLike:
        """Project out noise related to pair of particles close enough (undo the original projection).

        Parameters
        ----------
        x:
            Input matrix-vector product
        diagonal_zeroes_for_brownian:
            Number of neighbors for each particle, considering the lubrication cutoff

        Returns
        -------
        x

        """
        x = jnp.where(diagonal_zeroes_for_brownian == 0.0, 0.0, x)
        return x

    def precondition_brownian_rfu_multiply(psi: ArrayLike) -> ArrayLike:
        """Precondition for Lanczos iterative computation of square root of lubrication matrix.

        Parameters
        ----------
        psi:
            Input matrix-vector product

        Returns
        -------
        x

        """
        # Apply precodintion for large eigenvalues - part 1
        # psi = jscipy.linalg.solve_triangular(jnp.transpose(R_fu_prec_lower_triang), psi, lower=False)

        # Apply precodintion for small eigenvalues - part 1
        # psi = Precondition_DiagMult_kernel(psi, diagonal_elements_for_brownian, 1) #more preconditioning (not needed for now)

        z = compute_lubrication_fu(psi)
        z += precondition_projector_pair(psi, diagonal_zeroes_for_brownian)

        # Apply precodintion for small eigenvalues - part 2
        # z = Precondition_DiagMult_kernel(z,diagonal_elements_for_brownian, 1) #more preconditioning (not needed for now)

        # Apply precodintion for large eigenvalues - part 2
        # return jscipy.linalg.solve_triangular(R_fu_prec_lower_triang, z, lower=True)

        return z

    def precondition_brownian_undo(nf_brownian_force: ArrayLike) -> ArrayLike:
        """Undo the precondition used for Lanczos iterative computation of square root of lubrication matrix.

        Parameters
        ----------
        nf_brownian_force:
            Input Brownian forces from lubrication interactions

        Returns
        -------
        precondition_projector_pair_undo(nf_brownian_force, diagonal_zeroes_for_brownian)

        """
        # undo large eigenvalue precondition
        # nf_brownian_force = jnp.dot(R_fu_prec_lower_triang,nf_brownian_force)
        # undo small eigenvalue precondition
        # nf_brownian_force = Precondition_DiagMult_kernel(nf_brownian_force,diagonal_elements_for_brownian,-1)
        return precondition_projector_pair_undo(nf_brownian_force, diagonal_zeroes_for_brownian)

    def compute_lubrication_fu(velocities: ArrayLike) -> ArrayLike:
        """Compute matrix-vector product of R_FU with a velocity vector

        Parameters
        ----------
        velocities:
            Input particles velocities.

        Returns
        -------
        jnp.ravel(forces)

        """
        vel_i = (jnp.reshape(velocities, (num_particles, 6))).at[indices_i_lub].get()
        vel_j = (jnp.reshape(velocities, (num_particles, 6))).at[indices_j_lub].get()

        # Dot product of r and U, i.e. axisymmetric projection
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

        # Cross product of U and r, i.e. eps_ijk*r_k*U_j = Px dot U, (eps_ijk is the Levi-Civita symbol)
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

    def helper_compute_R12psi(
        n_iter_lanczos_nf: ArrayLike, trid: ArrayLike, vectors: ArrayLike
    ) -> ArrayLike:
        """Parameters
        ----------
        n_iter_lanczos_nf:
            Number of Lanczos iteration performed
        trid:
            Tridiagonal matrix obtained from Lanczos decomposition
        vectors:
            Vectors spanning Krylov subspace, obtained from Lanczos decomposition

        Returns
        -------
        jnp.dot(vectors.T,jnp.dot(a,betae1)) * jnp.sqrt(2.0*temperature/dt)

        """
        betae1 = jnp.zeros(n_iter_lanczos_nf)
        betae1 = betae1.at[0].add(1 * psinorm)
        a, b = jnp.linalg.eigh(trid)
        a = jnp.where(a < 0.0, 0.0, a)  # numerical cutoff to avoid small negative values
        a = jnp.dot((jnp.dot(b, jnp.diag(jnp.sqrt(a)))), b.T)

        return jnp.dot(vectors.T, jnp.dot(a, betae1))

    # Scale random numbers from [0,1] to [-sqrt(3),sqrt(3)]
    random_array = (2 * random_array - 1) * jnp.sqrt(3.0)

    psinorm = jnp.linalg.norm(random_array)
    trid, vectors = lanczos.lanczos_algorithm(
        precondition_brownian_rfu_multiply, 6 * num_particles, n_iter_lanczos_nf, random_array
    )

    r_fu12psi_old = helper_compute_R12psi(
        (n_iter_lanczos_nf - 1),
        trid[: (n_iter_lanczos_nf - 1), : (n_iter_lanczos_nf - 1)],
        vectors[: (n_iter_lanczos_nf - 1), :],
    )
    r_fu12psi = helper_compute_R12psi(n_iter_lanczos_nf, trid, vectors)

    buff = jnp.linalg.norm(r_fu12psi)
    stepnorm = jnp.linalg.norm((r_fu12psi - r_fu12psi_old)) / buff
    r_fu12psi = r_fu12psi * jnp.sqrt(2.0 * temperature / dt)
    r_fu12psi = precondition_brownian_undo(r_fu12psi)
    return r_fu12psi, stepnorm, trid


@partial(jit, static_argnums=[0])
def compute_bd_randomforce(num_particles: int, temperature: float, dt: float, random_array: ArrayLike) -> Array:
    """Compute thermal fluctuation for Brownian Dynamics.

    Parameters
    ----------
    num_particles: (int)
        Number of particles
    temperature: (float)
        Thermal energy
    dt: (float)
        Time step
    random_array: (float)
        Array (,6*num_particles) of random numbers with the proper variance

    Returns
    -------
    random_velocity

    """
    # scale random numbers from [0,1] to [-sqrt(3),sqrt(3)] to obtain
    # a random variable with zero mean and unit variance
    random_velocity = (2 * random_array - 1) * jnp.sqrt(3.0)

    # translational drag coefficient for a sphere is 1 in simulation units (6*pi*eta*a)
    drag_coeff = jnp.ones(6 * num_particles)
    # rotational drag coefficient for a sphere is 4/3 in simulation units (8*pi*eta*a)
    drag_coeff = drag_coeff.at[3::6].set(4 / 3)
    drag_coeff = drag_coeff.at[4::6].set(4 / 3)
    drag_coeff = drag_coeff.at[5::6].set(4 / 3)

    # scale by proper factor to satisfy fluctuation-dissipation theorem
    random_velocity = random_velocity * jnp.sqrt(2.0 * temperature / dt / drag_coeff)

    return random_velocity


@partial(jit, static_argnums=[0])
def convert_to_generalized(
    num_particles: int,
    ws_lin_vel: ArrayLike,
    rs_lin_vel: ArrayLike,
    ws_ang_vel_strain: ArrayLike,
    rs_ang_vel_strain: ArrayLike,
) -> Array:
    """Combine linear/angular velocities and rate of strain into a generalized velocity vector.

    Parameters
    ----------
    num_particles: (int)
        Number of particles
    ws_lin_vel: (float)
        Array (,3*num_particles) containing wave-space linear velocity
    rs_lin_vel: (float)
        Array (,3*num_particles) containing real-space linear velocity
    ws_ang_vel_strain: (float)
        Array (,8*num_particles) containing wave-space angular velocity and rate of strain
    ws_ang_vel_strain: (float)
        Array (,8*num_particles) containing real-space angular velocity and rate of strain

    Returns
    -------
    generalized_velocities

    """
    lin_vel = ws_lin_vel + rs_lin_vel
    ang_vel_and_strain = ws_ang_vel_strain + rs_ang_vel_strain

    # Convert to Generalized Velocities+strain
    generalized_velocities = jnp.zeros(
        11 * num_particles
    )  # First 6N entries for U and last 5N for strain rates

    generalized_velocities = generalized_velocities.at[0 : 6 * num_particles : 6].add(lin_vel.at[0::3].get())
    generalized_velocities = generalized_velocities.at[1 : 6 * num_particles : 6].add(lin_vel.at[1::3].get())
    generalized_velocities = generalized_velocities.at[2 : 6 * num_particles : 6].add(lin_vel.at[2::3].get())
    generalized_velocities = generalized_velocities.at[3 : 6 * num_particles : 6].add(
        ang_vel_and_strain.at[0::8].get()
    )
    generalized_velocities = generalized_velocities.at[4 : 6 * num_particles : 6].add(
        ang_vel_and_strain.at[1::8].get()
    )
    generalized_velocities = generalized_velocities.at[5 : 6 * num_particles : 6].add(
        ang_vel_and_strain.at[2::8].get()
    )
    generalized_velocities = generalized_velocities.at[(6 * num_particles + 0) :: 5].add(
        ang_vel_and_strain.at[3::8].get()
    )
    generalized_velocities = generalized_velocities.at[(6 * num_particles + 1) :: 5].add(
        ang_vel_and_strain.at[4::8].get()
    )
    generalized_velocities = generalized_velocities.at[(6 * num_particles + 2) :: 5].add(
        ang_vel_and_strain.at[5::8].get()
    )
    generalized_velocities = generalized_velocities.at[(6 * num_particles + 3) :: 5].add(
        ang_vel_and_strain.at[6::8].get()
    )
    generalized_velocities = generalized_velocities.at[(6 * num_particles + 4) :: 5].add(
        ang_vel_and_strain.at[7::8].get()
    )

    return generalized_velocities


def compute_exact_thermals(
    num_particles: int,
    m_self: ArrayLike,
    temperature: float,
    dt: float,
    random_array_nf: ArrayLike,
    random_array_real: ArrayLike,
    r: ArrayLike,
    indices_i: ArrayLike,
    indices_j: ArrayLike,
    f1,
    f2,
    g1,
    g2,
    h1,
    h2,
    h3: ArrayLike,
    r_lub: ArrayLike,
    indices_i_lub: ArrayLike,
    indices_j_lub: ArrayLike,
    xa11,
    xa12,
    ya11,
    ya12,
    yb11,
    yb12,
    xc11,
    xc12,
    yc11,
    yc12,
    yb21: ArrayLike,
) -> tuple[Array, Array]:
    """Compute square root of real-space granmobility and lubrication resistance using scipy functions.

    These are then used to test the correctness of the square roots (of these operators) obtained from Lanczos decomposition.

    Parameters
    ----------
    num_particles: (int)
        Number of Particles
    m_self: (float)
        Array (,2) containing mobility self contributions
    temperature: (float)
        Thermal energy
    dt: (float)
        Time step
    random_array_nf: (float)
        Array (,6*num_particles) of random numbers with the proper variance, for thermal lubrication
    random_array_real: (float)
        Array (,11*num_particles) of random numbers with the proper variance, for thermal far-field real space
    r: (float)
        Array (n_pair_ff,3) containing units vectors connecting each pair of particles in the far-field neighbor list
    indices_i: (int)
        Array (n_pair_ff) of indices of first particle in neighbor list pairs
    indices_j: (int)
        Array (n_pair_ff) of indices of second particle in neighbor list pairs
    f1,f2,g1,g2,h1,h2,h3: (float)
        Array (,n_pair_ff) containing mobility scalar function evaluated for the current particle configuration
    r_lub: (float)
        Array (n_pair_nf,3) containing units vectors connecting each pair of particles in the near-field neighbor list
    indices_i_lub: (int)
        Array (,n_pair_nf) containing indices of first particle in near-field neighbor list pairs
    indices_j_lub: (int)
        Array (,n_pair_nf) containing indices of second particle in near-field neighbor list pairs
    xa11,xa12,ya11,ya12,yb11,yb12,xc11,xc12,yc11,yc12,yb21: (float)
        Array (,n_pair_nf) containing resistance scalar function evaluated for the current particle configuration

    Returns
    -------
    generalized_velocities

    """

    @jit
    def convert_to_generalized(M12psi: ArrayLike) -> Array:
        """Combine linear/angular velocities and rate of strain into a generalized velocity vector.

        Parameters
        ----------
        M12psi:
            Velocity Linear+Angular/Strain

        Returns
        -------
        generalized_velocities

        """
        # combine w_lin_velocities, w_ang_vel_and_strain
        lin_vel = M12psi.at[: 3 * num_particles].get()
        ang_vel_and_strain = M12psi.at[3 * num_particles :].get()
        # Convert to Generalized Velocities+strain
        generalized_velocities = jnp.zeros(
            11 * num_particles
        )  # First 6N entries for U and last 5N for strain rates

        generalized_velocities = generalized_velocities.at[0 : 6 * num_particles : 6].set(
            lin_vel.at[0::3].get()
        )
        generalized_velocities = generalized_velocities.at[1 : 6 * num_particles : 6].set(
            lin_vel.at[1::3].get()
        )
        generalized_velocities = generalized_velocities.at[2 : 6 * num_particles : 6].set(
            lin_vel.at[2::3].get()
        )
        generalized_velocities = generalized_velocities.at[3 : 6 * num_particles : 6].set(
            ang_vel_and_strain.at[0::8].get()
        )
        generalized_velocities = generalized_velocities.at[4 : 6 * num_particles : 6].set(
            ang_vel_and_strain.at[1::8].get()
        )
        generalized_velocities = generalized_velocities.at[5 : 6 * num_particles : 6].set(
            ang_vel_and_strain.at[2::8].get()
        )
        generalized_velocities = generalized_velocities.at[(6 * num_particles + 0) :: 5].set(
            ang_vel_and_strain.at[3::8].get()
        )
        generalized_velocities = generalized_velocities.at[(6 * num_particles + 1) :: 5].set(
            ang_vel_and_strain.at[4::8].get()
        )
        generalized_velocities = generalized_velocities.at[(6 * num_particles + 2) :: 5].set(
            ang_vel_and_strain.at[5::8].get()
        )
        generalized_velocities = generalized_velocities.at[(6 * num_particles + 3) :: 5].set(
            ang_vel_and_strain.at[6::8].get()
        )
        generalized_velocities = generalized_velocities.at[(6 * num_particles + 4) :: 5].set(
            ang_vel_and_strain.at[7::8].get()
        )

        return generalized_velocities

    @jit
    def helper_reshape(Mpsi: ArrayLike) -> Array:
        """Reshape input array.

        Parameters
        ----------
        Mpsi:
            Input array

        Returns
        -------
        generalized_velocities

        """
        lin_vel = Mpsi[0]
        ang_vel_and_strain = Mpsi[1]
        reshaped_array = jnp.zeros(11 * num_particles)
        reshaped_array = reshaped_array.at[: 3 * num_particles].set(jnp.reshape(lin_vel, 3 * num_particles))
        reshaped_array = reshaped_array.at[3 * num_particles :].set(jnp.reshape(ang_vel_and_strain, 8 * num_particles))
        return reshaped_array

    @jit
    def compute_lubrication_fu(velocities):
        vel_i = (jnp.reshape(velocities, (num_particles, 6))).at[indices_i_lub].get()
        vel_j = (jnp.reshape(velocities, (num_particles, 6))).at[indices_j_lub].get()

        # Dot product of r and U, i.e. axisymmetric projection
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

        # Cross product of U and r, i.e. eps_ijk*r_k*U_j = Px dot U, (eps_ijk is the Levi-Civita symbol)
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

    @jit
    def helper_mpsi(random_array_real):
        # input is already in the format: [Forces, Torque+Stresslet] (not in the generalized format [Force+Torque,Stresslet] like in the saddle point solver)
        forces = random_array_real.at[: 3 * num_particles].get()

        couplets = jnp.zeros(8 * num_particles)
        couplets = couplets.at[::8].set(random_array_real.at[(3 * num_particles + 3) :: 8].get())  # C[0] = S[0]
        couplets = couplets.at[1::8].set(
            random_array_real.at[(3 * num_particles + 4) :: 8].get()
            + random_array_real.at[(3 * num_particles + 2) :: 8].get() * 0.5
        )  # C[1] = S[1] + L[2]/2
        couplets = couplets.at[2::8].set(
            random_array_real.at[(3 * num_particles + 5) :: 8].get()
            - random_array_real.at[(3 * num_particles + 1) :: 8].get() * 0.5
        )  # C[2] = S[2] - L[1]/2
        couplets = couplets.at[3::8].set(
            random_array_real.at[(3 * num_particles + 6) :: 8].get()
            + random_array_real.at[(3 * num_particles + 0) :: 8].get() * 0.5
        )  # C[3] = S[3] + L[0]/2
        couplets = couplets.at[4::8].set(
            random_array_real.at[(3 * num_particles + 7) :: 8].get()
        )  # C[4] = S[4]
        couplets = couplets.at[5::8].set(
            random_array_real.at[(3 * num_particles + 4) :: 8].get()
            - random_array_real.at[(3 * num_particles + 2) :: 8].get() * 0.5
        )  # C[5] = S[1] - L[2]/2
        couplets = couplets.at[6::8].set(
            random_array_real.at[(3 * num_particles + 5) :: 8].get()
            + random_array_real.at[(3 * num_particles + 1) :: 8].get() * 0.5
        )  # C[6] = S[2] + L[1]/2
        couplets = couplets.at[7::8].set(
            random_array_real.at[(3 * num_particles + 6) :: 8].get()
            - random_array_real.at[(3 * num_particles + 0) :: 8].get() * 0.5
        )  # C[7] = S[3] - L[0]/2

        # Allocate arrays for Linear velocities and Velocity gradients (from which we can extract angular velocities and rate of strain)
        r_lin_velocities = jnp.zeros((num_particles, 3), float)
        r_velocity_gradient = jnp.zeros((num_particles, 8), float)

        # SELF CONTRIBUTIONS
        r_lin_velocities = r_lin_velocities.at[:, 0].set(m_self.at[0].get() * forces.at[0::3].get())
        r_lin_velocities = r_lin_velocities.at[:, 1].set(m_self.at[0].get() * forces.at[1::3].get())
        r_lin_velocities = r_lin_velocities.at[:, 2].set(m_self.at[0].get() * forces.at[2::3].get())

        r_velocity_gradient = r_velocity_gradient.at[:, 0].set(
            m_self.at[1].get() * (couplets.at[0::8].get() - 4 * couplets.at[0::8].get())
        )
        r_velocity_gradient = r_velocity_gradient.at[:, 5].set(
            m_self.at[1].get() * (couplets.at[1::8].get() - 4 * couplets.at[5::8].get())
        )
        r_velocity_gradient = r_velocity_gradient.at[:, 6].set(
            m_self.at[1].get() * (couplets.at[2::8].get() - 4 * couplets.at[6::8].get())
        )
        r_velocity_gradient = r_velocity_gradient.at[:, 7].set(
            m_self.at[1].get() * (couplets.at[3::8].get() - 4 * couplets.at[7::8].get())
        )

        r_velocity_gradient = r_velocity_gradient.at[:, 4].set(
            m_self.at[1].get() * (couplets.at[4::8].get() - 4 * couplets.at[4::8].get())
        )
        r_velocity_gradient = r_velocity_gradient.at[:, 1].set(
            m_self.at[1].get() * (couplets.at[5::8].get() - 4 * couplets.at[1::8].get())
        )
        r_velocity_gradient = r_velocity_gradient.at[:, 2].set(
            m_self.at[1].get() * (couplets.at[6::8].get() - 4 * couplets.at[2::8].get())
        )
        r_velocity_gradient = r_velocity_gradient.at[:, 3].set(
            m_self.at[1].get() * (couplets.at[7::8].get() - 4 * couplets.at[3::8].get())
        )

        # # Geometric quantities
        rdotf_j = (
            r.at[:, 0].get() * forces.at[3 * indices_j + 0].get()
            + r.at[:, 1].get() * forces.at[3 * indices_j + 1].get()
            + r.at[:, 2].get() * forces.at[3 * indices_j + 2].get()
        )
        mrdotf_i = -(
            r.at[:, 0].get() * forces.at[3 * indices_i + 0].get()
            + r.at[:, 1].get() * forces.at[3 * indices_i + 1].get()
            + r.at[:, 2].get() * forces.at[3 * indices_i + 2].get()
        )

        cj_dotr = jnp.array(
            [
                couplets.at[8 * indices_j + 0].get() * r.at[:, 0].get()
                + couplets.at[8 * indices_j + 1].get() * r.at[:, 1].get()
                + couplets.at[8 * indices_j + 2].get() * r.at[:, 2].get(),
                couplets.at[8 * indices_j + 5].get() * r.at[:, 0].get()
                + couplets.at[8 * indices_j + 4].get() * r.at[:, 1].get()
                + couplets.at[8 * indices_j + 3].get() * r.at[:, 2].get(),
                couplets.at[8 * indices_j + 6].get() * r.at[:, 0].get()
                + couplets.at[8 * indices_j + 7].get() * r.at[:, 1].get()
                - (couplets.at[8 * indices_j + 0].get() + couplets.at[8 * indices_j + 4].get())
                * r.at[:, 2].get(),
            ]
        )

        ci_dotmr = jnp.array(
            [
                -couplets.at[8 * indices_i + 0].get() * r.at[:, 0].get()
                - couplets.at[8 * indices_i + 1].get() * r.at[:, 1].get()
                - couplets.at[8 * indices_i + 2].get() * r.at[:, 2].get(),
                -couplets.at[8 * indices_i + 5].get() * r.at[:, 0].get()
                - couplets.at[8 * indices_i + 4].get() * r.at[:, 1].get()
                - couplets.at[8 * indices_i + 3].get() * r.at[:, 2].get(),
                -couplets.at[8 * indices_i + 6].get() * r.at[:, 0].get()
                - couplets.at[8 * indices_i + 7].get() * r.at[:, 1].get()
                + (couplets.at[8 * indices_i + 0].get() + couplets.at[8 * indices_i + 4].get())
                * r.at[:, 2].get(),
            ]
        )

        rdotc_j = jnp.array(
            [
                couplets.at[8 * indices_j + 0].get() * r.at[:, 0].get()
                + couplets.at[8 * indices_j + 5].get() * r.at[:, 1].get()
                + couplets.at[8 * indices_j + 6].get() * r.at[:, 2].get(),
                couplets.at[8 * indices_j + 1].get() * r.at[:, 0].get()
                + couplets.at[8 * indices_j + 4].get() * r.at[:, 1].get()
                + couplets.at[8 * indices_j + 7].get() * r.at[:, 2].get(),
                couplets.at[8 * indices_j + 2].get() * r.at[:, 0].get()
                + couplets.at[8 * indices_j + 3].get() * r.at[:, 1].get()
                - (couplets.at[8 * indices_j + 0].get() + couplets.at[8 * indices_j + 4].get())
                * r.at[:, 2].get(),
            ]
        )

        mrdotc_i = jnp.array(
            [
                -couplets.at[8 * indices_i + 0].get() * r.at[:, 0].get()
                - couplets.at[8 * indices_i + 5].get() * r.at[:, 1].get()
                - couplets.at[8 * indices_i + 6].get() * r.at[:, 2].get(),
                -couplets.at[8 * indices_i + 1].get() * r.at[:, 0].get()
                - couplets.at[8 * indices_i + 4].get() * r.at[:, 1].get()
                - couplets.at[8 * indices_i + 7].get() * r.at[:, 2].get(),
                -couplets.at[8 * indices_i + 2].get() * r.at[:, 0].get()
                - couplets.at[8 * indices_i + 3].get() * r.at[:, 1].get()
                + (couplets.at[8 * indices_i + 0].get() + couplets.at[8 * indices_i + 4].get())
                * r.at[:, 2].get(),
            ]
        )

        rdotc_jj_dotr = (
            r.at[:, 0].get() * cj_dotr.at[0, :].get()
            + r.at[:, 1].get() * cj_dotr.at[1, :].get()
            + r.at[:, 2].get() * cj_dotr.at[2, :].get()
        )
        mrdotc_ii_dotmr = -(
            r.at[:, 0].get() * ci_dotmr.at[0, :].get()
            + r.at[:, 1].get() * ci_dotmr.at[1, :].get()
            + r.at[:, 2].get() * ci_dotmr.at[2, :].get()
        )

        # Compute Velocity for particles i
        r_lin_velocities = r_lin_velocities.at[indices_i, 0].add(
            f1 * forces.at[3 * indices_j].get() + (f2 - f1) * rdotf_j * r.at[:, 0].get()
        )
        r_lin_velocities = r_lin_velocities.at[indices_i, 1].add(
            f1 * forces.at[3 * indices_j + 1].get() + (f2 - f1) * rdotf_j * r.at[:, 1].get()
        )
        r_lin_velocities = r_lin_velocities.at[indices_i, 2].add(
            f1 * forces.at[3 * indices_j + 2].get() + (f2 - f1) * rdotf_j * r.at[:, 2].get()
        )
        r_lin_velocities = r_lin_velocities.at[indices_i, 0].add(
            g1 * (cj_dotr.at[0, :].get() - rdotc_jj_dotr * r.at[:, 0].get())
            + g2 * (rdotc_j.at[0, :].get() - 4.0 * rdotc_jj_dotr * r.at[:, 0].get())
        )
        r_lin_velocities = r_lin_velocities.at[indices_i, 1].add(
            g1 * (cj_dotr.at[1, :].get() - rdotc_jj_dotr * r.at[:, 1].get())
            + g2 * (rdotc_j.at[1, :].get() - 4.0 * rdotc_jj_dotr * r.at[:, 1].get())
        )
        r_lin_velocities = r_lin_velocities.at[indices_i, 2].add(
            g1 * (cj_dotr.at[2, :].get() - rdotc_jj_dotr * r.at[:, 2].get())
            + g2 * (rdotc_j.at[2, :].get() - 4.0 * rdotc_jj_dotr * r.at[:, 2].get())
        )
        # Compute Velocity for particles j
        r_lin_velocities = r_lin_velocities.at[indices_j, 0].add(
            f1 * forces.at[3 * indices_i].get() - (f2 - f1) * mrdotf_i * r.at[:, 0].get()
        )
        r_lin_velocities = r_lin_velocities.at[indices_j, 1].add(
            f1 * forces.at[3 * indices_i + 1].get() - (f2 - f1) * mrdotf_i * r.at[:, 1].get()
        )
        r_lin_velocities = r_lin_velocities.at[indices_j, 2].add(
            f1 * forces.at[3 * indices_i + 2].get() - (f2 - f1) * mrdotf_i * r.at[:, 2].get()
        )
        r_lin_velocities = r_lin_velocities.at[indices_j, 0].add(
            g1 * (ci_dotmr.at[0, :].get() + mrdotc_ii_dotmr * r.at[:, 0].get())
            + g2 * (mrdotc_i.at[0, :].get() + 4.0 * mrdotc_ii_dotmr * r.at[:, 0].get())
        )
        r_lin_velocities = r_lin_velocities.at[indices_j, 1].add(
            g1 * (ci_dotmr.at[1, :].get() + mrdotc_ii_dotmr * r.at[:, 1].get())
            + g2 * (mrdotc_i.at[1, :].get() + 4.0 * mrdotc_ii_dotmr * r.at[:, 1].get())
        )
        r_lin_velocities = r_lin_velocities.at[indices_j, 2].add(
            g1 * (ci_dotmr.at[2, :].get() + mrdotc_ii_dotmr * r.at[:, 2].get())
            + g2 * (mrdotc_i.at[2, :].get() + 4.0 * mrdotc_ii_dotmr * r.at[:, 2].get())
        )

        # Compute Velocity Gradient for particles i and j
        r_velocity_gradient = r_velocity_gradient.at[indices_i, 0].add(
            (-1)
            * g1
            * (
                r.at[:, 0].get() * forces.at[3 * indices_j + 0].get()
                - rdotf_j * r.at[:, 0].get() * r.at[:, 0].get()
            )
            + (-1)
            * g2
            * (
                rdotf_j
                + forces.at[3 * indices_j + 0].get() * r.at[:, 0].get()
                - 4.0 * rdotf_j * r.at[:, 0].get() * r.at[:, 0].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_j, 0].add(
            (-1)
            * g1
            * (
                -r.at[:, 0].get() * forces.at[3 * indices_i + 0].get()
                - mrdotf_i * r.at[:, 0].get() * r.at[:, 0].get()
            )
            + (-1)
            * g2
            * (
                mrdotf_i
                - forces.at[3 * indices_i + 0].get() * r.at[:, 0].get()
                - 4.0 * mrdotf_i * r.at[:, 0].get() * r.at[:, 0].get()
            )
        )
        r_velocity_gradient = r_velocity_gradient.at[indices_i, 0].add(
            h1 * (couplets.at[8 * indices_j + 0].get() - 4.0 * couplets.at[8 * indices_j + 0].get())
            + h2
            * (
                r.at[:, 0].get() * cj_dotr.at[0, :].get()
                - rdotc_jj_dotr * r.at[:, 0].get() * r.at[:, 0].get()
            )
            + h3
            * (
                rdotc_jj_dotr
                + cj_dotr.at[0, :].get() * r.at[:, 0].get()
                + r.at[:, 0].get() * rdotc_j.at[0, :].get()
                + rdotc_j.at[0, :].get() * r.at[:, 0].get()
                - 6.0 * rdotc_jj_dotr * r.at[:, 0].get() * r.at[:, 0].get()
                - couplets.at[8 * indices_j + 0].get()
            )
        )
        r_velocity_gradient = r_velocity_gradient.at[indices_j, 0].add(
            h1 * (couplets.at[8 * indices_i + 0].get() - 4.0 * couplets.at[8 * indices_i + 0].get())
            + h2
            * (
                -r.at[:, 0].get() * ci_dotmr.at[0, :].get()
                - mrdotc_ii_dotmr * r.at[:, 0].get() * r.at[:, 0].get()
            )
            + h3
            * (
                mrdotc_ii_dotmr
                - ci_dotmr.at[0, :].get() * r.at[:, 0].get()
                - r.at[:, 0].get() * mrdotc_i.at[0, :].get()
                - mrdotc_i.at[0, :].get() * r.at[:, 0].get()
                - 6.0 * mrdotc_ii_dotmr * r.at[:, 0].get() * r.at[:, 0].get()
                - couplets.at[8 * indices_i + 0].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_i, 1].add(
            (-1)
            * g1
            * (
                r.at[:, 1].get() * forces.at[3 * indices_j + 0].get()
                - rdotf_j * r.at[:, 1].get() * r.at[:, 0].get()
            )
            + (-1)
            * g2
            * (
                forces.at[3 * indices_j + 1].get() * r.at[:, 0].get()
                - 4.0 * rdotf_j * r.at[:, 1].get() * r.at[:, 0].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_j, 1].add(
            (-1)
            * g1
            * (
                -r.at[:, 1].get() * forces.at[3 * indices_i + 0].get()
                - mrdotf_i * r.at[:, 1].get() * r.at[:, 0].get()
            )
            + (-1)
            * g2
            * (
                -forces.at[3 * indices_i + 1].get() * r.at[:, 0].get()
                - 4.0 * mrdotf_i * r.at[:, 1].get() * r.at[:, 0].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_i, 1].add(
            h1 * (couplets.at[8 * indices_j + 5].get() - 4.0 * couplets.at[8 * indices_j + 1].get())
            + h2
            * (
                r.at[:, 1].get() * cj_dotr.at[0, :].get()
                - rdotc_jj_dotr * r.at[:, 1].get() * r.at[:, 0].get()
            )
            + h3
            * (
                cj_dotr.at[1, :].get() * r.at[:, 0].get()
                + r.at[:, 1].get() * rdotc_j.at[0, :].get()
                + rdotc_j.at[1, :].get() * r.at[:, 0].get()
                - 6.0 * rdotc_jj_dotr * r.at[:, 1].get() * r.at[:, 0].get()
                - couplets.at[8 * indices_j + 1].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_j, 1].add(
            h1 * (couplets.at[8 * indices_i + 5].get() - 4.0 * couplets.at[8 * indices_i + 1].get())
            + h2
            * (
                -r.at[:, 1].get() * ci_dotmr.at[0, :].get()
                - mrdotc_ii_dotmr * r.at[:, 1].get() * r.at[:, 0].get()
            )
            + h3
            * (
                -ci_dotmr.at[1, :].get() * r.at[:, 0].get()
                - r.at[:, 1].get() * mrdotc_i.at[0, :].get()
                - mrdotc_i.at[1, :].get() * r.at[:, 0].get()
                - 6.0 * mrdotc_ii_dotmr * r.at[:, 1].get() * r.at[:, 0].get()
                - couplets.at[8 * indices_i + 1].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_i, 2].add(
            (-1)
            * g1
            * (
                r.at[:, 2].get() * forces.at[3 * indices_j + 0].get()
                - rdotf_j * r.at[:, 2].get() * r.at[:, 0].get()
            )
            + (-1)
            * g2
            * (
                forces.at[3 * indices_j + 2].get() * r.at[:, 0].get()
                - 4.0 * rdotf_j * r.at[:, 2].get() * r.at[:, 0].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_j, 2].add(
            (-1)
            * g1
            * (
                -r.at[:, 2].get() * forces.at[3 * indices_i + 0].get()
                - mrdotf_i * r.at[:, 2].get() * r.at[:, 0].get()
            )
            + (-1)
            * g2
            * (
                -forces.at[3 * indices_i + 2].get() * r.at[:, 0].get()
                - 4.0 * mrdotf_i * r.at[:, 2].get() * r.at[:, 0].get()
            )
        )
        r_velocity_gradient = r_velocity_gradient.at[indices_i, 2].add(
            h1 * (couplets.at[8 * indices_j + 6].get() - 4.0 * couplets.at[8 * indices_j + 2].get())
            + h2
            * (
                r.at[:, 2].get() * cj_dotr.at[0, :].get()
                - rdotc_jj_dotr * r.at[:, 2].get() * r.at[:, 0].get()
            )
            + h3
            * (
                cj_dotr.at[2, :].get() * r.at[:, 0].get()
                + r.at[:, 2].get() * rdotc_j.at[0, :].get()
                + rdotc_j.at[2, :].get() * r.at[:, 0].get()
                - 6.0 * rdotc_jj_dotr * r.at[:, 2].get() * r.at[:, 0].get()
                - couplets.at[8 * indices_j + 2].get()
            )
        )
        r_velocity_gradient = r_velocity_gradient.at[indices_j, 2].add(
            h1 * (couplets.at[8 * indices_i + 6].get() - 4.0 * couplets.at[8 * indices_i + 2].get())
            + h2
            * (
                r.at[:, 2].get() * ci_dotmr.at[0, :].get() * (-1)
                - mrdotc_ii_dotmr * r.at[:, 2].get() * r.at[:, 0].get()
            )
            + h3
            * (
                -ci_dotmr.at[2, :].get() * r.at[:, 0].get()
                - r.at[:, 2].get() * mrdotc_i.at[0, :].get()
                - mrdotc_i.at[2, :].get() * r.at[:, 0].get()
                - 6.0 * mrdotc_ii_dotmr * r.at[:, 2].get() * r.at[:, 0].get()
                - couplets.at[8 * indices_i + 2].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_i, 3].add(
            (-1)
            * g1
            * (
                r.at[:, 2].get() * forces.at[3 * indices_j + 1].get()
                - rdotf_j * r.at[:, 2].get() * r.at[:, 1].get()
            )
            + (-1)
            * g2
            * (
                forces.at[3 * indices_j + 2].get() * r.at[:, 1].get()
                - 4.0 * rdotf_j * r.at[:, 2].get() * r.at[:, 1].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_j, 3].add(
            (-1)
            * g1
            * (
                -r.at[:, 2].get() * forces.at[3 * indices_i + 1].get()
                - mrdotf_i * r.at[:, 2].get() * r.at[:, 1].get()
            )
            + (-1)
            * g2
            * (
                -forces.at[3 * indices_i + 2].get() * r.at[:, 1].get()
                - 4.0 * mrdotf_i * r.at[:, 2].get() * r.at[:, 1].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_i, 3].add(
            h1 * (couplets.at[8 * indices_j + 7].get() - 4.0 * couplets.at[8 * indices_j + 3].get())
            + h2
            * (
                r.at[:, 2].get() * cj_dotr.at[1, :].get()
                - rdotc_jj_dotr * r.at[:, 2].get() * r.at[:, 1].get()
            )
            + h3
            * (
                cj_dotr.at[2, :].get() * r.at[:, 1].get()
                + r.at[:, 2].get() * rdotc_j.at[1, :].get()
                + rdotc_j.at[2, :].get() * r.at[:, 1].get()
                - 6.0 * rdotc_jj_dotr * r.at[:, 2].get() * r.at[:, 1].get()
                - couplets.at[8 * indices_j + 3].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_j, 3].add(
            h1 * (couplets.at[8 * indices_i + 7].get() - 4.0 * couplets.at[8 * indices_i + 3].get())
            + h2
            * (
                -r.at[:, 2].get() * ci_dotmr.at[1, :].get()
                - mrdotc_ii_dotmr * r.at[:, 2].get() * r.at[:, 1].get()
            )
            + h3
            * (
                -ci_dotmr.at[2, :].get() * r.at[:, 1].get()
                - r.at[:, 2].get() * mrdotc_i.at[1, :].get()
                - mrdotc_i.at[2, :].get() * r.at[:, 1].get()
                - 6.0 * mrdotc_ii_dotmr * r.at[:, 2].get() * r.at[:, 1].get()
                - couplets.at[8 * indices_i + 3].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_i, 4].add(
            (-1)
            * g1
            * (
                r.at[:, 1].get() * forces.at[3 * indices_j + 1].get()
                - rdotf_j * r.at[:, 1].get() * r.at[:, 1].get()
            )
            + (-1)
            * g2
            * (
                rdotf_j
                + forces.at[3 * indices_j + 1].get() * r.at[:, 1].get()
                - 4.0 * rdotf_j * r.at[:, 1].get() * r.at[:, 1].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_j, 4].add(
            (-1)
            * g1
            * (
                -r.at[:, 1].get() * forces.at[3 * indices_i + 1].get()
                - mrdotf_i * r.at[:, 1].get() * r.at[:, 1].get()
            )
            + (-1)
            * g2
            * (
                mrdotf_i
                - forces.at[3 * indices_i + 1].get() * r.at[:, 1].get()
                - 4.0 * mrdotf_i * r.at[:, 1].get() * r.at[:, 1].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_i, 4].add(
            h1 * (couplets.at[8 * indices_j + 4].get() - 4.0 * couplets.at[8 * indices_j + 4].get())
            + h2
            * (
                r.at[:, 1].get() * cj_dotr.at[1, :].get()
                - rdotc_jj_dotr * r.at[:, 1].get() * r.at[:, 1].get()
            )
            + h3
            * (
                rdotc_jj_dotr
                + cj_dotr.at[1, :].get() * r.at[:, 1].get()
                + r.at[:, 1].get() * rdotc_j.at[1, :].get()
                + rdotc_j.at[1, :].get() * r.at[:, 1].get()
                - 6.0 * rdotc_jj_dotr * r.at[:, 1].get() * r.at[:, 1].get()
                - couplets.at[8 * indices_j + 4].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_j, 4].add(
            h1 * (couplets.at[8 * indices_i + 4].get() - 4.0 * couplets.at[8 * indices_i + 4].get())
            + h2
            * (
                -r.at[:, 1].get() * ci_dotmr.at[1, :].get()
                - mrdotc_ii_dotmr * r.at[:, 1].get() * r.at[:, 1].get()
            )
            + h3
            * (
                mrdotc_ii_dotmr
                - ci_dotmr.at[1, :].get() * r.at[:, 1].get()
                - r.at[:, 1].get() * mrdotc_i.at[1, :].get()
                - mrdotc_i.at[1, :].get() * r.at[:, 1].get()
                - 6.0 * mrdotc_ii_dotmr * r.at[:, 1].get() * r.at[:, 1].get()
                - couplets.at[8 * indices_i + 4].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_i, 5].add(
            (-1)
            * g1
            * (
                r.at[:, 0].get() * forces.at[3 * indices_j + 1].get()
                - rdotf_j * r.at[:, 0].get() * r.at[:, 1].get()
            )
            + (-1)
            * g2
            * (
                forces.at[3 * indices_j + 0].get() * r.at[:, 1].get()
                - 4.0 * rdotf_j * r.at[:, 0].get() * r.at[:, 1].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_j, 5].add(
            (-1)
            * g1
            * (
                -r.at[:, 0].get() * forces.at[3 * indices_i + 1].get()
                - mrdotf_i * r.at[:, 0].get() * r.at[:, 1].get()
            )
            + (-1)
            * g2
            * (
                -forces.at[3 * indices_i + 0].get() * r.at[:, 1].get()
                - 4.0 * mrdotf_i * r.at[:, 0].get() * r.at[:, 1].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_i, 5].add(
            h1 * (couplets.at[8 * indices_j + 1].get() - 4.0 * couplets.at[8 * indices_j + 5].get())
            + h2
            * (
                r.at[:, 0].get() * cj_dotr.at[1, :].get()
                - rdotc_jj_dotr * r.at[:, 0].get() * r.at[:, 1].get()
            )
            + h3
            * (
                cj_dotr.at[0, :].get() * r.at[:, 1].get()
                + r.at[:, 0].get() * rdotc_j.at[1, :].get()
                + rdotc_j.at[0, :].get() * r.at[:, 1].get()
                - 6.0 * rdotc_jj_dotr * r.at[:, 0].get() * r.at[:, 1].get()
                - couplets.at[8 * indices_j + 5].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_j, 5].add(
            h1 * (couplets.at[8 * indices_i + 1].get() - 4.0 * couplets.at[8 * indices_i + 5].get())
            + h2
            * (
                -r.at[:, 0].get() * ci_dotmr.at[1, :].get()
                - mrdotc_ii_dotmr * r.at[:, 0].get() * r.at[:, 1].get()
            )
            + h3
            * (
                -ci_dotmr.at[0, :].get() * r.at[:, 1].get()
                - r.at[:, 0].get() * mrdotc_i.at[1, :].get()
                - mrdotc_i.at[0, :].get() * r.at[:, 1].get()
                - 6.0 * mrdotc_ii_dotmr * r.at[:, 0].get() * r.at[:, 1].get()
                - couplets.at[8 * indices_i + 5].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_i, 6].add(
            (-1)
            * g1
            * (
                r.at[:, 0].get() * forces.at[3 * indices_j + 2].get()
                - rdotf_j * r.at[:, 0].get() * r.at[:, 2].get()
            )
            + (-1)
            * g2
            * (
                forces.at[3 * indices_j + 0].get() * r.at[:, 2].get()
                - 4.0 * rdotf_j * r.at[:, 0].get() * r.at[:, 2].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_j, 6].add(
            (-1)
            * g1
            * (
                -r.at[:, 0].get() * forces.at[3 * indices_i + 2].get()
                - mrdotf_i * r.at[:, 0].get() * r.at[:, 2].get()
            )
            + (-1)
            * g2
            * (
                -forces.at[3 * indices_i + 0].get() * r.at[:, 2].get()
                - 4.0 * mrdotf_i * r.at[:, 0].get() * r.at[:, 2].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_i, 6].add(
            h1 * (couplets.at[8 * indices_j + 2].get() - 4.0 * couplets.at[8 * indices_j + 6].get())
            + h2
            * (
                r.at[:, 0].get() * cj_dotr.at[2, :].get()
                - rdotc_jj_dotr * r.at[:, 0].get() * r.at[:, 2].get()
            )
            + h3
            * (
                cj_dotr.at[0, :].get() * r.at[:, 2].get()
                + r.at[:, 0].get() * rdotc_j.at[2, :].get()
                + rdotc_j.at[0, :].get() * r.at[:, 2].get()
                - 6.0 * rdotc_jj_dotr * r.at[:, 0].get() * r.at[:, 2].get()
                - couplets.at[8 * indices_j + 6].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_j, 6].add(
            h1 * (couplets.at[8 * indices_i + 2].get() - 4.0 * couplets.at[8 * indices_i + 6].get())
            + h2
            * (
                -r.at[:, 0].get() * ci_dotmr.at[2, :].get()
                - mrdotc_ii_dotmr * r.at[:, 0].get() * r.at[:, 2].get()
            )
            + h3
            * (
                -ci_dotmr.at[0, :].get() * r.at[:, 2].get()
                - r.at[:, 0].get() * mrdotc_i.at[2, :].get()
                - mrdotc_i.at[0, :].get() * r.at[:, 2].get()
                - 6.0 * mrdotc_ii_dotmr * r.at[:, 0].get() * r.at[:, 2].get()
                - couplets.at[8 * indices_i + 6].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_i, 7].add(
            (-1)
            * g1
            * (
                r.at[:, 1].get() * forces.at[3 * indices_j + 2].get()
                - rdotf_j * r.at[:, 1].get() * r.at[:, 2].get()
            )
            + (-1)
            * g2
            * (
                forces.at[3 * indices_j + 1].get() * r.at[:, 2].get()
                - 4.0 * rdotf_j * r.at[:, 1].get() * r.at[:, 2].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_j, 7].add(
            (-1)
            * g1
            * (
                -r.at[:, 1].get() * forces.at[3 * indices_i + 2].get()
                - mrdotf_i * r.at[:, 1].get() * r.at[:, 2].get()
            )
            + (-1)
            * g2
            * (
                -forces.at[3 * indices_i + 1].get() * r.at[:, 2].get()
                - 4.0 * mrdotf_i * r.at[:, 1].get() * r.at[:, 2].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_i, 7].add(
            h1 * (couplets.at[8 * indices_j + 3].get() - 4.0 * couplets.at[8 * indices_j + 7].get())
            + h2
            * (
                r.at[:, 1].get() * cj_dotr.at[2, :].get()
                - rdotc_jj_dotr * r.at[:, 1].get() * r.at[:, 2].get()
            )
            + h3
            * (
                cj_dotr.at[1, :].get() * r.at[:, 2].get()
                + r.at[:, 1].get() * rdotc_j.at[2, :].get()
                + rdotc_j.at[1, :].get() * r.at[:, 2].get()
                - 6.0 * rdotc_jj_dotr * r.at[:, 1].get() * r.at[:, 2].get()
                - couplets.at[8 * indices_j + 7].get()
            )
        )

        r_velocity_gradient = r_velocity_gradient.at[indices_j, 7].add(
            h1 * (couplets.at[8 * indices_i + 3].get() - 4.0 * couplets.at[8 * indices_i + 7].get())
            + h2
            * (
                -r.at[:, 1].get() * ci_dotmr.at[2, :].get()
                - mrdotc_ii_dotmr * r.at[:, 1].get() * r.at[:, 2].get()
            )
            + h3
            * (
                -ci_dotmr.at[1, :].get() * r.at[:, 2].get()
                - r.at[:, 1].get() * mrdotc_i.at[2, :].get()
                - mrdotc_i.at[1, :].get() * r.at[:, 2].get()
                - 6.0 * mrdotc_ii_dotmr * r.at[:, 1].get() * r.at[:, 2].get()
                - couplets.at[8 * indices_i + 7].get()
            )
        )

        # # Convert to angular velocities and rate of strain
        r_ang_vel_and_strain = jnp.zeros((num_particles, 8))
        r_ang_vel_and_strain = r_ang_vel_and_strain.at[:, 0].set(
            (r_velocity_gradient.at[:, 3].get() - r_velocity_gradient.at[:, 7].get()) * 0.5
        )
        r_ang_vel_and_strain = r_ang_vel_and_strain.at[:, 1].set(
            (r_velocity_gradient.at[:, 6].get() - r_velocity_gradient.at[:, 2].get()) * 0.5
        )
        r_ang_vel_and_strain = r_ang_vel_and_strain.at[:, 2].set(
            (r_velocity_gradient.at[:, 1].get() - r_velocity_gradient.at[:, 5].get()) * 0.5
        )
        r_ang_vel_and_strain = r_ang_vel_and_strain.at[:, 3].set(
            2 * r_velocity_gradient.at[:, 0].get() + r_velocity_gradient.at[:, 4].get()
        )
        r_ang_vel_and_strain = r_ang_vel_and_strain.at[:, 4].set(
            r_velocity_gradient.at[:, 1].get() + r_velocity_gradient.at[:, 5].get()
        )
        r_ang_vel_and_strain = r_ang_vel_and_strain.at[:, 5].set(
            r_velocity_gradient.at[:, 2].get() + r_velocity_gradient.at[:, 6].get()
        )
        r_ang_vel_and_strain = r_ang_vel_and_strain.at[:, 6].set(
            r_velocity_gradient.at[:, 3].get() + r_velocity_gradient.at[:, 7].get()
        )
        r_ang_vel_and_strain = r_ang_vel_and_strain.at[:, 7].set(
            r_velocity_gradient.at[:, 0].get() + 2 * r_velocity_gradient.at[:, 4].get()
        )

        return r_lin_velocities, r_ang_vel_and_strain

    # obtain matrix form of linear operator Mpsi, by computing Mpsi(e_i) with e_i basis vectors (1,0,...,0), (0,1,0,...) ...
    random_array_nf = (2 * random_array_nf - 1) * jnp.sqrt(3.0)
    r_fu_matrix = np.zeros((6 * num_particles, 6 * num_particles))
    basis_vectors = np.eye(6 * num_particles, dtype=float)
    for iii in range(6 * num_particles):
        rei = compute_lubrication_fu(basis_vectors[iii, :])
        r_fu_matrix[:, iii] = rei
    sqrt_r_fu = scipy.linalg.sqrtm(
        r_fu_matrix
    )
    r_fu12psi_correct = jnp.dot(sqrt_r_fu, random_array_nf * np.sqrt(2.0 * temperature / dt))

    random_array_real = (2 * random_array_real - 1) * jnp.sqrt(3.0)
    matrix_m = np.zeros((11 * num_particles, 11 * num_particles))
    basis_vectors = np.eye(11 * num_particles, dtype=float)
    for iii in range(11 * num_particles):
        a = helper_mpsi(basis_vectors[iii, :])
        mei = helper_reshape(a)
        matrix_m[:, iii] = mei
    sqrt_m = scipy.linalg.sqrtm(matrix_m)
    m12psi_debug = jnp.dot(sqrt_m, random_array_real * jnp.sqrt(2.0 * temperature / dt))

    return convert_to_generalized(m12psi_debug), r_fu12psi_correct
