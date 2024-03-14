import main
import jax.numpy as jnp
import numpy as np


class TestClass:

    def test_one(self):
        reference_traj = np.load('files/dancing_spheres_ref.npy')
        traj = main.main(
            1000, 10, 0.05, 50, 50, 50, 3, 0.5,
            0., 1, 0.5, 0.001, 0., 1, 1.,
            jnp.array([[-5., 0., 0.], [0., 0., 0.], [7., 0., 0.]]),
            0, 0, 0, 0, 0., 0.,
            'None')
        error = jnp.linalg.norm(reference_traj-traj)
        assert (error < 1e-8)

    def test_two(self):
        reference_traj = np.load('files/shear_pair_ref.npy')
        dr = 0.01
        traj = main.main(
            1000, 10, 0.01, 50, 50, 50, 2, 0.5,
            0., 1, 0.5, 0.001, 0., 0, 1.,
            jnp.array([[0., 1.+dr, 0.], [0., -1.-dr, 0.]]),
            0, 0, 0, 0, 0.1, 0.,
            'None')
        error = jnp.linalg.norm(reference_traj-traj)
        assert (error < 1e-8)
