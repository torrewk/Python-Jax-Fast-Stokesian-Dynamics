import freud
import jax.numpy as jnp
import numpy as np
from jax.lib import xla_bridge

from jfsd import main


def jax_has_gpu():
    return (xla_bridge.get_backend().platform)


class TestClass:

    def test_deterministic_hydro(self):
    
        assert (jax_has_gpu() == 'gpu')
    
        reference_traj = np.load('files/dancing_spheres_ref.npy')
        traj, _, _ = main.main(
            1000, 10, 0.05, 50, 50, 50, 3, 0.5,
            0., 1, 0.5, 0.001, 0., 1, 1.,
            jnp.array([[-5., 0., 0.], [0., 0., 0.], [7., 0., 0.]]),
            0, 0, 0, 0, 0., 0.,
            'None', 0, 0, 0,np.array([0]), np.array([0]),
            1 )
        error = np.linalg.norm(reference_traj-traj)
        assert (error < 1e-8)

    def test_external_shear(self):
    
        assert (jax_has_gpu() == 'gpu')
    
        reference_traj = np.load('files/shear_pair_ref.npy')
        dr = 0.01
        traj, _, _ = main.main(
            1000, 10, 0.01, 50, 50, 50, 2, 0.5,
            0., 1, 0.5, 0.001, 0., 0, 1.,
            jnp.array([[0., 1.+dr, 0.], [0., -1.-dr, 0.]]),
            0, 0, 0, 0, 0.1, 0.,
            'None', 0, 0, 0,np.array([0]), np.array([0]),
            1 )
        error = np.linalg.norm(reference_traj-traj)
        assert (error < 1e-8)

    def test_thermal_1body(self):

        def theoretical_MSD_hashimoto(tempo, L):
            D_eff = (1-2.837/L + 4*np.pi/3 / (L*L*L))
            return tempo * 6*D_eff


        assert (jax_has_gpu() == 'gpu')
        #seeds for thermal noise
        rfd_seeds = [19989, 97182, 46075, 69177,
                     25312, 73247, 19735, 7738, 53116, 47587]
        ff_r_seeds = [70272, 3300, 4182, 72024,
                      55409, 13651, 83703, 55077, 64330, 20129]
        ff_w_seeds = [14863, 83909, 73247, 61582,
                      25154, 83909, 21343, 31196, 75367, 59341]
        msd = freud.msd.MSD()
        MSDs = []
        for i in range(10):
            traj, _, _ = main.main(
                100, 1, 0.1, 50, 50, 50, 1, 0.5,
                1., 1, 0.5, 0.001, 0., 0, 1.,
                jnp.array([[0., 0., 0.]]),
                rfd_seeds[i], ff_w_seeds[i], ff_r_seeds[i], 0,
                0., 0.,
                'None', 0, 0, 0, np.array([0]), np.array([0]), 
                1)
            msd.compute(positions=(traj))
            MSDs.append(msd.msd)
        MSDs_average = np.mean(MSDs, axis=0)
        error = np.abs((theoretical_MSD_hashimoto(
            np.arange(0, 100) * 0.1, 50))[:10] - MSDs_average[:10])
        assert (np.max(error) < 0.2)  # reference tolerance
