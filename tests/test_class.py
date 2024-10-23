import freud
import jax.numpy as jnp
import numpy as np
from jax.lib import xla_bridge
from jfsd import main

def jax_has_gpu():
    """Check that the machine in use has an available GPU for jax.

    """
    return (xla_bridge.get_backend().platform)

class TestClass:
    
    def test_deterministic_hydro(self):
        """Physical unit test for deterministic (shearless) part of hydrodynamic calculations. 
        
        Test sedimentation of three spheres against known results.

        """
            
        assert (jax_has_gpu() == 'gpu') #check that gpu works properly
        reference_traj = np.load('files/dancing_spheres_ref.npy') #load reference trajectory

        #test SD
        traj, _, _, _ = main.main(
            1000, 10, 0.05, 50, 50, 50, 3, 0.5,
            0., 1, 0.5, 0.001, 0., 1, 1.,
            jnp.array([[-5., 0., 0.], [0., 0., 0.], [7., 0., 0.]]),
            0, 0, 0, 0, 0., 0.,
            None, 0, 0, 0,np.array([0]), np.array([0]),
            2,0,
            0.1,0.1)
        error = np.linalg.norm(reference_traj-traj)
        assert (error < 1e-8)
        
        #test RPY
        traj, _, _, _ = main.main(
            1000, 10, 0.05, 50, 50, 50, 3, 0.5,
            0., 1, 0.5, 0.001, 0., 1, 1.,
            jnp.array([[-5., 0., 0.], [0., 0., 0.], [7., 0., 0.]]),
            0, 0, 0, 0, 0., 0.,
            None, 0, 0, 0,np.array([0]), np.array([0]),
            1,0,0,0)
        error = (np.linalg.norm(reference_traj-traj)) / np.linalg.norm(reference_traj)
        assert (error < 0.02)

    def test_external_shear(self):
        """Physical unit test for deterministic shear part of hydrodynamic calculations. 
        
        Test pair of particle in simple shear against analytical result.

        """
        
        assert (jax_has_gpu() == 'gpu') #check that gpu works properly
        reference_traj = np.load('files/shear_pair_ref.npy') #load reference trajectory
        dr = 0.01 #set initial surface-to-surface distance of pair of particle in simple shear
        
        #test SD
        traj, _, _, _ = main.main(
            1000, 10, 0.01, 50, 50, 50, 2, 0.5,
            0., 1, 0.5, 0.001, 0., 0, 1.,
            jnp.array([[0., 1.+dr, 0.], [0., -1.-dr, 0.]]),
            0, 0, 0, 0, 0.1, 0.,
            None, 0, 0, 0,np.array([0]), np.array([0]),
            2,0,0,0)
        error = np.linalg.norm(reference_traj-traj)
        assert (error < 1e-8)
        
        #test RPY
        traj, _, _, _ = main.main(
            1000, 10, 0.01, 50, 50, 50, 2, 0.5,
            0., 1, 0.5, 0.001, 0., 0, 1.,
            jnp.array([[0., 1.+dr, 0.], [0., -1.-dr, 0.]]),
            0, 0, 0, 0, 0.1, 0.,
            None, 0, 0, 0,np.array([0]), np.array([0]),
            1,0,0,0)
        error = np.linalg.norm(reference_traj-traj)/np.linalg.norm(reference_traj)
        assert (error < 0.17)

    def test_thermal_1body(self):
        """Physical unit test for non-deterministic part of hydrodynamic calculations. 
        
        Test Brownian motion of a single particle, in periodic boundary conditions.
        Thus, the particle will interact with its periodic images, generating deviation from the simple diffusion of a particle in an unbounded solvent. 
        This test mostly probes the wave-space far-field calculation of thermal motion. 

        """
        
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
            traj, _, _, _ = main.main(
                100, 1, 0.1, 50, 50, 50, 1, 0.5,
                1., 1, 0.5, 0.001, 0., 0, 1.,
                jnp.array([[0., 0., 0.]]),
                rfd_seeds[i], ff_w_seeds[i], ff_r_seeds[i], 0,
                0., 0.,
                None, 0, 0, 0, np.array([0]), np.array([0]), 
                2,0,0,0)
            msd.compute(positions=(traj))
            MSDs.append(msd.msd)
        MSDs_average = np.mean(MSDs, axis=0)
        error = np.abs((theoretical_MSD_hashimoto(
            np.arange(0, 100) * 0.1, 50))[:10] - MSDs_average[:10])
        assert (np.max(error) < 0.2)  # reference tolerance
        
    def test_thermal_realspace(self):
        """Physical unit test for non-deterministic part of hydrodynamic calculations. 
        
        Compare square root of resistance/mobility obtained using Lanczsos decomposition, with the square roots computed using scipy. 
        This test probes lubrication and real-space far-field calculation of thermal motion.

        """
        
        assert (jax_has_gpu() == 'gpu')
        _, _, _, testresults = main.main(
                    1, 1, 0.1, 20, 20, 20, 5, 0.5,
                    1., 1, 0.5, 0.001, 0., 0, 1.,
                    jnp.array([[0., 0., 0.],[0., 0., 2.1],[0., 0.,-2.1],[-5., 0., 0.],[5., 0., 0.]]),
                    19989, 3300, 83909, 41234,
                    0., 0.,
                    None, 0, 0, 0, np.array([0]), np.array([0]), 
                    2,1,0,0)
        error_nf = testresults[0]
        error_ff = testresults[1]
        assert (error_nf < 0.001)
        assert (error_ff < 0.001)
        
