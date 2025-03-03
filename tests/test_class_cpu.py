import jax.numpy as jnp
import numpy as np
from jax.lib import xla_bridge
from jfsd import main
import pytest 

def jax_has_cpu():
    """Check that jax is running on CPU.

    """
    return (xla_bridge.get_backend().platform)
    
class TestClassCPU:
    
    @pytest.mark.parametrize("delta", [0.0001,0.001,0.01,0.1,1,10])
    def test_sedimenting_triangle(self,delta):
        """Physical unit test for deterministic (shearless) part of hydrodynamic calculations, that runs on CPU. 
        
        Test instantaneous sedimentation of three spheres against reference values.

        """
        
        assert (jax_has_cpu() == 'cpu') #check that jax is running on cpu
                
        deltas = np.array([0.0001,0.001,0.01,0.1,1,10])
        index = np.searchsorted(deltas, delta)
        reference_traj = np.load('files/sedimenting_triangle_reference.npy') #load reference trajectory

        r1 = np.array([-2. - delta,0.,0.])
        r2 = np.array([0.,0.,0.])
        theta= np.pi/3
        alpha = np.pi - theta
        r3 = np.array([np.cos(alpha),np.sin(alpha),0.]) * (2.+delta)
        #test SD
        traj, _, _, _ = main.main(
                1, 1, 1, 50, 50, 50, 3, 0.5,
                0., 1, 0.5, 0.001, 0., 1, 1.,
                jnp.array([r1, r2, r3]),
                0, 0, 0, 0, 0., 0.,
                None, 0, 0, 0,np.array([0]), np.array([0]),
                2,0,0,0.,0.)
        error = np.linalg.norm(reference_traj[index,:,:] - traj[0, :, :])
        assert (error < 5 * 1e-4)
        
    def test_thermal_realspace(self):
        """Physical unit test for non-deterministic part of hydrodynamic calculations, that runs on CPU. 
        
        Compare square root of resistance/mobility obtained using Lanczsos decomposition, with the square roots computed using scipy. 
        This test probes lubrication and real-space far-field calculation of thermal motion.

        """
        assert (jax_has_cpu() == 'cpu') #check that jax is running on cpu
        num_particles = 25
        dr=2.0005
        _, _, _, testresults = main.main(
                    1, 1, 0.1, 35, 35, 35, num_particles, 0.5,
                    0.001, 1, 0.5, 0.001, 0., 0, 1.,
                    jnp.array([[0., dr, 0.],    [0., dr, dr],    [0., dr,-dr],    [-dr, dr, 0.],    [dr, dr, 0.],
                                [0., 2*dr, 0.],  [0., 2*dr, dr],  [0., 2*dr,-dr],  [-dr, 2*dr, 0.],  [dr, 2*dr, 0.],
                                [0., 0., 0.],    [0., 0., dr],    [0., 0.,-dr],    [-dr, 0., 0.],    [dr, 0., 0.],
                                [0., -dr, 0.],   [0., -dr, dr],   [0., -dr,-dr],   [-dr, -dr, 0.],   [dr, -dr, 0.],
                                [0., -2*dr, 0.], [0., -2*dr, dr], [0., -2*dr,-dr], [-dr, -2*dr, 0.], [dr, -2*dr, 0.]]),
                    19989, 3300, 83909, 41234,
                    0., 0.,
                    None, 0, 0, 0, np.array([0]), np.array([0]), 
                    2,0,1,0,0)
        error_nf = testresults[0]
        error_ff = testresults[1]
        assert (error_nf < 0.01)
        assert (error_ff < 0.01)
    
    @pytest.mark.parametrize("delta", [0.0001, 0.001, 0.01, 0.1, 1, 10])
    def test_shear(self, delta):
        """Physical unit test for deterministic shear part of hydrodynamic calculations, that runs on CPU.
        
        Test instantaneous response to shear of three spheres against reference values.
        """
        assert jax_has_cpu() == 'cpu'  # Check that jax is running on CPU
        
        deltas = np.array([0.0001, 0.001, 0.01, 0.1, 1, 10])
        index = np.searchsorted(deltas, delta)
        reference_traj = np.load('files/sheared_triplet_reference.npy')  # Load reference trajectory
        r1 = np.array([0., -2. - delta, 0.])
        r2 = np.array([0., 0., 0.])
        r3 = np.array([0., 2. + delta, 0.]) 
        traj, _, _, _ = main.main(
            1, 1, 0.1, 50, 50, 50, 3, 0.5,
            0., 1, 0.5, 0.001, 0., 0, 0.,
            jnp.array([r1, r2, r3]),
            0, 0, 0, 0, 1., 0.,
            None, 0, 0, 0, np.array([0]), np.array([0]),
            2, 0, 0, 0., 0.
            )
        # Only calculate error for this specific delta
        error = np.linalg.norm(reference_traj[index,:,:] - traj[0, :, :])
        assert (error < 5 * 1e-4)
