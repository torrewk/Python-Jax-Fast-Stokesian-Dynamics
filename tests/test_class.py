import freud
import jax.numpy as jnp
import numpy as np
from jax.lib import xla_bridge
from jfsd import main
import pytest
from scipy.stats import kstest


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
            0., 0.5, 0.001, 0., 1, 1.,
            jnp.array([[-5., 0., 0.], [0., 0., 0.], [7., 0., 0.]]),
            jnp.ones(3),
            0, 0, 0, 0, 0., 0.,
            None, 0, 0, 0,np.array([0]), np.array([0]),
            2,0,0,0.,0.)


        error = np.linalg.norm(reference_traj-traj) / np.linalg.norm(reference_traj)
        assert (error < 5*1e-5)

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
            0., 0.5, 0.001, 0., 0, 1.,
            jnp.array([[0., 1.+dr, 0.], [0., -1.-dr, 0.]]),
            jnp.ones(2),
            0, 0, 0, 0, 0.1, 0.,
            None, 0, 0, 0,np.array([0]), np.array([0]),
            2,0,0,0.,0.)
        error = np.linalg.norm(reference_traj-traj)/np.linalg.norm(reference_traj)
        print(error)
        assert (error < 0.000516)

    def test_thermal_1body(self):
        """Physical unit test for non-deterministic part of hydrodynamic calculations. 
        
        Test Brownian motion of a single particle, in periodic boundary conditions.
        Thus, the particle will interact with its periodic images, generating deviation from the simple diffusion of a particle in an unbounded solvent. 
        This test mostly probes the wave-space far-field calculation of thermal motion. 
        """
        
        assert (jax_has_gpu() == 'gpu'), "This test requires a GPU for JAX to run."
        
        n_runs = 20  # Number of simulation runs
        n_steps = 150 # Number of time steps in each run
        L = 50 # Box size in each dimension
        dt = 0.1 # Time step size
        
        # Calculate theoretical D_eff assuming D_0 = 1 in the simulation
        # First three terms of the Hasimoto (1959) expansion
        theoretical_D_eff = (1 - 2.837/L + 4*np.pi/3 / (L*L*L))
                
        # Calculate the std for each time step
        times = np.arange(n_steps) * dt
        # Standard deviation follows the Gaussian distribution for every dimension
        stds = np.sqrt(2 * theoretical_D_eff * times)
        
        # Skip t=0 to avoid division by zero in KS test (since σ=0 at t=0)
        timesteps_to_check = np.arange(1, n_steps, 10)
        critical_value = 0.10  # Significance level for KS test
        
        # Set a fixed seed for reproducibility
        np.random.seed(42)
        rfd_seeds = np.random.default_rng().integers(0, 2**31-1, n_runs)
        ff_w_seeds = np.random.default_rng().integers(0, 2**31-1, n_runs)
        ff_r_seeds = np.random.default_rng().integers(0, 2**31-1, n_runs)
        
        positions_to_check = np.zeros((n_runs, len(timesteps_to_check), 3))
                
        # Run the simulation multiple times to get a distribution of trajectories
        for i in range(n_runs):
            traj, _, _, _ = main.main(
                n_steps, 1, dt, L, L, L, 1, 0.5,
                1., 0.5, 0.001, 0., 0, 1.,
                jnp.array([[0., 0., 0.]]),
                jnp.ones(1),
                rfd_seeds[i], ff_w_seeds[i], ff_r_seeds[i], 0,
                0., 0.,
                None, 0, 0, 0, np.array([0]), np.array([0]), 
                2, 0, 0, 0., 0.)
            
            # Extract trajectory for the single particle (all timesteps)
            positions = np.array(traj)
            
            # Process only if we have a valid trajectory
            if positions.size > 0 and positions.shape[0] > 0:
                # Get all timesteps for particle 0
                positions = positions[:, 0, :]
                
                # Unwrap the trajectory to account for periodic boundary conditions
                # TODO: save image flags to avoid unwrapping
                unwrapped = np.zeros_like(positions)
                unwrapped[0] = positions[0]
                
                for j in range(1, len(positions)):
                    disp = positions[j] - positions[j-1]
                    
                    # Check for boundary crossings in each dimension
                    for dim in range(3):
                        if disp[dim] > L/2:
                            disp[dim] -= L
                        elif disp[dim] < -L/2:
                            disp[dim] += L
                    
                    unwrapped[j] = unwrapped[j-1] + disp
                
                # Extract unwrapped coordinates
                x = unwrapped[:, 0]
                y = unwrapped[:, 1]
                z = unwrapped[:, 2]
                
                # Save the unwrapped trajectory for the time steps we want to check
                for idx, t in enumerate(timesteps_to_check):
                    if t < len(x):  # Ensure we don't go out of bounds
                        positions_to_check[i, idx, 0] = x[t]
                        positions_to_check[i, idx, 1] = y[t]
                        positions_to_check[i, idx, 2] = z[t]
        
        # Calculate the KS test for each time step and each coordinate
        p_values_x = []
        p_values_y = []
        p_values_z = []
        
        for i, step in enumerate(timesteps_to_check):
            if i < positions_to_check.shape[1]:
                # Get the standard deviation for this time step
                sigma = stds[step]
                
                # Skip if sigma is zero
                if sigma == 0:
                    continue
                
                # Extract positions at the current time step
                positions_at_t = positions_to_check[:, i, :]
                
                # Standardize all coordinates (convert to z-scores)
                z_x = positions_at_t[:, 0] / sigma
                z_y = positions_at_t[:, 1] / sigma
                z_z = positions_at_t[:, 2] / sigma
                
                # Perform KS test for each coordinate against standard normal distribution
                _, p_value_x = kstest(z_x, 'norm')
                _, p_value_y = kstest(z_y, 'norm')
                _, p_value_z = kstest(z_z, 'norm')
                
                p_values_x.append(p_value_x)
                p_values_y.append(p_value_y)
                p_values_z.append(p_value_z)
        
        # Calculate the percentage of p-values above critical value for each coordinate
        percentage_above_critical_x = np.mean(np.array(p_values_x) > critical_value)
        percentage_above_critical_y = np.mean(np.array(p_values_y) > critical_value)
        percentage_above_critical_z = np.mean(np.array(p_values_z) > critical_value)
        
        # For critical_value=0.10, we expect approximately 90% of p-values to exceed it
        expected = 1.0 - critical_value  # 0.90
        tolerance = 0.05  # Allow a tolerance of 5% for the test
        
        # Verify that each coordinate follows the expected distribution
        for coord_name, percentage in [("X", percentage_above_critical_x), 
                                      ("Y", percentage_above_critical_y), 
                                      ("Z", percentage_above_critical_z)]:
            assert percentage >= (expected - tolerance), \
                f"{coord_name}: Expected at least {(expected-tolerance)*100:.0f}% of p-values > {critical_value}, but got only {percentage*100:.2f}%."
        
        # Also check the average across all coordinates
        avg_percentage = np.mean([percentage_above_critical_x, percentage_above_critical_y, percentage_above_critical_z])
        
        print(f"Average percentage above {critical_value:.2f}: {avg_percentage*100:.2f}%")
        
        assert avg_percentage >= (expected - tolerance/2), \
            f"Average percentage across all coordinates ({avg_percentage*100:.2f}%) is below the minimum expected threshold ({(expected-tolerance/2)*100:.0f}%)."
    
    def test_thermal_realspace(self):
        """Physical unit test for non-deterministic part of hydrodynamic calculations. 
        
        Compare square root of resistance/mobility obtained using Lanczsos decomposition, with the square roots computed using scipy. 
        This test probes lubrication and real-space far-field calculation of thermal motion.

        """
        num_particles = 50
        dr=2.0005
        assert (jax_has_gpu() == 'gpu')
        _, _, _, testresults = main.main(
                    1, 1, 0.1, 35, 35, 35, num_particles, 0.5,
                    0.001, 0.5, 0.001, 0., 0, 1.,
                    jnp.array([[0., dr, 0.],    [0., dr, dr],    [0., dr,-dr],    [-dr, dr, 0.],    [dr, dr, 0.],
                                [0., 2*dr, 0.],  [0., 2*dr, dr],  [0., 2*dr,-dr],  [-dr, 2*dr, 0.],  [dr, 2*dr, 0.],
                                [0., 3*dr, 0.],  [0., 3*dr, dr],  [0., 3*dr,-dr],  [-dr, 3*dr, 0.],  [dr, 3*dr, 0.],
                                [0., 4*dr, 0.],  [0., 4*dr, dr],  [0., 4*dr,-dr],  [-dr, 4*dr, 0.],  [dr, 4*dr, 0.],
                                [0., 0., 0.],    [0., 0., dr],    [0., 0.,-dr],    [-dr, 0., 0.],    [dr, 0., 0.],
                                [0., -dr, 0.],   [0., -dr, dr],   [0., -dr,-dr],   [-dr, -dr, 0.],   [dr, -dr, 0.],
                                [0., -2*dr, 0.], [0., -2*dr, dr], [0., -2*dr,-dr], [-dr, -2*dr, 0.], [dr, -2*dr, 0.],
                                [0., -3*dr, 0.], [0., -3*dr, dr], [0., -3*dr,-dr], [-dr, -3*dr, 0.], [dr, -3*dr, 0.],
                                [0., -4*dr, 0.], [0., -4*dr, dr], [0., -4*dr,-dr], [-dr, -4*dr, 0.], [dr, -4*dr, 0.],
                                [0., -5*dr, 0.], [0., -5*dr, dr], [0., -5*dr,-dr], [-dr, -5*dr, 0.], [dr, -5*dr, 0.]]),
                    jnp.ones(num_particles),
                    19989, 3300, 83909, 41234,
                    0., 0.,
                    None, 0, 0, 0, np.array([0]), np.array([0]), 
                    2,0,1,0,0)
        error_nf = testresults[0]
        error_ff = testresults[1]

        assert (error_nf < 0.01)
        assert (error_ff < 0.01)
            
    @pytest.mark.parametrize("delta", [0.0001,0.001,0.01,0.1,1,10])
    def test_sedimenting_triangle(self,delta):
        """Physical unit test for deterministic (shearless) part of hydrodynamic calculations, that can run also on CPU.
        
        Test instantaneous sedimentation of three spheres against reference values.

        """
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
                0., 0.5, 0.001, 0., 1, 1.,
                jnp.array([r1, r2, r3]),
                jnp.ones(3),
                0, 0, 0, 0, 0., 0.,
                None, 0, 0, 0,np.array([0]), np.array([0]),
                2,0,0,0.,0.)
        error = np.linalg.norm(reference_traj[index,:,:] - traj[0, :, :])
        assert (error < 9*1e-5)
    
    @pytest.mark.parametrize("delta", [0.0001, 0.001, 0.01, 0.1, 1, 10])
    # @pytest.mark.parametrize("delta", [1])
    def test_shear(self,delta):
        """Physical unit test for deterministic shear part of hydrodynamic calculations, that can run also on CPU. 
        
        Test instantaneous response to shear of three spheres against reference vaues.

        """
        deltas = np.array([0.0001, 0.001, 0.01, 0.1, 1, 10])
        index = np.searchsorted(deltas, delta)
        reference_traj = np.load('files/sheared_triplet_reference.npy')  # Load reference trajectory
        r1 = np.array([0., -2. - delta, 0.])
        r2 = np.array([0., 0., 0.])
        r3 = np.array([0., 2. + delta, 0.]) 
        traj, _, _, _ = main.main(
            1, 1, 0.1, 50, 50, 50, 3, 0.5,
            0., 0.5, 0.001, 0., 0, 0.,
            jnp.array([r1, r2, r3]),
            jnp.ones(3),
            0, 0, 0, 0, 1., 0.,
            None, 0, 0, 0, np.array([0]), np.array([0]),
            2, 0, 0, 0., 0.
            )
        # Only calculate error for this specific delta
        error = np.linalg.norm(reference_traj[index,:,:] - traj[0, :, :])
        assert error < 9*1e-5
