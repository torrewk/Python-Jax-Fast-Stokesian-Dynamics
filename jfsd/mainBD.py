import math
import os
import time

import jax.numpy as jnp
import jax.scipy as jscipy

from jfsd import jaxmd_space as space
from jfsd import jaxmd_partition as partition

import numpy as np
from jax import jit, random
from jax.config import config

from jfsd import appliedForces, shear, thermal, utils

config.update("jax_enable_x64", False) #disable double precision
np.set_printoptions(precision=8, suppress=True)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false' # avoid JAX allocating most of the GPU memory even if not needed


def main(
    Nsteps: int,
    writing_period: int,
    dt: float,
    Lx: float, Ly: float, Lz: float,
    N: int,
    max_strain: float,
    T: float,
    a: float,
    xi: float,
    error: float,
    U: float,
    buoyancy_flag: int,
    U_cutoff: float,
    positions: float,
    seed_RFD: int,
    seed_ffwave: int,
    seed_ffreal: int,
    seed: int,
    shear_rate_0: float,
    shear_freq: float,
    output: str,
    stresslet_flag: bool,
    velocity_flag: bool,
    orient_flag: bool,
    constant_applied_forces: float,
    constant_applied_torques: float,
    HIs_flag: int) -> tuple:
    
    """Integrate the particles equation of motions forward in time, 
    including only single-body hydrodynamics interactions.

    While the simulation is performed, trajectories data are saved into a .npy file.

    Parameters
    ----------
    Nsteps:
        Number of timesteps
    writing_period:
        Period for writing to file
    dt:
        Timestep
    Lx:
        Box size (x-direction)
    Ly:
        Box size (y-direction)
    Lz:
        Box size (z-direction)
    N:
        Number of particles
    max_strain:
        Max strain applied to the box (not needed in Brownian Dynamics)
    T:
        Thermal energy
    a:
        Particle radius
    xi:
        Ewald split parameter
    error:
        Tolerance error
    U:
        Interaction strength
    buoyancy_flag:
        Set to 1 to have gravity acting on colloids
    U_cutoff:
        Distance cutoff for interacting particles
    positions:
        Array of particles initial positions (N,3)
    seed_RFD:
        Seed for Brownian Drift calculation (not needed in Brownian Dynamics)
    seed_ffwave:
        Seed for wave space part of far-field velocity slip (not needed in Brownian Dynamics)
    seed_ffreal:
        Seed for real space part of far-field velocity slip (not needed in Brownian Dynamics)
    seed:
        Seed for random Brownian forces
    shear_rate_0:
        Axisymmetric shear rate amplitude
    shear_freq:
        Frequency of shear, set to zero to have simple shear
    output:
        File name for output
    stresslet_flag:
        To have stresslet in the output
    velocity_flag:
        To have velocities in the output/var/log/nvidia-installer.log
    orient_flag:
        To have particle orientations in the output
    constant_applied_forces:
        Array of external forces (N,3)
    constant_applied_torques:
        Array of external torques (N,3)
    HIs_flag:
        Flag used to set level of hydrodynamic interaction. 0 for BD, 1 for SD.

    Returns
    -------
        trajectory, stresslet_history, velocities

    """

    @jit
    def update_positions(
        shear_rate: float,
        positions: float,
        displacements_vector_matrix: float,
        nbrs_list: partition.NeighborList, 
        net_vel: float,
        dt: float) -> tuple: 
                              
        """Update particle positions and neighbor lists
        
        Parameters
        ----------
        shear_rate:
            Shear rate at current time step
        positions:
            Array (N,3) of particles positions
        displacements_vector_matrix:
            Array (N,N,3) of relative displacements between particles
        nbrs_list:
            Neighbor lists for precondition matrix
        net_vel:
            Array (6*N) of linear/angular velocities relative to the background flow
        dt:
            Discrete time step
        
        Returns
        -------
        positions, displacements_vector_matrix, nbrs_list
        
        """   

        #Define array of displacement r(t+dt)-r(t)
        dR = jnp.zeros((N, 3), float)
        #Compute actual displacement due to velocities (relative to background flow)
        dR = dR.at[:, 0].set(dt*net_vel.at[(0)::6].get())
        dR = dR.at[:, 1].set(dt*net_vel.at[(1)::6].get())
        dR = dR.at[:, 2].set(dt*net_vel.at[(2)::6].get())
        #Apply displacement and compute wrapped shift (Lees Edwards boundary conditions)
        # shift system origin to (0,0,0)
        positions = shift(positions+jnp.array([Lx, Ly, Lz])/2, dR)
        # re-shift origin back to box center
        positions = positions - jnp.array([Lx, Ly, Lz])*0.5

        #Define array of displacement r(t+dt)-r(t) (this time for displacement given by background flow)
        dR = jnp.zeros((N, 3), float)
        # Assuming y:gradient direction, x:background flow direction
        dR = dR.at[:, 0].set(dt * shear_rate * positions.at[:, 1].get())
        # Shift system origin to (0,0,0)
        positions = shift(positions+jnp.array([Lx, Ly, Lz])/2, dR)
        # re-shift origin back to box center
        positions = positions - jnp.array([Lx, Ly, Lz]) * 0.5

        #Compute new relative displacements between particles
        displacements_vector_matrix = (
            space.map_product(displacement))(positions, positions)

        #Update neighbor lists
        nbrs_list = nbrs_list.update(positions)

        return positions, displacements_vector_matrix, nbrs_list



    start_time = time.time()  # perfomances evaluation
    #######################################################################
   
    xy = 0.  # set box tilt factor to zero to begin (unsheared box)

    #  set INITIAL Periodic Space and Displacement Metric
    #  box is defined by an upper triangular matrix of the form: (xy,xz,yz are tilt factors)
    #    [Lx  Ly*xy  Lz*xz]
    #    [0   Ly     Lz*yz]
    #    [0   0      Lz   ]
    displacement, shift = space.periodic_general(
        jnp.array([[Lx, Ly*xy, Lz*0.], [0., Ly, Lz*0.], [0., 0., Lz]]), fractional_coordinates=False)

    # compute matrix of INITIAL displacements between particles (each element is a vector from particle j to i)
    displacements_vector_matrix = (
        space.map_product(displacement))(positions, positions)
    
    # set external applied forces/torques (no pair-interactions, will be added later)
    AppliedForce = jnp.zeros(3*N, float)
    AppliedTorques = jnp.zeros(3*N, float)
    if(buoyancy_flag == 1):  # apply buoyancy forces (in z direction)
        AppliedForce = AppliedForce.at[2::3].add(-1.)
    if(np.count_nonzero(constant_applied_forces) > 0):  # apply external forces
        AppliedForce += jnp.ravel(constant_applied_forces)
    if(np.count_nonzero(constant_applied_torques) > 0):  # apply external torques
        AppliedTorques += jnp.ravel(constant_applied_torques)
    
    # initialize neighborlists
    neighbor_fn = utils.initialize_single_neighborlist(
        U_cutoff, Lx, Ly, Lz, displacement)

    # allocate lists for first time - they will be updated at each timestep and if needed, re-allocated too.
    nbrs_list = neighbor_fn.allocate(
        positions+jnp.array([Lx, Ly, Lz])/2)
    nl = np.array(nbrs_list.idx)
    
    # set array for output trajectory, velocities and stresslet in time
    trajectory = np.zeros((int(Nsteps/writing_period), N, 3), float)
    stresslet_history = np.zeros((int(Nsteps/writing_period), N, 5), float)
    velocities = np.zeros((int(Nsteps/writing_period), N, 6), float)

    # create RNG states
    key = random.PRNGKey(seed)

    # initialize stresslet (for output, not needed to integrate trajectory in space)
    stresslet = jnp.zeros((N, 5), float)

    #measure compilation time of the first part
    compilation_time = time.time() - start_time
        
    # kwt debug: check if particles overlap
    overlaps, overlaps2 = utils.check_overlap(
        displacements_vector_matrix)
    print('Starting: initial overlaps are ', jnp.sum(overlaps)-N)
 
    start_time = time.time() #perfomances evaluation
    

    for step in range(Nsteps):
        nl = np.array(nbrs_list.idx)
        # initialize generalized velocity (6*N array, for linear and angular components)
        general_velocity = jnp.zeros(6*N, float) #this array stores the 6N dimensional generalized velocity
        saddle_b = jnp.zeros(17*N, float)


        if((stresslet_flag > 0) and ((step % writing_period) == 0)):
            #reset stresslet to zero
            stresslet = jnp.zeros((N, 5), float)
            
        
        # precompute quantities for far-field and near-field calculation
        (r, indices_i, indices_j) = utils.precomputeBD(positions, nl, displacements_vector_matrix, N, Lx, Ly, Lz)
        
        # compute shear-rate for current timestep: simple(shear_freq=0) or oscillatory(shear_freq>0)
        shear_rate = shear.update_shear_rate(
            dt, step, shear_rate_0, shear_freq, phase=0)
        
        # add applied forces and conservative (pair potential) forces to right-hand side of system
        saddle_b = appliedForces.sumAppliedForces(N, AppliedForce, AppliedTorques, saddle_b, U,
                                                  indices_i, indices_j, displacements_vector_matrix,
                                                  U_cutoff,HIs_flag)
        general_velocity = - saddle_b[11*N:]
        
        # compute Thermal Fluctuations only if temperature is not zero
        if(T > 0):

            # generate random numbers for far-field random forces
            
            key, random_array = utils.generate_random_array(
                key, (6*N))

            # compute far-field (wave space contribution) slip velocity and set in rhs of linear system
            random_force = thermal.compute_BD_randomforce(N, T, dt,random_array)
            general_velocity += random_force
            
        
        
        if((stresslet_flag > 0) and ((step % writing_period) == 0)):
            
            # save stresslet
            stresslet_history[int(step/writing_period), :, :] = stresslet
            if(output != 'None'):
                np.save('stresslet_'+output, stresslet_history)

        # update positions
        (positions, displacements_vector_matrix,nbrs_list) = update_positions(shear_rate,positions, 
                                                                       displacements_vector_matrix,
                                                                       nbrs_list,
                                                                       general_velocity,
                                                                       dt)

                                
        # if system is sheared update box tilt factor
        if(shear_rate_0 != 0):
            xy = shear.update_box_tilt_factor(
                dt, shear_rate_0, xy, step, shear_freq)
            displacement, shift = space.periodic_general(
                jnp.array([[Lx, Ly*xy, Lz*0.], [0., Ly, Lz*0.], [0., 0., Lz]]), fractional_coordinates=False)
            

        # compute compilation time
        if(step == 0):
            compilation_time2 = (time.time() - start_time)
            print('Compilation Time is ', compilation_time+compilation_time2)

        # write trajectory to file
        if((step % writing_period) == 0):

            #check that the position to save do not contain 'nan' or 'inf'
            if((jnp.isnan(positions)).any() or (jnp.isinf(positions)).any()):
                raise ValueError(
                    "Invalid particles positions. Abort!")

            #save trajectory to file
            trajectory[int(step/writing_period), :, :] = positions
            if(output != 'None'):
                np.save(output, trajectory)
                
            #store velocity (linear and angular) to file
            if( (velocity_flag > 0) ):
                velocities[int(step/writing_period), :, :] = jnp.reshape(general_velocity, (N,6))
                if (output != 'None'): 
                    np.save('velocities_'+output, velocities)
                    
            #store orientation
            #TODO
            
            # kwt debug: check if particles overlap
            # overlaps, overlaps2 = utils.check_overlap(
            #     displacements_vector_matrix)
            # print('Step= ', step, ' Overlaps are ', jnp.sum(overlaps)-N)
                
    end_time = time.time()
    print('Time for ', Nsteps, ' steps is ', end_time-start_time-compilation_time2,
          'or ', Nsteps/(end_time-start_time-compilation_time2), ' steps per second')

    return trajectory, stresslet_history, velocities
