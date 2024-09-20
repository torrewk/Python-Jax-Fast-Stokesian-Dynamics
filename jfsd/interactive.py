import numpy as np

from jfsd import main, utils

def interactive_main():

    print('This software performs Stokesian Dynamics simulations of colloidal particles in a tricyclic periodic box')
    print()

    HIs_flag = int(input('First, insert 0 for Brownian Dynamics, 1 for Rotne-Prager-Yamakawa, or 2 for Stokesian Dynamics and press enter: '))

    Nsteps = int(input('Insert number of simulation timesteps to perform and press enter: '))

    writing_period = int(input('Insert storing period (periodicity of saving data) and press enter: '))
    if(writing_period>Nsteps):
        raise ValueError("Periodicity of saving cannot be larger than the total amount of simulation timesteps. Abort!")

    N = int(input('Insert number of particles in the box and press enter: '))
    Lx = float(input('Insert x-length of simulation box and press enter: '))
    Ly = float(input('Insert y-length of simulation box and press enter: '))
    Lz = float(input('Insert z-length of simulation box and press enter: '))

    dt = float(input('Insert value of time step and press enter: '))


    kT = float(input('Insert value of temperature (set to 1 to have a single-particle diffusion coefficient = 1) and press enter: '))
    U = float(input('Insert value of interaction strength (in units of thermal energy) and press enter: '))
    U = U * kT
    U_cutoff = float(input('Insert value of space cutoff for pair-interactions (in units of particle radius) and press enter: '))

    shear_rate_0 = float(input('Insert value of shear rate amplitude and press enter: '))
    shear_freq = float(input('Insert value of shear rate frequency and press enter: '))
    alpha_fric = float(input('Insert value of friction coeff. and press enter: '))
    h0_fric = float(input('Insert value of friction range (in units of particle radius) and press enter: '))

    stresslet_flag = int(input('Insert 1 for storing the stresslet, or 0 for not storing it, and press enter: '))
    velocity_flag = int(input('Insert 1 for storing the velocity, or 0 for not storing it, and press enter: '))
    orientation_flag = int(input('Insert 1 for storing the orientation, or 0 for not storing it, and press enter: '))


    buffer = int(input('In case constant (and uniform in space) applied FORCES are needed, press 1 and enter. Otherwise 0 and enter:'))
    constant_applied_forces  = np.zeros((N,3))
    if( buffer == 1 ):    
        fx = float(input('Insert the value for the component x of the constant force, and press enter:'))            
        fy = float(input('Insert the value for the component y of the constant force, and press enter:'))            
        fz = float(input('Insert the value for the component z of the constant force, and press enter:'))                
        constant_applied_forces[:,0]  = fx
        constant_applied_forces[:,1]  = fy
        constant_applied_forces[:,2]  = fz
    buffer = int(input('In case constant (and uniform in space) applied TORQUES are needed, press 1 and enter. Otherwise 0 and enter:'))
    constant_applied_torques  = np.zeros((N,3))
    if( buffer == 1 ):    
        fx = float(input('Insert the value for the component x of the constant torque, and press enter:'))            
        fy = float(input('Insert the value for the component y of the constant torque, and press enter:'))            
        fz = float(input('Insert the value for the component z of the constant torque, and press enter:'))            
        constant_applied_torques[:,0]  = fx
        constant_applied_torques[:,1]  = fy
        constant_applied_torques[:,2]  = fz    
        
    print('Vol Fraction is ', N/(Lx*Ly*Lz)*4*np.pi/3)

    init_positions_seed = int(input("Insert a seed for initial particle configuration (set to 0 to set positions manually for each particle, or to load a numpy array) and press enter: "))
    if(init_positions_seed == 0):
        traj_name = str(input("Insert the name of the .npy array (must be in the same folder as this script), or leave empty to manually set the positions, and press enter: "))
        if( traj_name==''):
            positions = np.zeros((N,3))
            for i in range(N):
                positions[i,0]=float(input(f"Insert the x-coordinates of particle {i} and press enter: "))
                positions[i,1]=float(input(f"Insert the y-coordinates of particle {i} and press enter: "))
                positions[i,2]=float(input(f"Insert the z-coordinates of particle {i} and press enter: "))
        else:
            positions = np.load(traj_name)
    else:
        positions = utils.create_hardsphere_configuration(Lx,N,init_positions_seed,0.001)

    output_name = str(input('Insert the name of the output file and press enter:'))    
            
    if ((kT>0) and (HIs_flag==2)):
        print('Brownian motion needs 4 seeds.')
        seed_RFD = int(input("Insert a seed for the random-finite-difference and press enter: "))
        seed_ffwave = int(input("Insert a seed for wave space far-field Brownian motion and press enter: "))
        seed_ffreal = int(input("Insert a seed for real space far-field Brownian motion and press enter: "))
        seed_nf = int(input("Insert a seed for real space near-field Brownian motion and press enter: "))
    if ((kT>0) and (HIs_flag==1)):
        print('Brownian motion needs 2 seeds.')
        seed_ffwave = int(input("Insert a seed for wave space far-field Brownian motion and press enter: "))
        seed_ffreal = int(input("Insert a seed for real space far-field Brownian motion and press enter: "))
        seed_RFD = 0
        seed_nf = 0
        
    if ((kT>0) and (HIs_flag==0)):
        print('Brownian motion needs 1 seed.')
        seed_RFD = 0
        seed_ffwave = 0
        seed_ffreal = 0
        seed_nf = int(input("Insert a seed for Brownian forces and press enter: "))
    else:
        seed_RFD = seed_ffwave = seed_ffreal = seed_nf = 0

    main.main(
            Nsteps,
            writing_period,
            dt,  # simulation timestep
            Lx, Ly, Lz,  # box sizes
            N,  # number of particles
            0.5,  # max box strain
            kT,  # thermal energy
            1,  # radius of a colloid (leave to 1)
            0.5,  # ewald parameter (leave to 0.5)
            0.001,  # error tolerance
            U,  # strength of bonds
            0,  # buoyancy
            0, # potential cutoff
            positions,
            seed_RFD, seed_ffwave, seed_ffreal, seed_nf,
            shear_rate_0, shear_freq,
            output_name,  # file name for output )
            stresslet_flag,
            velocity_flag,
            orientation_flag,
            constant_applied_forces,
            constant_applied_torques,
            HIs_flag,
            0,alpha_fric,h0_fric)
