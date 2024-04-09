import main
import numpy as np
import utils

print('This software performs Stokesian Dynamics simulations of colloidal particles in a tricyclic periodic box')
print()

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

shear_rate_0 = float(input('Insert value of shear rate amplitude and press enter: '))
shear_freq = float(input('Insert value of shear rate frequency and press enter: '))

stresslet_flag = int(input('Insert 1 for storing the stresslet, or 0 for not storing it, and press enter: '))

print('Vol Fraction is ', N/(Lx*Ly*Lz)*4*np.pi/3)

init_positions_seed = int(input("Insert a seed for initial particle configuration (set to 0 to set positions manually for each particle) and press enter: "))
if(init_positions_seed == 0):
    positions = np.zeros((N,3))
    for i in range(N):
        positions[i,0]=float(input(f"Insert the x-coordinates of particle {i} and press enter: "))
        positions[i,1]=float(input(f"Insert the y-coordinates of particle {i} and press enter: "))
        positions[i,2]=float(input(f"Insert the z-coordinates of particle {i} and press enter: "))
else:
    positions = utils.CreateRandomConfiguration(
    Lx, N, init_positions_seed)  # (N,3) array of particle positions
    
        
if (kT>0):
    print('Brownian motion needs 4 seeds.')
    seed_RFD = int(input("Insert a seed for the random-finite-difference and press enter: "))
    seed_ffwave = int(input("Insert a seed for wave space far-field Brownian motion and press enter: "))
    seed_ffreal = int(input("Insert a seed for real space far-field Brownian motion and press enter: "))
    seed_nf = int(input("Insert a seed for real space near-field Brownian motion and press enter: "))
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
    0, # LJ potential cutoff  
    positions,
    seed_RFD, seed_ffwave, seed_ffreal, seed_nf,
    shear_rate_0, shear_freq,
    'trajectory',  # file name for output )
    stresslet_flag
)
