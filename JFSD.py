import main
import numpy as np
import utils
#This script performs Stokesian Dynamics simulations of colloidal particles in a tricyclic periodic box
#Specify the simulation parameters below and run

Nsteps = 1000         # (int) Number of simulation timestep to perform
N = 1000             # (int) Number of particles in the simulation
writing_period = 1  # (int) Period for writing to file
L = 20.                 # (float) Box size
dt = 0.01              # (float) timestep

# boolean flag to compute and save to file the stresslet (needed for rheology)
stresslet_flag = 0

print('Vol Fraction is ', N/(L*L*L)*4*np.pi/3)

# seed for creating initial configuration of non-overlapping particles
init_positions_seed = 864645
positions = utils.CreateRandomConfiguration(
    L, N, init_positions_seed)  # (N,3) array of particle positions

#seeds needed for the entire calculation
seed_RFD = 46075  # random-finite-difference calculation for grad(R_FU)
seed_ffwave = 73247  # far-field (wavespace) thermal fluctuations
seed_ffreal = 4182  # far-field (realspace) thermal fluctuations
seed_nf = 53465477  # near-field (realspace) thermal fluctuations

# (float) Brownian motion not fully tested, leave temperature to zero
kT = 1 / 60 / np.pi
shear_rate_0 = 0.1  # (float) Shear rate amplitude
shear_freq = 0.     # (float) Shear frequency
U = 10.*kT  # (float) Strength of colloidal bonds (AO interaction)

main.main(
    Nsteps,
    writing_period,
    dt,  # simulation timestep
    L, L, L,  # box sizes
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
