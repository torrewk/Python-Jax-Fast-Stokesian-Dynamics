import main
import numpy as np
import utils
#This script performs Stokesian Dynamics simulations of colloidal particles in a tricyclic periodic box
#Specify the simulation parameters below and run

Nsteps = 2          # (int) Number of simulation timestep to perform
N = 2000               # (int) Number of particles in the simulation
writing_period = 1  # (int) Period for writing to file
L = 75.                 # (float) Box size
dt = 0.000001              # (float) timestep

print('Vol Fraction is ', N/(L*L*L)*4*np.pi/3)

# seed for creating initial configuration of non-overlapping particles
init_positions_seed = 864645
positions = utils.CreateRandomConfiguration(
    L, N, init_positions_seed)  # (N,3) array of particle positions

#seeds needed for the entire calculation
seed_RFD = 73247  # random-finite-difference calculation for grad(R_FU)
seed_ffwave = 83909  # far-field (wavespace) thermal fluctuations
seed_ffreal = 13651  # far-field (realspace) thermal fluctuations
seed_nf = 53465477  # near-field (realspace) thermal fluctuations

kT = 0.             # (float) Brownian motion not fully tested, leave temperature to zero
shear_rate_0 = 0.5  # (float) Shear rate amplitude
shear_freq = 0     # (float) Shear frequency
U = 10.  # (float) Strength of colloidal bonds

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
    0,
    # potential cutoff (1.5*colloid_diameter for attractive, 1*colloid_diameter for purely repulsive)
    1.,
    positions,
    seed_RFD, seed_ffwave, seed_ffreal, seed_nf,
    shear_rate_0, shear_freq,
    'output'  # file name for output )
)
