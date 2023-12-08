import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
from jax import random

import time
import numpy as np
from functools import partial
from jax import jit
from jax import random
import jax.numpy as jnp
import jax.scipy as jscipy
from jax_md import partition, space
import sys
import math
from jax.config import config
config.update("jax_enable_x64", True)
import scipy

np.set_printoptions(precision=8,suppress=True)



def initialize_neighborlist(a, Lx, Ly, Lz, displacement, ewald_cut):
    # Set various Neighborlist
    # For Lubrication Hydrodynamic Forces Calculation
    lub_neighbor_fn = partition.neighbor_list(displacement,  # Displacement metric
                                              jnp.array([Lx, Ly, Lz]), # Box size
                                                r_cutoff=4.,  # Spatial cutoff for 2 particles to be neighbor
                                              dr_threshold=0.1,  # displacement of particles threshold to recompute neighbor list
                                              capacity_multiplier=1,
                                              format=partition.NeighborListFormat.OrderedSparse)
    # For Precondition of Lubrication Hydrodynamic Forces Calculation
    prec_lub_neighbor_fn = partition.neighbor_list(displacement,  # Displacement metric
                                                   jnp.array([Lx, Ly, Lz]), # Box size
                                                    r_cutoff=2.1,  # Spatial cutoff for 2 particles to be neighbor
                                                    dr_threshold=0.1,  # displacement of particles threshold to recompute neighbor list
                                                   capacity_multiplier=1,
                                                   format=partition.NeighborListFormat.OrderedSparse)

    # For Far-Field Real Space Hydrodynamic Forces Calculation
    ff_neighbor_fn = partition.neighbor_list(displacement,  # Displacement metric
                                             jnp.array([Lx, Ly, Lz]), # Box size
                                             r_cutoff=ewald_cut,  # Spatial cutoff for 2 particles to be neighbor
                                             dr_threshold=0.1,  # displacement of particles threshold to recompute neighbor list
                                             capacity_multiplier=1,
                                             format=partition.NeighborListFormat.OrderedSparse)
    # For Interparticle Potential Forces Calculation
    pot_neighbor_fn = partition.neighbor_list(displacement,  # Displacement metric
                                              jnp.array([Lx, Ly, Lz]), # Box size
                                              r_cutoff=3*a,  # Spatial cutoff for 2 particles to be neighbor
                                              dr_threshold=0.1,  # displacement of particles threshold to recompute neighbor list
                                              capacity_multiplier=1,
                                              format=partition.NeighborListFormat.OrderedSparse)
    return lub_neighbor_fn, prec_lub_neighbor_fn, ff_neighbor_fn, pot_neighbor_fn


def main(Nsteps,  # int: number of simulation steps to perform
         writing_period, #period of writing to file 
         dt,  # float: timestep
         Lx, Ly, Lz,  # float: Box size,
         N,  # int: number of particles in the simulation
         max_strain, # float: Maximus strain applied to the box (can set to 0.5)
         T,  # float: Temperature
         a,  # float: Particle Radius
         xi,  # float: Ewald split parameter (~0.5 for Soft Matter systems)
         error,  # float: tolerance error (set to 0.001)
         U,  # float: Interaction strength
         U_cutoff,  # float: distance cutoff for interacting particles
         potential,  # string: Type of interactions between particles
         positions,  # array: Particle positions (in input: initial positions)
         seed_RFD, # int: Simulation seed for Brownian Drift calculation
         seed_ffwave, # int: Simulation seed for wave space part of far-field velocity slip
         seed_ffreal, # int: Simulation seed for real space part of far-field velocity slip
         seed_nf,      # int: Simulation seed for near-field random forces 
         shear_rate,   # assuming simple axisymmetric shear
         gravity,
         outputfile
         ):

    
    def random_force_on_grid_indexing(Nx,Ny,Nz):
        normal_indices = []
        normal_conj_indices = []
        nyquist_indices = []
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    
                    if (not(2*k >= Nz+1) and 
                        not((k == 0) and (2*j >= Ny+1)) and 
                        not((k == 0) and (j == 0) and (2*i >= Nx+1)) and 
                        not((k == 0) and (j == 0) and (i == 0))):
                        
                        ii_nyquist = ( ( i == int(Nx/2) ) and ( int(Nx/2) == int((Nx+1)/2) ) )
                        jj_nyquist = ( ( j == int(Ny/2) ) and ( int(Ny/2) == int((Ny+1)/2) ) )
                        kk_nyquist = ( ( k == int(Nz/2) ) and ( int(Nz/2) == int((Nz+1)/2) ) )
                        if (   (i==0 and jj_nyquist and k==0) 
                            or (ii_nyquist and j==0 and k==0) 
                            or (ii_nyquist and jj_nyquist and k==0)
                            or (i==0 and j==0 and kk_nyquist)
                            or (i ==0 and jj_nyquist and kk_nyquist)
                            or (ii_nyquist and j==0 and kk_nyquist)
                            or (ii_nyquist and jj_nyquist and kk_nyquist)):
                            nyquist_indices.append([i,j,k])
                        else:
                            normal_indices.append([i,j,k])
                            if(ii_nyquist or (i==0)):
                                i_conj = i    
                            else:
                                i_conj = Nx - i
                            if(jj_nyquist or (j==0)):
                                j_conj = j    
                            else:
                                j_conj = Ny - j
                            if(kk_nyquist or (k==0)):
                                k_conj = k    
                            else:
                                k_conj = Nz - k
                            normal_conj_indices.append([i_conj,j_conj,k_conj])
        return jnp.array(normal_indices),jnp.array(normal_conj_indices),jnp.array(nyquist_indices)
    
    
    def compute_real_space_ewald_table():  # table (filled with zeroes as input)

            # Table discretization in double precision
            dr = np.longfloat(0.00100000000000000000000000000000)

            Imrr = np.zeros(nR)
            rr = np.zeros(nR)
            g1 = np.zeros(nR)
            g2 = np.zeros(nR)
            h1 = np.zeros(nR)
            h2 = np.zeros(nR)
            h3 = np.zeros(nR)

            xxi = np.longfloat(xi)
            a_64 = np.longfloat(a)
            Pi = np.longfloat(3.1415926535897932384626433832795)
            kk = np.arange(nR,dtype=np.longdouble)
            r_array = (kk * dr + dr)
            
            # Expression have been simplified assuming no overlap, touching, and overlap

            for i in range(nR):
                
                r = r_array[i]    
                
                if(r>2*a_64):
        
                    Imrr[i] = (-math.pow(a_64, -1) + (math.pow(a_64, 2)*math.pow(r, -3))/2. + (3*math.pow(r, -1))/4. + (
                        3*math.erfc(r*xxi)*math.pow(a_64, -2)*math.pow(r, -3)*(-12*math.pow(r, 4) + math.pow(xxi, -4)))/128
                    + math.pow(a_64, -2)*((9*r)/32. -
                                        (3*math.pow(r, -3)*math.pow(xxi, -4))/128.)
                    + (math.erfc((2*a_64 + r)*xxi)*(128*math.pow(a_64, -1) + 64*math.pow(a_64, 2)*math.pow(r, -3) +
                      96*math.pow(r, -1) + math.pow(a_64, -2)*(36*r - 3*math.pow(r, -3)*math.pow(xxi, -4))))/256.
                    + (math.erfc(2*a_64*xxi - r*xxi)*(128*math.pow(a_64, -1) - 64*math.pow(a_64, 2)*math.pow(r, -3) -
                                                          96*math.pow(r, -1) + math.pow(a_64, -2)*(-36*r + 3*math.pow(r, -3)*math.pow(xxi, -4))))/256.
                    + (3*math.exp(-(math.pow(r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -2)*math.pow(Pi, -0.5)
                        * math.pow(r, -2)*math.pow(xxi, -3)*(1 + 6*math.pow(r, 2)*math.pow(xxi, 2)))/64.
                    + (math.exp(-(math.pow(2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -2)*math.pow(Pi, -0.5)*math.pow(r, -3)*math.pow(xxi, -3)
                        * (8*r*math.pow(a_64, 2)*math.pow(xxi, 2) - 16*math.pow(a_64, 3)*math.pow(xxi, 2) + a_64*(2 - 28*math.pow(r, 2)*math.pow(xxi, 2)) - 3*(r + 6*math.pow(r, 3)*math.pow(xxi, 2))))/128.
                    + (math.exp(-(math.pow(-2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -2)*math.pow(Pi, -0.5)*math.pow(r, -3)*math.pow(xxi, -3) *
                        (8*r*math.pow(a_64, 2)*math.pow(xxi, 2) + 16*math.pow(a_64, 3)*math.pow(xxi, 2) + a_64*(-2 + 28*math.pow(r, 2)*math.pow(xxi, 2)) - 3*(r + 6*math.pow(r, 3)*math.pow(xxi, 2))))/128.)



                    rr[i] =( -math.pow(a_64, -1) - math.pow(a_64, 2)*math.pow(r, -3) + (3*math.pow(r, -1))/2. + (
                                3*math.pow(a_64, -2)*math.pow(r, -3)*(4*math.pow(r, 4) + math.pow(xxi, -4)))/64.
                    + (math.erfc(2*a_64*xxi - r*xxi)*(64*math.pow(a_64, -1) + 64*math.pow(a_64, 2)*math.pow(r, -3) -
                        96*math.pow(r, -1) + math.pow(a_64, -2)*(-12*r - 3*math.pow(r, -3)*math.pow(xxi, -4))))/128.
                    + (math.erfc((2*a_64 + r)*xxi)*(64*math.pow(a_64, -1) - 64*math.pow(a_64, 2)*math.pow(r, -3) +
                              96*math.pow(r, -1) + math.pow(a_64, -2)*(12*r + 3*math.pow(r, -3)*math.pow(xxi, -4))))/128.
                    + (3*math.exp(-(math.pow(r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -2)*math.pow(Pi, -0.5)
                              * math.pow(r, -2)*math.pow(xxi, -3)*(-1 + 2*math.pow(r, 2)*math.pow(xxi, 2)))/32.
                    - ((2*a_64 + 3*r)*math.exp(-(math.pow(-2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -2)*math.pow(Pi, -0.5)*math.pow(r, -3)*math.pow(xxi, -3) *
                                (-1 - 8*a_64*r*math.pow(xxi, 2) + 8*math.pow(a_64, 2)*math.pow(xxi, 2) + 2*math.pow(r, 2)*math.pow(xxi, 2)))/64.
                    + ((2*a_64 - 3*r)*math.exp(-(math.pow(2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -2)*math.pow(Pi, -0.5)*math.pow(r, -3)*math.pow(xxi, -3) *
                                (-1 + 8*a_64*r*math.pow(xxi, 2) + 8*math.pow(a_64, 2)*math.pow(xxi, 2) + 2*math.pow(r, 2)*math.pow(xxi, 2)))/64.
                    - (3*math.erfc(r*xxi)*math.pow(a_64, -2)*math.pow(r, -3)
                              * math.pow(xxi, -4)*(1 + 4*math.pow(r, 4)*math.pow(xxi, 4)))/64.)
                    
                    g1[i] =    (    (math.exp(-(math.pow(r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -4)*math.pow(Pi, -0.5)*math.pow(r, -3) *
                                    math.pow(xxi, -5)*(9 + 15*math.pow(r, 2)*math.pow(xxi, 2) - 30*math.pow(r, 4)*math.pow(xxi, 4)))/64.
                    + (math.exp(-(math.pow(2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -4)*math.pow(Pi, -0.5)*math.pow(r, -4)*math.pow(xxi, -5)
                                      * (18*a_64 - 45*r - 3*(2*a_64 + r)*(-16*a_64*r + 8*math.pow(a_64, 2) + 25*math.pow(r, 2))*math.pow(xxi, 2) + 6*(2*a_64 + r)*(-32*r*math.pow(a_64, 3) + 32*math.pow(a_64, 4) + 44*math.pow(a_64, 2)*math.pow(r, 2) - 36*a_64*math.pow(r, 3) + 25*math.pow(r, 4))*math.pow(xxi, 4))) / 640.
                    + (math.exp(-(math.pow(-2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -4)*math.pow(Pi, -0.5)*math.pow(r, -4)*math.pow(xxi, -5)
                                      * (-9*(2*a_64 + 5*r) + 3*(2*a_64 - r)*(16*a_64*r + 8*math.pow(a_64, 2) + 25*math.pow(r, 2))*math.pow(xxi, 2) - 6*(2*a_64 - r)*(32*r*math.pow(a_64, 3) + 32*math.pow(a_64, 4) + 44*math.pow(a_64, 2)*math.pow(r, 2) + 36*a_64*math.pow(r, 3) + 25*math.pow(r, 4))*math.pow(xxi, 4)))/640.
                    + (3*math.erfc(r*xxi)*math.pow(a_64, -4)*math.pow(r, -4)*math.pow(xxi, -6) *
                                      (3 + 3*math.pow(r, 2)*math.pow(xxi, 2) + 20*math.pow(r, 6)*math.pow(xxi, 6)))/128.
                    - (3*math.erfc((-2*a_64 + r)*xxi)*math.pow(a_64, -4)*math.pow(r, -4)*math.pow(xxi, -6)*(15 + 5*math.pow(r, 2)*math.pow(xxi, 2)*(3 + 64*math.pow(
                                        a_64, 4)*math.pow(xxi, 4)) + 512*math.pow(a_64, 6)*math.pow(xxi, 6) - 256*a_64*math.pow(r, 5)*math.pow(xxi, 6) + 100*math.pow(r, 6)*math.pow(xxi, 6)))/1280.
                    - (3*math.erfc((2*a_64 + r)*xxi)*math.pow(a_64, -4)*math.pow(r, -4)*math.pow(xxi, -6)*(15 + 5*math.pow(r, 2)*math.pow(xxi, 2)*(3 + 64*math.pow(a_64, 4)*math.pow(xxi, 4)) + 512*math.pow(a_64, 6)*math.pow(xxi, 6) + 256*a_64*math.pow(r, 5)*math.pow(xxi, 6)
                                                                                                                                + 100*math.pow(r, 6)*math.pow(xxi, 6)))/1280.)
                    
                    
                    g2[i] =  ( (-3*math.exp(-(math.pow(r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -4)*math.pow(Pi, -0.5)*math.pow(r, -3)
                                    * math.pow(xxi, -5)*(3 - math.pow(r, 2)*math.pow(xxi, 2) + 2*math.pow(r, 4)*math.pow(xxi, 4)))/64.
                    + (math.exp(-(math.pow(-2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -4)*math.pow(Pi, -0.5)*math.pow(r, -4)*math.pow(xxi, -5)
                                        * (18*a_64 + 45*r - 3*(24*r*math.pow(a_64, 2) + 16*math.pow(a_64, 3) + 14*a_64*math.pow(r, 2) + 5*math.pow(r, 3))*math.pow(xxi, 2) + 6*(24*r*math.pow(a_64, 2) + 16*math.pow(a_64, 3) + 14*a_64*math.pow(r, 2) + 5*math.pow(r, 3))*math.pow(-2*a_64 + r, 2)*math.pow(xxi, 4)))/640.
                    + (math.exp(-(math.pow(2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -4)*math.pow(Pi, -0.5)*math.pow(r, -4)*math.pow(xxi, -5)
                                      * (-18*a_64 + 45*r + 3*(-24*r*math.pow(a_64, 2) + 16*math.pow(a_64, 3) + 14*a_64*math.pow(r, 2) - 5*math.pow(r, 3))*math.pow(xxi, 2) - 6*(-24*r*math.pow(a_64, 2) + 16*math.pow(a_64, 3) + 14*a_64*math.pow(r, 2) - 5*math.pow(r, 3))*math.pow(2*a_64 + r, 2)*math.pow(xxi, 4)))/640.
                    + (3*math.erfc((-2*a_64 + r)*xxi)*math.pow(a_64, -4)*math.pow(r, -4)*math.pow(xxi, -6)*(15 - 15*math.pow(r, 2)*math.pow(xxi, 2) +
                                      4*(128*math.pow(a_64, 6) - 80*math.pow(a_64, 4)*math.pow(r, 2) + 16*a_64*math.pow(r, 5) - 5*math.pow(r, 6))*math.pow(xxi, 6)))/1280.
                    + (3*math.erfc(r*xxi)*math.pow(a_64, -4)*math.pow(r, -4)*math.pow(xxi, -6) *
                                      (-3 + 3*math.pow(r, 2)*math.pow(xxi, 2) + 4*math.pow(r, 6)*math.pow(xxi, 6)))/128.
                    - (3*math.erfc((2*a_64 + r)*xxi)*math.pow(a_64, -4)*math.pow(r, -4)*math.pow(xxi, -6)*(-15 + 15*math.pow(r, 2)*math.pow(xxi, 2) +
                                      4*(-128*math.pow(a_64, 6) + 80*math.pow(a_64, 4)*math.pow(r, 2) + 16*a_64*math.pow(r, 5) + 5*math.pow(r, 6))*math.pow(xxi, 6)))/1280.)
                                    
                    
                    h1[i] =   (     (3*math.exp(-(math.pow(r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -4)*math.pow(xxi, -7)
                                    * (27 - 2*math.pow(xxi, 2)*(15*math.pow(r, 2) + 2*math.pow(r, 4)*math.pow(xxi, 2) - 4*math.pow(r, 6)*math.pow(xxi, 4) + 48*math.pow(a_64, 2)*(3 - math.pow(r, 2)*math.pow(xxi, 2) + 2*math.pow(r, 4)*math.pow(xxi, 4)))))/4096.
                    + (3*math.exp(-(math.pow(-2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -5)*math.pow(xxi, -7)
                                      * (270*a_64 - 135*r + 6*(2*a_64 + 5*r)*(12*math.pow(a_64, 2) + 5*math.pow(r, 2))*math.pow(xxi, 2) - 4*(144*r*math.pow(a_64, 4) + 96*math.pow(a_64, 5) + 64*math.pow(a_64, 3)*math.pow(r, 2) - 30*a_64*math.pow(r, 4) - 5*math.pow(r, 5))*math.pow(xxi, 4)
                    + 8*math.pow(2*a_64 - r, 3)*(96*r*math.pow(a_64, 3) + 48*math.pow(a_64, 4) + 80*math.pow(a_64, 2)*math.pow(r, 2) + 40*a_64*math.pow(r, 3) + 5*math.pow(r, 4))*math.pow(xxi, 6)))/40960.
                    + (3*math.exp(-(math.pow(2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -5)*math.pow(xxi, -7)
                       * (-135*(2*a_64 + r) - 6*(2*a_64 - 5*r)*(12*math.pow(a_64, 2) + 5*math.pow(r, 2))*math.pow(xxi, 2) + 4*(-144*r*math.pow(a_64, 4) + 96*math.pow(a_64, 5) + 64*math.pow(a_64, 3)*math.pow(r, 2) - 30*a_64*math.pow(r, 4) + 5*math.pow(r, 5))*math.pow(xxi, 4)
                    - 8*(-96*r*math.pow(a_64, 3) + 48*math.pow(a_64, 4) + 80*math.pow(a_64, 2)*math.pow(r, 2) - 40*a_64*math.pow(r, 3) + 5*math.pow(r, 4))*math.pow(2*a_64 + r, 3)*math.pow(xxi, 6)))/40960.
                    + (3*math.erfc(r*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(27 + 8*math.pow(xxi, 2)*(-6*math.pow(r, 2) + 9*math.pow(r, 4)*math.pow(xxi, 2) - 2*math.pow(r, 8)*math.pow(xxi, 6)
                                                                                                                                                + 12*math.pow(a_64, 2)*(-3 + 3*math.pow(r, 2)*math.pow(xxi, 2) + 4*math.pow(r, 6)*math.pow(xxi, 6)))))/8192.
                    + (3*math.erfc((-2*a_64 + r)*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(-135 + 240*(6*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 2) - 360*math.pow(r, 2)*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 4)
                                                                                                                                + 16*(96*r*math.pow(a_64, 3) + 48*math.pow(a_64, 4) + 80*math.pow(a_64, 2)*math.pow(r, 2) + 40*a_64*math.pow(r, 3) + 5*math.pow(r, 4))*math.pow(-2*a_64 + r, 4)*math.pow(xxi, 8)))/81920.
                    + (3*math.erfc((2*a_64 + r)*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(-135 + 240*(6*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 2) - 360*math.pow(r, 2)*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 4)
                                                                                                                                + 16*(-96*r*math.pow(a_64, 3) + 48*math.pow(a_64, 4) + 80*math.pow(a_64, 2)*math.pow(r, 2) - 40*a_64*math.pow(r, 3) + 5*math.pow(r, 4))*math.pow(2*a_64 + r, 4)*math.pow(xxi, 8)))/81920.)
                   
                    
                    h2[i] =  (     (9*math.exp(-(math.pow(r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -4)*math.pow(xxi, -7)
                                    * (-45 - 78*math.pow(r, 2)*math.pow(xxi, 2) + 28*math.pow(r, 4)*math.pow(xxi, 4) + 32*math.pow(a_64, 2)*math.pow(xxi, 2)*(15 + 19*math.pow(r, 2)*math.pow(xxi, 2) + 10*math.pow(r, 4)*math.pow(xxi, 4)) - 56*math.pow(r, 6)*math.pow(xxi, 6)))/4096.
                    + (9*math.exp(-(math.pow(2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -5)*math.pow(xxi, -7)
                                      * (45*(2*a_64 + r) + 6*(-20*r*math.pow(a_64, 2) + 8*math.pow(a_64, 3) + 46*a_64*math.pow(r, 2) + 13*math.pow(r, 3))*math.pow(xxi, 2)
                    - 4*(2*a_64 + r)*(-32*r*math.pow(a_64, 3) + 16*math.pow(a_64, 4) + 48*math.pow(a_64, 2)
                                                        * math.pow(r, 2) - 56*a_64*math.pow(r, 3) + 7*math.pow(r, 4))*math.pow(xxi, 4)
                    + 8*(2*a_64 + r)*(16*math.pow(a_64, 4) + 16*math.pow(a_64, 2)*math.pow(r, 2) + 7*math.pow(r, 4))*math.pow(-2*a_64 + r, 2)*math.pow(xxi, 6)))/8192.
                    + (9*math.exp(-(math.pow(-2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -5)*math.pow(xxi, -7)
                                      * (45*(-2*a_64 + r) - 6*(20*r*math.pow(a_64, 2) + 8*math.pow(a_64, 3) + 46*a_64*math.pow(r, 2) - 13*math.pow(r, 3))*math.pow(xxi, 2)
                    + 4*(2*a_64 - r)*(32*r*math.pow(a_64, 3) + 16*math.pow(a_64, 4) + 48*math.pow(a_64, 2)
                                                        * math.pow(r, 2) + 56*a_64*math.pow(r, 3) + 7*math.pow(r, 4))*math.pow(xxi, 4)
                    - 8*(2*a_64 - r)*(16*math.pow(a_64, 4) + 16*math.pow(a_64, 2)*math.pow(r, 2) + 7*math.pow(r, 4))*math.pow(2*a_64 + r, 2)*math.pow(xxi, 6)))/8192.
                    - (9*math.erfc((-2*a_64 + r)*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(-45 + 8*math.pow(xxi, 2)*(60*math.pow(a_64, 2) - 6*math.pow(r, 2) + 9*math.pow(r, 2)*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 2)
                                                                                                                                                          + 2*(256*math.pow(a_64, 8) + 128*math.pow(a_64, 6)*math.pow(r, 2) - 40*math.pow(a_64, 2)*math.pow(r, 6) + 7*math.pow(r, 8))*math.pow(xxi, 6))))/16384.
                    - (9*math.erfc((2*a_64 + r)*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(-45 + 8*math.pow(xxi, 2)*(60*math.pow(a_64, 2) - 6*math.pow(r, 2) + 9*math.pow(r, 2)*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 2)
                                                                                                                                                          + 2*(256*math.pow(a_64, 8) + 128*math.pow(a_64, 6)*math.pow(r, 2) - 40*math.pow(a_64, 2)*math.pow(r, 6) + 7*math.pow(r, 8))*math.pow(xxi, 6))))/16384.
                    - (9*math.erfc(r*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(45 + 8*math.pow(xxi, 2)*(6*math.pow(r, 2) - 9*math.pow(r, 4)*math.pow(xxi, 2) - 14*math.pow(r, 8)*math.pow(xxi, 6)
                                                                                                                                                + 4*math.pow(a_64, 2)*(-15 - 9*math.pow(r, 2)*math.pow(xxi, 2) + 20*math.pow(r, 6)*math.pow(xxi, 6)))))/8192.)
                            
                    
                    h3[i] =   (     (9*math.exp(-(math.pow(r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -4)*math.pow(xxi, -7)
                                    * (-45 + 18*math.pow(r, 2)*math.pow(xxi, 2) - 4*math.pow(r, 4)*math.pow(xxi, 4) + 32*math.pow(a_64, 2)*math.pow(xxi, 2)*(15 + math.pow(r, 2)*math.pow(xxi, 2) - 2*math.pow(r, 4)*math.pow(xxi, 4)) + 8*math.pow(r, 6)*math.pow(xxi, 6)))/4096.
                    + (9*math.exp(-(math.pow(2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -5)*math.pow(xxi, -7)
                                      * (45*(2*a_64 + r) + 6*(2*a_64 - 3*r)*math.pow(-2*a_64 + r, 2)*math.pow(xxi, 2) - 4*math.pow(2*a_64 - r, 3)*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 4) + 8*math.pow(2*a_64 - r, 3)*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(2*a_64 + r, 2)*math.pow(xxi, 6)))/8192.
                    + (9*math.exp(-(math.pow(-2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -5)*math.pow(xxi, -7)
                                      * (45*(-2*a_64 + r) - 6*(2*a_64 + 3*r)*math.pow(2*a_64 + r, 2)*math.pow(xxi, 2) + 4*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(2*a_64 + r, 3)*math.pow(xxi, 4) - 8*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(-2*a_64 + r, 2)*math.pow(2*a_64 + r, 3)*math.pow(xxi, 6)))/8192.
                    - (9*math.erfc((-2*a_64 + r)*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(-45 + 8*math.pow(xxi, 2)
                                                                                                                                * (60*math.pow(a_64, 2) + 6*math.pow(r, 2) - 3*math.pow(r, 2)*(12*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 2) + 2*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(4*math.pow(a_64, 2) - math.pow(r, 2), 3)*math.pow(xxi, 6))))/16384.
                    - (9*math.erfc((2*a_64 + r)*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(-45 + 8*math.pow(xxi, 2)*(60*math.pow(a_64, 2) + 6*math.pow(r, 2) - 3*math.pow(r, 2)*(12*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 2)
                                                                                                                                                          + 2*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(4*math.pow(a_64, 2) - math.pow(r, 2), 3)*math.pow(xxi, 6))))/16384.
                    + (9*math.erfc(r*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(-45 + 8*math.pow(xxi, 2)*(6*math.pow(r, 2) - 3*math.pow(r, 4)*math.pow(
                                        xxi, 2) - 2*math.pow(r, 8)*math.pow(xxi, 6) + 4*math.pow(a_64, 2)*(15 - 9*math.pow(r, 2)*math.pow(xxi, 2) + 4*math.pow(r, 6)*math.pow(xxi, 6)))))/8192.)
                                    
                    
                elif(r == 2*a_64):
                    
                    Imrr[i] = -((math.pow(a_64, -5)*(3 + 16*a_64*xxi*math.pow(Pi, -0.5))*math.pow(xxi, -4))/2048. 
                    + (3*math.erfc(2*a_64*xxi)*math.pow(a_64, -5) * (-192*math.pow(a_64, 4) + math.pow(xxi, -4)))/1024.
                    + math.erfc(4*a_64*xxi)*(math.pow(a_64, -1) -
                                                  (3*math.pow(a_64, -5)*math.pow(xxi, -4))/2048.)
                    + (math.exp(-16*math.pow(a_64, 2)*math.pow(xxi, 2))*math.pow(a_64, -4)*math.pow(
                        Pi, -0.5)*math.pow(xxi, -3)*(-1 - 64*math.pow(a_64, 2)*math.pow(xxi, 2)))/256.
                    + (3*math.exp(-4*math.pow(a_64, 2)*math.pow(xxi, 2))*math.pow(a_64, -4)*math.pow(
                        Pi, -0.5)*math.pow(xxi, -3)*(1 + 24*math.pow(a_64, 2)*math.pow(xxi, 2)))/256.)
                    
                    
                    rr[i] = ((math.pow(a_64, -5)*(3 + 16*a_64*xxi*math.pow(Pi, -0.5))*math.pow(xxi, -4))/1024. 
                    + math.erfc(
                        2*a_64*xxi)*((-3*math.pow(a_64, -1))/8. - (3*math.pow(a_64, -5)*math.pow(xxi, -4))/512.)
                    + math.erfc(4*a_64*xxi)*(math.pow(a_64, -1) +
                                                  (3*math.pow(a_64, -5)*math.pow(xxi, -4))/1024.)
                    + (math.exp(-16*math.pow(a_64, 2)*math.pow(xxi, 2))*math.pow(a_64, -4)*math.pow(
                        Pi, -0.5)*math.pow(xxi, -3)*(1 - 32*math.pow(a_64, 2)*math.pow(xxi, 2)))/128.
                    + (3*math.exp(-4*math.pow(a_64, 2)*math.pow(xxi, 2))*math.pow(a_64, -4)*math.pow(
                        Pi, -0.5)*math.pow(xxi, -3)*(-1 + 8*math.pow(a_64, 2)*math.pow(xxi, 2)))/128.)
                    
                    
                    g1[i] = ( (math.exp(-(math.pow(r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -4)*math.pow(Pi, -0.5)*math.pow(r, -3) *
                                    math.pow(xxi, -5)*(9 + 15*math.pow(r, 2)*math.pow(xxi, 2) - 30*math.pow(r, 4)*math.pow(xxi, 4)))/64.
                    + (math.exp(-(math.pow(2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -4)*math.pow(Pi, -0.5)*math.pow(r, -4)*math.pow(xxi, -5)
                                      * (18*a_64 - 45*r - 3*(2*a_64 + r)*(-16*a_64*r + 8*math.pow(a_64, 2) + 25*math.pow(r, 2))*math.pow(xxi, 2) + 6*(2*a_64 + r)*(-32*r*math.pow(a_64, 3) + 32*math.pow(a_64, 4) + 44*math.pow(a_64, 2)*math.pow(r, 2) - 36*a_64*math.pow(r, 3) + 25*math.pow(r, 4))*math.pow(xxi, 4)))/640.
                    + (math.exp(-(math.pow(-2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -4)*math.pow(Pi, -0.5)*math.pow(r, -4)*math.pow(xxi, -5)
                                      * (-9*(2*a_64 + 5*r) + 3*(2*a_64 - r)*(16*a_64*r + 8*math.pow(a_64, 2) + 25*math.pow(r, 2))*math.pow(xxi, 2) - 6*(2*a_64 - r)*(32*r*math.pow(a_64, 3) + 32*math.pow(a_64, 4) + 44*math.pow(a_64, 2)*math.pow(r, 2) + 36*a_64*math.pow(r, 3) + 25*math.pow(r, 4))*math.pow(xxi, 4)))/640.
                    + (3*math.erfc(r*xxi)*math.pow(a_64, -4)*math.pow(r, -4)*math.pow(xxi, -6) *
                                      (3 + 3*math.pow(r, 2)*math.pow(xxi, 2) + 20*math.pow(r, 6)*math.pow(xxi, 6)))/128.
                    - (3*math.erfc((-2*a_64 + r)*xxi)*math.pow(a_64, -4)*math.pow(r, -4)*math.pow(xxi, -6)*(15 + 5*math.pow(r, 2)*math.pow(xxi, 2)*(3 + 64*math.pow(
                                        a_64, 4)*math.pow(xxi, 4)) + 512*math.pow(a_64, 6)*math.pow(xxi, 6) - 256*a_64*math.pow(r, 5)*math.pow(xxi, 6) + 100*math.pow(r, 6)*math.pow(xxi, 6)))/1280.
                    - (3*math.erfc((2*a_64 + r)*xxi)*math.pow(a_64, -4)*math.pow(r, -4)*math.pow(xxi, -6)*(15 + 5*math.pow(r, 2)*math.pow(xxi, 2)*(3 + 64*math.pow(a_64, 4)*math.pow(xxi, 4)) + 512*math.pow(a_64, 6)*math.pow(xxi, 6) + 256*a_64*math.pow(r, 5)*math.pow(xxi, 6)
                                                                                                                                + 100*math.pow(r, 6)*math.pow(xxi, 6)))/1280.)
                    
                    g2[i] =  (      (-3*math.exp(-(math.pow(r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -4)*math.pow(Pi, -0.5)*math.pow(r, -3)
                                    * math.pow(xxi, -5)*(3 - math.pow(r, 2)*math.pow(xxi, 2) + 2*math.pow(r, 4)*math.pow(xxi, 4)))/64.
                    + (math.exp(-(math.pow(-2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -4)*math.pow(Pi, -0.5)*math.pow(r, -4)*math.pow(xxi, -5)
                                      * (18*a_64 + 45*r - 3*(24*r*math.pow(a_64, 2) + 16*math.pow(a_64, 3) + 14*a_64*math.pow(r, 2) + 5*math.pow(r, 3))*math.pow(xxi, 2) + 6*(24*r*math.pow(a_64, 2) + 16*math.pow(a_64, 3) + 14*a_64*math.pow(r, 2) + 5*math.pow(r, 3))*math.pow(-2*a_64 + r, 2)*math.pow(xxi, 4)))/640.
                    + (math.exp(-(math.pow(2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -4)*math.pow(Pi, -0.5)*math.pow(r, -4)*math.pow(xxi, -5)
                                      * (-18*a_64 + 45*r + 3*(-24*r*math.pow(a_64, 2) + 16*math.pow(a_64, 3) + 14*a_64*math.pow(r, 2) - 5*math.pow(r, 3))*math.pow(xxi, 2) - 6*(-24*r*math.pow(a_64, 2) + 16*math.pow(a_64, 3) + 14*a_64*math.pow(r, 2) - 5*math.pow(r, 3))*math.pow(2*a_64 + r, 2)*math.pow(xxi, 4)))/640.
                    + (3*math.erfc((-2*a_64 + r)*xxi)*math.pow(a_64, -4)*math.pow(r, -4)*math.pow(xxi, -6)*(15 - 15*math.pow(r, 2)*math.pow(xxi, 2) +
                                      4*(128*math.pow(a_64, 6) - 80*math.pow(a_64, 4)*math.pow(r, 2) + 16*a_64*math.pow(r, 5) - 5*math.pow(r, 6))*math.pow(xxi, 6)))/1280.
                    + (3*math.erfc(r*xxi)*math.pow(a_64, -4)*math.pow(r, -4)*math.pow(xxi, -6) *
                                      (-3 + 3*math.pow(r, 2)*math.pow(xxi, 2) + 4*math.pow(r, 6)*math.pow(xxi, 6)))/128.
                    - (3*math.erfc((2*a_64 + r)*xxi)*math.pow(a_64, -4)*math.pow(r, -4)*math.pow(xxi, -6)*(-15 + 15*math.pow(r, 2)*math.pow(xxi, 2) +
                                      4*(-128*math.pow(a_64, 6) + 80*math.pow(a_64, 4)*math.pow(r, 2) + 16*a_64*math.pow(r, 5) + 5*math.pow(r, 6))*math.pow(xxi, 6)))/1280.)
                                    
                    
                    
                    h1[i] =   (     (3*math.exp(-(math.pow(r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -4)*math.pow(xxi, -7) *
                                    (27 - 2*math.pow(xxi, 2)*(15*math.pow(r, 2) + 2*math.pow(r, 4)*math.pow(xxi, 2) - 4*math.pow(r, 6)*math.pow(xxi, 4) + 48*math.pow(a_64, 2)*(3 - math.pow(r, 2)*math.pow(xxi, 2) + 2*math.pow(r, 4)*math.pow(xxi, 4)))))/4096.
                    + (3*math.exp(-(math.pow(-2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -5)*math.pow(xxi, -7) *
                                      (270*a_64 - 135*r + 6*(2*a_64 + 5*r)*(12*math.pow(a_64, 2) + 5*math.pow(r, 2))*math.pow(xxi, 2) - 4*(144*r*math.pow(a_64, 4) + 96*math.pow(a_64, 5) + 64*math.pow(a_64, 3)*math.pow(r, 2) - 30*a_64*math.pow(r, 4) - 5*math.pow(r, 5))*math.pow(xxi, 4)
                    + 8*math.pow(2*a_64 - r, 3)*(96*r*math.pow(a_64, 3) + 48*math.pow(a_64, 4) + 80*math.pow(a_64, 2)*math.pow(r, 2) + 40*a_64*math.pow(r, 3) + 5*math.pow(r, 4))*math.pow(xxi, 6)))/40960.
                    + (3*math.exp(-(math.pow(2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -5)*math.pow(xxi, -7) *
                                      (-135*(2*a_64 + r) - 6*(2*a_64 - 5*r)*(12*math.pow(a_64, 2) + 5*math.pow(r, 2))*math.pow(xxi, 2) + 4*(-144*r*math.pow(a_64, 4) + 96*math.pow(a_64, 5) + 64*math.pow(a_64, 3)*math.pow(r, 2) - 30*a_64*math.pow(r, 4) + 5*math.pow(r, 5))*math.pow(xxi, 4)
                                        - 8*(-96*r*math.pow(a_64, 3) + 48*math.pow(a_64, 4) + 80*math.pow(a_64, 2)*math.pow(r, 2) - 40*a_64*math.pow(r, 3) + 5*math.pow(r, 4))*math.pow(2*a_64 + r, 3)*math.pow(xxi, 6)))/40960.
                    + (3*math.erfc(r*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(27 + 8*math.pow(xxi, 2)*(-6*math.pow(r, 2) + 9*math.pow(r, 4)*math.pow(xxi, 2) - 2*math.pow(r, 8)*math.pow(xxi, 6)
                                                                                                                                                + 12*math.pow(a_64, 2)*(-3 + 3*math.pow(r, 2)*math.pow(xxi, 2) + 4*math.pow(r, 6)*math.pow(xxi, 6)))))/8192.
                    + (3*math.erfc((-2*a_64 + r)*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(-135 + 240*(6*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 2) - 360*math.pow(r, 2)*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 4)
                                                                                                                                + 16*(96*r*math.pow(a_64, 3) + 48*math.pow(a_64, 4) + 80*math.pow(a_64, 2)*math.pow(r, 2) + 40*a_64*math.pow(r, 3) + 5*math.pow(r, 4))*math.pow(-2*a_64 + r, 4)*math.pow(xxi, 8)))/81920.
                    + (3*math.erfc((2*a_64 + r)*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(-135 + 240*(6*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 2) - 360*math.pow(r, 2)*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 4)
                                                                                                                                + 16*(-96*r*math.pow(a_64, 3) + 48*math.pow(a_64, 4) + 80*math.pow(a_64, 2)*math.pow(r, 2) - 40*a_64*math.pow(r, 3) + 5*math.pow(r, 4))*math.pow(2*a_64 + r, 4)*math.pow(xxi, 8)))/81920.)
                             
                    
                    h2[i] =  (   (9*math.exp(-(math.pow(r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -4)*math.pow(xxi, -7) *
                                    (-45 - 78*math.pow(r, 2)*math.pow(xxi, 2) + 28*math.pow(r, 4)*math.pow(xxi, 4) + 32*math.pow(a_64, 2)*math.pow(xxi, 2)*(15 + 19*math.pow(r, 2)*math.pow(xxi, 2) + 10*math.pow(r, 4)*math.pow(xxi, 4)) - 56*math.pow(r, 6)*math.pow(xxi, 6)))/4096.
                    + (9*math.exp(-(math.pow(2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -5)*math.pow(xxi, -7) *
                                      (45*(2*a_64 + r) + 6*(-20*r*math.pow(a_64, 2) + 8*math.pow(a_64, 3) + 46*a_64*math.pow(r, 2) + 13*math.pow(r, 3))*math.pow(xxi, 2)
                    - 4*(2*a_64 + r)*(-32*r*math.pow(a_64, 3) + 16*math.pow(a_64, 4) + 48*math.pow(a_64, 2)
                                                      * math.pow(r, 2) - 56*a_64*math.pow(r, 3) + 7*math.pow(r, 4))*math.pow(xxi, 4)
                    + 8*(2*a_64 + r)*(16*math.pow(a_64, 4) + 16*math.pow(a_64, 2)*math.pow(r, 2) + 7*math.pow(r, 4))*math.pow(-2*a_64 + r, 2)*math.pow(xxi, 6)))/8192.
                    + (9*math.exp(-(math.pow(-2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -5)*math.pow(xxi, -7) *
                                      (45*(-2*a_64 + r) - 6*(20*r*math.pow(a_64, 2) + 8*math.pow(a_64, 3) + 46*a_64*math.pow(r, 2) - 13*math.pow(r, 3))*math.pow(xxi, 2)
                    + 4*(2*a_64 - r)*(32*r*math.pow(a_64, 3) + 16*math.pow(a_64, 4) + 48*math.pow(a_64, 2)
                                                      * math.pow(r, 2) + 56*a_64*math.pow(r, 3) + 7*math.pow(r, 4))*math.pow(xxi, 4)
                    - 8*(2*a_64 - r)*(16*math.pow(a_64, 4) + 16*math.pow(a_64, 2)*math.pow(r, 2) + 7*math.pow(r, 4))*math.pow(2*a_64 + r, 2)*math.pow(xxi, 6)))/8192.
                    - (9*math.erfc((-2*a_64 + r)*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(-45 + 8*math.pow(xxi, 2)*(60*math.pow(a_64, 2) - 6*math.pow(r, 2) + 9*math.pow(r, 2)*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 2)
                                                                                                                                                          + 2*(256*math.pow(a_64, 8) + 128*math.pow(a_64, 6)*math.pow(r, 2) - 40*math.pow(a_64, 2)*math.pow(r, 6) + 7*math.pow(r, 8))*math.pow(xxi, 6))))/16384.
                    - (9*math.erfc((2*a_64 + r)*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(-45 + 8*math.pow(xxi, 2)*(60*math.pow(a_64, 2) - 6*math.pow(r, 2) + 9*math.pow(r, 2)*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 2)
                                                                                                                                                          + 2*(256*math.pow(a_64, 8) + 128*math.pow(a_64, 6)*math.pow(r, 2) - 40*math.pow(a_64, 2)*math.pow(r, 6) + 7*math.pow(r, 8))*math.pow(xxi, 6))))/16384.
                    - (9*math.erfc(r*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(45 + 8*math.pow(xxi, 2)*(6*math.pow(r, 2) - 9*math.pow(r, 4)*math.pow(xxi, 2) - 14*math.pow(r, 8)*math.pow(xxi, 6)
                                                                                                                                                + 4*math.pow(a_64, 2)*(-15 - 9*math.pow(r, 2)*math.pow(xxi, 2) + 20*math.pow(r, 6)*math.pow(xxi, 6)))))/8192.)
                                    
                    
                    h3[i] =   (      (9*math.exp(-(math.pow(r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -4)*math.pow(xxi, -7) *
                                    (-45 + 18*math.pow(r, 2)*math.pow(xxi, 2) - 4*math.pow(r, 4)*math.pow(xxi, 4) + 32*math.pow(a_64, 2)*math.pow(xxi, 2)*(15 + math.pow(r, 2)*math.pow(xxi, 2) - 2*math.pow(r, 4)*math.pow(xxi, 4)) + 8*math.pow(r, 6)*math.pow(xxi, 6)))/4096.
                    + (9*math.exp(-(math.pow(2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -5)*math.pow(xxi, -7) *
                                      (45*(2*a_64 + r) + 6*(2*a_64 - 3*r)*math.pow(-2*a_64 + r, 2)*math.pow(xxi, 2) - 4*math.pow(2*a_64 - r, 3)*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 4) + 8*math.pow(2*a_64 - r, 3)*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(2*a_64 + r, 2)*math.pow(xxi, 6)))/8192.
                    + (9*math.exp(-(math.pow(-2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -5)*math.pow(xxi, -7) *
                                      (45*(-2*a_64 + r) - 6*(2*a_64 + 3*r)*math.pow(2*a_64 + r, 2)*math.pow(xxi, 2) + 4*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(2*a_64 + r, 3)*math.pow(xxi, 4) - 8*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(-2*a_64 + r, 2)*math.pow(2*a_64 + r, 3)*math.pow(xxi, 6)))/8192.
                    - (9*math.erfc((-2*a_64 + r)*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(-45 + 8*math.pow(xxi, 2) *
                                                                                                                                (60*math.pow(a_64, 2) + 6*math.pow(r, 2) - 3*math.pow(r, 2)*(12*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 2) + 2*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(4*math.pow(a_64, 2) - math.pow(r, 2), 3)*math.pow(xxi, 6))))/16384.
                    - (9*math.erfc((2*a_64 + r)*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(-45 + 8*math.pow(xxi, 2)*(60*math.pow(a_64, 2) + 6*math.pow(r, 2) - 3*math.pow(r, 2)*(12*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 2)
                                                                                                                                                          + 2*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(4*math.pow(a_64, 2) - math.pow(r, 2), 3)*math.pow(xxi, 6))))/16384.
                    + (9*math.erfc(r*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(-45 + 8*math.pow(xxi, 2)*(6*math.pow(r, 2) - 3*math.pow(r, 4)*math.pow(
                                        xxi, 2) - 2*math.pow(r, 8)*math.pow(xxi, 6) + 4*math.pow(a_64, 2)*(15 - 9*math.pow(r, 2)*math.pow(xxi, 2) + 4*math.pow(r, 6)*math.pow(xxi, 6)))))/8192.)
                                    
                    
                elif(r < 2*a_64):
                    
                    Imrr[i] = ( (-9*r*math.pow(a_64, -2))/32 
                    + math.pow(a_64, -1) 
                    - (math.pow(a_64, 2)*math.pow(r, -3)) /2 
                    - (3*math.pow(r, -1))/4 
                    + (3*math.erfc(r*xxi)*math.pow(a_64, -2)*math.pow(r, -3)* (-12*math.pow(r, 4)
                    + math.pow(xxi, -4)))/128
                    + (math.erfc((-2*a_64 + r)*xxi)*(-128*math.pow(a_64, -1) + 64*math.pow(a_64, 2)*math.pow(r, - 3) + 96*math.pow(r, -1) + math.pow(a_64, -2)*(36*r - 3*math.pow(r, -3)*math.pow(xxi, -4))))/256
                    + (math.erfc((2*a_64 + r)*xxi)*(128*math.pow(a_64, -1) + 64*math.pow(a_64, 2)*math.pow(r, -3) + 96*math.pow(r, -1) + math.pow(a_64, -2)*(36*r - 3*math.pow(r, -3)*math.pow(xxi, -4))))/256
                    + (3*math.exp(-(math.pow(r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -2)*math.pow(Pi, -0.5) * math.pow(r, -2)*math.pow(xxi, -3)*(1 + 6*math.pow(r, 2)*math.pow(xxi, 2)))/64
                    + (math.exp(-(math.pow(2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -2)*math.pow(Pi, -0.5)*math.pow(r, -3)*math.pow(xxi, -3) * (8*r*math.pow(a_64, 2)*math.pow(xxi, 2) - 16*math.pow(a_64, 3)*math.pow(xxi, 2) + a_64*(2 - 28*math.pow(r, 2)*math.pow(xxi, 2)) - 3*(r + 6*math.pow(r, 3)*math.pow(xxi, 2))))/128
                    + (math.exp(-(math.pow(-2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -2)*math.pow(Pi, -0.5)*math.pow(r, -3)*math.pow(xxi, -3) * (8*r*math.pow(a_64, 2)*math.pow(xxi, 2) + 16*math.pow(a_64, 3)*math.pow(xxi, 2) + a_64*(-2 + 28*math.pow(r, 2)*math.pow(xxi, 2)) - 3*(r + 6*math.pow(r, 3)*math.pow(xxi, 2))))/128 )
                    

                    rr[i] = (((2*a_64 + 3*r)*math.pow(a_64, -2) *
                    math.pow(2*a_64 - r, 3)*math.pow(r, -3))/16.
                    + (math.erfc((-2*a_64 + r)*xxi)*(-64*math.pow(a_64, -1) - 64*math.pow(a_64, 2)*math.pow(r, -
                      3) + 96*math.pow(r, -1) + math.pow(a_64, -2)*(12*r + 3*math.pow(r, -3)*math.pow(xxi, -4))))/128.
                    + (math.erfc((2*a_64 + r)*xxi)*(64*math.pow(a_64, -1) - 64*math.pow(a_64, 2)*math.pow(r, -3) +
                      96*math.pow(r, -1) + math.pow(a_64, -2)*(12*r + 3*math.pow(r, -3)*math.pow(xxi, -4))))/128.
                    + (3*math.exp(-(math.pow(r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -2)*math.pow(Pi, -0.5)
                      * math.pow(r, -2)*math.pow(xxi, -3)*(-1 + 2*math.pow(r, 2)*math.pow(xxi, 2)))/32.
                    - ((2*a_64 + 3*r)*math.exp(-(math.pow(-2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -2)*math.pow(Pi, -0.5)*math.pow(r, -3)*math.pow(xxi, -3) *
                        (-1 - 8*a_64*r*math.pow(xxi, 2) + 8*math.pow(a_64, 2)*math.pow(xxi, 2) + 2*math.pow(r, 2)*math.pow(xxi, 2)))/64.
                    + ((2*a_64 - 3*r)*math.exp(-(math.pow(2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -2)*math.pow(Pi, -0.5)*math.pow(r, -3)*math.pow(xxi, -3) *
                        (-1 + 8*a_64*r*math.pow(xxi, 2) + 8*math.pow(a_64, 2)*math.pow(xxi, 2) + 2*math.pow(r, 2)*math.pow(xxi, 2)))/64.
                    - (3*math.erfc(r*xxi)*math.pow(a_64, -2)*math.pow(r, -3)
                      * math.pow(xxi, -4)*(1 + 4*math.pow(r, 4)*math.pow(xxi, 4)))/64.)

                    g1[i] =     ((-9*math.pow(a_64, -4)*math.pow(r, -4)*math.pow(xxi, -6))/128. 
                    - (9*math.pow(a_64, -4)*math.pow(r, -2)*math.pow(xxi, -4))/128.
                    + (math.exp(-(math.pow(r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -4)*math.pow(Pi, -0.5)*math.pow(r, -3)
                                  * math.pow(xxi, -5)*(9 + 15*math.pow(r, 2)*math.pow(xxi, 2) - 30*math.pow(r, 4)*math.pow(xxi, 4)))/64.
                    + (math.exp(-(math.pow(2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -4)*math.pow(Pi, -0.5)*math.pow(r, -4)*math.pow(xxi, -5) *
                                    (18*a_64 - 45*r - 3*(2*a_64 + r)*(-16*a_64*r + 8*math.pow(a_64, 2) + 25*math.pow(r, 2))*math.pow(xxi, 2) + 6*(2*a_64 + r)*(-32*r*math.pow(a_64, 3) + 32*math.pow(a_64, 4) + 44*math.pow(a_64, 2)*math.pow(r, 2) - 36*a_64*math.pow(r, 3) + 25*math.pow(r, 4))*math.pow(xxi, 4)))/640.
                    + (math.exp(-(math.pow(-2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -4)*math.pow(Pi, -0.5)*math.pow(r, -4)*math.pow(xxi, -5) *
                                  (-9*(2*a_64 + 5*r) + 3*(2*a_64 - r)*(16*a_64*r + 8*math.pow(a_64, 2) + 25*math.pow(r, 2))*math.pow(xxi, 2) - 6*(2*a_64 - r)*(32*r*math.pow(a_64, 3) + 32*math.pow(a_64, 4) + 44*math.pow(a_64, 2)*math.pow(r, 2) + 36*a_64*math.pow(r, 3) + 25*math.pow(r, 4))*math.pow(xxi, 4)))/640.
                    + (3*math.erfc(r*xxi)*math.pow(a_64, -4)*math.pow(r, -4)*math.pow(xxi, -6) *
                                  (3 + 3*math.pow(r, 2)*math.pow(xxi, 2) + 20*math.pow(r, 6)*math.pow(xxi, 6)))/128.
                    + (3*math.erfc(2*a_64*xxi - r*xxi)*math.pow(a_64, -4)*math.pow(r, -4)*math.pow(xxi, -6)*(15 + 5*math.pow(r, 2)*math.pow(xxi, 2)*(3 + 64*math.pow(
                                    a_64, 4)*math.pow(xxi, 4)) + 512*math.pow(a_64, 6)*math.pow(xxi, 6) - 256*a_64*math.pow(r, 5)*math.pow(xxi, 6) + 100*math.pow(r, 6)*math.pow(xxi, 6)))/1280.
                    - (3*math.erfc((2*a_64 + r)*xxi)*math.pow(a_64, -4)*math.pow(r, -4)*math.pow(xxi, -6)*(15 + 5*math.pow(r, 2)*math.pow(xxi, 2)*(3 + 64*math.pow(a_64, 4)*math.pow(xxi, 4)) + 512*math.pow(a_64, 6)*math.pow(xxi, 6) + 256*a_64*math.pow(r, 5)*math.pow(xxi, 6)
                                                                                                                            + 100*math.pow(r, 6)*math.pow(xxi, 6)))/1280.)
        
                    g2[i] =   ((-3*r*math.pow(a_64, -3))/10. - (12*math.pow(a_64, 2)*math.pow(r, -4)) /5. 
                    + (3*math.pow(r, -2))/2. + (3*math.pow(a_64, -4)*math.pow(r, 2))/32.
                    - (3*math.exp(-(math.pow(r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -4)*math.pow(Pi, -0.5)*math.pow(r, -3)
                                      * math.pow(xxi, -5)*(3 - math.pow(r, 2)*math.pow(xxi, 2) + 2*math.pow(r, 4)*math.pow(xxi, 4)))/64.
                    + (math.exp(-(math.pow(-2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -4)*math.pow(Pi, -0.5)*math.pow(r, -4)*math.pow(xxi, -5) *
                                        (18*a_64 + 45*r - 3*(24*r*math.pow(a_64, 2) + 16*math.pow(a_64, 3) + 14*a_64*math.pow(r, 2) + 5*math.pow(r, 3))*math.pow(xxi, 2) + 6*(24*r*math.pow(a_64, 2) + 16*math.pow(a_64, 3) + 14*a_64*math.pow(r, 2) + 5*math.pow(r, 3))*math.pow(-2*a_64 + r, 2)*math.pow(xxi, 4)))/640.
                    + (math.exp(-(math.pow(2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -4)*math.pow(Pi, -0.5)*math.pow(r, -4)*math.pow(xxi, -5) *
                                        (-18*a_64 + 45*r + 3*(-24*r*math.pow(a_64, 2) + 16*math.pow(a_64, 3) + 14*a_64*math.pow(r, 2) - 5*math.pow(r, 3))*math.pow(xxi, 2) - 6*(-24*r*math.pow(a_64, 2) + 16*math.pow(a_64, 3) + 14*a_64*math.pow(r, 2) - 5*math.pow(r, 3))*math.pow(2*a_64 + r, 2)*math.pow(xxi, 4)))/640.
                    + (3*math.erfc((-2*a_64 + r)*xxi)*math.pow(a_64, -4)*math.pow(r, -4)*math.pow(xxi, -6)*(15 - 15*math.pow(r, 2)*math.pow(xxi, 2) +
                                      4*(128*math.pow(a_64, 6) - 80*math.pow(a_64, 4)*math.pow(r, 2) + 16*a_64*math.pow(r, 5) - 5*math.pow(r, 6))*math.pow(xxi, 6)))/1280.
                    + (3*math.erfc(r*xxi)*math.pow(a_64, -4)*math.pow(r, -4)*math.pow(xxi, -6) *
                                      (-3 + 3*math.pow(r, 2)*math.pow(xxi, 2) + 4*math.pow(r, 6)*math.pow(xxi, 6)))/128.
                    - (3*math.erfc((2*a_64 + r)*xxi)*math.pow(a_64, -4)*math.pow(r, -4)*math.pow(xxi, -6)*(-15 + 15*math.pow(r, 2)*math.pow(xxi, 2) +
                                      4*(-128*math.pow(a_64, 6) + 80*math.pow(a_64, 4)*math.pow(r, 2) + 16*a_64*math.pow(r, 5) + 5*math.pow(r, 6))*math.pow(xxi, 6)))/1280.)
                                    
                    
                    h1[i] =        ( (9*r*math.pow(a_64, -4))/64. - (3*math.pow(a_64, -3))/10. - (9*math.pow(a_64, 2)*math.pow(
                                        r, -5))/10. + (3*math.pow(r, -3))/4. - (3*math.pow(a_64, -6)*math.pow(r, 3))/512.
                    + (3*math.exp(-(math.pow(r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -4)*math.pow(xxi, -7) *
                                        (27 - 2*math.pow(xxi, 2)*(15*math.pow(r, 2) + 2*math.pow(r, 4)*math.pow(xxi, 2) - 4*math.pow(r, 6)*math.pow(xxi, 4) + 48*math.pow(a_64, 2)*(3 - math.pow(r, 2)*math.pow(xxi, 2) + 2*math.pow(r, 4)*math.pow(xxi, 4)))))/4096.
                    + (3*math.exp(-(math.pow(-2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -5)*math.pow(xxi, -7) *
                                        (270*a_64 - 135*r + 6*(2*a_64 + 5*r)*(12*math.pow(a_64, 2) + 5*math.pow(r, 2))*math.pow(xxi, 2) - 4*(144*r*math.pow(a_64, 4) + 96*math.pow(a_64, 5) + 64*math.pow(a_64, 3)*math.pow(r, 2) - 30*a_64*math.pow(r, 4) - 5*math.pow(r, 5))*math.pow(xxi, 4)
                    + 8*math.pow(2*a_64 - r, 3)*(96*r*math.pow(a_64, 3) + 48*math.pow(a_64, 4) + 80*math.pow(a_64, 2)*math.pow(r, 2) + 40*a_64*math.pow(r, 3) + 5*math.pow(r, 4))*math.pow(xxi, 6)))/40960.
                    + (3*math.exp(-(math.pow(2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -5)*math.pow(xxi, -7) *
                                        (-135*(2*a_64 + r) - 6*(2*a_64 - 5*r)*(12*math.pow(a_64, 2) + 5*math.pow(r, 2))*math.pow(xxi, 2) + 4*(-144*r*math.pow(a_64, 4) + 96*math.pow(a_64, 5) + 64*math.pow(a_64, 3)*math.pow(r, 2) - 30*a_64*math.pow(r, 4) + 5*math.pow(r, 5))*math.pow(xxi, 4)
                                        - 8*(-96*r*math.pow(a_64, 3) + 48*math.pow(a_64, 4) + 80*math.pow(a_64, 2)*math.pow(r, 2) - 40*a_64*math.pow(r, 3) + 5*math.pow(r, 4))*math.pow(2*a_64 + r, 3)*math.pow(xxi, 6)))/40960.
                    + (3*math.erfc(r*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(27 + 8*math.pow(xxi, 2)*(-6*math.pow(r, 2) + 9*math.pow(r, 4)*math.pow(xxi, 2) - 2*math.pow(r, 8)*math.pow(xxi, 6)
                                                                                                                                                + 12*math.pow(a_64, 2)*(-3 + 3*math.pow(r, 2)*math.pow(xxi, 2) + 4*math.pow(r, 6)*math.pow(xxi, 6)))))/8192.
                    + (3*math.erfc((-2*a_64 + r)*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(-135 + 240*(6*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 2) - 360*math.pow(r, 2)*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 4)
                                                                                                                                + 16*(96*r*math.pow(a_64, 3) + 48*math.pow(a_64, 4) + 80*math.pow(a_64, 2)*math.pow(r, 2) + 40*a_64*math.pow(r, 3) + 5*math.pow(r, 4))*math.pow(-2*a_64 + r, 4)*math.pow(xxi, 8)))/81920.
                    + (3*math.erfc((2*a_64 + r)*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(-135 + 240*(6*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 2) - 360*math.pow(r, 2)*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 4)
                                                                                                                                + 16*(-96*r*math.pow(a_64, 3) + 48*math.pow(a_64, 4) + 80*math.pow(a_64, 2)*math.pow(r, 2) - 40*a_64*math.pow(r, 3) + 5*math.pow(r, 4))*math.pow(2*a_64 + r, 4)*math.pow(xxi, 8)))/81920.)
                                    
                    
                    h2[i] =    (   (63*r*math.pow(a_64, -4))/64. - (3*math.pow(a_64, -3))/2. + (9*math.pow(a_64, 2)*math.pow(r, -5))/2. - (3*math.pow(r, -3)
                                                                                                                                      )/4. - (33*math.pow(a_64, -6)*math.pow(r, 3))/512. + (9*math.pow(a_64, -6)*math.pow(r, -3)*math.pow(xxi, -6))/128.
                    - (27*math.pow(a_64, -4)*math.pow(r, -3)*math.pow(xxi, -4))/64. + (9*math.exp(-(math.pow(r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -4)*math.pow(xxi, -7) *
                                                                                                      (-45 - 78*math.pow(r, 2)*math.pow(xxi, 2) + 28*math.pow(r, 4)*math.pow(xxi, 4) + 32*math.pow(a_64, 2)*math.pow(xxi, 2)*(15 + 19*math.pow(r, 2)*math.pow(xxi, 2) + 10*math.pow(r, 4)*math.pow(xxi, 4)) - 56*math.pow(r, 6)*math.pow(xxi, 6)))/4096.
                    + (3*math.erfc(2*a_64*xxi - r*xxi)*math.pow(a_64, -6)*math.pow(r, -3)*math.pow(xxi, -6)*(-3 + 18*math.pow(a_64, 2)*math.pow(xxi, 2)*(1 - 4*math.pow(r, 4)*math.pow(xxi, 4)) + 128*math.pow(a_64, 6)*math.pow(xxi, 6) + 64*math.pow(a_64, 3)*math.pow(r, 3)*math.pow(xxi, 6)
                                                                                                                                + 8*math.pow(r, 6)*math.pow(xxi, 6)))/256. + (9*math.exp(-(math.pow(2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -5)*math.pow(xxi, -7) *
                                                                                                                                                                                (45*(2*a_64 + r) + 6*(-20*r*math.pow(a_64, 2) + 8*math.pow(a_64, 3) + 46*a_64*math.pow(r, 2) + 13*math.pow(r, 3))*math.pow(xxi, 2)
                                                                                                                                                                                - 4*(2*a_64 + r)*(-32*r*math.pow(a_64, 3) + 16*math.pow(a_64, 4) + 48*math.pow(
                                                                                                                                                                                    a_64, 2)*math.pow(r, 2) - 56*a_64*math.pow(r, 3) + 7*math.pow(r, 4))*math.pow(xxi, 4)
                                                                                                                                                                                + 8*(2*a_64 + r)*(16*math.pow(a_64, 4) + 16*math.pow(a_64, 2)*math.pow(r, 2) + 7*math.pow(r, 4))*math.pow(-2*a_64 + r, 2)*math.pow(xxi, 6)))/8192.
                    + (9*math.exp(-(math.pow(-2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -5)*math.pow(xxi, -7) *
                                        (45*(-2*a_64 + r) - 6*(20*r*math.pow(a_64, 2) + 8*math.pow(a_64, 3) + 46*a_64*math.pow(r, 2) - 13*math.pow(r, 3))*math.pow(xxi, 2)
                    + 4*(2*a_64 - r)*(32*r*math.pow(a_64, 3) + 16*math.pow(a_64, 4) + 48*math.pow(a_64, 2)
                                                        * math.pow(r, 2) + 56*a_64*math.pow(r, 3) + 7*math.pow(r, 4))*math.pow(xxi, 4)
                                        - 8*(2*a_64 - r)*(16*math.pow(a_64, 4) + 16*math.pow(a_64, 2)*math.pow(r, 2) + 7*math.pow(r, 4))*math.pow(2*a_64 + r, 2)*math.pow(xxi, 6)))/8192.
                    - (9*math.erfc((2*a_64 + r)*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(-45 + 8*math.pow(xxi, 2)*(60*math.pow(a_64, 2) - 6*math.pow(r, 2) + 9*math.pow(r, 2)*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 2)
                                                                                                                                                          + 2*(256*math.pow(a_64, 8) + 128*math.pow(a_64, 6)*math.pow(r, 2) - 40*math.pow(a_64, 2)*math.pow(r, 6) + 7*math.pow(r, 8))*math.pow(xxi, 6))))/16384.
                    + (3*math.erfc((-2*a_64 + r)*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(135 + 8*math.pow(xxi, 2)*(-6*(30*math.pow(a_64, 2) + math.pow(r, 2)) + 9*(4*math.pow(a_64, 2) - 3*math.pow(r, 2))*math.pow(r, 2)*math.pow(xxi, 2)
                                                                                                                                                          + 2*(-768*math.pow(a_64, 8) + 128*math.pow(a_64, 6)*math.pow(r, 2) + 256*math.pow(a_64, 3)*math.pow(r, 5) - 168*math.pow(a_64, 2)*math.pow(r, 6) + 11*math.pow(r, 8))*math.pow(xxi, 6))))/16384.
                    - (9*math.erfc(r*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(45 + 8*math.pow(xxi, 2)*(6*math.pow(r, 2) - 9*math.pow(r, 4)*math.pow(xxi, 2) - 14*math.pow(r, 8)*math.pow(xxi, 6)
                                                                                                                                                + 4*math.pow(a_64, 2)*(-15 - 9*math.pow(r, 2)*math.pow(xxi, 2) + 20*math.pow(r, 6)*math.pow(xxi, 6)))))/8192.)
                                    

                    h3[i] =       (  (9*r*math.pow(a_64, -4))/64. + (9*math.pow(a_64, 2)*math.pow(r, -5))/2. 
                    - (9*math.pow(r, -3))/4. - (9*math.pow(a_64, -6)*math.pow(r, 3))/512.
                    + (9*math.exp(-(math.pow(r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -4)*math.pow(xxi, -7) *
                                        (-45 + 18*math.pow(r, 2)*math.pow(xxi, 2) - 4*math.pow(r, 4)*math.pow(xxi, 4) + 32*math.pow(a_64, 2)*math.pow(xxi, 2)*(15 + math.pow(r, 2)*math.pow(xxi, 2) - 2*math.pow(r, 4)*math.pow(xxi, 4)) + 8*math.pow(r, 6)*math.pow(xxi, 6)))/4096.
                    + (9*math.exp(-(math.pow(2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -5)*math.pow(xxi, -7) *
                                        (45*(2*a_64 + r) + 6*(2*a_64 - 3*r)*math.pow(-2*a_64 + r, 2)*math.pow(xxi, 2) - 4*math.pow(2*a_64 - r, 3)*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 4) + 8*math.pow(2*a_64 - r, 3)*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(2*a_64 + r, 2)*math.pow(xxi, 6)))/8192.
                    + (9*math.exp(-(math.pow(-2*a_64 + r, 2)*math.pow(xxi, 2)))*math.pow(a_64, -6)*math.pow(Pi, -0.5)*math.pow(r, -5)*math.pow(xxi, -7) *
                                        (45*(-2*a_64 + r) - 6*(2*a_64 + 3*r)*math.pow(2*a_64 + r, 2)*math.pow(xxi, 2) + 4*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(2*a_64 + r, 3)*math.pow(xxi, 4) - 8*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(-2*a_64 + r, 2)*math.pow(2*a_64 + r, 3)*math.pow(xxi, 6)))/8192.
                    - (9*math.erfc((-2*a_64 + r)*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(-45 + 8*math.pow(xxi, 2) *
                                                                                                                                (60*math.pow(a_64, 2) + 6*math.pow(r, 2) - 3*math.pow(r, 2)*(12*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 2) + 2*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(4*math.pow(a_64, 2) - math.pow(r, 2), 3)*math.pow(xxi, 6))))/16384.
                    - (9*math.erfc((2*a_64 + r)*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(-45 + 8*math.pow(xxi, 2)*(60*math.pow(a_64, 2) + 6*math.pow(r, 2) - 3*math.pow(r, 2)*(12*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(xxi, 2)
                                                                                                                                                          + 2*(4*math.pow(a_64, 2) + math.pow(r, 2))*math.pow(4*math.pow(a_64, 2) - math.pow(r, 2), 3)*math.pow(xxi, 6))))/16384.
                    + (9*math.erfc(r*xxi)*math.pow(a_64, -6)*math.pow(r, -5)*math.pow(xxi, -8)*(-45 + 8*math.pow(xxi, 2)*(6*math.pow(r, 2) - 3*math.pow(r, 4)*math.pow(
                                        xxi, 2) - 2*math.pow(r, 8)*math.pow(xxi, 6) + 4*math.pow(a_64, 2)*(15 - 9*math.pow(r, 2)*math.pow(xxi, 2) + 4*math.pow(r, 6)*math.pow(xxi, 6)))))/8192.)
                                   

           
      

            return Imrr, rr, g1, g2, h1, h2, h3
    
    


    def compute_k_gridpoint_number(kmax):
        # Set initial number of grid points in wave space
        Nx = int(kmax * Lx / (2.0 * 3.1415926536) * 2.0) + 1
        Ny = int(kmax * Ly / (2.0 * 3.1415926536) * 2.0) + 1
        Nz = int(kmax * Lz / (2.0 * 3.1415926536) * 2.0) + 1

        # Get list of int values between 8 and 512 that can be written as (2^a)*(3^b)*(5^c)
        # Then sort list from low to high and figure out how many entries there are
        Mlist = []
        for ii in range(0, 10):
            pow2 = int(1)
            for i in range(0, ii):
                pow2 *= 2
            for jj in range(0, 6):
                pow3 = int(1)
                for j in range(0, jj):
                    pow3 *= 3
                for kk in range(0, 4):
                    pow5 = 1
                    for k in range(0, kk):
                        pow5 *= 5
                    Mcurr = int(pow2 * pow3 * pow5)
                    if ((Mcurr >= 8) and (Mcurr <= 512)):
                        Mlist.append(Mcurr)
        Mlist = jnp.array(Mlist)
        Mlist = jnp.sort(Mlist)
        nmult = len(Mlist)

        # Compute the number of grid points in each direction (should be a power of 2,3,5 for most efficient FFTs)
        for ii in range(0, nmult):
            if(Nx <= Mlist[ii]):
                Nx = Mlist[ii]
                break
        for ii in range(0, nmult):
            if(Ny <= Mlist[ii]):
                Ny = Mlist[ii]
                break
        for ii in range(0, nmult):
            if(Nz <= Mlist[ii]):
                Nz = Mlist[ii]
                break

        # Maximum number of FFT nodes is limited by available memory = 512 * 512 * 512 = 134217728
        if (Nx*Ny*Nz > 134217728):
            sys.exit("Requested Number of Fourier Nodes Exceeds Max Dimension of 512^3")
        return Nx, Ny, Nz
    
    # Check that ewald_cut is small enough to avoid interaction with the particles images (in real space part of calculation)
    def check_ewald_cut(ewald_cut):
        if ((ewald_cut > Lx/2) or (ewald_cut > Ly/2) or (ewald_cut > Lz/2)):
            print(
                'WARNING: Real space Ewald cutoff radius is too large! Increase xi and retry.')
            max_cut = max([Lx, Ly, Lz]) / 2.0
            new_xi = jnp.sqrt(-jnp.log(error)) / max_cut
            print('Try with ,', new_xi)
            sys.exit("Exit the program!!")
        return
    
    # Maximum eigenvalue of A'*A to scale support, P, for spreading on deformed grids (Fiore and Swan, J. Chem. Phys., 2018)
    def check_max_shear(gridh, xisq, Nx, Ny, Nz):

        gamma = max_strain
        lambdaa = 1 + gamma*gamma / 2 + gamma * jnp.sqrt(1+gamma*gamma/4)

        # Parameters for the Spectral Ewald Method (Lindbo and Tornberg, J. Comp. Phys., 2011)
        gaussm = 1.0
        while (math.erfc(gaussm / jnp.sqrt(2*lambdaa)) > error):
            gaussm += 0.01
        gaussP = int(gaussm*gaussm / 3.1415926536) + 1
        w = gaussP*gridh[0] / 2.0  # Gaussian width in simulation units
        eta = (2.0*w/gaussm)*(2.0*w/gaussm) * \
            (xisq)  # Gaussian splitting parameter
        
        # Check that the support size isn't somehow larger than the grid
        if (gaussP > min(Nx, min(Ny, Nz))):
            print("Quadrature Support Exceeds Available Grid")
            print("(Mx, My, Mz) = (", Nx, '), (', Ny, '), (', Nz, ')')
            print("Support Size, P =", gaussP)
            sys.exit()
        return eta, gaussP
    
    @partial(jit, static_argnums=[0,1,2])
    def compute_sheared_grid(Nx,Nz,Ny,tilt_factor):
        gridk = jnp.zeros((Nx*Ny*Nz, 4),float) 
        # Here create arrays that store the indices that we would have if this function was not vectorized (using for loop instead)
        Nxx = jnp.repeat(jnp.repeat(jnp.arange(Nz), Ny), Nx)
        Nzz = jnp.resize(jnp.arange(Nx), Nx*Ny*Nz)
        Nyy = jnp.resize(jnp.repeat(jnp.arange(Nz), Ny), Nx*Ny*Nz)


        gridk_x = jnp.where(Nxx < (Nx+1)/2, Nxx, (Nxx-Nx))
        gridk_y = (jnp.where(Nyy < (Ny+1)/2, Nyy, (Nyy - Ny)) - tilt_factor * gridk_x *Ly/Lx) / Ly
        gridk_x = gridk_x/Lx 
        gridk_z = jnp.where(Nzz < (Nz+1)/2, Nzz,(Nzz-Nz)) / Lz
        gridk_x *= 2.0*3.1416926536
        gridk_y *= 2.0*3.1416926536
        gridk_z *= 2.0*3.1416926536

        
                
        # k dot k and fourth component (contains the scaling factor of the FFT)
        k_sq = gridk_x*gridk_x + gridk_y*gridk_y + gridk_z*gridk_z
        gridk_w = jnp.where(k_sq > 0, 6.0*3.1415926536 * (1.0 + k_sq/4.0/xisq)
                            * jnp.exp(-(1-eta) * k_sq/4.0/xisq) / (k_sq) / (Nx*Ny*Nz), 0)

        # store the results
        gridk = gridk.at[:, 0].set(gridk_x)
        gridk = gridk.at[:, 1].set(gridk_y)
        gridk = gridk.at[:, 2].set(gridk_z)
        gridk = gridk.at[:, 3].set(gridk_w)
        
        #Reshape to match the gridded quantities
        gridk = jnp.reshape(gridk, (Nx, Ny, Nz, 4))
        gridk = jnp.array(gridk)
        
        return gridk
    
    
    @jit
    def RFU_Precondition(
        ichol_relaxer,
        displacements_vector_matrix
        ):
        #Define empty matrix
        R_fu_precondition = jnp.zeros((6*N, 6*N),float)
             
        # By definition, R points from particle 1 to particle 2 (i to j), otherwise the handedness and symmetry of the lubrication functions is lost
        R = displacements_vector_matrix.at[nl_lub_prec[0,:],nl_lub_prec[1,:]].get() #array of vectors from particle i to j (size = npairs)
        dist = space.distance(R)  # distance between particle i and j
        r = R / dist[:, None]  # unit vector from particle j to i
        
        # # Indices in resistance table
        ind = (jnp.log10((dist - 2.0) / ResTable_min) / ResTable_dr)
        ind = ind.astype(int)
        
        dist_lower = ResTable_dist[ind]
        dist_upper = ResTable_dist[ind+1]
        
        # # Linear interpolation of the Table values
        fac = (dist - dist_lower) / (dist_upper - dist_lower)
        
        XA11 = jnp.where(dist > 0, ResTable_vals[0], 0.)
        XA11 = jnp.where(dist >= 2+ResTable_min, (ResTable_vals[22*(ind)+0] + (
            ResTable_vals[22*(ind+1)+0]-ResTable_vals[22*(ind)+0]) * fac), XA11)
        
        XA12 = jnp.where(dist > 0, ResTable_vals[1], 0.)
        XA12 = jnp.where(dist >= 2+ResTable_min, (ResTable_vals[22*(ind)+1] + (
            ResTable_vals[22*(ind+1)+1]-ResTable_vals[22*(ind)+1]) * fac), XA12)

        YA11 = jnp.where(dist > 0, ResTable_vals[2], 0.)
        YA11 = jnp.where(dist >= 2+ResTable_min, (ResTable_vals[22*(ind)+2] + (
            ResTable_vals[22*(ind+1)+2]-ResTable_vals[22*(ind)+2]) * fac), YA11)

        YA12 = jnp.where(dist > 0, ResTable_vals[3], 0.)
        YA12 = jnp.where(dist >= 2+ResTable_min, (ResTable_vals[22*(ind)+3] + (
            ResTable_vals[22*(ind+1)+3]-ResTable_vals[22*(ind)+3]) * fac), YA12)

        YB11 = jnp.where(dist > 0, ResTable_vals[4], 0.)
        YB11 = jnp.where(dist >= 2+ResTable_min, (ResTable_vals[22*(ind)+4] + (
            ResTable_vals[22*(ind+1)+4]-ResTable_vals[22*(ind)+4]) * fac), YB11)

        YB12 = jnp.where(dist > 0, ResTable_vals[5], 0.)
        YB12 = jnp.where(dist >= 2+ResTable_min, (ResTable_vals[22*(ind)+5] + (
            ResTable_vals[22*(ind+1)+5]-ResTable_vals[22*(ind)+5]) * fac), YB12)

        XC11 = jnp.where(dist > 0, ResTable_vals[6], 0.)
        XC11 = jnp.where(dist >= 2+ResTable_min, (ResTable_vals[22*(ind)+6] + (
            ResTable_vals[22*(ind+1)+6]-ResTable_vals[22*(ind)+6]) * fac), XC11)

        XC12 = jnp.where(dist > 0, ResTable_vals[7], 0.)
        XC12 = jnp.where(dist >= 2+ResTable_min, (ResTable_vals[22*(ind)+7] + (
            ResTable_vals[22*(ind+1)+7]-ResTable_vals[22*(ind)+7]) * fac), XC12)

        YC11 = jnp.where(dist > 0, ResTable_vals[8], 0.)
        YC11 = jnp.where(dist >= 2+ResTable_min, (ResTable_vals[22*(ind)+8] + (
            ResTable_vals[22*(ind+1)+8]-ResTable_vals[22*(ind)+8]) * fac), YC11)

        YC12 = jnp.where(dist > 0, ResTable_vals[9], 0.)
        YC12 = jnp.where(dist >= 2+ResTable_min, (ResTable_vals[22*(ind)+9] + (
            ResTable_vals[22*(ind+1)+9]-ResTable_vals[22*(ind)+9]) * fac), YC12)


        epsr = jnp.array(
            [[jnp.zeros(n_pairs_lub_prec), r[:,2], -r[:,1]], [-r[:,2], jnp.zeros(n_pairs_lub_prec), r[:,0]], [r[:,1], -r[:,0], jnp.zeros(n_pairs_lub_prec)]])
        Imrr = jnp.array(
            [[jnp.ones(n_pairs_lub_prec), jnp.zeros(n_pairs_lub_prec), jnp.zeros(n_pairs_lub_prec)],
              [jnp.zeros(n_pairs_lub_prec), jnp.ones(n_pairs_lub_prec),jnp.zeros(n_pairs_lub_prec)], 
              [jnp.zeros(n_pairs_lub_prec), jnp.zeros(n_pairs_lub_prec), jnp.ones(n_pairs_lub_prec)]])
        rr = jnp.array([[r[:,0]*r[:,0],r[:,0]*r[:,1],r[:,0]*r[:,2]],[r[:,1]*r[:,0],r[:,1]*r[:,1],r[:,1]*r[:,2]],[r[:,2]*r[:,0],r[:,2]*r[:,1],r[:,2]*r[:,2]]])
        Imrr = Imrr - rr

        A_neigh = XA12 * (rr) + YA12 * (Imrr)
        A_self = XA11 * (rr) + YA11 * (Imrr)
        B_neigh = YB12 * (epsr)
        B_self = YB11 * (epsr)
        C_neigh = XC12 * (rr) + YC12 * (Imrr)
        C_self = XC11 * (rr) + YC11 * (Imrr)
        
        
        # Fill in matrix (pair contributions)
        # # this is for all the A12 blocks 
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:],6*nl_lub_prec[1,:]].set(A_neigh[0,0,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:],6*nl_lub_prec[1,:]+1].set(A_neigh[0,1,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:],6*nl_lub_prec[1,:]+2].set(A_neigh[0,2,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+1,6*nl_lub_prec[1,:]].set(A_neigh[1,0,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+1,6*nl_lub_prec[1,:]+1].set(A_neigh[1,1,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+1,6*nl_lub_prec[1,:]+2].set(A_neigh[1,2,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+2,6*nl_lub_prec[1,:]].set(A_neigh[2,0,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+2,6*nl_lub_prec[1,:]+1].set(A_neigh[2,1,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+2,6*nl_lub_prec[1,:]+2].set(A_neigh[2,2,:])
        
        # # this is for all the C12 blocks 
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+3,6*nl_lub_prec[1,:]+3].set(C_neigh[0,0,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+3,6*nl_lub_prec[1,:]+4].set(C_neigh[0,1,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+3,6*nl_lub_prec[1,:]+5].set(C_neigh[0,2,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+4,6*nl_lub_prec[1,:]+3].set(C_neigh[1,0,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+4,6*nl_lub_prec[1,:]+4].set(C_neigh[1,1,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+4,6*nl_lub_prec[1,:]+5].set(C_neigh[1,2,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+5,6*nl_lub_prec[1,:]+3].set(C_neigh[2,0,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+5,6*nl_lub_prec[1,:]+4].set(C_neigh[2,1,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+5,6*nl_lub_prec[1,:]+5].set(C_neigh[2,2,:])
       
        # # this is for all the Bt12 blocks (Bt12 = B12)
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:],6*nl_lub_prec[1,:]+3].set(B_neigh[0,0,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:],6*nl_lub_prec[1,:]+4].set(B_neigh[0,1,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:],6*nl_lub_prec[1,:]+5].set(B_neigh[0,2,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+1,6*nl_lub_prec[1,:]+3].set(B_neigh[1,0,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+1,6*nl_lub_prec[1,:]+4].set(B_neigh[1,1,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+1,6*nl_lub_prec[1,:]+5].set(B_neigh[1,2,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+2,6*nl_lub_prec[1,:]+3].set(B_neigh[2,0,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+2,6*nl_lub_prec[1,:]+4].set(B_neigh[2,1,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+2,6*nl_lub_prec[1,:]+5].set(B_neigh[2,2,:])
        
        # # this is for all the B12 blocks (Bt12 = B12)
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+3,6*nl_lub_prec[1,:]].set(B_neigh[0,0,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+3,6*nl_lub_prec[1,:]+1].set(B_neigh[0,1,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+3,6*nl_lub_prec[1,:]+2].set(B_neigh[0,2,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+4,6*nl_lub_prec[1,:]].set(B_neigh[1,0,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+4,6*nl_lub_prec[1,:]+1].set(B_neigh[1,1,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+4,6*nl_lub_prec[1,:]+2].set(B_neigh[1,2,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+5,6*nl_lub_prec[1,:]].set(B_neigh[2,0,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+5,6*nl_lub_prec[1,:]+1].set(B_neigh[2,1,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+5,6*nl_lub_prec[1,:]+2].set(B_neigh[2,2,:])
        
        # Fill in matrix (self contributions) (there are a sum of contributions from each pairs: so particle 0 self contribution will be a sum over all neighboring particles, and so on...)
        
        #A11 Block         
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:],6*nl_lub_prec[0,:]].add(A_self[0,0,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:],6*nl_lub_prec[0,:]+1].add(A_self[0,1,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:],6*nl_lub_prec[0,:]+2].add(A_self[0,2,:])
        # R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+1,6*nl_lub_prec[0,:]].add(A_self[1,0,:]) #below diagonal
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+1,6*nl_lub_prec[0,:]+1].add(A_self[1,1,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+1,6*nl_lub_prec[0,:]+2].add(A_self[1,2,:])
        # R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+2,6*nl_lub_prec[0,:]].add(A_self[2,0,:]) #below diagonal
        # R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+2,6*nl_lub_prec[0,:]+1].add(A_self[2,1,:]) #below diagonal
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+2,6*nl_lub_prec[0,:]+2].add(A_self[2,2,:])
        
        #A22 Block        
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,:],6*nl_lub_prec[1,:]].add(A_self[0,0,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,:],6*nl_lub_prec[1,:]+1].add(A_self[0,1,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,:],6*nl_lub_prec[1,:]+2].add(A_self[0,2,:])
        # R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,:]+1,6*nl_lub_prec[1,:]].add(A_self[1,0,:])#below diagonal
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,:]+1,6*nl_lub_prec[1,:]+1].add(A_self[1,1,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,:]+1,6*nl_lub_prec[1,:]+2].add(A_self[1,2,:])
        # R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,:]+2,6*nl_lub_prec[1,:]].add(A_self[2,0,:])#below diagonal
        # R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,:]+2,6*nl_lub_prec[1,:]+1].add(A_self[2,1,:])#below diagonal
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,:]+2,6*nl_lub_prec[1,:]+2].add(A_self[2,2,:])
        
        #C11 Block         
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+3,6*nl_lub_prec[0,:]+3].add(C_self[0,0,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+3,6*nl_lub_prec[0,:]+4].add(C_self[0,1,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+3,6*nl_lub_prec[0,:]+5].add(C_self[0,2,:])
        # R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+4,6*nl_lub_prec[0,:]+3].add(C_self[1,0,:]) #below diagonal
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+4,6*nl_lub_prec[0,:]+4].add(C_self[1,1,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+4,6*nl_lub_prec[0,:]+5].add(C_self[1,2,:])
        # R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+5,6*nl_lub_prec[0,:]+3].add(C_self[2,0,:]) #below diagonal
        # R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+5,6*nl_lub_prec[0,:]+4].add(C_self[2,1,:]) #below diagonal
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+5,6*nl_lub_prec[0,:]+5].add(C_self[2,2,:])
        
        #C22 Block        
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,:]+3,6*nl_lub_prec[1,:]+3].add(C_self[0,0,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,:]+3,6*nl_lub_prec[1,:]+4].add(C_self[0,1,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,:]+3,6*nl_lub_prec[1,:]+5].add(C_self[0,2,:])
        # R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,:]+4,6*nl_lub_prec[1,:]+3].add(C_self[1,0,:])#below diagonal
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,:]+4,6*nl_lub_prec[1,:]+5].add(C_self[1,1,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,:]+4,6*nl_lub_prec[1,:]+6].add(C_self[1,2,:])
        # R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,:]+5,6*nl_lub_prec[1,:]+3].add(C_self[2,0,:])#below diagonal
        # R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,:]+5,6*nl_lub_prec[1,:]+4].add(C_self[2,1,:])#below diagonal
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,:]+5,6*nl_lub_prec[1,:]+5].add(C_self[2,2,:])
        
        #Bt11 Block
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:],6*nl_lub_prec[0,:]+3].add(-B_self[0,0,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:],6*nl_lub_prec[0,:]+4].add(-B_self[0,1,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:],6*nl_lub_prec[0,:]+5].add(-B_self[0,2,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+1,6*nl_lub_prec[0,:]+3].add(-B_self[1,0,:]) 
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+1,6*nl_lub_prec[0,:]+4].add(-B_self[1,1,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+1,6*nl_lub_prec[0,:]+5].add(-B_self[1,2,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+2,6*nl_lub_prec[0,:]+3].add(-B_self[2,0,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+2,6*nl_lub_prec[0,:]+4].add(-B_self[2,1,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[0,:]+2,6*nl_lub_prec[0,:]+5].add(-B_self[2,2,:])
        
        #Bt22 Block        
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,:],6*nl_lub_prec[1,:]+3].add(B_self[0,0,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,:],6*nl_lub_prec[1,:]+4].add(B_self[0,1,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,:],6*nl_lub_prec[1,:]+5].add(B_self[0,2,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,:]+1,6*nl_lub_prec[1,:]+3].add(B_self[1,0,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,:]+1,6*nl_lub_prec[1,:]+5].add(B_self[1,1,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,:]+1,6*nl_lub_prec[1,:]+6].add(B_self[1,2,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,:]+2,6*nl_lub_prec[1,:]+3].add(B_self[2,0,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,:]+2,6*nl_lub_prec[1,:]+4].add(B_self[2,1,:])
        R_fu_precondition = R_fu_precondition.at[6*nl_lub_prec[1,:]+2,6*nl_lub_prec[1,:]+5].add(B_self[2,2,:])
        
        # Symmetrize R_fu_nf
        diagonal_elements = R_fu_precondition.diagonal()  # extract the diagonal
        R_fu_precondition += jnp.transpose(R_fu_precondition-jnp.diag(diagonal_elements))        
        # now we have a symmetrix matrix 

        # Compress diagonal values (needed later for brownian calculations, to have more numerical stability)
        diagonal_elements_for_brownian = jnp.where((diagonal_elements >= 1) | (
            diagonal_elements == 0), 1, jnp.sqrt(1/diagonal_elements))  # compress diagonal elements

        # Add identity for far field contribution and scale it properly
        # Because all values are made dimensionless on 6*pi*eta*a, the diagonal elements for FU (forces - velocities) are 1, but those for LW are 4/3(torques - angular velocities)
        diagonal_elements = jnp.where((jnp.arange(6*N)-6*(jnp.repeat(jnp.arange(N), 6))) <
                                      3, ichol_relaxer, ichol_relaxer*1.33333333333)

        # sum the diagonal to the rest of the precondition matrix
        R_fu_precondition += jnp.diag(diagonal_elements)

        #Perform cholesky factorization and obtain lower triangle cholesky factor of R_fu
        R_fu_precondition = jnp.linalg.cholesky(R_fu_precondition) 

        return R_fu_precondition, diagonal_elements_for_brownian
    
    
    
    # given a support size (for gaussian spread) -> compute distances in a gaussP x gaussP x gaussP grid (where gauss P might be corrected in it is odd or even...)
    def precompute_grid_distancing(gauss_P, gridh, tilt_factor,positions):
        # SEE Molbility.cu (line 265) for tips on how to change the distances when we have shear
        grid = np.zeros((gauss_P, gauss_P, gauss_P,N))
        center_offsets = (jnp.array(positions)+jnp.array([Lx,Ly,Lz])/2)
        center_offsets = center_offsets.at[:,0].add(-tilt_factor*center_offsets.at[:,1].get())
        center_offsets = center_offsets / jnp.array([Lx,Ly,Lz])  * jnp.array([Nx, Ny, Nz])
        center_offsets -= jnp.array(center_offsets, dtype = int) #in grid units, not particle radius units
        center_offsets = jnp.where(center_offsets>0.5, -(1-center_offsets), center_offsets)
        for i in range(gauss_P):
            for j in range(gauss_P):
                for k in range(gauss_P):
                    grid[i, j, k, :] = gridh*gridh* ( (i - int(gauss_P/2) - center_offsets[:,0] + tilt_factor*(j - int(gauss_P/2) - center_offsets[:,1])   )**2 
                                                     + (j - int(gauss_P/2) - center_offsets[:,1] )**2 
                                                     + (k - int(gauss_P/2) - center_offsets[:,2] )**2) 
        return grid
    
    @jit
    def update_neigh_list_and_displacements(shear_rate,positions, displacements_vector_matrix, nbrs_lub, nbrs_lub_prec, nbrs_ff,net_vel,dt):
        
        dR = jnp.zeros((N,3),float)
        dR = dR.at[:,0].set(dt*net_vel.at[(0)::6].get())
        dR = dR.at[:,1].set(dt*net_vel.at[(1)::6].get())
        dR = dR.at[:,2].set(dt*net_vel.at[(2)::6].get())
        
        positions = shift(positions+jnp.array([Lx,Ly,Lz])/2,dR)
        positions = positions - jnp.array([Lx,Ly,Lz])*0.5
        dR = jnp.zeros((N,3),float)
        dR = dR.at[:,0].set(dt * shear_rate * positions.at[:,1].get())
        positions = shift(positions+jnp.array([Lx,Ly,Lz])/2,dR)
        positions = positions - jnp.array([Lx,Ly,Lz])*0.5
        
        displacements_vector_matrix = (
            space.map_product(displacement))(positions, positions)
        
        nbrs_lub = nbrs_lub.update(positions)
        nbrs_lub_prec = nbrs_lub_prec.update(positions)
        nbrs_ff = nbrs_ff.update(positions)
         
        return positions, displacements_vector_matrix, nbrs_lub, nbrs_lub_prec, nbrs_ff
    
    @partial(jit, static_argnums=[1])
    def generate_random_array(key,size):
        key, subkey = random.split(key) #advance RNG state (otherwise will get same random numbers)
        return subkey,  (random.uniform(subkey, (size,)) ) 


    @jit
    def solver(
            rhs,  # rhs vector of the linear system
            gridk,
            R_fu_prec_lower_triang,
            all_indices_x,all_indices_y,all_indices_z,gaussian_grid_spacing1,gaussian_grid_spacing2,
            r,indices_i,indices_j,f1,f2,g1,g2,h1,h2,h3,
            r_lub,indices_i_lub,indices_j_lub,XA11,XA12,YA11,YA12,YB11,YB12,XC11,XC12,YC11,YC12,YB21
            ):

        
        def GeneralizedMobility(
            # output: the generalized velocities vector of particles (velocities+angular velocities+rateOfStrain: 11N)
            # input: the generalized forces (force+torque+stresslet: 11*N) of particles
            generalized_forces,
            ):
            
            # Helper function
            def swap_real_imag(cplx_arr):
                return -jnp.imag(cplx_arr) + 1j * jnp.real(cplx_arr)

            # Get forces,torques,couplets from generalized forces (3*N vector: f1x,f1y,f1z, ... , fNx,fNy,fNz and same for torque, while couplet is 5N) 
            forces = jnp.zeros(3*N)
            forces = forces.at[0::3].set(generalized_forces.at[0:(6*N):6].get())
            forces = forces.at[1::3].set(generalized_forces.at[1:(6*N):6].get())
            forces = forces.at[2::3].set(generalized_forces.at[2:(6*N):6].get())
            torques = jnp.zeros(3*N)
            torques = torques.at[0::3].set(generalized_forces.at[3:(6*N):6].get())
            torques = torques.at[1::3].set(generalized_forces.at[4:(6*N):6].get())
            torques = torques.at[2::3].set(generalized_forces.at[5:(6*N):6].get())
            stresslet = jnp.zeros(5*N)
            stresslet = stresslet.at[0::5].set(generalized_forces.at[(6*N+0)::5].get()) #Sxx
            stresslet = stresslet.at[1::5].set(generalized_forces.at[(6*N+1)::5].get()) #Sxy
            stresslet = stresslet.at[2::5].set(generalized_forces.at[(6*N+2)::5].get()) #Sxz
            stresslet = stresslet.at[3::5].set(generalized_forces.at[(6*N+3)::5].get()) #Syz
            stresslet = stresslet.at[4::5].set(generalized_forces.at[(6*N+4)::5].get()) #Syy

            # Get 'couplet' out from generalized forces (8*N vector)    
            couplets = jnp.zeros(8*N)
            couplets = couplets.at[::8].set(stresslet.at[::5].get())  # C[0] = S[0]
            couplets = couplets.at[1::8].set(
                stresslet.at[1::5].get()+torques.at[2::3].get()*0.5)  # C[1] = S[1] + L[2]/2
            couplets = couplets.at[2::8].set(
                stresslet.at[2::5].get()-torques.at[1::3].get()*0.5)  # C[2] = S[2] - L[1]/2
            couplets = couplets.at[3::8].set(
                stresslet.at[3::5].get()+torques.at[::3].get()*0.5)  # C[3] = S[3] + L[0]/2
            couplets = couplets.at[4::8].set(stresslet.at[4::5].get())  # C[4] = S[4]
            couplets = couplets.at[5::8].set(
                stresslet.at[1::5].get()-torques.at[2::3].get()*0.5)  # C[5] = S[1] - L[2]/2
            couplets = couplets.at[6::8].set(
                stresslet.at[2::5].get()+torques.at[1::3].get()*0.5)  # C[6] = S[2] + L[1]/2
            couplets = couplets.at[7::8].set(
                stresslet.at[3::5].get()-torques.at[::3].get()*0.5)  # C[7] = S[3] - L[0]/2

            ##########################################################################################################################################
            ######################################## WAVE SPACE CONTRIBUTION #########################################################################
            ##########################################################################################################################################
            
            #Create Grids for current iteration
            gridX = jnp.zeros((Nx, Ny, Nz))
            gridY = jnp.zeros((Nx, Ny, Nz))
            gridZ = jnp.zeros((Nx, Ny, Nz))
            gridXX = jnp.zeros((Nx, Ny, Nz))
            gridXY = jnp.zeros((Nx, Ny, Nz))
            gridXZ = jnp.zeros((Nx, Ny, Nz))
            gridYX = jnp.zeros((Nx, Ny, Nz))
            gridYY = jnp.zeros((Nx, Ny, Nz))
            gridYZ = jnp.zeros((Nx, Ny, Nz))
            gridZX = jnp.zeros((Nx, Ny, Nz))
            gridZY = jnp.zeros((Nx, Ny, Nz))
            
            gridX = gridX.at[all_indices_x, all_indices_y, all_indices_z].add(jnp.ravel((jnp.swapaxes(jnp.swapaxes(jnp.swapaxes(gaussian_grid_spacing1*jnp.resize(forces.at[0::3].get(), (gaussP,gaussP,gaussP,N) ),3,2),2,1),1,0))))
            gridY = gridY.at[all_indices_x, all_indices_y, all_indices_z].add(jnp.ravel((jnp.swapaxes(jnp.swapaxes(jnp.swapaxes(gaussian_grid_spacing1*jnp.resize(forces.at[1::3].get(), (gaussP,gaussP,gaussP,N) ),3,2),2,1),1,0))))            
            gridZ = gridZ.at[all_indices_x, all_indices_y, all_indices_z].add(jnp.ravel((jnp.swapaxes(jnp.swapaxes(jnp.swapaxes(gaussian_grid_spacing1*jnp.resize(forces.at[2::3].get(), (gaussP,gaussP,gaussP,N) ),3,2),2,1),1,0))))
            gridXX = gridXX.at[all_indices_x, all_indices_y, all_indices_z].add(jnp.ravel((jnp.swapaxes(jnp.swapaxes(jnp.swapaxes(gaussian_grid_spacing1*jnp.resize(couplets.at[0::8].get(), (gaussP,gaussP,gaussP,N) ),3,2),2,1),1,0))))
            gridXY = gridXY.at[all_indices_x, all_indices_y, all_indices_z].add(jnp.ravel((jnp.swapaxes(jnp.swapaxes(jnp.swapaxes(gaussian_grid_spacing1*jnp.resize(couplets.at[1::8].get(), (gaussP,gaussP,gaussP,N) ),3,2),2,1),1,0))))
            gridXZ = gridXZ.at[all_indices_x, all_indices_y, all_indices_z].add(jnp.ravel((jnp.swapaxes(jnp.swapaxes(jnp.swapaxes(gaussian_grid_spacing1*jnp.resize(couplets.at[2::8].get(), (gaussP,gaussP,gaussP,N) ),3,2),2,1),1,0)))) 
            gridYZ = gridYZ.at[all_indices_x, all_indices_y, all_indices_z].add(jnp.ravel((jnp.swapaxes(jnp.swapaxes(jnp.swapaxes(gaussian_grid_spacing1*jnp.resize(couplets.at[3::8].get(), (gaussP,gaussP,gaussP,N) ),3,2),2,1),1,0))))
            gridYY = gridYY.at[all_indices_x, all_indices_y, all_indices_z].add(jnp.ravel((jnp.swapaxes(jnp.swapaxes(jnp.swapaxes(gaussian_grid_spacing1*jnp.resize(couplets.at[4::8].get(), (gaussP,gaussP,gaussP,N) ),3,2),2,1),1,0))))
            gridYX = gridYX.at[all_indices_x, all_indices_y, all_indices_z].add(jnp.ravel((jnp.swapaxes(jnp.swapaxes(jnp.swapaxes(gaussian_grid_spacing1*jnp.resize(couplets.at[5::8].get(), (gaussP,gaussP,gaussP,N) ),3,2),2,1),1,0))))
            gridZX = gridZX.at[all_indices_x, all_indices_y, all_indices_z].add(jnp.ravel((jnp.swapaxes(jnp.swapaxes(jnp.swapaxes(gaussian_grid_spacing1*jnp.resize(couplets.at[6::8].get(), (gaussP,gaussP,gaussP,N) ),3,2),2,1),1,0))))
            gridZY = gridZY.at[all_indices_x, all_indices_y, all_indices_z].add(jnp.ravel((jnp.swapaxes(jnp.swapaxes(jnp.swapaxes(gaussian_grid_spacing1*jnp.resize(couplets.at[7::8].get(), (gaussP,gaussP,gaussP,N) ),3,2),2,1),1,0))))

            # Apply FFT
            gridX = jnp.fft.fftn(gridX)
            gridY = jnp.fft.fftn(gridY)
            gridZ = jnp.fft.fftn(gridZ)
            gridXX = jnp.fft.fftn(gridXX)
            gridXY = jnp.fft.fftn(gridXY)
            gridXZ = jnp.fft.fftn(gridXZ)
            gridYZ = jnp.fft.fftn(gridYZ)
            gridYY = jnp.fft.fftn(gridYY)
            gridYX = jnp.fft.fftn(gridYX)
            gridZX = jnp.fft.fftn(gridZX)
            gridZY = jnp.fft.fftn(gridZY)
            gridZZ = - gridXX - gridYY
            
            gridk_sqr = (gridk.at[:, :, :, 0].get()*gridk.at[:, :, :, 0].get()+gridk.at[:, :, :, 1].get()
                         * gridk.at[:, :, :, 1].get()+gridk.at[:, :, :, 2].get()*gridk.at[:, :, :, 2].get())
            gridk_mod = jnp.sqrt(gridk_sqr)
            
            kdF = jnp.where(gridk_mod > 0, (gridk.at[:, :, :, 0].get()*gridX +
                            gridk.at[:, :, :, 1].get()*gridY+gridk.at[:, :, :, 2].get()*gridZ)/gridk_sqr, 0)

            Cdkx = (gridk.at[:, :, :, 0].get()*gridXX + gridk.at[:, :, :, 1].get()*gridXY + gridk.at[:, :, :, 2].get()*gridXZ)
            Cdky = (gridk.at[:, :, :, 0].get()*gridYX + gridk.at[:, :, :, 1].get()*gridYY + gridk.at[:, :, :, 2].get()*gridYZ)
            Cdkz = (gridk.at[:, :, :, 0].get()*gridZX + gridk.at[:, :, :, 1].get()*gridZY + gridk.at[:, :, :, 2].get()*gridZZ)

            kdcdk = jnp.where(
                gridk_mod > 0, ( gridk.at[:, :, :, 0].get()*Cdkx
                                +gridk.at[:, :, :, 1].get()*Cdky
                                +gridk.at[:, :, :, 2].get()*Cdkz)/gridk_sqr, 0)
            
            Fkxx = (gridk.at[:, :, :, 0].get()*gridX)
            Fkxy = (gridk.at[:, :, :, 1].get()*gridX)
            Fkxz = (gridk.at[:, :, :, 2].get()*gridX)
            Fkyx = (gridk.at[:, :, :, 0].get()*gridY)
            Fkyy = (gridk.at[:, :, :, 1].get()*gridY)
            Fkyz = (gridk.at[:, :, :, 2].get()*gridY)
            Fkzx = (gridk.at[:, :, :, 0].get()*gridZ)
            Fkzy = (gridk.at[:, :, :, 1].get()*gridZ)
            kkxx = gridk.at[:, :, :, 0].get()*gridk.at[:, :, :, 0].get()
            kkxy = gridk.at[:, :, :, 0].get()*gridk.at[:, :, :, 1].get()
            kkxz = gridk.at[:, :, :, 0].get()*gridk.at[:, :, :, 2].get()
            kkyx = gridk.at[:, :, :, 1].get()*gridk.at[:, :, :, 0].get()
            kkyy = gridk.at[:, :, :, 1].get()*gridk.at[:, :, :, 1].get()
            kkyz = gridk.at[:, :, :, 1].get()*gridk.at[:, :, :, 2].get()
            kkzx = gridk.at[:, :, :, 2].get()*gridk.at[:, :, :, 0].get()
            kkzy = gridk.at[:, :, :, 2].get()*gridk.at[:, :, :, 1].get()
            Cdkkxx = gridk.at[:, :, :, 0].get() * Cdkx
            Cdkkxy = gridk.at[:, :, :, 1].get() * Cdkx
            Cdkkxz = gridk.at[:, :, :, 2].get() * Cdkx
            Cdkkyx = gridk.at[:, :, :, 0].get() * Cdky
            Cdkkyy = gridk.at[:, :, :, 1].get() * Cdky
            Cdkkyz = gridk.at[:, :, :, 2].get() * Cdky
            Cdkkzx = gridk.at[:, :, :, 0].get() * Cdkz
            Cdkkzy = gridk.at[:, :, :, 1].get() * Cdkz

            # UF part
            B = jnp.where(gridk_mod > 0, gridk.at[:, :, :, 3].get()*(jnp.sin(gridk_mod)/gridk_mod)*(
                jnp.sin(gridk_mod)/gridk_mod), 0)  # scaling factor
            gridX = B * (gridX - gridk.at[:, :, :, 0].get() * kdF)
            gridY = B * (gridY - gridk.at[:, :, :, 1].get() * kdF)
            gridZ = B * (gridZ - gridk.at[:, :, :, 2].get() * kdF)

            # UC part (here B is imaginary so we absorb the imaginary unit in the funtion 'swap_real_imag()' which returns -Im(c)+i*Re(c)
            B = jnp.where(gridk_mod > 0, gridk.at[:, :, :, 3].get()*(jnp.sin(gridk_mod)/gridk_mod)*(3 * (jnp.sin(
                gridk_mod) - gridk_mod*jnp.cos(gridk_mod)) / (gridk_mod*gridk_sqr)), 0)  # scaling factor

            gridX += B * swap_real_imag( (Cdkx - kdcdk * gridk.at[:, :, :, 0].get()))
            gridY += B * swap_real_imag( (Cdky - kdcdk * gridk.at[:, :, :, 1].get()))
            gridZ += B * swap_real_imag( (Cdkz - kdcdk * gridk.at[:, :, :, 2].get()))

            # DF part
            B = jnp.where(gridk_mod > 0, gridk.at[:, :, :, 3].get()*(-1)*(jnp.sin(gridk_mod)/gridk_mod)*(3 * (
                jnp.sin(gridk_mod) - gridk_mod*jnp.cos(gridk_mod)) / (gridk_mod*gridk_sqr)), 0)  # scaling factor
            gridXX = B * swap_real_imag((Fkxx - kkxx * kdF))
            gridXY = B * swap_real_imag((Fkxy - kkxy * kdF))
            gridXZ = B * swap_real_imag((Fkxz - kkxz * kdF))
            gridYX = B * swap_real_imag((Fkyx - kkyx * kdF))
            gridYY = B * swap_real_imag((Fkyy - kkyy * kdF))
            gridYZ = B * swap_real_imag((Fkyz - kkyz * kdF))
            gridZX = B * swap_real_imag((Fkzx - kkzx * kdF))
            gridZY = B * swap_real_imag((Fkzy - kkzy * kdF))

            # DC part
            B = jnp.where(gridk_mod > 0, gridk.at[:, :, :, 3].get()*(9)*((jnp.sin(gridk_mod) - gridk_mod*jnp.cos(gridk_mod)) / (
                gridk_mod*gridk_sqr)) * ((jnp.sin(gridk_mod) - gridk_mod*jnp.cos(gridk_mod)) / (gridk_mod*gridk_sqr)), 0)  # scaling factor
            gridXX += B * (Cdkkxx - kkxx * kdcdk)
            gridXY += B * (Cdkkxy - kkxy * kdcdk)
            gridXZ += B * (Cdkkxz - kkxz * kdcdk)
            gridYX += B * (Cdkkyx - kkyx * kdcdk)
            gridYY += B * (Cdkkyy - kkyy * kdcdk)
            gridYZ += B * (Cdkkyz - kkyz * kdcdk)
            gridZX += B * (Cdkkzx - kkzx * kdcdk)
            gridZY += B * (Cdkkzy - kkzy * kdcdk)
            
            # Inverse FFT
            gridX = jnp.real(jnp.fft.ifftn(gridX,norm='forward'))
            gridY = jnp.real(jnp.fft.ifftn(gridY,norm='forward'))
            gridZ = jnp.real(jnp.fft.ifftn(gridZ,norm='forward'))
            gridXX = jnp.real(jnp.fft.ifftn(gridXX,norm='forward'))
            gridXY = jnp.real(jnp.fft.ifftn(gridXY,norm='forward'))
            gridXZ = jnp.real(jnp.fft.ifftn(gridXZ,norm='forward'))
            gridYX = jnp.real(jnp.fft.ifftn(gridYX,norm='forward'))
            gridYY = jnp.real(jnp.fft.ifftn(gridYY,norm='forward'))
            gridYZ = jnp.real(jnp.fft.ifftn(gridYZ,norm='forward'))
            gridZX = jnp.real(jnp.fft.ifftn(gridZX,norm='forward'))
            gridZY = jnp.real(jnp.fft.ifftn(gridZY,norm='forward'))
            
            # Compute Linear velocities and Velocity gradients (from which we can extract angular velocities and rate of strain)
            w_lin_velocities = jnp.zeros((N, 3),float)
            w_velocity_gradient = jnp.zeros((N, 8),float)
               
             
            w_lin_velocities = w_lin_velocities.at[:, 0].add(
                jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
                    gridX.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3)))
            w_lin_velocities = w_lin_velocities.at[:, 1].add(
                jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
                    gridY.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3))) 
            w_lin_velocities = w_lin_velocities.at[:, 2].add(
                jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
                    gridZ.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3)))              


            w_velocity_gradient = w_velocity_gradient.at[:, 0].add(
                jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
                    gridXX.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3)))
            
            #index might be 1 instead if 5 (originally is 1, and so on for the ones below)
            w_velocity_gradient = w_velocity_gradient.at[:, 1].add(
                jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
                    gridXY.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3))) 
            #index might be 2 instead if 6           
            w_velocity_gradient = w_velocity_gradient.at[:, 2].add(
                  jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
                    gridXZ.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3)))  
            #index might be 3 instead if 7
            w_velocity_gradient = w_velocity_gradient.at[:, 3].add(
                jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
                    gridYZ.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3)))
            
            
            w_velocity_gradient = w_velocity_gradient.at[:, 4].add(
                jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
                    gridYY.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3))) 
            
            #index might be 5 instead if 1
            w_velocity_gradient = w_velocity_gradient.at[:, 5].add(
                jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
                    gridYX.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3)))     
            #index might be 6 instead if 2                        
            w_velocity_gradient = w_velocity_gradient.at[:, 6].add(
                  jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
                    gridZX.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3))) 
            #index might be 7 instead if 3
            w_velocity_gradient = w_velocity_gradient.at[:, 7].add(
                jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
                    gridZY.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3)))              
            
            
            ##########################################################################################################################################
            ######################################## REAL SPACE CONTRIBUTION #########################################################################
            ##########################################################################################################################################
            
            
            # Allocate arrays for Linear velocities and Velocity gradients (from which we can extract angular velocities and rate of strain)
            r_lin_velocities = jnp.zeros((N, 3),float)
            r_velocity_gradient = jnp.zeros((N, 8),float)
            
            # SELF CONTRIBUTIONS
            r_lin_velocities = r_lin_velocities.at[:, 0].set(
                m_self.at[0].get() * forces.at[0::3].get())
            r_lin_velocities = r_lin_velocities.at[:, 1].set(
                m_self.at[0].get() * forces.at[1::3].get())
            r_lin_velocities = r_lin_velocities.at[:, 2].set(
                m_self.at[0].get() * forces.at[2::3].get())
            
            r_velocity_gradient = r_velocity_gradient.at[:, 0].set(
                m_self.at[1].get()*(couplets.at[0::8].get() - 4 * couplets.at[0::8].get()))
            r_velocity_gradient = r_velocity_gradient.at[:, 5].set(
                m_self.at[1].get()*(couplets.at[1::8].get() - 4 * couplets.at[5::8].get()))
            r_velocity_gradient = r_velocity_gradient.at[:, 6].set(
                m_self.at[1].get()*(couplets.at[2::8].get() - 4 * couplets.at[6::8].get()))
            r_velocity_gradient = r_velocity_gradient.at[:, 7].set(
                m_self.at[1].get()*(couplets.at[3::8].get() - 4 * couplets.at[7::8].get()))
            
            r_velocity_gradient = r_velocity_gradient.at[:, 4].set(
                m_self.at[1].get()*(couplets.at[4::8].get() - 4 * couplets.at[4::8].get()))
            r_velocity_gradient = r_velocity_gradient.at[:, 1].set(
                m_self.at[1].get()*(couplets.at[5::8].get() - 4 * couplets.at[1::8].get()))
            r_velocity_gradient = r_velocity_gradient.at[:, 2].set(
                m_self.at[1].get()*(couplets.at[6::8].get() - 4 * couplets.at[2::8].get()))
            r_velocity_gradient = r_velocity_gradient.at[:, 3].set(
                m_self.at[1].get()*(couplets.at[7::8].get() - 4 * couplets.at[3::8].get()))
            
            
 
            # # Pair contributions
    
            # # Geometric quantities
            rdotf_j =   (r.at[:,0].get() * forces.at[3*indices_j + 0].get() + r.at[:,1].get() * forces.at[3*indices_j + 1].get() + r.at[:,2].get() * forces.at[3*indices_j + 2].get())
            mrdotf_i = -(r.at[:,0].get() * forces.at[3*indices_i + 0].get() + r.at[:,1].get() * forces.at[3*indices_i + 1].get() + r.at[:,2].get() * forces.at[3*indices_i + 2].get())
            
            Cj_dotr = jnp.array( [couplets.at[8*indices_j + 0].get() * r.at[:,0].get() + couplets.at[8*indices_j + 1].get() * r.at[:,1].get() + couplets.at[8*indices_j + 2].get() * r.at[:,2].get(),
                                  couplets.at[8*indices_j + 5].get() * r.at[:,0].get() + couplets.at[8*indices_j + 4].get() * r.at[:,1].get() + couplets.at[8*indices_j + 3].get() * r.at[:,2].get(),
                                  couplets.at[8*indices_j + 6].get() * r.at[:,0].get() + couplets.at[8*indices_j + 7].get() * r.at[:,1].get() -(couplets.at[8*indices_j + 0].get() + couplets.at[8*indices_j + 4].get()) * r.at[:,2].get()])
            
            Ci_dotmr=jnp.array( [-couplets.at[8*indices_i + 0].get() * r.at[:,0].get() - couplets.at[8*indices_i + 1].get() * r.at[:,1].get() - couplets.at[8*indices_i + 2].get() * r.at[:,2].get(),
                                  -couplets.at[8*indices_i + 5].get() * r.at[:,0].get() - couplets.at[8*indices_i + 4].get() * r.at[:,1].get() - couplets.at[8*indices_i + 3].get() * r.at[:,2].get(),
                                  -couplets.at[8*indices_i + 6].get() * r.at[:,0].get() - couplets.at[8*indices_i + 7].get() * r.at[:,1].get() +(couplets.at[8*indices_i + 0].get() + couplets.at[8*indices_i + 4].get()) * r.at[:,2].get()])


            rdotC_j = jnp.array([couplets.at[8*indices_j + 0].get() * r.at[:,0].get() + couplets.at[8*indices_j + 5].get() * r.at[:,1].get() + couplets.at[8*indices_j + 6].get() * r.at[:,2].get(),
                                  couplets.at[8*indices_j + 1].get() * r.at[:,0].get() + couplets.at[8*indices_j + 4].get() * r.at[:,1].get() + couplets.at[8*indices_j + 7].get() * r.at[:,2].get(),
                                  couplets.at[8*indices_j + 2].get() * r.at[:,0].get() + couplets.at[8*indices_j + 3].get() * r.at[:,1].get() -(couplets.at[8*indices_j + 0].get() + couplets.at[8*indices_j + 4].get()) * r.at[:,2].get()])
            
            mrdotC_i=jnp.array([-couplets.at[8*indices_i + 0].get() * r.at[:,0].get() - couplets.at[8*indices_i + 5].get() * r.at[:,1].get() - couplets.at[8*indices_i + 6].get() * r.at[:,2].get(),
                                  -couplets.at[8*indices_i + 1].get() * r.at[:,0].get() - couplets.at[8*indices_i + 4].get() * r.at[:,1].get() - couplets.at[8*indices_i + 7].get() * r.at[:,2].get(),
                                  -couplets.at[8*indices_i + 2].get() * r.at[:,0].get() - couplets.at[8*indices_i + 3].get() * r.at[:,1].get() +(couplets.at[8*indices_i + 0].get() + couplets.at[8*indices_i + 4].get()) * r.at[:,2].get()])
                        
            rdotC_jj_dotr   =  (r.at[:,0].get()*Cj_dotr.at[0,:].get()  + r.at[:,1].get()*Cj_dotr.at[1,:].get()  + r.at[:,2].get()*Cj_dotr.at[2,:].get())
            mrdotC_ii_dotmr = -(r.at[:,0].get()*Ci_dotmr.at[0,:].get() + r.at[:,1].get()*Ci_dotmr.at[1,:].get() + r.at[:,2].get()*Ci_dotmr.at[2,:].get())
            
            
            # Compute Velocity for particles i
            r_lin_velocities = r_lin_velocities.at[indices_i, 0].add(f1 * forces.at[3*indices_j].get() + (f2 - f1) * rdotf_j * r.at[:,0].get())
            r_lin_velocities = r_lin_velocities.at[indices_i, 1].add(f1 * forces.at[3*indices_j+1].get() + (f2 - f1) * rdotf_j * r.at[:,1].get())
            r_lin_velocities = r_lin_velocities.at[indices_i, 2].add(f1 * forces.at[3*indices_j+2].get() + (f2 - f1) * rdotf_j * r.at[:,2].get())
            r_lin_velocities = r_lin_velocities.at[indices_i, 0].add(g1 * (Cj_dotr.at[0,:].get() - rdotC_jj_dotr * r.at[:,0].get()) + g2 * (rdotC_j.at[0,:].get() - 4.*rdotC_jj_dotr * r.at[:,0].get()))
            r_lin_velocities = r_lin_velocities.at[indices_i, 1].add(g1 * (Cj_dotr.at[1,:].get() - rdotC_jj_dotr * r.at[:,1].get()) + g2 * (rdotC_j.at[1,:].get() - 4.*rdotC_jj_dotr * r.at[:,1].get()))
            r_lin_velocities = r_lin_velocities.at[indices_i, 2].add(g1 * (Cj_dotr.at[2,:].get() - rdotC_jj_dotr * r.at[:,2].get()) + g2 * (rdotC_j.at[2,:].get() - 4.*rdotC_jj_dotr * r.at[:,2].get()))
            # Compute Velocity for particles j
            r_lin_velocities = r_lin_velocities.at[indices_j, 0].add(f1 * forces.at[3*indices_i].get() - (f2 - f1) * mrdotf_i * r.at[:,0].get())
            r_lin_velocities = r_lin_velocities.at[indices_j, 1].add(f1 * forces.at[3*indices_i+1].get() - (f2 - f1) * mrdotf_i * r.at[:,1].get())
            r_lin_velocities = r_lin_velocities.at[indices_j, 2].add(f1 * forces.at[3*indices_i+2].get() - (f2 - f1) * mrdotf_i * r.at[:,2].get())
            r_lin_velocities = r_lin_velocities.at[indices_j, 0].add(g1 * (Ci_dotmr.at[0,:].get() + mrdotC_ii_dotmr * r.at[:,0].get()) + g2 * (mrdotC_i.at[0,:].get() + 4.*mrdotC_ii_dotmr * r.at[:,0].get()))
            r_lin_velocities = r_lin_velocities.at[indices_j, 1].add(g1 * (Ci_dotmr.at[1,:].get() + mrdotC_ii_dotmr * r.at[:,1].get()) + g2 * (mrdotC_i.at[1,:].get() + 4.*mrdotC_ii_dotmr * r.at[:,1].get()))
            r_lin_velocities = r_lin_velocities.at[indices_j, 2].add(g1 * (Ci_dotmr.at[2,:].get() + mrdotC_ii_dotmr * r.at[:,2].get()) + g2 * (mrdotC_i.at[2,:].get() + 4.*mrdotC_ii_dotmr * r.at[:,2].get()))
            
            
            # Compute Velocity Gradient for particles i and j
            r_velocity_gradient = r_velocity_gradient.at[indices_i, 0].add(
                (-1) * g1 * (r.at[:,0].get() * forces.at[3*indices_j+0].get() - rdotf_j * r.at[:,0].get()*r.at[:,0].get())
                +(-1)*g2 * (rdotf_j + forces.at[3*indices_j+0].get()*r.at[:,0].get() - 4.*rdotf_j * r.at[:,0].get()*r.at[:,0].get()))
            
            r_velocity_gradient = r_velocity_gradient.at[indices_j, 0].add(
                (-1) * g1 * (-r.at[:,0].get() * forces.at[3*indices_i+0].get() - mrdotf_i * r.at[:,0].get()*r.at[:,0].get()) 
                +(-1)*g2 * (mrdotf_i - forces.at[3*indices_i+0].get()*r.at[:,0].get() - 4.*mrdotf_i * r.at[:,0].get()*r.at[:,0].get()))
            r_velocity_gradient = r_velocity_gradient.at[indices_i, 0].add(
                h1 * (couplets.at[8*indices_j+0].get() - 4. * couplets.at[8*indices_j+0].get()) 
                + h2 * (r.at[:,0].get()*Cj_dotr.at[0,:].get() - rdotC_jj_dotr * r.at[:,0].get()*r.at[:,0].get()) 
                + h3 * (rdotC_jj_dotr + Cj_dotr.at[0,:].get()*r.at[:,0].get() + r.at[:,0].get()*rdotC_j.at[0,:].get() + rdotC_j.at[0,:].get()*r.at[:,0].get() 
                                                                        - 6.*rdotC_jj_dotr*r.at[:,0].get()*r.at[:,0].get() - couplets.at[8*indices_j+0].get()))
            r_velocity_gradient = r_velocity_gradient.at[indices_j, 0].add(
                h1 * (couplets.at[8*indices_i+0].get() - 4. * couplets.at[8*indices_i+0].get()) 
                + h2 * (-r.at[:,0].get()*Ci_dotmr.at[0,:].get() - mrdotC_ii_dotmr * r.at[:,0].get()*r.at[:,0].get())
                + h3 * (mrdotC_ii_dotmr - Ci_dotmr.at[0,:].get()*r.at[:,0].get() - r.at[:,0].get()*mrdotC_i.at[0,:].get() - mrdotC_i.at[0,:].get()*r.at[:,0].get() 
                                                                        - 6.*mrdotC_ii_dotmr*r.at[:,0].get()*r.at[:,0].get() - couplets.at[8*indices_i+0].get()))



            r_velocity_gradient = r_velocity_gradient.at[indices_i, 1].add(
                (-1) * g1 * (r.at[:,1].get() * forces.at[3*indices_j+0].get() - rdotf_j * r.at[:,1].get()*r.at[:,0].get()) 
                + (-1)*g2 * (forces.at[3*indices_j+1].get()*r.at[:,0].get() - 4.*rdotf_j * r.at[:,1].get()*r.at[:,0].get()))
            
            r_velocity_gradient = r_velocity_gradient.at[indices_j, 1].add(
                (-1) * g1 * (-r.at[:,1].get() * forces.at[3*indices_i+0].get() - mrdotf_i * r.at[:,1].get()*r.at[:,0].get()) 
                + (-1)*g2 * (-forces.at[3*indices_i+1].get()*r.at[:,0].get() - 4.*mrdotf_i * r.at[:,1].get()*r.at[:,0].get()))
            
            r_velocity_gradient = r_velocity_gradient.at[indices_i, 1].add(
                h1 * (couplets.at[8*indices_j+5].get() - 4. * couplets.at[8*indices_j+1].get()) 
                + h2 * (r.at[:,1].get()*Cj_dotr.at[0,:].get() - rdotC_jj_dotr * r.at[:,1].get()*r.at[:,0].get())
                + h3 * (Cj_dotr.at[1,:].get()*r.at[:,0].get() + r.at[:,1].get()*rdotC_j.at[0,:].get() + rdotC_j.at[1,:].get()*r.at[:,0].get()
                                                    - 6.*rdotC_jj_dotr*r.at[:,1].get()*r.at[:,0].get() - couplets.at[8*indices_j+1].get()))
            
            r_velocity_gradient = r_velocity_gradient.at[indices_j, 1].add(
                h1 * (couplets.at[8*indices_i+5].get() - 4. * couplets.at[8*indices_i+1].get())
                + h2 * (-r.at[:,1].get()*Ci_dotmr.at[0,:].get() - mrdotC_ii_dotmr * r.at[:,1].get()*r.at[:,0].get())
                + h3 * (-Ci_dotmr.at[1,:].get()*r.at[:,0].get() - r.at[:,1].get()*mrdotC_i.at[0,:].get() - mrdotC_i.at[1,:].get()*r.at[:,0].get()
                                                        - 6.*mrdotC_ii_dotmr*r.at[:,1].get()*r.at[:,0].get() - couplets.at[8*indices_i+1].get()))



            r_velocity_gradient = r_velocity_gradient.at[indices_i, 2].add(
                      (-1) * g1 * (r.at[:,2].get() * forces.at[3*indices_j+0].get() - rdotf_j * r.at[:,2].get()*r.at[:,0].get()) 
                      + (-1)*g2 * (forces.at[3*indices_j+2].get()*r.at[:,0].get() - 4.*rdotf_j * r.at[:,2].get()*r.at[:,0].get()))
            
            r_velocity_gradient = r_velocity_gradient.at[indices_j, 2].add(
                      (-1) * g1 * (-r.at[:,2].get() * forces.at[3*indices_i+0].get() - mrdotf_i * r.at[:,2].get()*r.at[:,0].get()) 
                      + (-1)*g2 * (-forces.at[3*indices_i+2].get()*r.at[:,0].get() - 4.*mrdotf_i * r.at[:,2].get()*r.at[:,0].get()))
            r_velocity_gradient = r_velocity_gradient.at[indices_i, 2].add(
                h1 * (couplets.at[8*indices_j+6].get() - 4. * couplets.at[8*indices_j+2].get())
                + h2 * (r.at[:,2].get()*Cj_dotr.at[0,:].get() - rdotC_jj_dotr * r.at[:,2].get()*r.at[:,0].get())
                + h3 * (Cj_dotr.at[2,:].get()*r.at[:,0].get() + r.at[:,2].get()*rdotC_j.at[0,:].get() + rdotC_j.at[2,:].get()*r.at[:,0].get()
                        - 6.*rdotC_jj_dotr*r.at[:,2].get()*r.at[:,0].get() - couplets.at[8*indices_j+2].get()))
            r_velocity_gradient = r_velocity_gradient.at[indices_j, 2].add(
                h1 * (couplets.at[8*indices_i+6].get() - 4. * couplets.at[8*indices_i+2].get())
                + h2 * (r.at[:,2].get()*Ci_dotmr.at[0,:].get()*(-1) - mrdotC_ii_dotmr * r.at[:,2].get()*r.at[:,0].get())
                + h3 * (-Ci_dotmr.at[2,:].get()*r.at[:,0].get() - r.at[:,2].get()*mrdotC_i.at[0,:].get() - mrdotC_i.at[2,:].get()*r.at[:,0].get()
                        - 6.*mrdotC_ii_dotmr*r.at[:,2].get()*r.at[:,0].get() - couplets.at[8*indices_i+2].get()))
           
        
        
        
            r_velocity_gradient = r_velocity_gradient.at[indices_i, 3].add(
                    (-1) * g1 * (r.at[:,2].get() * forces.at[3*indices_j+1].get() - rdotf_j * r.at[:,2].get()*r.at[:,1].get())
                    + (-1)*g2 * (forces.at[3*indices_j+2].get()*r.at[:,1].get() - 4.*rdotf_j * r.at[:,2].get()*r.at[:,1].get()))
            
            r_velocity_gradient = r_velocity_gradient.at[indices_j, 3].add(
                    (-1) * g1 * (-r.at[:,2].get() * forces.at[3*indices_i+1].get() - mrdotf_i * r.at[:,2].get()*r.at[:,1].get())
                    + (-1)*g2 * (-forces.at[3*indices_i+2].get()*r.at[:,1].get() - 4.*mrdotf_i * r.at[:,2].get()*r.at[:,1].get()))
            
            r_velocity_gradient = r_velocity_gradient.at[indices_i, 3].add(
                h1 * (couplets.at[8*indices_j+7].get() - 4. * couplets.at[8*indices_j+3].get())
                + h2 * (r.at[:,2].get()*Cj_dotr.at[1,:].get() - rdotC_jj_dotr * r.at[:,2].get()*r.at[:,1].get())
                + h3 * (Cj_dotr.at[2,:].get()*r.at[:,1].get() + r.at[:,2].get()*rdotC_j.at[1,:].get() + rdotC_j.at[2,:].get()*r.at[:,1].get()
                        - 6.*rdotC_jj_dotr*r.at[:,2].get()*r.at[:,1].get() - couplets.at[8*indices_j+3].get()))

            r_velocity_gradient = r_velocity_gradient.at[indices_j, 3].add(
                h1 * (couplets.at[8*indices_i+7].get() - 4. * couplets.at[8*indices_i+3].get())
                + h2 * (-r.at[:,2].get()*Ci_dotmr.at[1,:].get() - mrdotC_ii_dotmr * r.at[:,2].get()*r.at[:,1].get()) 
                + h3 * (-Ci_dotmr.at[2,:].get()*r.at[:,1].get() - r.at[:,2].get()*mrdotC_i.at[1,:].get() - mrdotC_i.at[2,:].get()*r.at[:,1].get()
                        - 6.*mrdotC_ii_dotmr*r.at[:,2].get()*r.at[:,1].get() - couplets.at[8*indices_i+3].get()))



            r_velocity_gradient = r_velocity_gradient.at[indices_i, 4].add(
                    (-1) * g1 * (r.at[:,1].get() * forces.at[3*indices_j+1].get() - rdotf_j * r.at[:,1].get()*r.at[:,1].get())
                    + (-1)*g2 * (rdotf_j + forces.at[3*indices_j+1].get()*r.at[:,1].get() - 4.*rdotf_j * r.at[:,1].get()*r.at[:,1].get()))
            
            r_velocity_gradient = r_velocity_gradient.at[indices_j, 4].add(
                    (-1) * g1 * (-r.at[:,1].get() * forces.at[3*indices_i+1].get() - mrdotf_i * r.at[:,1].get()*r.at[:,1].get())
                    + (-1)*g2 * (mrdotf_i - forces.at[3*indices_i+1].get()*r.at[:,1].get() - 4.*mrdotf_i * r.at[:,1].get()*r.at[:,1].get()))
             
            r_velocity_gradient = r_velocity_gradient.at[indices_i, 4].add(
                h1 * (couplets.at[8*indices_j+4].get() - 4. * couplets.at[8*indices_j+4].get())
                + h2 * (r.at[:,1].get()*Cj_dotr.at[1,:].get() - rdotC_jj_dotr * r.at[:,1].get()*r.at[:,1].get()) 
                + h3 * (rdotC_jj_dotr + Cj_dotr.at[1,:].get()*r.at[:,1].get() + r.at[:,1].get()*rdotC_j.at[1,:].get() + rdotC_j.at[1,:].get()*r.at[:,1].get()
                        - 6.*rdotC_jj_dotr*r.at[:,1].get()*r.at[:,1].get() - couplets.at[8*indices_j+4].get()))
            
            r_velocity_gradient = r_velocity_gradient.at[indices_j, 4].add(
                h1 * (couplets.at[8*indices_i+4].get() - 4. * couplets.at[8*indices_i+4].get())
                + h2 * (-r.at[:,1].get()*Ci_dotmr.at[1,:].get() - mrdotC_ii_dotmr * r.at[:,1].get()*r.at[:,1].get())
                + h3 * (mrdotC_ii_dotmr - Ci_dotmr.at[1,:].get()*r.at[:,1].get() - r.at[:,1].get()*mrdotC_i.at[1,:].get() - mrdotC_i.at[1,:].get()*r.at[:,1].get()
                        - 6.*mrdotC_ii_dotmr*r.at[:,1].get()*r.at[:,1].get() - couplets.at[8*indices_i+4].get()))



            r_velocity_gradient = r_velocity_gradient.at[indices_i, 5].add(
                    (-1) * g1 * (r.at[:,0].get() * forces.at[3*indices_j+1].get() - rdotf_j * r.at[:,0].get()*r.at[:,1].get())
                    + (-1)*g2 * (forces.at[3*indices_j+0].get()*r.at[:,1].get() - 4.*rdotf_j * r.at[:,0].get()*r.at[:,1].get()))
            
            r_velocity_gradient = r_velocity_gradient.at[indices_j, 5].add(
                    (-1) * g1 * (-r.at[:,0].get() * forces.at[3*indices_i+1].get() - mrdotf_i * r.at[:,0].get()*r.at[:,1].get())
                    + (-1)*g2 * (-forces.at[3*indices_i+0].get()*r.at[:,1].get() - 4.*mrdotf_i * r.at[:,0].get()*r.at[:,1].get()))
            
            r_velocity_gradient = r_velocity_gradient.at[indices_i, 5].add(
                h1 * (couplets.at[8*indices_j+1].get() - 4. * couplets.at[8*indices_j+5].get())
                + h2 * (r.at[:,0].get()*Cj_dotr.at[1,:].get() - rdotC_jj_dotr * r.at[:,0].get()*r.at[:,1].get())
                + h3 * (Cj_dotr.at[0,:].get()*r.at[:,1].get() + r.at[:,0].get()*rdotC_j.at[1,:].get() + rdotC_j.at[0,:].get()*r.at[:,1].get()
                        - 6.*rdotC_jj_dotr*r.at[:,0].get()*r.at[:,1].get() - couplets.at[8*indices_j+5].get()))
            
            r_velocity_gradient = r_velocity_gradient.at[indices_j, 5].add(
                h1 * (couplets.at[8*indices_i+1].get() - 4. * couplets.at[8*indices_i+5].get())
                + h2 * (-r.at[:,0].get()*Ci_dotmr.at[1,:].get() - mrdotC_ii_dotmr * r.at[:,0].get()*r.at[:,1].get())
                + h3 * (-Ci_dotmr.at[0,:].get()*r.at[:,1].get() - r.at[:,0].get()*mrdotC_i.at[1,:].get() - mrdotC_i.at[0,:].get()*r.at[:,1].get()
                        - 6.*mrdotC_ii_dotmr*r.at[:,0].get()*r.at[:,1].get() - couplets.at[8*indices_i+5].get()))



            r_velocity_gradient = r_velocity_gradient.at[indices_i, 6].add(
                      (-1) * g1 * (r.at[:,0].get() * forces.at[3*indices_j+2].get() - rdotf_j * r.at[:,0].get()*r.at[:,2].get())
                      + (-1)*g2 * (forces.at[3*indices_j+0].get()*r.at[:,2].get() - 4.*rdotf_j * r.at[:,0].get()*r.at[:,2].get()))
            
            r_velocity_gradient = r_velocity_gradient.at[indices_j, 6].add(
                      (-1) * g1 * (-r.at[:,0].get() * forces.at[3*indices_i+2].get() - mrdotf_i * r.at[:,0].get()*r.at[:,2].get())
                      + (-1)*g2 * (-forces.at[3*indices_i+0].get()*r.at[:,2].get() - 4.*mrdotf_i * r.at[:,0].get()*r.at[:,2].get()))
             
            r_velocity_gradient = r_velocity_gradient.at[indices_i, 6].add(
                h1 * (couplets.at[8*indices_j+2].get() - 4. * couplets.at[8*indices_j+6].get())
                + h2 * (r.at[:,0].get()*Cj_dotr.at[2,:].get() - rdotC_jj_dotr * r.at[:,0].get()*r.at[:,2].get())
                + h3 * (Cj_dotr.at[0,:].get()*r.at[:,2].get() + r.at[:,0].get()*rdotC_j.at[2,:].get() + rdotC_j.at[0,:].get()*r.at[:,2].get()
                        - 6.*rdotC_jj_dotr*r.at[:,0].get()*r.at[:,2].get() - couplets.at[8*indices_j+6].get()))
            
            r_velocity_gradient = r_velocity_gradient.at[indices_j, 6].add(
                h1 * (couplets.at[8*indices_i+2].get() - 4. * couplets.at[8*indices_i+6].get())
                + h2 * (-r.at[:,0].get()*Ci_dotmr.at[2,:].get() - mrdotC_ii_dotmr * r.at[:,0].get()*r.at[:,2].get())
                + h3 * (-Ci_dotmr.at[0,:].get()*r.at[:,2].get() - r.at[:,0].get()*mrdotC_i.at[2,:].get() - mrdotC_i.at[0,:].get()*r.at[:,2].get()
                        - 6.*mrdotC_ii_dotmr*r.at[:,0].get()*r.at[:,2].get() - couplets.at[8*indices_i+6].get()))
            
            
            
            r_velocity_gradient = r_velocity_gradient.at[indices_i, 7].add(
                    (-1) * g1 * (r.at[:,1].get() * forces.at[3*indices_j+2].get() - rdotf_j * r.at[:,1].get()*r.at[:,2].get())
                    + (-1)*g2 * (forces.at[3*indices_j+1].get()*r.at[:,2].get() - 4.*rdotf_j * r.at[:,1].get()*r.at[:,2].get()))
            
            r_velocity_gradient = r_velocity_gradient.at[indices_j, 7].add(
                    (-1) * g1 * (-r.at[:,1].get() * forces.at[3*indices_i+2].get() - mrdotf_i * r.at[:,1].get()*r.at[:,2].get())
                    + (-1)*g2 * (-forces.at[3*indices_i+1].get()*r.at[:,2].get() - 4.*mrdotf_i * r.at[:,1].get()*r.at[:,2].get()))
            
            r_velocity_gradient = r_velocity_gradient.at[indices_i, 7].add(
                h1 * (couplets.at[8*indices_j+3].get() - 4. * couplets.at[8*indices_j+7].get())
                + h2 * (r.at[:,1].get()*Cj_dotr.at[2,:].get() - rdotC_jj_dotr * r.at[:,1].get()*r.at[:,2].get())
                + h3 * (Cj_dotr.at[1,:].get()*r.at[:,2].get() + r.at[:,1].get()*rdotC_j.at[2,:].get() + rdotC_j.at[1,:].get()*r.at[:,2].get()
                        - 6.*rdotC_jj_dotr*r.at[:,1].get()*r.at[:,2].get() - couplets.at[8*indices_j+7].get()))
            
            r_velocity_gradient = r_velocity_gradient.at[indices_j, 7].add(
                h1 * (couplets.at[8*indices_i+3].get() - 4. * couplets.at[8*indices_i+7].get())
                + h2 * (-r.at[:,1].get()*Ci_dotmr.at[2,:].get() - mrdotC_ii_dotmr * r.at[:,1].get()*r.at[:,2].get())
                + h3 * (-Ci_dotmr.at[1,:].get()*r.at[:,2].get() - r.at[:,1].get()*mrdotC_i.at[2,:].get() - mrdotC_i.at[1,:].get()*r.at[:,2].get()
                        - 6.*mrdotC_ii_dotmr*r.at[:,1].get()*r.at[:,2].get() - couplets.at[8*indices_i+7].get()))
            
                
            ##########################################################################################################################################
            ##########################################################################################################################################

            # Add wave and real space part together
            lin_vel = w_lin_velocities + r_lin_velocities
            velocity_gradient = w_velocity_gradient + r_velocity_gradient

            # # Convert to angular velocities and rate of strain
            ang_vel_and_strain = jnp.zeros((N, 8))
            ang_vel_and_strain = ang_vel_and_strain.at[:, 0].set((velocity_gradient.at[:, 3].get()-velocity_gradient.at[:, 7].get()) * 0.5)
            ang_vel_and_strain = ang_vel_and_strain.at[:, 1].set((velocity_gradient.at[:, 6].get()-velocity_gradient.at[:, 2].get()) * 0.5)
            ang_vel_and_strain = ang_vel_and_strain.at[:, 2].set((velocity_gradient.at[:, 1].get()-velocity_gradient.at[:, 5].get()) * 0.5)
            ang_vel_and_strain = ang_vel_and_strain.at[:, 3].set(2*velocity_gradient.at[:, 0].get()+velocity_gradient.at[:, 4].get())
            ang_vel_and_strain = ang_vel_and_strain.at[:, 4].set(velocity_gradient.at[:, 1].get()+velocity_gradient.at[:, 5].get())
            ang_vel_and_strain = ang_vel_and_strain.at[:, 5].set(velocity_gradient.at[:, 2].get()+velocity_gradient.at[:, 6].get())
            ang_vel_and_strain = ang_vel_and_strain.at[:, 6].set(velocity_gradient.at[:, 3].get()+velocity_gradient.at[:, 7].get())
            ang_vel_and_strain = ang_vel_and_strain.at[:, 7].set(velocity_gradient.at[:, 0].get()+2*velocity_gradient.at[:, 4].get())

            # Convert to Generalized Velocities+Strain 
            generalized_velocities = jnp.zeros(11*N) #First 6N entries for U and last 5N for strain rates

            generalized_velocities = generalized_velocities.at[0:6*N:6].set(
                lin_vel.at[:, 0].get())
            generalized_velocities = generalized_velocities.at[1:6*N:6].set(
                lin_vel.at[:, 1].get())
            generalized_velocities = generalized_velocities.at[2:6*N:6].set(
                lin_vel.at[:, 2].get())
            generalized_velocities = generalized_velocities.at[3:6*N:6].set(
                ang_vel_and_strain.at[:, 0].get())
            generalized_velocities = generalized_velocities.at[4:6*N:6].set(
                ang_vel_and_strain.at[:, 1].get())
            generalized_velocities = generalized_velocities.at[5:6*N:6].set(
                ang_vel_and_strain.at[:, 2].get())
            generalized_velocities = generalized_velocities.at[(6*N+0)::5].set(
                ang_vel_and_strain.at[:, 3].get())
            generalized_velocities = generalized_velocities.at[(6*N+1)::5].set(
                ang_vel_and_strain.at[:, 4].get())
            generalized_velocities = generalized_velocities.at[(6*N+2)::5].set(
                ang_vel_and_strain.at[:, 5].get())
            generalized_velocities = generalized_velocities.at[(6*N+3)::5].set(
                ang_vel_and_strain.at[:, 6].get())
            generalized_velocities = generalized_velocities.at[(6*N+4)::5].set(
                ang_vel_and_strain.at[:, 7].get())
            
            #Clean Grids for next iteration
            gridX = jnp.zeros((Nx, Ny, Nz))
            gridY = jnp.zeros((Nx, Ny, Nz))
            gridZ = jnp.zeros((Nx, Ny, Nz))
            gridXX = jnp.zeros((Nx, Ny, Nz))
            gridXY = jnp.zeros((Nx, Ny, Nz))
            gridXZ = jnp.zeros((Nx, Ny, Nz))
            gridYX = jnp.zeros((Nx, Ny, Nz))
            gridYY = jnp.zeros((Nx, Ny, Nz))
            gridYZ = jnp.zeros((Nx, Ny, Nz))
            gridZX = jnp.zeros((Nx, Ny, Nz))
            gridZY = jnp.zeros((Nx, Ny, Nz))
            
            return generalized_velocities

        
        def ComputeLubricationFU(velocities):

            vel_i = (jnp.reshape(velocities,(N,6))).at[indices_i_lub].get()
            vel_j = (jnp.reshape(velocities,(N,6))).at[indices_j_lub].get()
            
            # Dot product of r and U, i.e. axisymmetric projection (minus sign of rj is taken into account at the end of calculation)
            rdui = r_lub.at[:,0].get()*vel_i.at[:,0].get()+r_lub.at[:,1].get()*vel_i.at[:,1].get()+r_lub.at[:,2].get()*vel_i.at[:,2].get()
            rduj = r_lub.at[:,0].get()*vel_j.at[:,0].get()+r_lub.at[:,1].get()*vel_j.at[:,1].get()+r_lub.at[:,2].get()*vel_j.at[:,2].get()
            rdwi = r_lub.at[:,0].get()*vel_i.at[:,3].get()+r_lub.at[:,1].get()*vel_i.at[:,4].get()+r_lub.at[:,2].get()*vel_i.at[:,5].get()
            rdwj = r_lub.at[:,0].get()*vel_j.at[:,3].get()+r_lub.at[:,1].get()*vel_j.at[:,4].get()+r_lub.at[:,2].get()*vel_j.at[:,5].get()

            # Cross product of U and r, i.e. eps_ijk*r_k*U_j = Px dot U, (eps_ijk is the Levi-Civita symbol)
            epsrdui = jnp.array([r_lub.at[:,2].get() * vel_i.at[:,1].get() - r_lub.at[:,1].get() * vel_i.at[:,2].get(),
                                -r_lub.at[:,2].get() * vel_i.at[:,0].get() + r_lub.at[:,0].get() * vel_i.at[:,2].get(),
                                r_lub.at[:,1].get() * vel_i.at[:,0].get() - r_lub.at[:,0].get() * vel_i.at[:,1].get()])
            
            epsrdwi = jnp.array([r_lub.at[:,2].get() * vel_i.at[:,4].get() - r_lub.at[:,1].get() * vel_i.at[:,5].get(), 
                                -r_lub.at[:,2].get() * vel_i.at[:,3].get() + r_lub.at[:,0].get() * vel_i.at[:,5].get(), 
                                r_lub.at[:,1].get() * vel_i.at[:,3].get() - r_lub.at[:,0].get() * vel_i.at[:,4].get()])
            
            epsrduj = jnp.array([r_lub.at[:,2].get() * vel_j.at[:,1].get() - r_lub.at[:,1].get() * vel_j.at[:,2].get(), 
                                -r_lub.at[:,2].get() * vel_j.at[:,0].get() + r_lub.at[:,0].get() * vel_j.at[:,2].get(), 
                                r_lub.at[:,1].get() * vel_j.at[:,0].get() - r_lub.at[:,0].get() * vel_j.at[:,1].get()])
            
            epsrdwj = jnp.array([r_lub.at[:,2].get() * vel_j.at[:,4].get() - r_lub.at[:,1].get() * vel_j.at[:,5].get(), 
                                -r_lub.at[:,2].get() * vel_j.at[:,3].get() + r_lub.at[:,0].get() * vel_j.at[:,5].get(), 
                                r_lub.at[:,1].get() * vel_j.at[:,3].get() - r_lub.at[:,0].get() * vel_j.at[:,4].get()])
            
            forces = jnp.zeros((N,6),float)
            
            # Compute the contributions to the force for particles i (Fi = A11*Ui + A12*Uj + BT11*Wi + BT12*Wj)
            f = ((XA11 - YA11).at[:,None].get() * rdui.at[:,None].get() * r_lub + YA11.at[:,None].get() * vel_i.at[:,:3].get() 
            + (XA12 - YA12).at[:,None].get() * rduj.at[:,None].get() * r_lub + YA12.at[:,None].get() * vel_j.at[:,:3].get() + YB11.at[:,None].get() * (-epsrdwi.T) + YB21.at[:,None].get() * (-epsrdwj.T))
            forces = forces.at[indices_i_lub, :3].add(f)
            # Compute the contributions to the force for particles j (Fj = A11*Uj + A12*Ui + BT11*Wj + BT12*Wi)
            f = ((XA11 - YA11).at[:,None].get() * rduj.at[:,None].get() * r_lub + YA11.at[:,None].get() * vel_j.at[:,:3].get() 
            + (XA12 - YA12).at[:,None].get() * rdui.at[:,None].get() * r_lub + YA12.at[:,None].get() * vel_i.at[:,:3].get() + YB11.at[:,None].get() * (epsrdwj.T) + YB21.at[:,None].get() * (epsrdwi.T))
            forces = forces.at[indices_j_lub, :3].add(f)
            # Compute the contributions to the torque for particles i (Li = B11*Ui + B12*Uj + C11*Wi + C12*Wj)
            l = (YB11.at[:,None].get() * epsrdui.T + YB12.at[:,None].get() * epsrduj.T 
            + (XC11 - YC11).at[:,None].get() * rdwi.at[:,None].get() * r_lub + YC11.at[:,None].get() * vel_i.at[:,3:].get() 
            + (XC12 - YC12).at[:,None].get() * rdwj.at[:,None].get() * r_lub + YC12.at[:,None].get() * vel_j.at[:,3:].get())
            forces = forces.at[indices_i_lub, 3:].add(l)
            # Compute the contributions to the torque for particles j (Lj = B11*Uj + B12*Ui + C11*Wj + C12*Wi)
            l = (-YB11.at[:,None].get() * epsrduj.T - YB12.at[:,None].get() * epsrdui.T 
            + (XC11 - YC11).at[:,None].get() * rdwj.at[:,None].get() * r_lub + YC11.at[:,None].get() * vel_j.at[:,3:].get() 
            + (XC12 - YC12).at[:,None].get() * rdwi.at[:,None].get() * r_lub + YC12.at[:,None].get() * vel_i.at[:,3:].get())           
            forces = forces.at[indices_j_lub, 3:].add(l)
            
            return jnp.ravel(forces)

        
        def compute_saddle(x):

            # set output to zero to start
            Ax = jnp.zeros(N*axis,float)

            # compute far-field contribution (M * F): output velocities+torques (first 6N) and strain rates (last 5N)
            Ax = Ax.at[:11*N].set(GeneralizedMobility(x.at[:11*N].get()))

            # Add B*U (M*F + B*U): modify the first 6N entries of output
            Ax = Ax.at[:6*N].add(x.at[11*N:].get())
                       
            # compute near-field contribution (- R^nf_FU * U)
            Ax = Ax.at[11*N:].set(ComputeLubricationFU(x[11*N:]) * (-1))
            
            # Add (B^T * F - RFU * U): modify the last 6N entries of output
            Ax = Ax.at[11*N:].add(x.at[:6*N].get())
            
            return Ax
        

        def compute_precond(x):

            # set output to zero to start
            Px = jnp.zeros(17*N,float)

            # action of precondition matrix on the first 11*N entries of x is the same as the identity (indeed, the identity is the precondition matrix for the far field granmobility M)
            Px = Px.at[:11*N].set(x[:11*N])

            # -R_FU^-1 * x[:6N]
            # # First solve L^T * y = x, for y, where L^T is the lower triangular Chol. factor of R^nf_FU
            buffer = jscipy.linalg.solve_triangular(R_fu_prec_lower_triang, x.at[:6*N].get(), lower=True)
            # # Then solve L * z = y, for z, where L is the upper triangular Chol. factor of R^nf_FU
            buffer = jscipy.linalg.solve_triangular(jnp.transpose(R_fu_prec_lower_triang), buffer, lower=False)
            Px = Px.at[:6*N].add(-buffer)
            Px = Px.at[11*N:].add(buffer)

            # -R_FU^-1 * x[11N:]
            # # First solve L^T * y = x, for y, where L^T is the lower triangular Chol. factor of R^nf_FU
            buffer = jscipy.linalg.solve_triangular(R_fu_prec_lower_triang, x.at[11*N:].get(), lower=True)
            # # Then solve L * z = y, for z, where L is the upper triangular Chol. factor of R^nf_FU
            buffer = jscipy.linalg.solve_triangular(jnp.transpose(R_fu_prec_lower_triang), buffer, lower=False)
            Px = Px.at[:6*N].add(buffer)
            Px = Px.at[11*N:].add(-buffer)

            return Px
        
        axis = 17
        ###################################################################################################################
        # solve the linear system
        x, exitCode = jscipy.sparse.linalg.gmres(
            A = compute_saddle, b=rhs, tol=1e-5, restart=50, M=compute_precond) 
        return x, exitCode

    def compute_farfield_slipvelocity(
            n_iter_Lanczos_ff,gridk,random_array_wave,random_array_real,
            all_indices_x,all_indices_y,all_indices_z,gaussian_grid_spacing1,gaussian_grid_spacing2,
            r,indices_i,indices_j,f1,f2,g1,g2,h1,h2,h3
            ):
        
        @jit
        def helper_Mpsi(random_array_real):
            
            #input is already in the format: [Forces, Torque+Stresslet] (not in the generalized format [Force+Torque,Stresslet] like in the saddle point solver)
            forces = random_array_real.at[:3*N].get()
            
            couplets = jnp.zeros(8*N)
            couplets = couplets.at[::8].set(random_array_real.at[(3*N+3)::8].get())  # C[0] = S[0]
            couplets = couplets.at[1::8].set(
                random_array_real.at[(3*N+4)::8].get()+random_array_real.at[(3*N+2)::8].get()*0.5)  # C[1] = S[1] + L[2]/2
            couplets = couplets.at[2::8].set(
                random_array_real.at[(3*N+5)::8].get()-random_array_real.at[(3*N+1)::8].get()*0.5)  # C[2] = S[2] - L[1]/2
            couplets = couplets.at[3::8].set(
                random_array_real.at[(3*N+6)::8].get()+random_array_real.at[(3*N+0)::8].get()*0.5)  # C[3] = S[3] + L[0]/2
            couplets = couplets.at[4::8].set(random_array_real.at[(3*N+7)::8].get())  # C[4] = S[4]
            couplets = couplets.at[5::8].set(
                random_array_real.at[(3*N+4)::8].get()-random_array_real.at[(3*N+2)::8].get()*0.5)  # C[5] = S[1] - L[2]/2
            couplets = couplets.at[6::8].set(
                random_array_real.at[(3*N+5)::8].get()+random_array_real.at[(3*N+1)::8].get()*0.5)  # C[6] = S[2] + L[1]/2
            couplets = couplets.at[7::8].set(
                random_array_real.at[(3*N+6)::8].get()-random_array_real.at[(3*N+0)::8].get()*0.5)  # C[7] = S[3] - L[0]/2
            
            
            # Allocate arrays for Linear velocities and Velocity gradients (from which we can extract angular velocities and rate of strain)
            r_lin_velocities = jnp.zeros((N, 3),float)
            r_velocity_gradient = jnp.zeros((N, 8),float)
            
            # SELF CONTRIBUTIONS
            r_lin_velocities = r_lin_velocities.at[:, 0].set(
                m_self.at[0].get() * forces.at[0::3].get())
            r_lin_velocities = r_lin_velocities.at[:, 1].set(
                m_self.at[0].get() * forces.at[1::3].get())
            r_lin_velocities = r_lin_velocities.at[:, 2].set(
                m_self.at[0].get() * forces.at[2::3].get())
            
            r_velocity_gradient = r_velocity_gradient.at[:, 0].set(
                m_self.at[1].get()*(couplets.at[0::8].get() - 4 * couplets.at[0::8].get()))
            r_velocity_gradient = r_velocity_gradient.at[:, 5].set(
                m_self.at[1].get()*(couplets.at[1::8].get() - 4 * couplets.at[5::8].get()))
            r_velocity_gradient = r_velocity_gradient.at[:, 6].set(
                m_self.at[1].get()*(couplets.at[2::8].get() - 4 * couplets.at[6::8].get()))
            r_velocity_gradient = r_velocity_gradient.at[:, 7].set(
                m_self.at[1].get()*(couplets.at[3::8].get() - 4 * couplets.at[7::8].get()))
            
            r_velocity_gradient = r_velocity_gradient.at[:, 4].set(
                m_self.at[1].get()*(couplets.at[4::8].get() - 4 * couplets.at[4::8].get()))
            r_velocity_gradient = r_velocity_gradient.at[:, 1].set(
                m_self.at[1].get()*(couplets.at[5::8].get() - 4 * couplets.at[1::8].get()))
            r_velocity_gradient = r_velocity_gradient.at[:, 2].set(
                m_self.at[1].get()*(couplets.at[6::8].get() - 4 * couplets.at[2::8].get()))
            r_velocity_gradient = r_velocity_gradient.at[:, 3].set(
                m_self.at[1].get()*(couplets.at[7::8].get() - 4 * couplets.at[3::8].get()))
            
            # # Geometric quantities
            rdotf_j =   (r.at[:,0].get() * forces.at[3*indices_j + 0].get() + r.at[:,1].get() * forces.at[3*indices_j + 1].get() + r.at[:,2].get() * forces.at[3*indices_j + 2].get())
            mrdotf_i = -(r.at[:,0].get() * forces.at[3*indices_i + 0].get() + r.at[:,1].get() * forces.at[3*indices_i + 1].get() + r.at[:,2].get() * forces.at[3*indices_i + 2].get())
            
            Cj_dotr = jnp.array( [couplets.at[8*indices_j + 0].get() * r.at[:,0].get() + couplets.at[8*indices_j + 1].get() * r.at[:,1].get() + couplets.at[8*indices_j + 2].get() * r.at[:,2].get(),
                                  couplets.at[8*indices_j + 5].get() * r.at[:,0].get() + couplets.at[8*indices_j + 4].get() * r.at[:,1].get() + couplets.at[8*indices_j + 3].get() * r.at[:,2].get(),
                                  couplets.at[8*indices_j + 6].get() * r.at[:,0].get() + couplets.at[8*indices_j + 7].get() * r.at[:,1].get() -(couplets.at[8*indices_j + 0].get() + couplets.at[8*indices_j + 4].get()) * r.at[:,2].get()])
            
            Ci_dotmr=jnp.array( [-couplets.at[8*indices_i + 0].get() * r.at[:,0].get() - couplets.at[8*indices_i + 1].get() * r.at[:,1].get() - couplets.at[8*indices_i + 2].get() * r.at[:,2].get(),
                                  -couplets.at[8*indices_i + 5].get() * r.at[:,0].get() - couplets.at[8*indices_i + 4].get() * r.at[:,1].get() - couplets.at[8*indices_i + 3].get() * r.at[:,2].get(),
                                  -couplets.at[8*indices_i + 6].get() * r.at[:,0].get() - couplets.at[8*indices_i + 7].get() * r.at[:,1].get() +(couplets.at[8*indices_i + 0].get() + couplets.at[8*indices_i + 4].get()) * r.at[:,2].get()])


            rdotC_j = jnp.array([couplets.at[8*indices_j + 0].get() * r.at[:,0].get() + couplets.at[8*indices_j + 5].get() * r.at[:,1].get() + couplets.at[8*indices_j + 6].get() * r.at[:,2].get(),
                                  couplets.at[8*indices_j + 1].get() * r.at[:,0].get() + couplets.at[8*indices_j + 4].get() * r.at[:,1].get() + couplets.at[8*indices_j + 7].get() * r.at[:,2].get(),
                                  couplets.at[8*indices_j + 2].get() * r.at[:,0].get() + couplets.at[8*indices_j + 3].get() * r.at[:,1].get() -(couplets.at[8*indices_j + 0].get() + couplets.at[8*indices_j + 4].get()) * r.at[:,2].get()])
            
            mrdotC_i=jnp.array([-couplets.at[8*indices_i + 0].get() * r.at[:,0].get() - couplets.at[8*indices_i + 5].get() * r.at[:,1].get() - couplets.at[8*indices_i + 6].get() * r.at[:,2].get(),
                                  -couplets.at[8*indices_i + 1].get() * r.at[:,0].get() - couplets.at[8*indices_i + 4].get() * r.at[:,1].get() - couplets.at[8*indices_i + 7].get() * r.at[:,2].get(),
                                  -couplets.at[8*indices_i + 2].get() * r.at[:,0].get() - couplets.at[8*indices_i + 3].get() * r.at[:,1].get() +(couplets.at[8*indices_i + 0].get() + couplets.at[8*indices_i + 4].get()) * r.at[:,2].get()])
                        
            rdotC_jj_dotr   =  (r.at[:,0].get()*Cj_dotr.at[0,:].get()  + r.at[:,1].get()*Cj_dotr.at[1,:].get()  + r.at[:,2].get()*Cj_dotr.at[2,:].get())
            mrdotC_ii_dotmr = -(r.at[:,0].get()*Ci_dotmr.at[0,:].get() + r.at[:,1].get()*Ci_dotmr.at[1,:].get() + r.at[:,2].get()*Ci_dotmr.at[2,:].get())
            
            
            # Compute Velocity for particles i
            r_lin_velocities = r_lin_velocities.at[indices_i, 0].add(f1 * forces.at[3*indices_j].get() + (f2 - f1) * rdotf_j * r.at[:,0].get())
            r_lin_velocities = r_lin_velocities.at[indices_i, 1].add(f1 * forces.at[3*indices_j+1].get() + (f2 - f1) * rdotf_j * r.at[:,1].get())
            r_lin_velocities = r_lin_velocities.at[indices_i, 2].add(f1 * forces.at[3*indices_j+2].get() + (f2 - f1) * rdotf_j * r.at[:,2].get())
            r_lin_velocities = r_lin_velocities.at[indices_i, 0].add(g1 * (Cj_dotr.at[0,:].get() - rdotC_jj_dotr * r.at[:,0].get()) + g2 * (rdotC_j.at[0,:].get() - 4.*rdotC_jj_dotr * r.at[:,0].get()))
            r_lin_velocities = r_lin_velocities.at[indices_i, 1].add(g1 * (Cj_dotr.at[1,:].get() - rdotC_jj_dotr * r.at[:,1].get()) + g2 * (rdotC_j.at[1,:].get() - 4.*rdotC_jj_dotr * r.at[:,1].get()))
            r_lin_velocities = r_lin_velocities.at[indices_i, 2].add(g1 * (Cj_dotr.at[2,:].get() - rdotC_jj_dotr * r.at[:,2].get()) + g2 * (rdotC_j.at[2,:].get() - 4.*rdotC_jj_dotr * r.at[:,2].get()))
            # Compute Velocity for particles j
            r_lin_velocities = r_lin_velocities.at[indices_j, 0].add(f1 * forces.at[3*indices_i].get() - (f2 - f1) * mrdotf_i * r.at[:,0].get())
            r_lin_velocities = r_lin_velocities.at[indices_j, 1].add(f1 * forces.at[3*indices_i+1].get() - (f2 - f1) * mrdotf_i * r.at[:,1].get())
            r_lin_velocities = r_lin_velocities.at[indices_j, 2].add(f1 * forces.at[3*indices_i+2].get() - (f2 - f1) * mrdotf_i * r.at[:,2].get())
            r_lin_velocities = r_lin_velocities.at[indices_j, 0].add(g1 * (Ci_dotmr.at[0,:].get() + mrdotC_ii_dotmr * r.at[:,0].get()) + g2 * (mrdotC_i.at[0,:].get() + 4.*mrdotC_ii_dotmr * r.at[:,0].get()))
            r_lin_velocities = r_lin_velocities.at[indices_j, 1].add(g1 * (Ci_dotmr.at[1,:].get() + mrdotC_ii_dotmr * r.at[:,1].get()) + g2 * (mrdotC_i.at[1,:].get() + 4.*mrdotC_ii_dotmr * r.at[:,1].get()))
            r_lin_velocities = r_lin_velocities.at[indices_j, 2].add(g1 * (Ci_dotmr.at[2,:].get() + mrdotC_ii_dotmr * r.at[:,2].get()) + g2 * (mrdotC_i.at[2,:].get() + 4.*mrdotC_ii_dotmr * r.at[:,2].get()))
            
            
            # Compute Velocity Gradient for particles i and j
            r_velocity_gradient = r_velocity_gradient.at[indices_i, 0].add(
                (-1) * g1 * (r.at[:,0].get() * forces.at[3*indices_j+0].get() - rdotf_j * r.at[:,0].get()*r.at[:,0].get())
                +(-1)*g2 * (rdotf_j + forces.at[3*indices_j+0].get()*r.at[:,0].get() - 4.*rdotf_j * r.at[:,0].get()*r.at[:,0].get()))
            
            r_velocity_gradient = r_velocity_gradient.at[indices_j, 0].add(
                (-1) * g1 * (-r.at[:,0].get() * forces.at[3*indices_i+0].get() - mrdotf_i * r.at[:,0].get()*r.at[:,0].get()) 
                +(-1)*g2 * (mrdotf_i - forces.at[3*indices_i+0].get()*r.at[:,0].get() - 4.*mrdotf_i * r.at[:,0].get()*r.at[:,0].get()))
            r_velocity_gradient = r_velocity_gradient.at[indices_i, 0].add(
                h1 * (couplets.at[8*indices_j+0].get() - 4. * couplets.at[8*indices_j+0].get()) 
                + h2 * (r.at[:,0].get()*Cj_dotr.at[0,:].get() - rdotC_jj_dotr * r.at[:,0].get()*r.at[:,0].get()) 
                + h3 * (rdotC_jj_dotr + Cj_dotr.at[0,:].get()*r.at[:,0].get() + r.at[:,0].get()*rdotC_j.at[0,:].get() + rdotC_j.at[0,:].get()*r.at[:,0].get() 
                                                                        - 6.*rdotC_jj_dotr*r.at[:,0].get()*r.at[:,0].get() - couplets.at[8*indices_j+0].get()))
            r_velocity_gradient = r_velocity_gradient.at[indices_j, 0].add(
                h1 * (couplets.at[8*indices_i+0].get() - 4. * couplets.at[8*indices_i+0].get()) 
                + h2 * (-r.at[:,0].get()*Ci_dotmr.at[0,:].get() - mrdotC_ii_dotmr * r.at[:,0].get()*r.at[:,0].get())
                + h3 * (mrdotC_ii_dotmr - Ci_dotmr.at[0,:].get()*r.at[:,0].get() - r.at[:,0].get()*mrdotC_i.at[0,:].get() - mrdotC_i.at[0,:].get()*r.at[:,0].get() 
                                                                        - 6.*mrdotC_ii_dotmr*r.at[:,0].get()*r.at[:,0].get() - couplets.at[8*indices_i+0].get()))



            r_velocity_gradient = r_velocity_gradient.at[indices_i, 1].add(
                (-1) * g1 * (r.at[:,1].get() * forces.at[3*indices_j+0].get() - rdotf_j * r.at[:,1].get()*r.at[:,0].get()) 
                + (-1)*g2 * (forces.at[3*indices_j+1].get()*r.at[:,0].get() - 4.*rdotf_j * r.at[:,1].get()*r.at[:,0].get()))
            
            r_velocity_gradient = r_velocity_gradient.at[indices_j, 1].add(
                (-1) * g1 * (-r.at[:,1].get() * forces.at[3*indices_i+0].get() - mrdotf_i * r.at[:,1].get()*r.at[:,0].get()) 
                + (-1)*g2 * (-forces.at[3*indices_i+1].get()*r.at[:,0].get() - 4.*mrdotf_i * r.at[:,1].get()*r.at[:,0].get()))
            
            r_velocity_gradient = r_velocity_gradient.at[indices_i, 1].add(
                h1 * (couplets.at[8*indices_j+5].get() - 4. * couplets.at[8*indices_j+1].get()) 
                + h2 * (r.at[:,1].get()*Cj_dotr.at[0,:].get() - rdotC_jj_dotr * r.at[:,1].get()*r.at[:,0].get())
                + h3 * (Cj_dotr.at[1,:].get()*r.at[:,0].get() + r.at[:,1].get()*rdotC_j.at[0,:].get() + rdotC_j.at[1,:].get()*r.at[:,0].get()
                                                    - 6.*rdotC_jj_dotr*r.at[:,1].get()*r.at[:,0].get() - couplets.at[8*indices_j+1].get()))
            
            r_velocity_gradient = r_velocity_gradient.at[indices_j, 1].add(
                h1 * (couplets.at[8*indices_i+5].get() - 4. * couplets.at[8*indices_i+1].get())
                + h2 * (-r.at[:,1].get()*Ci_dotmr.at[0,:].get() - mrdotC_ii_dotmr * r.at[:,1].get()*r.at[:,0].get())
                + h3 * (-Ci_dotmr.at[1,:].get()*r.at[:,0].get() - r.at[:,1].get()*mrdotC_i.at[0,:].get() - mrdotC_i.at[1,:].get()*r.at[:,0].get()
                                                        - 6.*mrdotC_ii_dotmr*r.at[:,1].get()*r.at[:,0].get() - couplets.at[8*indices_i+1].get()))



            r_velocity_gradient = r_velocity_gradient.at[indices_i, 2].add(
                      (-1) * g1 * (r.at[:,2].get() * forces.at[3*indices_j+0].get() - rdotf_j * r.at[:,2].get()*r.at[:,0].get()) 
                      + (-1)*g2 * (forces.at[3*indices_j+2].get()*r.at[:,0].get() - 4.*rdotf_j * r.at[:,2].get()*r.at[:,0].get()))
            
            r_velocity_gradient = r_velocity_gradient.at[indices_j, 2].add(
                      (-1) * g1 * (-r.at[:,2].get() * forces.at[3*indices_i+0].get() - mrdotf_i * r.at[:,2].get()*r.at[:,0].get()) 
                      + (-1)*g2 * (-forces.at[3*indices_i+2].get()*r.at[:,0].get() - 4.*mrdotf_i * r.at[:,2].get()*r.at[:,0].get()))
            r_velocity_gradient = r_velocity_gradient.at[indices_i, 2].add(
                h1 * (couplets.at[8*indices_j+6].get() - 4. * couplets.at[8*indices_j+2].get())
                + h2 * (r.at[:,2].get()*Cj_dotr.at[0,:].get() - rdotC_jj_dotr * r.at[:,2].get()*r.at[:,0].get())
                + h3 * (Cj_dotr.at[2,:].get()*r.at[:,0].get() + r.at[:,2].get()*rdotC_j.at[0,:].get() + rdotC_j.at[2,:].get()*r.at[:,0].get()
                        - 6.*rdotC_jj_dotr*r.at[:,2].get()*r.at[:,0].get() - couplets.at[8*indices_j+2].get()))
            r_velocity_gradient = r_velocity_gradient.at[indices_j, 2].add(
                h1 * (couplets.at[8*indices_i+6].get() - 4. * couplets.at[8*indices_i+2].get())
                + h2 * (r.at[:,2].get()*Ci_dotmr.at[0,:].get()*(-1) - mrdotC_ii_dotmr * r.at[:,2].get()*r.at[:,0].get())
                + h3 * (-Ci_dotmr.at[2,:].get()*r.at[:,0].get() - r.at[:,2].get()*mrdotC_i.at[0,:].get() - mrdotC_i.at[2,:].get()*r.at[:,0].get()
                        - 6.*mrdotC_ii_dotmr*r.at[:,2].get()*r.at[:,0].get() - couplets.at[8*indices_i+2].get()))
           
        
        
        
            r_velocity_gradient = r_velocity_gradient.at[indices_i, 3].add(
                    (-1) * g1 * (r.at[:,2].get() * forces.at[3*indices_j+1].get() - rdotf_j * r.at[:,2].get()*r.at[:,1].get())
                    + (-1)*g2 * (forces.at[3*indices_j+2].get()*r.at[:,1].get() - 4.*rdotf_j * r.at[:,2].get()*r.at[:,1].get()))
            
            r_velocity_gradient = r_velocity_gradient.at[indices_j, 3].add(
                    (-1) * g1 * (-r.at[:,2].get() * forces.at[3*indices_i+1].get() - mrdotf_i * r.at[:,2].get()*r.at[:,1].get())
                    + (-1)*g2 * (-forces.at[3*indices_i+2].get()*r.at[:,1].get() - 4.*mrdotf_i * r.at[:,2].get()*r.at[:,1].get()))
            
            r_velocity_gradient = r_velocity_gradient.at[indices_i, 3].add(
                h1 * (couplets.at[8*indices_j+7].get() - 4. * couplets.at[8*indices_j+3].get())
                + h2 * (r.at[:,2].get()*Cj_dotr.at[1,:].get() - rdotC_jj_dotr * r.at[:,2].get()*r.at[:,1].get())
                + h3 * (Cj_dotr.at[2,:].get()*r.at[:,1].get() + r.at[:,2].get()*rdotC_j.at[1,:].get() + rdotC_j.at[2,:].get()*r.at[:,1].get()
                        - 6.*rdotC_jj_dotr*r.at[:,2].get()*r.at[:,1].get() - couplets.at[8*indices_j+3].get()))

            r_velocity_gradient = r_velocity_gradient.at[indices_j, 3].add(
                h1 * (couplets.at[8*indices_i+7].get() - 4. * couplets.at[8*indices_i+3].get())
                + h2 * (-r.at[:,2].get()*Ci_dotmr.at[1,:].get() - mrdotC_ii_dotmr * r.at[:,2].get()*r.at[:,1].get()) 
                + h3 * (-Ci_dotmr.at[2,:].get()*r.at[:,1].get() - r.at[:,2].get()*mrdotC_i.at[1,:].get() - mrdotC_i.at[2,:].get()*r.at[:,1].get()
                        - 6.*mrdotC_ii_dotmr*r.at[:,2].get()*r.at[:,1].get() - couplets.at[8*indices_i+3].get()))



            r_velocity_gradient = r_velocity_gradient.at[indices_i, 4].add(
                    (-1) * g1 * (r.at[:,1].get() * forces.at[3*indices_j+1].get() - rdotf_j * r.at[:,1].get()*r.at[:,1].get())
                    + (-1)*g2 * (rdotf_j + forces.at[3*indices_j+1].get()*r.at[:,1].get() - 4.*rdotf_j * r.at[:,1].get()*r.at[:,1].get()))
            
            r_velocity_gradient = r_velocity_gradient.at[indices_j, 4].add(
                    (-1) * g1 * (-r.at[:,1].get() * forces.at[3*indices_i+1].get() - mrdotf_i * r.at[:,1].get()*r.at[:,1].get())
                    + (-1)*g2 * (mrdotf_i - forces.at[3*indices_i+1].get()*r.at[:,1].get() - 4.*mrdotf_i * r.at[:,1].get()*r.at[:,1].get()))
             
            r_velocity_gradient = r_velocity_gradient.at[indices_i, 4].add(
                h1 * (couplets.at[8*indices_j+4].get() - 4. * couplets.at[8*indices_j+4].get())
                + h2 * (r.at[:,1].get()*Cj_dotr.at[1,:].get() - rdotC_jj_dotr * r.at[:,1].get()*r.at[:,1].get()) 
                + h3 * (rdotC_jj_dotr + Cj_dotr.at[1,:].get()*r.at[:,1].get() + r.at[:,1].get()*rdotC_j.at[1,:].get() + rdotC_j.at[1,:].get()*r.at[:,1].get()
                        - 6.*rdotC_jj_dotr*r.at[:,1].get()*r.at[:,1].get() - couplets.at[8*indices_j+4].get()))
            
            r_velocity_gradient = r_velocity_gradient.at[indices_j, 4].add(
                h1 * (couplets.at[8*indices_i+4].get() - 4. * couplets.at[8*indices_i+4].get())
                + h2 * (-r.at[:,1].get()*Ci_dotmr.at[1,:].get() - mrdotC_ii_dotmr * r.at[:,1].get()*r.at[:,1].get())
                + h3 * (mrdotC_ii_dotmr - Ci_dotmr.at[1,:].get()*r.at[:,1].get() - r.at[:,1].get()*mrdotC_i.at[1,:].get() - mrdotC_i.at[1,:].get()*r.at[:,1].get()
                        - 6.*mrdotC_ii_dotmr*r.at[:,1].get()*r.at[:,1].get() - couplets.at[8*indices_i+4].get()))



            r_velocity_gradient = r_velocity_gradient.at[indices_i, 5].add(
                    (-1) * g1 * (r.at[:,0].get() * forces.at[3*indices_j+1].get() - rdotf_j * r.at[:,0].get()*r.at[:,1].get())
                    + (-1)*g2 * (forces.at[3*indices_j+0].get()*r.at[:,1].get() - 4.*rdotf_j * r.at[:,0].get()*r.at[:,1].get()))
            
            r_velocity_gradient = r_velocity_gradient.at[indices_j, 5].add(
                    (-1) * g1 * (-r.at[:,0].get() * forces.at[3*indices_i+1].get() - mrdotf_i * r.at[:,0].get()*r.at[:,1].get())
                    + (-1)*g2 * (-forces.at[3*indices_i+0].get()*r.at[:,1].get() - 4.*mrdotf_i * r.at[:,0].get()*r.at[:,1].get()))
            
            r_velocity_gradient = r_velocity_gradient.at[indices_i, 5].add(
                h1 * (couplets.at[8*indices_j+1].get() - 4. * couplets.at[8*indices_j+5].get())
                + h2 * (r.at[:,0].get()*Cj_dotr.at[1,:].get() - rdotC_jj_dotr * r.at[:,0].get()*r.at[:,1].get())
                + h3 * (Cj_dotr.at[0,:].get()*r.at[:,1].get() + r.at[:,0].get()*rdotC_j.at[1,:].get() + rdotC_j.at[0,:].get()*r.at[:,1].get()
                        - 6.*rdotC_jj_dotr*r.at[:,0].get()*r.at[:,1].get() - couplets.at[8*indices_j+5].get()))
            
            r_velocity_gradient = r_velocity_gradient.at[indices_j, 5].add(
                h1 * (couplets.at[8*indices_i+1].get() - 4. * couplets.at[8*indices_i+5].get())
                + h2 * (-r.at[:,0].get()*Ci_dotmr.at[1,:].get() - mrdotC_ii_dotmr * r.at[:,0].get()*r.at[:,1].get())
                + h3 * (-Ci_dotmr.at[0,:].get()*r.at[:,1].get() - r.at[:,0].get()*mrdotC_i.at[1,:].get() - mrdotC_i.at[0,:].get()*r.at[:,1].get()
                        - 6.*mrdotC_ii_dotmr*r.at[:,0].get()*r.at[:,1].get() - couplets.at[8*indices_i+5].get()))



            r_velocity_gradient = r_velocity_gradient.at[indices_i, 6].add(
                      (-1) * g1 * (r.at[:,0].get() * forces.at[3*indices_j+2].get() - rdotf_j * r.at[:,0].get()*r.at[:,2].get())
                      + (-1)*g2 * (forces.at[3*indices_j+0].get()*r.at[:,2].get() - 4.*rdotf_j * r.at[:,0].get()*r.at[:,2].get()))
            
            r_velocity_gradient = r_velocity_gradient.at[indices_j, 6].add(
                      (-1) * g1 * (-r.at[:,0].get() * forces.at[3*indices_i+2].get() - mrdotf_i * r.at[:,0].get()*r.at[:,2].get())
                      + (-1)*g2 * (-forces.at[3*indices_i+0].get()*r.at[:,2].get() - 4.*mrdotf_i * r.at[:,0].get()*r.at[:,2].get()))
             
            r_velocity_gradient = r_velocity_gradient.at[indices_i, 6].add(
                h1 * (couplets.at[8*indices_j+2].get() - 4. * couplets.at[8*indices_j+6].get())
                + h2 * (r.at[:,0].get()*Cj_dotr.at[2,:].get() - rdotC_jj_dotr * r.at[:,0].get()*r.at[:,2].get())
                + h3 * (Cj_dotr.at[0,:].get()*r.at[:,2].get() + r.at[:,0].get()*rdotC_j.at[2,:].get() + rdotC_j.at[0,:].get()*r.at[:,2].get()
                        - 6.*rdotC_jj_dotr*r.at[:,0].get()*r.at[:,2].get() - couplets.at[8*indices_j+6].get()))
            
            r_velocity_gradient = r_velocity_gradient.at[indices_j, 6].add(
                h1 * (couplets.at[8*indices_i+2].get() - 4. * couplets.at[8*indices_i+6].get())
                + h2 * (-r.at[:,0].get()*Ci_dotmr.at[2,:].get() - mrdotC_ii_dotmr * r.at[:,0].get()*r.at[:,2].get())
                + h3 * (-Ci_dotmr.at[0,:].get()*r.at[:,2].get() - r.at[:,0].get()*mrdotC_i.at[2,:].get() - mrdotC_i.at[0,:].get()*r.at[:,2].get()
                        - 6.*mrdotC_ii_dotmr*r.at[:,0].get()*r.at[:,2].get() - couplets.at[8*indices_i+6].get()))
            
            
            
            r_velocity_gradient = r_velocity_gradient.at[indices_i, 7].add(
                    (-1) * g1 * (r.at[:,1].get() * forces.at[3*indices_j+2].get() - rdotf_j * r.at[:,1].get()*r.at[:,2].get())
                    + (-1)*g2 * (forces.at[3*indices_j+1].get()*r.at[:,2].get() - 4.*rdotf_j * r.at[:,1].get()*r.at[:,2].get()))
            
            r_velocity_gradient = r_velocity_gradient.at[indices_j, 7].add(
                    (-1) * g1 * (-r.at[:,1].get() * forces.at[3*indices_i+2].get() - mrdotf_i * r.at[:,1].get()*r.at[:,2].get())
                    + (-1)*g2 * (-forces.at[3*indices_i+1].get()*r.at[:,2].get() - 4.*mrdotf_i * r.at[:,1].get()*r.at[:,2].get()))
            
            r_velocity_gradient = r_velocity_gradient.at[indices_i, 7].add(
                h1 * (couplets.at[8*indices_j+3].get() - 4. * couplets.at[8*indices_j+7].get())
                + h2 * (r.at[:,1].get()*Cj_dotr.at[2,:].get() - rdotC_jj_dotr * r.at[:,1].get()*r.at[:,2].get())
                + h3 * (Cj_dotr.at[1,:].get()*r.at[:,2].get() + r.at[:,1].get()*rdotC_j.at[2,:].get() + rdotC_j.at[1,:].get()*r.at[:,2].get()
                        - 6.*rdotC_jj_dotr*r.at[:,1].get()*r.at[:,2].get() - couplets.at[8*indices_j+7].get()))
            
            r_velocity_gradient = r_velocity_gradient.at[indices_j, 7].add(
                h1 * (couplets.at[8*indices_i+3].get() - 4. * couplets.at[8*indices_i+7].get())
                + h2 * (-r.at[:,1].get()*Ci_dotmr.at[2,:].get() - mrdotC_ii_dotmr * r.at[:,1].get()*r.at[:,2].get())
                + h3 * (-Ci_dotmr.at[1,:].get()*r.at[:,2].get() - r.at[:,1].get()*mrdotC_i.at[2,:].get() - mrdotC_i.at[1,:].get()*r.at[:,2].get()
                        - 6.*mrdotC_ii_dotmr*r.at[:,1].get()*r.at[:,2].get() - couplets.at[8*indices_i+7].get()))
            
            # # Convert to angular velocities and rate of strain
            r_ang_vel_and_strain = jnp.zeros((N, 8))
            r_ang_vel_and_strain = r_ang_vel_and_strain.at[:, 0].set((r_velocity_gradient.at[:, 3].get()-r_velocity_gradient.at[:, 7].get()) * 0.5)
            r_ang_vel_and_strain = r_ang_vel_and_strain.at[:, 1].set((r_velocity_gradient.at[:, 6].get()-r_velocity_gradient.at[:, 2].get()) * 0.5)
            r_ang_vel_and_strain = r_ang_vel_and_strain.at[:, 2].set((r_velocity_gradient.at[:, 1].get()-r_velocity_gradient.at[:, 5].get()) * 0.5)
            r_ang_vel_and_strain = r_ang_vel_and_strain.at[:, 3].set(2*r_velocity_gradient.at[:, 0].get()+r_velocity_gradient.at[:, 4].get())
            r_ang_vel_and_strain = r_ang_vel_and_strain.at[:, 4].set(r_velocity_gradient.at[:, 1].get()+r_velocity_gradient.at[:, 5].get())
            r_ang_vel_and_strain = r_ang_vel_and_strain.at[:, 5].set(r_velocity_gradient.at[:, 2].get()+r_velocity_gradient.at[:, 6].get())
            r_ang_vel_and_strain = r_ang_vel_and_strain.at[:, 6].set(r_velocity_gradient.at[:, 3].get()+r_velocity_gradient.at[:, 7].get())
            r_ang_vel_and_strain = r_ang_vel_and_strain.at[:, 7].set(r_velocity_gradient.at[:, 0].get()+2*r_velocity_gradient.at[:, 4].get())            
            return r_lin_velocities, r_ang_vel_and_strain
        
        @jit
        def helper_reshape(lin_vel, ang_vel_and_strain):
            reshaped_array = jnp.zeros(11*N)
            reshaped_array = reshaped_array.at[:3*N].set(jnp.reshape(lin_vel,3*N))
            reshaped_array = reshaped_array.at[3*N:].set(jnp.reshape(ang_vel_and_strain,8*N))
            return reshaped_array

        @jit
        def helper_wavespace_calc(random_array_wave):
            gridX = jnp.zeros((Nx, Ny, Nz))
            gridY = jnp.zeros((Nx, Ny, Nz))
            gridZ = jnp.zeros((Nx, Ny, Nz))
            gridXX = jnp.zeros((Nx, Ny, Nz))
            gridXY = jnp.zeros((Nx, Ny, Nz))
            gridXZ = jnp.zeros((Nx, Ny, Nz))
            gridYX = jnp.zeros((Nx, Ny, Nz))
            gridYY = jnp.zeros((Nx, Ny, Nz))
            gridYZ = jnp.zeros((Nx, Ny, Nz))
            gridZX = jnp.zeros((Nx, Ny, Nz))
            gridZY = jnp.zeros((Nx, Ny, Nz))
            
            ### WAVE SPACE part
            gridX = jnp.array(gridX, complex)
            gridY = jnp.array(gridX, complex)
            gridZ = jnp.array(gridX, complex)
     
            fac = jnp.sqrt( 3.0 * kT / dt / (gridh[0] * gridh[1] * gridh[2]) )
            random_array_wave = (2*random_array_wave-1) * fac

            gridX = gridX.at[normal_indices[:,0],normal_indices[:,1],normal_indices[:,2]].set(
                random_array_wave.at[:len(normal_indices)].get() + 1j * random_array_wave.at[len(normal_indices):2*len(normal_indices)].get() )
            gridX = gridX.at[normal_conj_indices[:,0],normal_conj_indices[:,1],normal_conj_indices[:,2]].set(
                random_array_wave.at[:len(normal_indices)].get() - 1j * random_array_wave.at[len(normal_indices):2*len(normal_indices)].get() )
            gridX = gridX.at[nyquist_indices[:,0],nyquist_indices[:,1],nyquist_indices[:,2]].set(
                random_array_wave.at[6*len(normal_indices):(6*len(normal_indices)+ 1 * len(nyquist_indices))].get() *1.414213562373095 + 0j )

            gridY = gridY.at[normal_indices[:,0],normal_indices[:,1],normal_indices[:,2]].set(
                random_array_wave.at[(2*len(normal_indices)):(3*len(normal_indices))].get() + 1j * random_array_wave.at[(3*len(normal_indices)):(4*len(normal_indices))].get() )
            gridY = gridY.at[normal_conj_indices[:,0],normal_conj_indices[:,1],normal_conj_indices[:,2]].set(
                random_array_wave.at[(2*len(normal_indices)):(3*len(normal_indices))].get() - 1j * random_array_wave.at[(3*len(normal_indices)):(4*len(normal_indices))].get() )
            gridY = gridY.at[nyquist_indices[:,0],nyquist_indices[:,1],nyquist_indices[:,2]].set(
                random_array_wave.at[(6*len(normal_indices)+ 1 * len(nyquist_indices)):(6*len(normal_indices)+ 2 * len(nyquist_indices))].get() *1.414213562373095 + 0j )

            gridZ = gridZ.at[normal_indices[:,0],normal_indices[:,1],normal_indices[:,2]].set(
                random_array_wave.at[(4*len(normal_indices)):(5*len(normal_indices))].get() + 1j * random_array_wave.at[(5*len(normal_indices)):(6*len(normal_indices))].get() )
            gridZ = gridZ.at[normal_conj_indices[:,0],normal_conj_indices[:,1],normal_conj_indices[:,2]].set(
                random_array_wave.at[(4*len(normal_indices)):(5*len(normal_indices))].get() - 1j * random_array_wave.at[(5*len(normal_indices)):(6*len(normal_indices))].get() )
            gridZ = gridZ.at[nyquist_indices[:,0],nyquist_indices[:,1],nyquist_indices[:,2]].set(
                random_array_wave.at[(6*len(normal_indices)+ 2 * len(nyquist_indices)):(6*len(normal_indices)+ 3 * len(nyquist_indices))].get() *1.414213562373095 + 0j )
            
            #Compute k^2 and (|| k ||)
            gridk_sqr = (gridk.at[:, :, :, 0].get()*gridk.at[:, :, :, 0].get()+gridk.at[:, :, :, 1].get()
                         * gridk.at[:, :, :, 1].get()+gridk.at[:, :, :, 2].get()*gridk.at[:, :, :, 2].get())
            gridk_mod = jnp.sqrt(gridk_sqr)
            
            #Scaling factors
            B = jnp.where(gridk_mod > 0, jnp.sqrt(gridk.at[:, :, :, 3].get()), 0.)
            SU = jnp.where(gridk_mod > 0, jnp.sin(gridk_mod) / gridk_mod   , 0.)        
            SD = jnp.where(gridk_mod > 0, 3. * (jnp.sin(gridk_mod) - gridk_mod * jnp.cos(gridk_mod)) / (gridk_sqr * gridk_mod)   , 0.)        
            
            #Conjugate
            SD = -1. * SD
            
     		#Square root of Green's function times dW
            kdF = jnp.where(gridk_mod > 0., (gridk.at[:, :, :, 0].get()*gridX +
                            gridk.at[:, :, :, 1].get()*gridY+gridk.at[:, :, :, 2].get()*gridZ)/gridk_sqr, 0)

            BdWx = (gridX - gridk.at[:, :, :, 0].get() * kdF) * B
            BdWy = (gridY - gridk.at[:, :, :, 1].get() * kdF) * B
            BdWz = (gridZ - gridk.at[:, :, :, 2].get() * kdF) * B

            BdWkxx = BdWx * gridk.at[:, :, :, 0].get()
            BdWkxy = BdWx * gridk.at[:, :, :, 1].get()
            BdWkxz = BdWx * gridk.at[:, :, :, 2].get()
            BdWkyx = BdWy * gridk.at[:, :, :, 0].get()
            BdWkyy = BdWy * gridk.at[:, :, :, 1].get()
            BdWkyz = BdWy * gridk.at[:, :, :, 2].get()
            BdWkzx = BdWz * gridk.at[:, :, :, 0].get()
            BdWkzy = BdWz * gridk.at[:, :, :, 1].get()
            
            gridX = SU * BdWx
            gridY = SU * BdWy
            gridZ = SU * BdWz
            gridXX = SD * (-jnp.imag(BdWkxx) + 1j * jnp.real(BdWkxx))
            gridXY = SD * (-jnp.imag(BdWkxy) + 1j * jnp.real(BdWkxy))
            gridXZ = SD * (-jnp.imag(BdWkxz) + 1j * jnp.real(BdWkxz))
            gridYX = SD * (-jnp.imag(BdWkyx) + 1j * jnp.real(BdWkyx))
            gridYY = SD * (-jnp.imag(BdWkyy) + 1j * jnp.real(BdWkyy))
            gridYZ = SD * (-jnp.imag(BdWkyz) + 1j * jnp.real(BdWkyz))
            gridZX = SD * (-jnp.imag(BdWkzx) + 1j * jnp.real(BdWkzx))
            gridZY = SD * (-jnp.imag(BdWkzy) + 1j * jnp.real(BdWkzy))
            
            #Return rescaled forces to real space (Inverse FFT)
            gridX = jnp.real(jnp.fft.ifftn(gridX,norm='forward'))
            gridY = jnp.real(jnp.fft.ifftn(gridY,norm='forward'))
            gridZ = jnp.real(jnp.fft.ifftn(gridZ,norm='forward'))
            gridXX = jnp.real(jnp.fft.ifftn(gridXX,norm='forward'))
            gridXY = jnp.real(jnp.fft.ifftn(gridXY,norm='forward'))
            gridXZ = jnp.real(jnp.fft.ifftn(gridXZ,norm='forward'))
            gridYX = jnp.real(jnp.fft.ifftn(gridYX,norm='forward'))
            gridYY = jnp.real(jnp.fft.ifftn(gridYY,norm='forward'))
            gridYZ = jnp.real(jnp.fft.ifftn(gridYZ,norm='forward'))
            gridZX = jnp.real(jnp.fft.ifftn(gridZX,norm='forward'))
            gridZY = jnp.real(jnp.fft.ifftn(gridZY,norm='forward'))

            
            # Compute Linear velocities and Velocity gradients (from which we can extract angular velocities and rate of strain)
            w_lin_velocities = jnp.zeros((N, 3),float)
            w_velocity_gradient = jnp.zeros((N, 8),float)
               
             
            w_lin_velocities = w_lin_velocities.at[:, 0].add(
                jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
                    gridX.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3)))
            w_lin_velocities = w_lin_velocities.at[:, 1].add(
                jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
                    gridY.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3))) 
            w_lin_velocities = w_lin_velocities.at[:, 2].add(
                jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
                    gridZ.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3)))              


            w_velocity_gradient = w_velocity_gradient.at[:, 0].add(
                jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
                    gridXX.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3)))
            
            #index might be 1 instead if 5 (originally is 1, and so on for the ones below)
            w_velocity_gradient = w_velocity_gradient.at[:, 1].add(
                jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
                    gridXY.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3))) 
            #index might be 2 instead if 6           
            w_velocity_gradient = w_velocity_gradient.at[:, 2].add(
                  jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
                    gridXZ.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3)))  
            #index might be 3 instead if 7
            w_velocity_gradient = w_velocity_gradient.at[:, 3].add(
                jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
                    gridYZ.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3)))
            
            
            w_velocity_gradient = w_velocity_gradient.at[:, 4].add(
                jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
                    gridYY.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3))) 
            
            #index might be 5 instead if 1
            w_velocity_gradient = w_velocity_gradient.at[:, 5].add(
                jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
                    gridYX.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3)))     
            #index might be 6 instead if 2                        
            w_velocity_gradient = w_velocity_gradient.at[:, 6].add(
                  jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
                    gridZX.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3))) 
            #index might be 7 instead if 3
            w_velocity_gradient = w_velocity_gradient.at[:, 7].add(
                jnp.sum(gaussian_grid_spacing2 * jnp.reshape(
                    gridZY.at[all_indices_x, all_indices_y, all_indices_z].get(),(N,gaussP,gaussP,gaussP)),axis=(1,2,3)))    
            
            # Convert to angular velocities and rate of strain
            w_ang_vel_and_strain = jnp.zeros((N, 8))
            w_ang_vel_and_strain = w_ang_vel_and_strain.at[:, 0].set((w_velocity_gradient.at[:, 3].get()-w_velocity_gradient.at[:, 7].get()) * 0.5)
            w_ang_vel_and_strain = w_ang_vel_and_strain.at[:, 1].set((w_velocity_gradient.at[:, 6].get()-w_velocity_gradient.at[:, 2].get()) * 0.5)
            w_ang_vel_and_strain = w_ang_vel_and_strain.at[:, 2].set((w_velocity_gradient.at[:, 1].get()-w_velocity_gradient.at[:, 5].get()) * 0.5)
            w_ang_vel_and_strain = w_ang_vel_and_strain.at[:, 3].set(2*w_velocity_gradient.at[:, 0].get()+w_velocity_gradient.at[:, 4].get())
            w_ang_vel_and_strain = w_ang_vel_and_strain.at[:, 4].set(w_velocity_gradient.at[:, 1].get()+w_velocity_gradient.at[:, 5].get())
            w_ang_vel_and_strain = w_ang_vel_and_strain.at[:, 5].set(w_velocity_gradient.at[:, 2].get()+w_velocity_gradient.at[:, 6].get())
            w_ang_vel_and_strain = w_ang_vel_and_strain.at[:, 6].set(w_velocity_gradient.at[:, 3].get()+w_velocity_gradient.at[:, 7].get())
            w_ang_vel_and_strain = w_ang_vel_and_strain.at[:, 7].set(w_velocity_gradient.at[:, 0].get()+2*w_velocity_gradient.at[:, 4].get())
            
            return w_lin_velocities, w_ang_vel_and_strain
        
        @jit
        def helper_dot(a,b):
            return jnp.dot(a,b)
        @jit
        def helper_sqrt(a):
            return jnp.sqrt(a)
        @jit
        def helper_multip(a,b):
            return a*b      
        @jit
        def helper_lincomb(a,b,c,d):
            return a*b +c*d
        ############################################################################################################################################################
        #Wave Space Part
        lin_velocities, ang_vel_and_strain = helper_wavespace_calc(random_array_wave)
        
        ############################################################################################################################################################
        ### Real Space part
        
        #Scale random numbers from [0,1] to [-sqrt(3),-sqrt(3)]
        random_array_real = (2*random_array_real-1)*jnp.sqrt(3.)
        #obtain matrix form of linear operator Mpsi, by computing Mpsi(e_i) with e_i basis vectors (1,0,...,0), (0,1,0,...) ...
        Matrix_M = np.zeros((11*N,11*N))
        basis_vectors = np.eye(11*N, dtype = float)
        for iii in range(11*N):
            a,b = helper_Mpsi(basis_vectors[iii,:]); Mei = helper_reshape(a,b)
            Matrix_M[:,iii] =  Mei
        sqrt_M = scipy.linalg.sqrtm(Matrix_M) #EXTEMELY NOT EFFICIENT! need to be replaced with faster method
        M12psi = helper_dot(sqrt_M,random_array_real*np.sqrt(2. * kT / dt))
        
        ###############LANCZOS METHOD (not working...)###############
        
        #combine w_lin_velocities, w_ang_vel_and_strain
        lin_vel = jnp.reshape(lin_velocities,3*N) + M12psi.at[:3*N].get()
        ang_vel_and_strain = jnp.reshape(ang_vel_and_strain,8*N) + M12psi.at[3*N:].get()
        
        # Convert to Generalized Velocities+strain 
        generalized_velocities = jnp.zeros(11*N) #First 6N entries for U and last 5N for strain rates

        generalized_velocities = generalized_velocities.at[0:6*N:6].set(
            lin_vel.at[0::3].get())
        generalized_velocities = generalized_velocities.at[1:6*N:6].set(
            lin_vel.at[1::3].get())
        generalized_velocities = generalized_velocities.at[2:6*N:6].set(
            lin_vel.at[2::3].get())
        generalized_velocities = generalized_velocities.at[3:6*N:6].set(
            ang_vel_and_strain.at[0::8].get())
        generalized_velocities = generalized_velocities.at[4:6*N:6].set(
            ang_vel_and_strain.at[1::8].get())
        generalized_velocities = generalized_velocities.at[5:6*N:6].set(
            ang_vel_and_strain.at[2::8].get())
        generalized_velocities = generalized_velocities.at[(6*N+0)::5].set(
            ang_vel_and_strain.at[3::8].get())
        generalized_velocities = generalized_velocities.at[(6*N+1)::5].set(
            ang_vel_and_strain.at[4::8].get())
        generalized_velocities = generalized_velocities.at[(6*N+2)::5].set(
            ang_vel_and_strain.at[5::8].get())
        generalized_velocities = generalized_velocities.at[(6*N+3)::5].set(
            ang_vel_and_strain.at[6::8].get())
        generalized_velocities = generalized_velocities.at[(6*N+4)::5].set(
            ang_vel_and_strain.at[7::8].get())
        
        return generalized_velocities
        
    
    def compute_nearfield_brownianforce(
            random_array,
            r_lub,indices_i_lub,indices_j_lub,XA11,XA12,YA11,YA12,YB11,YB12,XC11,XC12,YC11,YC12,YB21
            ):
        
        @jit
        def ComputeLubricationFU(velocities):

            vel_i = (jnp.reshape(velocities,(N,6))).at[indices_i_lub].get()
            vel_j = (jnp.reshape(velocities,(N,6))).at[indices_j_lub].get()
            
            # Dot product of r and U, i.e. axisymmetric projection
            rdui = r_lub.at[:,0].get()*vel_i.at[:,0].get()+r_lub.at[:,1].get()*vel_i.at[:,1].get()+r_lub.at[:,2].get()*vel_i.at[:,2].get()
            rduj = r_lub.at[:,0].get()*vel_j.at[:,0].get()+r_lub.at[:,1].get()*vel_j.at[:,1].get()+r_lub.at[:,2].get()*vel_j.at[:,2].get()
            rdwi = r_lub.at[:,0].get()*vel_i.at[:,3].get()+r_lub.at[:,1].get()*vel_i.at[:,4].get()+r_lub.at[:,2].get()*vel_i.at[:,5].get()
            rdwj = r_lub.at[:,0].get()*vel_j.at[:,3].get()+r_lub.at[:,1].get()*vel_j.at[:,4].get()+r_lub.at[:,2].get()*vel_j.at[:,5].get()

            # Cross product of U and r, i.e. eps_ijk*r_k*U_j = Px dot U, (eps_ijk is the Levi-Civita symbol)
            epsrdui = jnp.array([r_lub.at[:,2].get() * vel_i.at[:,1].get() - r_lub.at[:,1].get() * vel_i.at[:,2].get(),
                                -r_lub.at[:,2].get() * vel_i.at[:,0].get() + r_lub.at[:,0].get() * vel_i.at[:,2].get(),
                                r_lub.at[:,1].get() * vel_i.at[:,0].get() - r_lub.at[:,0].get() * vel_i.at[:,1].get()])
            
            epsrdwi = jnp.array([r_lub.at[:,2].get() * vel_i.at[:,4].get() - r_lub.at[:,1].get() * vel_i.at[:,5].get(), 
                                -r_lub.at[:,2].get() * vel_i.at[:,3].get() + r_lub.at[:,0].get() * vel_i.at[:,5].get(), 
                                r_lub.at[:,1].get() * vel_i.at[:,3].get() - r_lub.at[:,0].get() * vel_i.at[:,4].get()])
            
            epsrduj = jnp.array([r_lub.at[:,2].get() * vel_j.at[:,1].get() - r_lub.at[:,1].get() * vel_j.at[:,2].get(), 
                                -r_lub.at[:,2].get() * vel_j.at[:,0].get() + r_lub.at[:,0].get() * vel_j.at[:,2].get(), 
                                r_lub.at[:,1].get() * vel_j.at[:,0].get() - r_lub.at[:,0].get() * vel_j.at[:,1].get()])
            
            epsrdwj = jnp.array([r_lub.at[:,2].get() * vel_j.at[:,4].get() - r_lub.at[:,1].get() * vel_j.at[:,5].get(), 
                                -r_lub.at[:,2].get() * vel_j.at[:,3].get() + r_lub.at[:,0].get() * vel_j.at[:,5].get(), 
                                r_lub.at[:,1].get() * vel_j.at[:,3].get() - r_lub.at[:,0].get() * vel_j.at[:,4].get()])
            
            forces = jnp.zeros((N,6),float)
            
            # Compute the contributions to the force for particles i (Fi = A11*Ui + A12*Uj + BT11*Wi + BT12*Wj)
            f = ((XA11 - YA11).at[:,None].get() * rdui.at[:,None].get() * r_lub + YA11.at[:,None].get() * vel_i.at[:,:3].get() 
            + (XA12 - YA12).at[:,None].get() * rduj.at[:,None].get() * r_lub + YA12.at[:,None].get() * vel_j.at[:,:3].get() + YB11.at[:,None].get() * (-epsrdwi.T) + YB21.at[:,None].get() * (-epsrdwj.T))
            forces = forces.at[indices_i_lub, :3].add(f)
            # Compute the contributions to the force for particles j (Fj = A11*Uj + A12*Ui + BT11*Wj + BT12*Wi)
            f = ((XA11 - YA11).at[:,None].get() * rduj.at[:,None].get() * r_lub + YA11.at[:,None].get() * vel_j.at[:,:3].get() 
            + (XA12 - YA12).at[:,None].get() * rdui.at[:,None].get() * r_lub + YA12.at[:,None].get() * vel_i.at[:,:3].get() + YB11.at[:,None].get() * (epsrdwj.T) + YB21.at[:,None].get() * (epsrdwi.T))
            forces = forces.at[indices_j_lub, :3].add(f)
            # Compute the contributions to the torque for particles i (Li = B11*Ui + B12*Uj + C11*Wi + C12*Wj)
            l = (YB11.at[:,None].get() * epsrdui.T + YB12.at[:,None].get() * epsrduj.T 
            + (XC11 - YC11).at[:,None].get() * rdwi.at[:,None].get() * r_lub + YC11.at[:,None].get() * vel_i.at[:,3:].get() 
            + (XC12 - YC12).at[:,None].get() * rdwj.at[:,None].get() * r_lub + YC12.at[:,None].get() * vel_j.at[:,3:].get())
            forces = forces.at[indices_i_lub, 3:].add(l)
            # Compute the contributions to the torque for particles j (Lj = B11*Uj + B12*Ui + C11*Wj + C12*Wi)
            l = (-YB11.at[:,None].get() * epsrduj.T - YB12.at[:,None].get() * epsrdui.T 
            + (XC11 - YC11).at[:,None].get() * rdwj.at[:,None].get() * r_lub + YC11.at[:,None].get() * vel_j.at[:,3:].get() 
            + (XC12 - YC12).at[:,None].get() * rdwi.at[:,None].get() * r_lub + YC12.at[:,None].get() * vel_i.at[:,3:].get())           
            forces = forces.at[indices_j_lub, 3:].add(l)
            
            return jnp.ravel(forces)
        
        @jit
        def helper_dot(a,b):
            return jnp.dot(a,b)

        
        ############################################################################################################################################################
        
        
        #Scale random numbers from [0,1] to [-sqrt(3),-sqrt(3)]
        random_array = (2*random_array-1)*jnp.sqrt(3.)
        #obtain matrix form of linear operator Mpsi, by computing Mpsi(e_i) with e_i basis vectors (1,0,...,0), (0,1,0,...) ...
        R_FU_Matrix = np.zeros((6*N,6*N))
        basis_vectors = np.eye(6*N, dtype = float)
        for iii in range(6*N):
            Rei = ComputeLubricationFU(basis_vectors[iii,:])
            R_FU_Matrix[:,iii] =  Rei
        sqrt_R_FU = scipy.linalg.sqrtm(R_FU_Matrix) #EXTEMELY NOT EFFICIENT! need to be replaced with faster method
        R_FU12psi = helper_dot(sqrt_R_FU,random_array*np.sqrt(2. * kT / dt))
        
        return R_FU12psi
    
    
    @jit
    def precompute(positions,gaussian_grid_spacing,nl_ff,nl_lub,displacements_vector_matrix,tilt_factor):
        #Wave Space calculation quantities
        
        #Compute fractional coordinates
        pos = positions + jnp.array([Lx, Ly, Lz])/2
        pos = pos.at[:,0].add(-tilt_factor*pos.at[:,1].get())
        pos = pos / jnp.array([Lx,Ly,Lz])  * jnp.array([Nx, Ny, Nz])
        ###convert positions in the box in indices in the grid
        # pos = (positions+np.array([Lx, Ly, Lz])/2)/np.array([Lx, Ly, Lz]) * np.array([Nx, Ny, Nz])
        intpos = (jnp.array(pos, int))  # integer positions
        intpos = jnp.where(pos-intpos>0.5,intpos+1,intpos)

        # actual values to put in the grid
        gaussian_grid_spacing1 = prefac * jnp.exp(-expfac * gaussian_grid_spacing)
        gaussian_grid_spacing2 = jnp.swapaxes(jnp.swapaxes(jnp.swapaxes(gaussian_grid_spacing1*quadW,3,2),2,1),1,0)
        
        #Starting indices on the grid from particle positions
        start_index_x = (intpos.at[:,0].get()-(gaussPd2)) % Nx
        start_index_y = (intpos.at[:,1].get()-(gaussPd2)) % Ny
        start_index_z = (intpos.at[:,2].get()-(gaussPd2)) % Nz
        #All indices on the grids from particle positions, ordered such that x is the 'slowest' index to change and z the 'fastest'
        all_indices_x = jnp.repeat(jnp.repeat(jnp.repeat(start_index_x, gaussP), gaussP), gaussP) + jnp.resize(jnp.repeat(jnp.repeat(jnp.arange(gaussP), gaussP), gaussP),gaussP*gaussP*gaussP*N)
        all_indices_y = jnp.repeat(jnp.repeat(jnp.repeat(start_index_y, gaussP), gaussP), gaussP) + jnp.resize(jnp.resize(jnp.repeat(jnp.arange(gaussP), gaussP), gaussP*gaussP*gaussP),gaussP*gaussP*gaussP*N)
        all_indices_z = jnp.repeat(jnp.repeat(jnp.repeat(start_index_z, gaussP), gaussP), gaussP) + jnp.resize(jnp.resize(jnp.resize(jnp.arange(gaussP), gaussP*gaussP), gaussP*gaussP*gaussP),gaussP*gaussP*gaussP*N)
        all_indices_x = all_indices_x % Nx
        all_indices_y = all_indices_y % Ny
        all_indices_z = all_indices_z % Nz
        
        ###################################################################################################################       
        #Real Space (far-field) calculation quantities
        indices_i = nl_ff[0,:] #Pair indices (i<j always)
        indices_j = nl_ff[1,:]
        R = displacements_vector_matrix.at[indices_i,indices_j].get() #array of vectors from particles i to j (size = npairs)
        dist = space.distance(R)  # distances between particles i and j
        r = -R / dist.at[:, None].get()  # unit vector from particle j to i

        # Interpolate scalar mobility functions values from ewald table
        r_ind = (ewald_n * (dist - ewald_dr) / (ewald_cut - ewald_dr))  # index in ewald table
        r_ind = r_ind.astype(int) #truncate decimal part
        offset1 = 2 * r_ind  # even indices
        offset2 = 2 * r_ind + 1  # odd indices

        tewaldC1m = ewaldC1.at[offset1].get()  # UF and UC
        tewaldC1p = ewaldC1.at[offset1+2].get()
        tewaldC2m = ewaldC1.at[offset2].get()  # DC
        tewaldC2p = ewaldC1.at[offset2+2].get()

        fac_ff = dist / ewald_dr - r_ind - 1.0 #interpolation factor
                    
        f1 = tewaldC1m.at[:,0].get() + (tewaldC1p.at[:,0].get() - tewaldC1m.at[:,0].get()) * fac_ff
        f2 = tewaldC1m.at[:,1].get() + (tewaldC1p.at[:,1].get() - tewaldC1m.at[:,1].get()) * fac_ff

        g1 = tewaldC1m.at[:,2].get() + (tewaldC1p.at[:,2].get() - tewaldC1m.at[:,2].get()) * fac_ff
        g2 = tewaldC1m.at[:,3].get() + (tewaldC1p.at[:,3].get() - tewaldC1m.at[:,3].get()) * fac_ff

        h1 = tewaldC2m.at[:,0].get() + (tewaldC2p.at[:,0].get() - tewaldC2m.at[:,0].get()) * fac_ff
        h2 = tewaldC2m.at[:,1].get() + (tewaldC2p.at[:,1].get() - tewaldC2m.at[:,1].get()) * fac_ff
        h3 = tewaldC2m.at[:,2].get() + (tewaldC2p.at[:,2].get() - tewaldC2m.at[:,2].get()) * fac_ff
        
        ###################################################################################################################
        #Lubrication calculation quantities
        indices_i_lub = nl_lub[0,:] #Pair indices (i<j always)
        indices_j_lub = nl_lub[1,:]
        R_lub = displacements_vector_matrix.at[nl_lub[0,:],nl_lub[1,:]].get() #array of vectors from particle i to j (size = npairs)
        dist_lub = space.distance(R_lub)  # distance between particle i and j
        r_lub = R_lub / dist_lub.at[:, None].get()  # unit vector from particle j to i
        
        # # Indices in resistance table
        ind = (jnp.log10((dist_lub - 2.0) / ResTable_min) / ResTable_dr)
        ind = ind.astype(int)
        dist_lub_lower = ResTable_dist.at[ind].get()
        dist_lub_upper = ResTable_dist.at[ind+1].get()
        # # Linear interpolation of the Table values
        fac_lub = (dist_lub - dist_lub_lower) / (dist_lub_upper - dist_lub_lower)
        
        XA11 = jnp.where(dist_lub > 0, ResTable_vals[0], 0.)
        XA11 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+0] + (
            ResTable_vals[22*(ind+1)+0]-ResTable_vals[22*(ind)+0]) * fac_lub), XA11)

        XA12 = jnp.where(dist_lub > 0, ResTable_vals[1], 0.)
        XA12 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+1] + (
            ResTable_vals[22*(ind+1)+1]-ResTable_vals[22*(ind)+1]) * fac_lub), XA12)

        YA11 = jnp.where(dist_lub > 0, ResTable_vals[2], 0.)
        YA11 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+2] + (
            ResTable_vals[22*(ind+1)+2]-ResTable_vals[22*(ind)+2]) * fac_lub), YA11)

        YA12 = jnp.where(dist_lub > 0, ResTable_vals[3], 0.)
        YA12 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+3] + (
            ResTable_vals[22*(ind+1)+3]-ResTable_vals[22*(ind)+3]) * fac_lub), YA12)

        YB11 = jnp.where(dist_lub > 0, ResTable_vals[4], 0.)
        YB11 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+4] + (
            ResTable_vals[22*(ind+1)+4]-ResTable_vals[22*(ind)+4]) * fac_lub), YB11)

        YB12 = jnp.where(dist_lub > 0, ResTable_vals[5], 0.)
        YB12 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+5] + (
            ResTable_vals[22*(ind+1)+5]-ResTable_vals[22*(ind)+5]) * fac_lub), YB12)

        XC11 = jnp.where(dist_lub > 0, ResTable_vals[6], 0.)
        XC11 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+6] + (
            ResTable_vals[22*(ind+1)+6]-ResTable_vals[22*(ind)+6]) * fac_lub), XC11)

        XC12 = jnp.where(dist_lub > 0, ResTable_vals[7], 0.)
        XC12 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+7] + (
            ResTable_vals[22*(ind+1)+7]-ResTable_vals[22*(ind)+7]) * fac_lub), XC12)

        YC11 = jnp.where(dist_lub > 0, ResTable_vals[8], 0.)
        YC11 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+8] + (
            ResTable_vals[22*(ind+1)+8]-ResTable_vals[22*(ind)+8]) * fac_lub), YC11)

        YC12 = jnp.where(dist_lub > 0, ResTable_vals[9], 0.)
        YC12 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+9] + (
            ResTable_vals[22*(ind+1)+9]-ResTable_vals[22*(ind)+9]) * fac_lub), YC12)
        
        YB21 = -YB12 # symmetry condition
        
        XG11 = jnp.where(dist_lub > 0, ResTable_vals[10], 0.)
        XG11 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+10] + (
            ResTable_vals[22*(ind+1)+10]-ResTable_vals[22*(ind)+10]) * fac_lub), YC12)
        
        XG12 = jnp.where(dist_lub > 0, ResTable_vals[11], 0.)
        XG12 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+11] + (
            ResTable_vals[22*(ind+1)+11]-ResTable_vals[22*(ind)+11]) * fac_lub), YC12)
        
        YG11 = jnp.where(dist_lub > 0, ResTable_vals[12], 0.)
        YG11 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+12] + (
            ResTable_vals[22*(ind+1)+12]-ResTable_vals[22*(ind)+12]) * fac_lub), YC12)
        
        YG12 = jnp.where(dist_lub > 0, ResTable_vals[13], 0.)
        YG12 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+13] + (
            ResTable_vals[22*(ind+1)+13]-ResTable_vals[22*(ind)+13]) * fac_lub), YC12)
        
        YH11 = jnp.where(dist_lub > 0, ResTable_vals[14], 0.)
        YH11 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+14] + (
            ResTable_vals[22*(ind+1)+14]-ResTable_vals[22*(ind)+14]) * fac_lub), YC12)
        
        YH12 = jnp.where(dist_lub > 0, ResTable_vals[15], 0.)
        YH12 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+15] + (
            ResTable_vals[22*(ind+1)+15]-ResTable_vals[22*(ind)+15]) * fac_lub), YC12)
        
        
        return (all_indices_x,all_indices_y,all_indices_z,gaussian_grid_spacing1,gaussian_grid_spacing2,
                r,indices_i,indices_j,f1,f2,g1,g2,h1,h2,h3,
                r_lub,indices_i_lub,indices_j_lub,XA11,XA12,YA11,YA12,YB11,YB12,XC11,XC12,YC11,YC12,YB21,
                XG11,XG12,YG11,YG12,YH11,YH12)


    @jit
    def compute_RFE(shear_rate,r_lub,indices_i_lub,indices_j_lub,XG11,XG12,YG11,YG12,YH11,YH12):
        #These simulations are constructed so that, if there is strain,
		# x is the flow direction
		# y is the gradient direction
		# z is the vorticity direction
	    # therefore,
		# Einf = [ 0 g 0 ]
		#	     [ g 0 0 ]
	    #		 [ 0 0 0 ]
        
        #symmetry conditions
        XG21 = -XG12
        YG21 = -YG12
        YH21 = YH12
        
        #define single particle strain (each particle experience same shear)        
        E = jnp.zeros((3,3),float)
        E = E.at[0,1].set(shear_rate/2)
        E = E.at[1,0].set(shear_rate/2) 
        
        Edri = jnp.array( [shear_rate/2 * r_lub.at[:,1].get(),
                           shear_rate/2 * r_lub.at[:,0].get(), r_lub.at[:,0].get() * 0. ]  )
        Edrj = Edri
        rdEdri = r_lub.at[:,0].get() * Edri.at[0].get() + r_lub.at[:,1].get() * Edri.at[1].get()    
        rdEdrj = rdEdri 
        epsrdEdri = jnp.array( [ r_lub.at[:,2].get() * Edri[1] - r_lub.at[:,1].get() * Edri[2] ,
                                 -r_lub.at[:,2].get() * Edri[0] + r_lub.at[:,0].get() * Edri[2] ,
                                 r_lub.at[:,1].get() * Edri[0] - r_lub.at[:,0].get() * Edri[1] ]  )
        epsrdEdrj = epsrdEdri
        
        forces = jnp.zeros((N,6),float)
        
        f = ( (XG11 - 2.0*YG11).at[:,None].get() * (-rdEdri.at[:,None].get()) * r_lub 
            + 2.0 * YG11.at[:,None].get() * (-Edri.T)                  
            + (XG21 - 2.0*YG21).at[:,None].get() * (-rdEdrj.at[:,None].get()) * r_lub 
            + 2.0 * YG21.at[:,None].get() * (-Edrj.T) )
        l= ( YH11.at[:,None].get() * (2.0 * epsrdEdri.T) 
            + YH21.at[:,None].get() * (2.0 * epsrdEdrj.T))
        
        forces = forces.at[indices_i_lub, :3].add(f)
        forces = forces.at[indices_i_lub, 3:].add(l)
        
        
        
        
        f = ( (XG11 - 2.0*YG11).at[:,None].get() * (rdEdrj.at[:,None].get()) * r_lub 
            + 2.0 * YG11.at[:,None].get() * (Edrj.T)                  
            + (XG21 - 2.0*YG21).at[:,None].get() * (rdEdri.at[:,None].get()) * r_lub 
            + 2.0 * YG21.at[:,None].get() * (Edri.T) )
        l= ( YH11.at[:,None].get() * (2.0 * epsrdEdrj.T) 
            + YH21.at[:,None].get() * (2.0 * epsrdEdri.T))
        
        forces = forces.at[indices_j_lub, :3].add(f)
        forces = forces.at[indices_j_lub, 3:].add(l)
        
        
        return jnp.ravel(forces)
        
    def update_box_tilt_factor(dt,shear_rate,tilt_factor):
        tilt_factor = tilt_factor+dt*shear_rate
        if (tilt_factor >= 0.5):
            tilt_factor = -0.5 + (tilt_factor-0.5)
        return tilt_factor
    
    #######################################################################
    # Set/Calculate a bunch of parameters needed for the entire calculation, while doing some checks
    # Compute the Real Space cutoff for the Ewald Summation in the Far-Field computation
    ewald_cut = jnp.sqrt(- jnp.log(error)) / xi
    # parameter needed to make Chol. decomposition of R_FU converge (initially to 1)
    ichol_relaxer = 1.0
    kmax = int(2.0 * jnp.sqrt(- jnp.log(error)) * xi) + 1  # Max wave number
    # Set number of Lanczos iterations (initially) to 2, for both far- and near-field
    n_iter_Lanczos_ff = 2
    # n_iter_Lanczos_nf = 2 #not needed for now
    xisq = xi * xi
    # Compute number of grid points in k space
    Nx, Ny, Nz = compute_k_gridpoint_number(kmax); 
    gridh = jnp.array([Lx, Ly, Lz]) / jnp.array([Nx, Ny, Nz])  # Set Grid Spacing
    quadW = gridh[0] * gridh[1] * gridh[2]
    xy = 0. #set box tilt factor to zero to begin (unsheared box)
    
    # Check that ewald_cut is small enough to avoid interaction with periodic images)
    check_ewald_cut(ewald_cut)
    
    # (Check) Maximum eigenvalue of A'*A to scale support, P, for spreading on deformed grids
    eta, gaussP = check_max_shear(gridh, xisq, Nx, Ny, Nz)
    prefac = (2.0 * xisq / 3.1415926536 / eta) * jnp.sqrt(2.0 * xisq / 3.1415926536 / eta)
    expfac = 2.0 * xisq / eta
    gaussPd2 = jnp.array(gaussP/2, int)
    
    # Get list of reciprocal space vectors, and scaling factor for the wave space calculation
    gridk = compute_sheared_grid(int(Nx),int(Ny),int(Nz),xy)   
    #######################################################################

    # Set Periodic Space and Displacement Metric
    displacement, shift = space.periodic_general(
        jnp.array([Lx, Ly, Lz]), fractional_coordinates=False)
    
    # compute matrix of displacements between particles (each element is a vector from particle j to i) - INITIAL POSITIONS
    displacements_vector_matrix = (
        space.map_product(displacement))(positions, positions)
    #######################################################################
    # Store the coefficients for the real space part of Ewald summation
    # Will precompute scaling factors for real space component of summation for a given discretization to speed up GPU calculations
    # NOTE: Potential sensitivity of real space functions at small xi, tabulation computed in double prec., then truncated to single

    ewald_dr = 0.001  # Distance resolution
    ewald_n = ewald_cut / ewald_dr - 1  # Number of entries in tabulation
    
    # Factors needed to compute self contribution
    pi12 = 1.77245385091  # square root of pi
    pi = 3.1415926536  # pi
    a = a *1.0 # radius
    axi = a * xi  # a * xi
    axi2 = axi * axi  # ( a * xi )^2

    # Compute self contribution
    m_self = jnp.zeros(2,float)
    m_self = m_self.at[0].set(
        (1 + 4*pi12*axi*math.erfc(2.*axi) - math.exp(-4.*axi2))/(4.*pi12*axi*a))
    
    
    m_self = m_self.at[1].set( (-3.*math.erfc(2.*a*xi)*math.pow(a,-3.))/10. - (3.*math.pow(a,-6.)*math.pow(pi,-0.5)*math.pow(xi,-3.))/80.
                              -  (9.*math.pow(a, -4)*math.pow(pi, -0.5)*math.pow(xi, -1))/40 
                              + (3.*math.exp(-4 * math.pow(a, 2)*math.pow(xi, 2))*math.pow(a, -6)*math.pow(pi, -0.5)*math.pow(xi, -3)
                                  *(1+10 * math.pow(a, 2)*math.pow(xi, 2))) / 80)
    #######################################################################
    # Create real space Ewald table
    nR = int(ewald_n + 1)  # number of entries in ewald table
    ewaldC1 = jnp.zeros((2*nR, 4))

    Imrr, rr, g1, g2, h1, h2, h3 = compute_real_space_ewald_table() #this version uses numpy long float
           
    ewaldC1 = ewaldC1.at[0::2, 0].set((Imrr))  # UF1
    ewaldC1 = ewaldC1.at[0::2, 1].set((rr))  # UF2
    ewaldC1 = ewaldC1.at[0::2, 2].set((g1/2))  # UC1
    ewaldC1 = ewaldC1.at[0::2, 3].set((-g2/2))  # UC2
    ewaldC1 = ewaldC1.at[1::2, 0].set((h1))  # DC1
    ewaldC1 = ewaldC1.at[1::2, 1].set((h2))  # DC2
    ewaldC1 = ewaldC1.at[1::2, 2].set((h3))  # DC3

    #######################################################################
    # Load resistance table
    ResTable_dist = jnp.load('ResTableDist.npy')
    ResTable_vals = jnp.load('ResTableVals.npy')
    ResTable_min = 0.000100
    ResTable_dr = 0.004305 # Table discretization (log space), i.e. ind * dr.set(log10( h[ind] / min )
    #######################################################################
    AppliedForce = jnp.zeros(3*N,float)
    AppliedTorques = jnp.zeros(3*N,float)
    AppliedForce = AppliedForce.at[2::3].add(gravity)
    # AppliedTorques = AppliedTorques.at[:].set(0.) #here can add applied torques to particles, if needed
    ######################################################################
    # Initialize neighborlists
    lub_neighbor_fn, prec_lub_neighbor_fn, ff_neighbor_fn, pot_neighbor_fn = initialize_neighborlist(
        a, Lx, Ly, Lz, displacement, ewald_cut)

    # Allocate lists for first time - they will be updated at each timestep and if needed, re-allocated too.
    # (see jax_md documentation for more detail about the difference between updating and allocating a neighbor list)
    nbrs_lub = lub_neighbor_fn.allocate(positions+jnp.array([Lx,Ly,Lz])/2)
    nbrs_lub_prec = prec_lub_neighbor_fn.allocate(positions+jnp.array([Lx,Ly,Lz])/2)
    nbrs_ff = ff_neighbor_fn.allocate(positions+jnp.array([Lx,Ly,Lz])/2)
    # nbrs_pot = pot_neighbor_fn.allocate(positions+jnp.array([Lx,Ly,Lz])/2) #interparticle potential not implemented yet
    nl_lub = np.array(nbrs_lub.idx)
    # nl_lub_prec = np.array(nbrs_lub_prec.idx) #probably not needed
    nl_ff = np.array(nbrs_ff.idx)
    # n_pairs_ff = len(nbrs_ff.idx[0]) #probably not needed
    # n_pairs_lub = len(nbrs_lub.idx[0]) #probably not needed
    # n_pairs_lub_prec = len(nbrs_lub_prec.idx[0]) #probably not needed
        
    #######################################################################  
    trajectory = np.zeros((int(Nsteps/writing_period + 1),N,3),float)
    
    # precompute grid distances for FFT (same for each gaussian support)
    gaussian_grid_spacing = precompute_grid_distancing(gaussP, gridh[0], xy, positions)

    #create RNG states
    key_RFD = random.PRNGKey(seed_RFD)
    key_ffwave = random.PRNGKey(seed_ffwave)
    key_ffreal = random.PRNGKey(seed_ffreal)
    key_nf = random.PRNGKey(seed_nf)
    
    #define epsilon for RFD
    epsilon = error
    
    #create indices list for spreading random forces on the wave space grid (far field brownian velocity, wave space calculation)    
    normal_indices,normal_conj_indices,nyquist_indices = random_force_on_grid_indexing(Nx,Ny,Nz)
    
    for step in range(Nsteps):
        
        #initialize generalized velocity (6*N array, for linear and angular components)
        general_velocity = jnp.zeros(6*N, float)
        
        #Define arrays for the saddle point solvers (solve linear system Ax=b, and A is the saddle point matrix)
        saddle_b = jnp.zeros(17*N,float)
        
        # # compute precondition resistance lubrication matrix
        nl_lub_prec = np.array(nbrs_lub_prec.idx)
        n_pairs_lub_prec = len(nbrs_lub_prec.idx[0])
        R_fu_prec_lower_triang, diagonal_elements_for_brownian = RFU_Precondition(ichol_relaxer,displacements_vector_matrix)
        
        # If T>0, compute Brownian Drift and use it to initialize the velocity
        if(T>0): 
            # general_velocity = jnp.zeros(6*N, float)
            key_RFD, random_array = generate_random_array(key_RFD,6*N) #get array of random variables
            random_array = -((2*random_array-1)*jnp.sqrt(3))
            #Add random displacement to RHS of linear system
            # saddle_b = saddle_b.at[11*N:].set(-((2*random_array-1)*jnp.sqrt(3)))
            saddle_b = saddle_b.at[11*N:].set(random_array)
            
            #Perform a displacement in the positive random directions and save it to a buffer
            buffer_positions, buffer_displacements_vector_matrix, buffer_nbrs_lub, buffer_nbrs_lub_prec, buffer_nbrs_ff = update_neigh_list_and_displacements(shear_rate,positions, displacements_vector_matrix, nbrs_lub, nbrs_lub_prec, nbrs_ff, random_array,epsilon/2.)
            buffer_gaussian_grid_spacing = precompute_grid_distancing(gaussP, gridh[0], xy, buffer_positions)
            buffer_nl_lub = np.array(buffer_nbrs_lub.idx)
            buffer_nl_ff = np.array(buffer_nbrs_ff.idx)     
            
            # Precompute quantities for far-field and near-field calculation
            (all_indices_x,all_indices_y,all_indices_z,gaussian_grid_spacing1,gaussian_grid_spacing2,
            r,indices_i,indices_j,f1,f2,g1,g2,h1,h2,h3,
            r_lub,indices_i_lub,indices_j_lub,XA11,XA12,YA11,YA12,YB11,YB12,XC11,XC12,YC11,YC12,YB21,
            XG11,XG12,YG11,YG12,YH11,YH12) = precompute(buffer_positions,buffer_gaussian_grid_spacing,buffer_nl_ff,buffer_nl_lub,buffer_displacements_vector_matrix,xy)
            
            
            #Solve the saddle point problem in the positive direction
            saddle_x, exitcode = solver(
                saddle_b,  # rhs vector of the linear system
                gridk,
                R_fu_prec_lower_triang,
                all_indices_x,all_indices_y,all_indices_z,gaussian_grid_spacing1,gaussian_grid_spacing2,
                r,indices_i,indices_j,f1,f2,g1,g2,h1,h2,h3,
                r_lub,indices_i_lub,indices_j_lub,XA11,XA12,YA11,YA12,YB11,YB12,XC11,XC12,YC11,YC12,YB21
                )
            general_velocity = saddle_x.at[11*N:].get()
            
            #Perform a displacement in the negative random directions and save it to a buffer
            buffer_positions, buffer_displacements_vector_matrix, buffer_nbrs_lub, buffer_nbrs_lub_prec, buffer_nbrs_ff = update_neigh_list_and_displacements(shear_rate,positions, displacements_vector_matrix, nbrs_lub, nbrs_lub_prec, nbrs_ff, random_array, -epsilon/2.)
            buffer_gaussian_grid_spacing = precompute_grid_distancing(gaussP, gridh[0], xy, buffer_positions)
            buffer_nl_lub = np.array(buffer_nbrs_lub.idx)
            buffer_nl_ff = np.array(buffer_nbrs_ff.idx)     
            
            # Precompute quantities for far-field and near-field calculation
            (all_indices_x,all_indices_y,all_indices_z,gaussian_grid_spacing1,gaussian_grid_spacing2,
            r,indices_i,indices_j,f1,f2,g1,g2,h1,h2,h3,
            r_lub,indices_i_lub,indices_j_lub,XA11,XA12,YA11,YA12,YB11,YB12,XC11,XC12,YC11,YC12,YB21,
            XG11,XG12,YG11,YG12,YH11,YH12) = precompute(buffer_positions,buffer_gaussian_grid_spacing,buffer_nl_ff,buffer_nl_lub,buffer_displacements_vector_matrix,xy)
                        
            #Solve the saddle point problem in the positive direction
            saddle_x, exitcode = solver(
                saddle_b,  # rhs vector of the linear system
                gridk,
                R_fu_prec_lower_triang,
                all_indices_x,all_indices_y,all_indices_z,gaussian_grid_spacing1,gaussian_grid_spacing2,
                r,indices_i,indices_j,f1,f2,g1,g2,h1,h2,h3,
                r_lub,indices_i_lub,indices_j_lub,XA11,XA12,YA11,YA12,YB11,YB12,XC11,XC12,YC11,YC12,YB21
                )
            general_velocity += (-saddle_x.at[11*N:].get()) #take the difference            
            general_velocity = general_velocity * T / epsilon #apply scaling
            saddle_b = jnp.zeros(17*N,float) #reset rhs to 0 for next solver
            
            
        # Precompute quantities for far-field and near-field calculation
        (all_indices_x,all_indices_y,all_indices_z,gaussian_grid_spacing1,gaussian_grid_spacing2,
        r,indices_i,indices_j,f1,f2,g1,g2,h1,h2,h3,
        r_lub,indices_i_lub,indices_j_lub,XA11,XA12,YA11,YA12,YB11,YB12,XC11,XC12,YC11,YC12,YB21,
        XG11,XG12,YG11,YG12,YH11,YH12) = precompute(positions,gaussian_grid_spacing,nl_ff,nl_lub,displacements_vector_matrix,xy)
        #Set up RHS for another saddle point solver    
        saddle_b = saddle_b.at[(11*N+0)::6].add(-AppliedForce.at[0::3].get()) #Add imposed (-forces) to rhs of linear system
        saddle_b = saddle_b.at[(11*N+1)::6].add(-AppliedForce.at[1::3].get())
        saddle_b = saddle_b.at[(11*N+2)::6].add(-AppliedForce.at[2::3].get())
        saddle_b = saddle_b.at[(11*N+3)::6].add(-AppliedTorques.at[0::3].get()) #Add imposed (-torques) to rhs of linear system
        saddle_b = saddle_b.at[(11*N+4)::6].add(-AppliedTorques.at[1::3].get())
        saddle_b = saddle_b.at[(11*N+5)::6].add(-AppliedTorques.at[2::3].get())
        
        #Add the (-) ambient rate of strain to the right-hand side
        saddle_b = saddle_b.at[(6*N+1):(11*N):5].add(-shear_rate) #see 'computational tricks' from Andrew Fiore's paper
        
        # #Compute near field shear contribution and add it to the rhs of the system
        saddle_b = saddle_b.at[11*N:].add(compute_RFE(shear_rate,r_lub,indices_i_lub,indices_j_lub,XG11,XG12,YG11,YG12,YH11,YH12))
        
        if(T>0):
            #Generate random numbers for far-field random forces
            key_ffwave, random_array_wave = generate_random_array(key_ffwave, (3 * 2 * len(normal_indices)+ 3 * len(nyquist_indices)) )
            key_ffreal, random_array_real = generate_random_array(key_ffreal, (11*N) )
            key_nf, random_array_nf = generate_random_array(key_nf, (6*N) )
            #Compute far-field slip velocity and set in rhs of linear system
            saddle_b = saddle_b.at[:11*N].set(compute_farfield_slipvelocity(n_iter_Lanczos_ff,gridk,random_array_wave,random_array_real,
                                                                            all_indices_x,all_indices_y,all_indices_z,gaussian_grid_spacing1,gaussian_grid_spacing2,
                                                                            r,indices_i,indices_j,f1,f2,g1,g2,h1,h2,h3))
            #Compute near-field random forces and set in rhs of linear system
            saddle_b = saddle_b.at[11*N:].set(compute_nearfield_brownianforce(random_array_nf, r_lub, indices_i_lub, indices_j_lub, XA11, XA12, YA11, YA12, YB11, YB12, XC11, XC12, YC11, YC12, YB21))
        
        saddle_x, exitcode = solver(
            saddle_b,  # rhs vector of the linear system
            gridk,
            R_fu_prec_lower_triang,
            all_indices_x,all_indices_y,all_indices_z,gaussian_grid_spacing1,gaussian_grid_spacing2,
            r,indices_i,indices_j,f1,f2,g1,g2,h1,h2,h3,
            r_lub,indices_i_lub,indices_j_lub,XA11,XA12,YA11,YA12,YB11,YB12,XC11,XC12,YC11,YC12,YB21
            )

        # update positions        
        positions, displacements_vector_matrix, nbrs_lub, nbrs_lub_prec, nbrs_ff = update_neigh_list_and_displacements(shear_rate,positions, displacements_vector_matrix, nbrs_lub, nbrs_lub_prec, nbrs_ff,saddle_x.at[11*N:].get(),dt)
        
        # update grid distances for FFT (same for each gaussian support)
        gaussian_grid_spacing = precompute_grid_distancing(gaussP, gridh[0], xy, positions)
        
        #Compute/Update k grid
        if(shear_rate != 0):
            xy = update_box_tilt_factor(dt,shear_rate,xy)
            gridk = compute_sheared_grid(int(Nx),int(Ny),int(Nz),xy)  
        
        n_pairs_lub_prec = len(nbrs_lub_prec.idx[0])
        nl_lub = np.array(nbrs_lub.idx)
        nl_lub_prec = np.array(nbrs_lub_prec.idx)
        nl_ff = np.array(nbrs_ff.idx)
        
        if((step%writing_period)==0):
            trajectory[int(step/writing_period),:,:] = positions
            np.save(outputfile, trajectory)  
            print('Step is ',step, ' out of ', Nsteps) 
        
    return 



positions = jnp.array([[-5., 0., 0.],
                         [0., 0., 0.],
                         [7., 0., 0.]],float)

Nsteps = 1000 # number of simulation steps 
N=len(positions) #number of particles
writing_period = 10
L=50.0 #box size
dt=0.05 #time step
kT = 0. #temperature
shear_rate = 0.0 #shear rate
gravity = -1 #set gravitational accelleration due to density mismatch (if any)
outputfile = 'DancingSpheres' #name of file where trajectory will be stored 


seed_RFD = 73247 #matters only if kt>0
seed_ffwave = 83909 #matters only if kt>0
seed_ffreal = 53116 #matters only if kt>0
seed_nf = 20129 #matters only if kt>0


ewald = main(Nsteps, writing_period, dt, L, L, L, N, 0.5, kT, 1,
     0.5, 0.001, 0, 0, 'None', positions,seed_RFD,seed_ffwave,seed_ffreal,seed_nf,shear_rate,gravity,outputfile)


