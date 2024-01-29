import sys
import numpy as np
import math
import random

# Check that ewald_cut is small enough to avoid interaction with the particles images (in real space part of calculation)
def Check_ewald_cut(ewald_cut,Lx,Ly,Lz,error):
    if ((ewald_cut > Lx/2) or (ewald_cut > Ly/2) or (ewald_cut > Lz/2)):
        print(
            'WARNING: Real space Ewald cutoff radius is too large! Increase xi and retry.')
        max_cut = max([Lx, Ly, Lz]) / 2.0
        new_xi = np.sqrt(-np.log(error)) / max_cut
        print('Try with ,', new_xi)
        sys.exit("Exit the program!!")
    else:
        print('Ewald Cutoff is ', ewald_cut)
    return

# Maximum eigenvalue of A'*A to scale support, P, for spreading on deformed grids (Fiore and Swan, J. Chem. Phys., 2018)
def Check_max_shear(gridh, xisq, Nx, Ny, Nz, max_strain, error):

    gamma = max_strain
    lambdaa = 1 + gamma*gamma / 2 + gamma * np.sqrt(1+gamma*gamma/4)

    # Parameters for the Spectral Ewald Method (Lindbo and Tornberg, J. Comp. Phys., 2011)
    gaussm = 1.0
    while (math.erfc(gaussm / np.sqrt(2*lambdaa)) > error):
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

def Compute_k_gridpoint_number(kmax,Lx,Ly,Lz):
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
    Mlist = np.array(Mlist)
    Mlist = np.sort(Mlist)
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

# given a support size (for gaussian spread) -> compute distances in a gaussP x gaussP x gaussP grid (where gauss P might be corrected in it is odd or even...)
def Precompute_grid_distancing(gauss_P, gridh, tilt_factor,positions,N,Nx,Ny,Nz,Lx,Ly,Lz):
    # SEE Molbility.cu (line 265) for tips on how to change the distances when we have shear
    grid = np.zeros((gauss_P, gauss_P, gauss_P,N))
    # center_offsets = (jnp.array(positions)+jnp.array([Lx,Ly,Lz])/2)*jnp.array([Nx,Ny,Nz])/jnp.array([Lx,Ly,Lz]) 
    center_offsets = (np.array(positions)+np.array([Lx,Ly,Lz])/2)
    center_offsets[:,0] += (-tilt_factor*center_offsets[:,1])
    # center_offsets = center_offsets.at[:,0].add(-tilt_factor*center_offsets.at[:,1].get())
    center_offsets = center_offsets / np.array([Lx,Ly,Lz])  * np.array([Nx, Ny, Nz])
    center_offsets -= np.array(center_offsets, dtype = int) #in grid units, not particle radius units
    center_offsets = np.where(center_offsets>0.5, -(1-center_offsets), center_offsets)
    for i in range(gauss_P):
        for j in range(gauss_P):
            for k in range(gauss_P):
                grid[i, j, k, :] = gridh*gridh* ( (i - int(gauss_P/2) - center_offsets[:,0] + tilt_factor*(j - int(gauss_P/2) - center_offsets[:,1])   )**2 
                                                 + (j - int(gauss_P/2) - center_offsets[:,1] )**2 
                                                 + (k - int(gauss_P/2) - center_offsets[:,2] )**2) 
    return grid



#Create configurations of non-overlapping spheres in a cubic box of length L
#The code is slow, but considering the amount of particle in a configuration (<10000), it is still feasible (order of minutes)
#Also, will be extremely slow if used to create high density configurations (limit to 10-20%)
def CreateRandomConfiguration(L,N,seed):
    positions = np.zeros((N,3),float)
    max_attempts = 100000
    attempts=0
    n=0
    random.seed(seed)
    while(n<N):
        attempts += 1
        x = random.uniform(-L/2, L/2)
        y = random.uniform(-L/2, L/2)
        z = random.uniform(-L/2, L/2)
        overlap=0
        for i in range(n):            
            d =  np.linalg.norm(positions[i,:]-np.array([x,y,z]))
            if(d<2.1):
                overlap=1
        if(overlap<1):
            n += 1
            positions[n-1,:] = np.array([x,y,z])
        if(attempts>max_attempts):
            print('Computation too long, abort.')
            break;
    return positions










