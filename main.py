import os
from functools import partial
from jax import random
from jax import jit
import jax.numpy as jnp
import jax.scipy as jscipy
from jax_md import partition, space
from jax.config import config
import math
import numpy as np
# import scipy
import time
import Resistance
import Mobility
import Thermal
import EwaldTables
import Helper
import gmres

config.update("jax_enable_x64", True)
np.set_printoptions(precision=8,suppress=True)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


def check_distances(pos):
    for i in range(len(pos)):
        for j in range(len(pos)):
            dist = np.linalg.norm(pos[i]-pos[j])
            if (i != j):
                if(dist<2.):
                    print('OVERLAP',i,j,dist)
    return

def initialize_neighborlist(a, Lx, Ly, Lz, displacement, ewald_cut):
    # Set various Neighborlist
    # For Lubrication Hydrodynamic Forces Calculation
    lub_neighbor_fn = partition.neighbor_list(displacement,  # Displacement metric
                                              jnp.array([Lx, Ly, Lz]), # Box size
                                                # r_cutoff=0.,  # Spatial cutoff for 2 particles to be neighbor
                                                r_cutoff=4.,  # Spatial cutoff for 2 particles to be neighbor
                                               dr_threshold=0.1,  # displacement of particles threshold to recompute neighbor list
                                              capacity_multiplier=1,
                                              format=partition.NeighborListFormat.OrderedSparse)
    # For Precondition of Lubrication Hydrodynamic Forces Calculation
    prec_lub_neighbor_fn = partition.neighbor_list(displacement,  # Displacement metric
                                                   jnp.array([Lx, Ly, Lz]), # Box size
                                                    r_cutoff=2.1,  # Spatial cutoff for 2 particles to be neighbor
                                                    # r_cutoff=0.,  # Spatial cutoff for 2 particles to be neighbor
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
         shear_rate   # assuming simple axisymmetric shear
         ):


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
        positions = positions - jnp.array([Lx,Ly,Lz]) * 0.5
        
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
            precomputed
            ):
       
        
        def compute_saddle(x):

            # set output to zero to start
            Ax = jnp.zeros(N*axis,float)

            # compute far-field contribution (M * F): output velocities+torques (first 6N) and strain rates (last 5N)
            Ax = Ax.at[:11*N].set(Mobility.GeneralizedMobility(N,int(Nx),int(Ny),int(Nz),
                                                               int(gaussP),gridk,m_self,
                                                               all_indices_x,all_indices_y,all_indices_z,gaussian_grid_spacing1,gaussian_grid_spacing2,
                                                               r,indices_i,indices_j,f1,f2,g1,g2,h1,h2,h3,
                                                               x.at[:11*N].get()))

            # Add B*U (M*F + B*U): modify the first 6N entries of output
            Ax = Ax.at[:6*N].add(x.at[11*N:].get())
                       
            # compute near-field contribution (- R^nf_FU * U)
            Ax = Ax.at[11*N:].set((Resistance.ComputeLubricationFU(x[11*N:],indices_i_lub,indices_j_lub,ResFunctions,r_lub,N )) * (-1))
            
            # Add (B^T * F - RFU * U): modify the last 6N entries of output
            Ax = Ax.at[11*N:].add(x.at[:6*N].get())
            
            return Ax

        def compute_precond(x):

            # set output to zero to start
            Px = jnp.zeros(17*N,float)

            # action of precondition matrix on the first 11*N entries of x is the same as the 
            # identity (indeed, the identity is the precondition matrix for the far field granmobility M)
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
        
        (all_indices_x,all_indices_y,all_indices_z,gaussian_grid_spacing1,gaussian_grid_spacing2,
        r,indices_i,indices_j,f1,f2,g1,g2,h1,h2,h3,
        r_lub,indices_i_lub,indices_j_lub,ResFunctions) = precomputed
        
        
        #TODO: make 'axis' input variable in order to switch between BD-RPY-SD (or a combination of them)
        axis = 17
        ###################################################################################################################
        # solve the linear system
        x, exitCode = jscipy.sparse.linalg.gmres(
            A = compute_saddle, b=rhs, tol=1e-5, restart=50, M=compute_precond) 
            # A = compute_saddle, b=rhs, tol=1e-5, restart=50) 
            
        # x = gmres.gmres(A=compute_saddle, b=rhs, x0=None, n=5, M=compute_precond)    
            
        return x



    
    @jit
    def precompute(positions,gaussian_grid_spacing,nl_ff,nl_lub,displacements_vector_matrix,tilt_factor):
        
        ###Wave Space calculation quantities
        
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
        ind = (jnp.log10((dist_lub - 2.) / ResTable_min) / ResTable_dr)
        ind = ind.astype(int)
        dist_lub_lower = ResTable_dist.at[ind].get()
        dist_lub_upper = ResTable_dist.at[ind+1].get()
        # # Linear interpolation of the Table values
        fac_lub = jnp.where(dist_lub_upper - dist_lub_lower>0.,
            (dist_lub - dist_lub_lower) / (dist_lub_upper - dist_lub_lower),0.)
        

        XA11 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+0] + (
            ResTable_vals[22*(ind+1)+0]-ResTable_vals[22*(ind)+0]) * fac_lub), ResTable_vals[0])

        XA12 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+1] + (
            ResTable_vals[22*(ind+1)+1]-ResTable_vals[22*(ind)+1]) * fac_lub), ResTable_vals[1])

        YA11 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+2] + (
            ResTable_vals[22*(ind+1)+2]-ResTable_vals[22*(ind)+2]) * fac_lub), ResTable_vals[2])

        YA12 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+3] + (
            ResTable_vals[22*(ind+1)+3]-ResTable_vals[22*(ind)+3]) * fac_lub), ResTable_vals[3])

        YB11 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+4] + (
            ResTable_vals[22*(ind+1)+4]-ResTable_vals[22*(ind)+4]) * fac_lub), ResTable_vals[4])

        YB12 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+5] + (
            ResTable_vals[22*(ind+1)+5]-ResTable_vals[22*(ind)+5]) * fac_lub), ResTable_vals[5])

        XC11 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+6] + (
            ResTable_vals[22*(ind+1)+6]-ResTable_vals[22*(ind)+6]) * fac_lub), ResTable_vals[6])

        XC12 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+7] + (
            ResTable_vals[22*(ind+1)+7]-ResTable_vals[22*(ind)+7]) * fac_lub), ResTable_vals[7])

        YC11 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+8] + (
            ResTable_vals[22*(ind+1)+8]-ResTable_vals[22*(ind)+8]) * fac_lub), ResTable_vals[8])

        YC12 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+9] + (
            ResTable_vals[22*(ind+1)+9]-ResTable_vals[22*(ind)+9]) * fac_lub), ResTable_vals[9])
        
        YB21 = -YB12 # symmetry condition
        
        XG11 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+10] + (
            ResTable_vals[22*(ind+1)+10]-ResTable_vals[22*(ind)+10]) * fac_lub), ResTable_vals[10])
        
        XG12 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+11] + (
            ResTable_vals[22*(ind+1)+11]-ResTable_vals[22*(ind)+11]) * fac_lub), ResTable_vals[11])
        
        YG11 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+12] + (
            ResTable_vals[22*(ind+1)+12]-ResTable_vals[22*(ind)+12]) * fac_lub), ResTable_vals[12])
        
        YG12 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+13] + (
            ResTable_vals[22*(ind+1)+13]-ResTable_vals[22*(ind)+13]) * fac_lub), ResTable_vals[13])
        
        YH11 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+14] + (
            ResTable_vals[22*(ind+1)+14]-ResTable_vals[22*(ind)+14]) * fac_lub), ResTable_vals[14])
        
        YH12 = jnp.where(dist_lub >= 2+ResTable_min, (ResTable_vals[22*(ind)+15] + (
            ResTable_vals[22*(ind+1)+15]-ResTable_vals[22*(ind)+15]) * fac_lub), ResTable_vals[15])

        ResFunc = jnp.array([XA11,XA12,YA11,YA12,YB11,YB12,XC11,XC12,YC11,YC12,YB21,
        XG11,XG12,YG11,YG12,YH11,YH12])

        return ((all_indices_x),(all_indices_y),(all_indices_z),gaussian_grid_spacing1,gaussian_grid_spacing2,
                r,indices_i,indices_j,f1,f2,g1,g2,h1,h2,h3,
                r_lub,indices_i_lub,indices_j_lub,ResFunc)

        
    def update_box_tilt_factor(dt,shear_rate,tilt_factor):
        tilt_factor = tilt_factor+dt*shear_rate
        if (tilt_factor >= 0.5):
            tilt_factor = -0.5 + (tilt_factor-0.5)
        return tilt_factor
    
    
    @jit
    def compute_pair_potential(U,indices_i,indices_j,dist):
        Fp = jnp.zeros((N,N,3))
        dist_mod = jnp.sqrt(dist[:,:,0]*dist[:,:,0]+dist[:,:,1]*dist[:,:,1]+dist[:,:,2]*dist[:,:,2]) 
        
        sigma = 2. #particle diameter
        sigma += sigma*0.01 # 1% shift to avoid overlaps for the hydrodynamic integrator (and non-positive def operators) 
        
        sigmdr = sigma / jnp.where(indices_i != indices_j, dist_mod[indices_i,indices_j], 0.)
        sigmdr = jnp.power(sigmdr,48)
        Fp_mod = -(96*U/(dist_mod[indices_i,indices_j]*dist_mod[indices_i,indices_j]) * sigmdr * (1 - sigmdr))
        # print(Fp_mod.shape,dist.shape,indices_i.shape,Fp.shape)
        Fp = Fp.at[indices_i,indices_j,0].set(Fp_mod*dist[indices_i,indices_j,0])
        Fp = Fp.at[indices_i,indices_j,1].set(Fp_mod*dist[indices_i,indices_j,1])
        Fp = Fp.at[indices_i,indices_j,2].set(Fp_mod*dist[indices_i,indices_j,2])
        Fp = Fp.at[indices_j,indices_i,0].set(Fp_mod*dist[indices_j,indices_i,0])
        Fp = Fp.at[indices_j,indices_i,1].set(Fp_mod*dist[indices_j,indices_i,1])
        Fp = Fp.at[indices_j,indices_i,2].set(Fp_mod*dist[indices_j,indices_i,2])
        
        
        
        Fp = jnp.sum(Fp,1)
        
        return 
    
    
    start_time = time.time()
    #######################################################################
    # Set/Calculate a bunch of parameters needed for the entire calculation, while doing some checks
    # Compute the Real Space cutoff for the Ewald Summation in the Far-Field computation
    ewald_cut = jnp.sqrt(- jnp.log(error)) / xi
    # parameter needed to make Chol. decomposition of R_FU converge (initially to 1)
    ichol_relaxer = 1.0
    kmax = int(2.0 * jnp.sqrt(- jnp.log(error)) * xi) + 1  # Max wave number
    # Set number of Lanczos iterations (initially) to 2, for both far- and near-field
    n_iter_Lanczos_ff = 2
    n_iter_Lanczos_nf = 2
    xisq = xi * xi
    # Compute number of grid points in k space
    Nx, Ny, Nz = Helper.Compute_k_gridpoint_number(kmax,Lx,Ly,Lz)
    gridh = jnp.array([Lx, Ly, Lz]) / jnp.array([Nx, Ny, Nz])  # Set Grid Spacing
    quadW = gridh[0] * gridh[1] * gridh[2]
    xy = 0. #set box tilt factor to zero to begin (unsheared box)
    # xy = update_box_tilt_factor(dt,shear_rate,0.)
    
    # Check that ewald_cut is small enough to avoid interaction with periodic images)
    Helper.Check_ewald_cut(ewald_cut,Lx,Ly,Lz,error)
    
    # (Check) Maximum eigenvalue of A'*A to scale support, P, for spreading on deformed grids
    eta, gaussP = Helper.Check_max_shear(gridh, xisq, Nx, Ny, Nz, max_strain, error)
    prefac = (2.0 * xisq / 3.1415926536 / eta) * jnp.sqrt(2.0 * xisq / 3.1415926536 / eta)
    expfac = 2.0 * xisq / eta
    gaussPd2 = jnp.array(gaussP/2, int)
    
    # Get list of reciprocal space vectors, and scaling factor for the wave space calculation
    gridk = compute_sheared_grid(int(Nx),int(Ny),int(Nz),xy)   
    #######################################################################

    ## Set INITIAL Periodic Space and Displacement Metric
    #Box is defined by an upper triangular matrix of the form:
    #    [Lx  Ly*xy  Lz*xz]
    #    [0   Ly     Lz*yz]
    #    [0   0      Lz   ]
    displacement, shift = space.periodic_general(
        jnp.array([[Lx, Ly*xy, Lz*0.],[0., Ly, Lz*0.],[0., 0., Lz]]), fractional_coordinates=False)
    
    # compute matrix of INITIAL displacements between particles (each element is a vector from particle j to i)
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
    ewaldC1 = EwaldTables.Compute_real_space_ewald_table(nR,a,xi) #this version uses numpy long float
    ewaldC1 = jnp.array(ewaldC1)
    
    #######################################################################
    # Load resistance table
    ResTable_dist = jnp.load('ResTableDist.npy')
    ResTable_vals = jnp.load('ResTableVals.npy')
    ResTable_min = 0.0001000000000000
    ResTable_dr = 0.0043050000000000 # Table discretization (log space), i.e. ind * dr.set(log10( h[ind] / min )
    #######################################################################
    AppliedForce = jnp.zeros(3*N,float)
    AppliedTorques = jnp.zeros(3*N,float)
    # AppliedForce = AppliedForce.at[2].set(-1.)
    # AppliedTorques = AppliedTorques.at[0].set(k_n)
    ######################################################################
    # Initialize neighborlists
    lub_neighbor_fn, prec_lub_neighbor_fn, ff_neighbor_fn, pot_neighbor_fn = initialize_neighborlist(
        a, Lx, Ly, Lz, displacement, ewald_cut)

    # Allocate lists for first time - they will be updated at each timestep and if needed, re-allocated too.
    nbrs_lub = lub_neighbor_fn.allocate(positions+jnp.array([Lx,Ly,Lz])/2)
    nbrs_lub_prec = prec_lub_neighbor_fn.allocate(positions+jnp.array([Lx,Ly,Lz])/2)
    nbrs_ff = ff_neighbor_fn.allocate(positions+jnp.array([Lx,Ly,Lz])/2)
    nl_lub = np.array(nbrs_lub.idx)
    nl_ff = np.array(nbrs_ff.idx)
    #######################################################################  
    trajectory = np.zeros((int(Nsteps/writing_period + 1),N,3),float)
    
    # precompute grid distances for FFT (same for each gaussian support)
    gaussian_grid_spacing = Helper.Precompute_grid_distancing(gaussP, gridh[0], xy, positions,N,Nx,Ny,Nz,Lx,Ly,Lz)

    #create RNG states
    key_RFD = random.PRNGKey(seed_RFD)
    key_ffwave = random.PRNGKey(seed_ffwave)
    key_ffreal = random.PRNGKey(seed_ffreal)
    key_nf = random.PRNGKey(seed_nf)
    
    #define epsilon for RFD
    epsilon = error
    
    # #create indices list for spreading random forces on the wave space grid (far field brownian velocity, wave space calculation)    
    normal_indices,normal_conj_indices,nyquist_indices = Thermal.Random_force_on_grid_indexing(Nx,Ny,Nz)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Compilation of first part took ',elapsed_time, 'seconds')
    
    for step in range(Nsteps):
        full_start_time = time.time()

        
        start_time1 = time.time()    
        #initialize generalized velocity (6*N array, for linear and angular components)
        general_velocity = jnp.zeros(6*N, float)
        
        #Define arrays for the saddle point solvers (solve linear system Ax=b, and A is the saddle point matrix)
        saddle_b = jnp.zeros(17*N,float)
        
        # Precompute quantities for far-field and near-field calculation
        (all_indices_x,all_indices_y,all_indices_z,gaussian_grid_spacing1,gaussian_grid_spacing2,
        r,indices_i,indices_j,f1,f2,g1,g2,h1,h2,h3,
        r_lub,indices_i_lub,indices_j_lub,ResFunction) = precompute(positions,gaussian_grid_spacing,nl_ff,nl_lub,displacements_vector_matrix,xy)

        # # compute precondition resistance lubrication matrix
        R_fu_prec_lower_triang, diagonal_elements_for_brownian, diagonal_zeroes_for_brownian = Resistance.RFU_Precondition(
            ichol_relaxer,
            displacements_vector_matrix.at[np.array(nbrs_lub_prec.idx)[0,:],np.array(nbrs_lub_prec.idx)[1,:]].get(),
            N,
            len(nbrs_lub_prec.idx[0]),
            np.array(nbrs_lub_prec.idx)
            )
        end_time1 = time.time()
        
        # print(R_fu_prec_lower_triang)
        # If T>0, compute Brownian Drift and use it to initialize the velocity
        if(T>0): 
            
            start_time2 = time.time()
            
            key_RFD, random_array = generate_random_array(key_RFD,6*N) #get array of random variables
            random_array = -((2*random_array-1)*jnp.sqrt(3))
            #Add random displacement to RHS of linear system
            saddle_b = saddle_b.at[11*N:].set(random_array)
            
            #Perform a displacement in the positive random directions and save it to a buffer
            buffer_positions, buffer_displacements_vector_matrix, buffer_nbrs_lub, buffer_nbrs_lub_prec, buffer_nbrs_ff = update_neigh_list_and_displacements(shear_rate,positions, displacements_vector_matrix, nbrs_lub, nbrs_lub_prec, nbrs_ff, random_array,epsilon/2.)
            buffer_gaussian_grid_spacing = Helper.Precompute_grid_distancing(gaussP, gridh[0], xy, buffer_positions,N,Nx,Ny,Nz,Lx,Ly,Lz)
            buffer_nl_lub = np.array(buffer_nbrs_lub.idx)
            buffer_nl_ff = np.array(buffer_nbrs_ff.idx)     

            #Solve the saddle point problem in the positive direction
            saddle_x = solver(
                saddle_b,  # rhs vector of the linear system
                gridk,
                R_fu_prec_lower_triang,
                precompute(buffer_positions,buffer_gaussian_grid_spacing,buffer_nl_ff,buffer_nl_lub,buffer_displacements_vector_matrix,xy)[::]
                )
            general_velocity = saddle_x.at[11*N:].get()

            
            #Perform a displacement in the negative random directions and save it to a buffer
            buffer_positions, buffer_displacements_vector_matrix, buffer_nbrs_lub, buffer_nbrs_lub_prec, buffer_nbrs_ff = update_neigh_list_and_displacements(shear_rate,positions, displacements_vector_matrix, nbrs_lub, nbrs_lub_prec, nbrs_ff, random_array, -epsilon/2.)
            buffer_gaussian_grid_spacing = Helper.Precompute_grid_distancing(gaussP, gridh[0], xy, buffer_positions,N,Nx,Ny,Nz,Lx,Ly,Lz)
            buffer_nl_lub = np.array(buffer_nbrs_lub.idx)
            buffer_nl_ff = np.array(buffer_nbrs_ff.idx)     
            
            #Solve the saddle point problem in the negative direction
            saddle_x = solver(
                saddle_b,  # rhs vector of the linear system
                gridk,
                R_fu_prec_lower_triang,
                precompute(buffer_positions,buffer_gaussian_grid_spacing,buffer_nl_ff,buffer_nl_lub,buffer_displacements_vector_matrix,xy)
                )
            # general_velocity = saddle_x.at[11*N:].get()
            
            #Take Difference and apply scaling
            general_velocity += (-saddle_x.at[11*N:].get())  
            general_velocity = general_velocity * T / epsilon 
            
            #Reset rhs to zero for next solver
            saddle_b = jnp.zeros(17*N,float) 
            end_time2 = time.time() 
            

        start_time3 = time.time() 
            
        #Set up RHS for another saddle point solver 
        
        AppliedForce = AppliedForce.at[::].add(compute_pair_potential(10*T,indices_i_lub,indices_j_lub,displacements_vector_matrix))
        
        saddle_b = saddle_b.at[(11*N+0)::6].add(-AppliedForce.at[0::3].get()) #Add imposed (-forces) to rhs of linear system
        saddle_b = saddle_b.at[(11*N+1)::6].add(-AppliedForce.at[1::3].get())
        saddle_b = saddle_b.at[(11*N+2)::6].add(-AppliedForce.at[2::3].get())
        saddle_b = saddle_b.at[(11*N+3)::6].add(-AppliedTorques.at[0::3].get()) #Add imposed (-torques) to rhs of linear system
        saddle_b = saddle_b.at[(11*N+4)::6].add(-AppliedTorques.at[1::3].get())
        saddle_b = saddle_b.at[(11*N+5)::6].add(-AppliedTorques.at[2::3].get())
        
        #Add the (-) ambient rate of strain to the right-hand side
        saddle_b = saddle_b.at[(6*N+1):(11*N):5].add(-shear_rate) #see 'computational tricks' from Andrew Fiore's paper
        
        # #Compute near field shear contribution R_FE and add it to the rhs of the system
        saddle_b = saddle_b.at[11*N:].add(Resistance.compute_RFE(N,shear_rate,r_lub,indices_i_lub,indices_j_lub,ResFunction[11],ResFunction[12],
                                                      ResFunction[13],ResFunction[14],ResFunction[15],ResFunction[16]))
        end_time3 = time.time() 
        

        if(T>0):
            
            #Generate random numbers for far-field random forces
            key_ffwave, random_array_wave = generate_random_array(key_ffwave, (3 * 2 * len(normal_indices)+ 3 * len(nyquist_indices)) )
            key_ffreal, random_array_real = generate_random_array(key_ffreal, (11*N) )
            key_nf, random_array_nf = generate_random_array(key_nf, (6*N) )
            
            
            start_time4 = time.time()

            #Compute far-field slip velocity and set in rhs of linear system
            buffer, n_iter_Lanczos_ff = Thermal.compute_farfield_slipvelocity(
                N,m_self,int(Nx),int(Ny),int(Nz),int(gaussP),T,dt,gridh,
                normal_indices,normal_conj_indices,nyquist_indices,
                n_iter_Lanczos_ff,gridk,random_array_wave,random_array_real,
                all_indices_x,all_indices_y,all_indices_z,gaussian_grid_spacing1,gaussian_grid_spacing2,
                r,indices_i,indices_j,f1,f2,g1,g2,h1,h2,h3,
                )
            saddle_b = saddle_b.at[:11*N].set(buffer)
            end_time4 = time.time()
            
            
            # start_time5 = time.time()
            # # #Compute near-field random forces and set in rhs of linear system
            # buffer, n_iter_Lanczos_nf = Thermal.compute_nearfield_brownianforce(
            # N,T,dt,
            # random_array_nf, r_lub, indices_i_lub, indices_j_lub, 
            # ResFunction[0], ResFunction[1], ResFunction[2], ResFunction[3], ResFunction[4], ResFunction[5], 
            # ResFunction[6], ResFunction[7], ResFunction[8], ResFunction[9], ResFunction[10], 
            # diagonal_elements_for_brownian, R_fu_prec_lower_triang, diagonal_zeroes_for_brownian,n_iter_Lanczos_nf)
            # saddle_b = saddle_b.at[11*N:].set(buffer)
            # end_time5 = time.time()
            

            
        start_time6 = time.time()
        saddle_x = solver(
            saddle_b,  # rhs vector of the linear system
            gridk,
            R_fu_prec_lower_triang,
            [all_indices_x,all_indices_y,all_indices_z,gaussian_grid_spacing1,gaussian_grid_spacing2,
            r,indices_i,indices_j,f1,f2,g1,g2,h1,h2,h3,
            r_lub,indices_i_lub,indices_j_lub,ResFunction]
            )
        
        
        
        end_time6 = time.time()
        
        # update positions        
        positions, displacements_vector_matrix, nbrs_lub, nbrs_lub_prec, nbrs_ff = update_neigh_list_and_displacements(shear_rate,positions, 
                                                                                                                       displacements_vector_matrix, 
                                                                                                                       nbrs_lub, nbrs_lub_prec, nbrs_ff,
                                                                                                                       saddle_x.at[11*N:].get()+general_velocity,dt)
        
        # update grid distances for FFT (same for each gaussian support)
        gaussian_grid_spacing = Helper.Precompute_grid_distancing(gaussP, gridh[0], xy, positions,N,Nx,Ny,Nz,Lx,Ly,Lz)

        #Compute/Update k grid and box tilt
        if(shear_rate != 0):
            xy = update_box_tilt_factor(dt,shear_rate,xy)
            displacement, shift = space.periodic_general(
                jnp.array([[Lx, Ly*xy, Lz*0.],[0., Ly, Lz*0.],[0., 0., Lz]]), fractional_coordinates=False)
            gridk = compute_sheared_grid(int(Nx),int(Ny),int(Nz),xy)  
        
        nl_lub = np.array(nbrs_lub.idx)
        nl_ff = np.array(nbrs_ff.idx)
        
        full_end_time = time.time()
        
        if((step%writing_period)==0):
            
            n_iter_Lanczos_ff=2
            n_iter_Lanczos_nf=2
            
            trajectory[int(step/writing_period),:,:] = positions
            # np.save('testshear_Final', trajectory)  
            # print(positions,step)
            print(saddle_x.at[11*N:].get())
            print(end_time1-start_time1, 'I (precompute + RFU precond)')
            print(end_time2-start_time2, 'II (RFD)')
            print(end_time3-start_time3, 'III (pair_potential + shear lubrication and strain rate)')
            print(end_time4-start_time4, 'IV.1 (far-field thermal)')
            # print(end_time5-start_time5, 'IV.2 (near-field thermal)')
            print(end_time6-start_time6, 'V (saddle solver and update)')
            print(full_end_time-full_start_time, 'Total')
        

        
    return 

# positions = jnp.array([[0., 1.01, 0.],[0.,-1.01,0.]], float)
# positions = jnp.array([[0., 1.01, 0.],[0.,-1.01,0.],[0.,-3.02,0.]])


Nsteps = 10
# Nsteps = 2
# N=len(positions)
N=1000
writing_period = 1
L=50.0
dt=0.1
init_positions_seed= 543767

positions = Helper.CreateRandomConfiguration(L, N, init_positions_seed)
positions = jnp.array(positions)
print('Vol Fraction is ', N/(L*L*L)*4*np.pi/3)

print(positions)
# check_distances(positions)

seed_RFD = 73247
seed_ffwave = 83909
seed_ffreal = 13651
seed_nf = 53465477 

kT = 0.01
shear_rate = 0.1
main(Nsteps, writing_period, dt, L, L, L, N, 0.5, kT, 1 ,
     0.5, 0.001, 0, 0,'None', positions,seed_RFD,seed_ffwave,seed_ffreal,seed_nf,shear_rate)
