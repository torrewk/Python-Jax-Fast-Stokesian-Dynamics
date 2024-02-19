import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
from functools import partial
from jax import jit
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)


@partial(jit, static_argnums=[0])
def AppliedForces(N, AppliedForce, AppliedTorques, saddle_b, U,indices_i,indices_j,dist, Ucutoff):
    
    def compute_pair_potential(U,indices_i,indices_j,dist):
        Fp = jnp.zeros((N,N,3))
        dist_mod = jnp.sqrt(dist[:,:,0]*dist[:,:,0]+dist[:,:,1]*dist[:,:,1]+dist[:,:,2]*dist[:,:,2]) 
        sigma = 2. #particle diameter
        sigma += sigma*0.01 # 1% shift to avoid overlaps for the hydrodynamic integrator (and non-positive def operators) 
        
        #compute sigma/delta_r for each pair
        sigmdr = sigma / jnp.where(indices_i != indices_j, dist_mod[indices_i,indices_j], 0.)
        sigmdr = jnp.power(sigmdr,48)
        
        #compute forces for each pair
        Fp_mod = (96*U/(dist_mod[indices_i,indices_j]*dist_mod[indices_i,indices_j]) * sigmdr * (1 - sigmdr))
        Fp_mod = jnp.where((dist_mod[indices_i,indices_j]) > sigma*Ucutoff, 0., Fp_mod )
        
        #get forces in components
        Fp = Fp.at[indices_i,indices_j,0].add(Fp_mod*dist[indices_i,indices_j,0])
        Fp = Fp.at[indices_i,indices_j,1].add(Fp_mod*dist[indices_i,indices_j,1])
        Fp = Fp.at[indices_i,indices_j,2].add(Fp_mod*dist[indices_i,indices_j,2])
        Fp = Fp.at[indices_j,indices_i,0].add(Fp_mod*dist[indices_j,indices_i,0])
        Fp = Fp.at[indices_j,indices_i,1].add(Fp_mod*dist[indices_j,indices_i,1])
        Fp = Fp.at[indices_j,indices_i,2].add(Fp_mod*dist[indices_j,indices_i,2])    
        
        #sum all forces in each particle
        Fp = jnp.sum(Fp,1)
        
        return Fp
    
    Fp = compute_pair_potential(U,indices_i,indices_j,dist)
    
    saddle_b = saddle_b.at[(11*N+0)::6].add(-AppliedForce.at[0::3].get() - Fp.at[:,0].get()) #Add imposed (-forces) to rhs of linear system
    saddle_b = saddle_b.at[(11*N+1)::6].add(-AppliedForce.at[1::3].get() - Fp.at[:,1].get())
    saddle_b = saddle_b.at[(11*N+2)::6].add(-AppliedForce.at[2::3].get() - Fp.at[:,2].get())
    saddle_b = saddle_b.at[(11*N+3)::6].add(-AppliedTorques.at[0::3].get()) #Add imposed (-torques) to rhs of linear system
    saddle_b = saddle_b.at[(11*N+4)::6].add(-AppliedTorques.at[1::3].get())
    saddle_b = saddle_b.at[(11*N+5)::6].add(-AppliedTorques.at[2::3].get())
    
    return saddle_b