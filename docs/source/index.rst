jfsd documentation
==================

Welcome to the JFSD (Jax-Fast-Stokesian-Dynamics) documentation page!

This repository contains a Python implementation of the Fast Stokesian Dynamics methods (A. Fiore, J. Swan, 2019, https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/fast-stokesian-dynamics/970BD1B80B43E21CD355C7BAD4644D46) which leverages the library JAX to just-in-time compile and run on GPU the simulation. The accuracy of hydrodynamic interactions can be scaled down in order to obtain Rotne-Prager-Yamakawa (up to 2-body effects, and no lubrication) or Brownian Dynamics (only 1-body effects). Boundary conditions can be switch from periodic to open, and the average height of asperities on particles can be specified, enabling simulations of rough colloids.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
