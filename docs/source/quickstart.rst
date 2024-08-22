Quickstart
==========

Run the main code via:
```
python jfsd/JFSD.py
```

During the simulation, the particles trajectories,velocities, and stresslets are saved in a numpy array of shape (N_s, N_p, N_c), with N_s the number of frames stored, N_p the number of particles and N_c the number of d.o.f. (3 for trajectories, 6 for velocities, 5 for stresslets).
