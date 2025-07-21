###################################################################################################
##                      Parameters: https://github.com/cchandre/GC2D_intranet                    ##
###################################################################################################

import numpy as xp
from gc2d_classes import mock_potential, extract_potential, GC2D
from pathlib import Path

# potential
#M = 25
#A = 0.6
#Nx, Ny = 64, 59
#potential = mock_potential(A, M, Nx, Ny)

desktop = Path("/Users/c.chandre/Desktop")
filename = desktop / 'PHI_filtered.h5'
potential = extract_potential(filename)

# parameters
rho, eta = 0, 0
Ntraj = 10
n_max = 1000
traj_type = "gc" # 'gc' (guiding centers) or 'fo' (full orbits) 

traj = {"type": traj_type, "rho": rho, "eta": eta, "CheckEnergy": False}
hs = GC2D(potential, traj)
z0 = hs.initial_conditions(Ntraj, type="random")

t_eval = 2 * xp.pi * xp.arange(n_max)
sol = hs.integrate(z0, t_eval, timestep=1e-1)

# Plot of the Poincaré section
#hs.plot_sol(sol, wrap=True)
hs.plot_sol(sol)

