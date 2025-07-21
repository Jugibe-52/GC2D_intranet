###################################################################################################
##                      Parameters: https://github.com/cchandre/GC2D_intranet                    ##
###################################################################################################

import numpy as xp
from gc2d_classes import mock_potential, extract_potential, GC2D
from pathlib import Path
import matplotlib.pyplot as plt

# potential
#M = 25
#A = 0.6
Nx, Ny = 64, 59
#potential = mock_potential(A, M, Nx, Ny)

desktop = Path("/Users/c.chandre/Desktop")
filename = desktop / 'PHI_filtered.h5'
potential = extract_potential(filename, nx=Nx, ny=Ny)

# parameters
rho, eta = 0, 0
Ntraj = 20
n_max = 500
traj_type = "gc" # 'gc' (guiding centers) or 'fo' (full orbits) 

traj = {"type": traj_type, "rho": rho, "eta": eta, "CheckEnergy": True}
hs = GC2D(potential, traj, k=5)
z0 = hs.initial_conditions(Ntraj, type="random")

#hs.plot_potential()

t_eval = 2 * xp.pi * xp.arange(n_max)
sol = hs.integrate(z0, t_eval, timestep=5e-2, omega=10)

# Plot of the Poincaré section
#hs.plot_sol(sol, wrap=True)
hs.plot_sol(sol)

