###################################################################################################
##                      Parameters: https://github.com/cchandre/GC2D_intranet                    ##
###################################################################################################

import numpy as xp
from gc2d_classes import mock_potential, extract_potential, GC2D
from pathlib import Path

# potential
M = 25
A = 0.6
Nx, Ny = 64, 64
potential = mock_potential(A, M, Nx, Ny)

#desktop = Path("/Users/c.chandre/Desktop")
#filename = desktop / 'PHI_filtered.h5'
#potential = extract_potential(filename, nx=Nx, ny=Ny)

# parameters
rho, eta = 0, 0
Ntraj = 20
n_max = 500
traj_type = "gc" # 'gc' (guiding centers) or 'fo' (full orbits) 

extension = True if traj_type == 'gc' else False

traj = {"type": traj_type, "rho": rho, "eta": eta, "CheckEnergy": True}
hs = GC2D(potential, traj, k=5)
z0 = hs.initial_conditions(Ntraj, type="random")

#hs.plot_potential(dx=1)

t_eval = 2 * xp.pi * xp.arange(n_max)

#lyap = hs.compute_lyapunov(2 * xp.pi * n_max, z0, reortho_dt=1, tol=1e-10, solver='RK45')
#print(lyap)

# Poincaré section
sol = hs.integrate(z0, t_eval, timestep=0.1, solver='BM4', extension=extension, check_energy=True)
hs.plot_sol(sol, wrap=True)
#hs.plot_sol(sol)
