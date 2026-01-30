###################################################################################################
##                      Parameters: https://github.com/cchandre/GC2D_intranet                    ##
###################################################################################################

import numpy as np
from gc2d_classes import mock_potential, extract_potential, GC2D
from pathlib import Path

# potential
Nx, Ny = 128, 128
B = 2

# M = 25
# potential = mock_potential(1 / B, M, Nx, Ny)

path = Path("/Users/c.chandre/Desktop")
filename = path / 'PHI.h5'
indx = [0, 1]
potential = extract_potential(filename, B=B, indx=indx, nx=Nx, ny=Ny)

# parameters
rho, eta = 0, 0
Ntraj = 20
n_max = 50
traj_type = "gc" # 'gc' (guiding centers) or 'fo' (full orbits) 

traj = {"type": traj_type, "rho": rho, "eta": eta, "CheckEnergy": True}
hs = GC2D(potential, traj, k=5)
z0 = hs.initial_conditions(Ntraj, type="random")

# hs.plot_potential()

t_eval = 2 * np.pi * np.arange(n_max)

# lyap = hs.compute_lyapunov(2 * np.pi * n_max, z0, reortho_dt=1, tol=1e-10, solver='RK45')
# print(lyap)

# Poincaré section
sol = hs.integrate(z0, t_eval, solver='BM4', timestep=2e-2,\
                   extension=True if traj_type == 'gc' else False, check_energy=True)
hs.plot_sol(sol, wrap=True)
