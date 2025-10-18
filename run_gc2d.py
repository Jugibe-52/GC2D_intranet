###################################################################################################
##                      Parameters: https://github.com/cchandre/GC2D_intranet                    ##
###################################################################################################

import numpy as xp
from gc2d_classes import mock_potential, extract_potential, GC2D
from pathlib import Path

# potential
Nx, Ny = 128, 128
B = 1.5

# M = 25
# potential = mock_potential(A, M, Nx, Ny)

path = Path("/Users/cchandre/Desktop")
filename = path / 'PHI_filtered.h5'
potential = extract_potential(filename, B=B, nx=Nx, ny=Ny)

# parameters
rho, eta = 0, 0
Ntraj = 20
n_max = 500
traj_type = "gc" # 'gc' (guiding centers) or 'fo' (full orbits) 

traj = {"type": traj_type, "rho": rho, "eta": eta, "CheckEnergy": True}
hs = GC2D(potential, traj, k=5)
z0 = hs.initial_conditions(Ntraj, x=[0.19, 0.21], y=[0.19, 0.21], type="random")

# hs.plot_potential()

t_eval = 2 * xp.pi * xp.arange(n_max)

#lyap = hs.compute_lyapunov(2 * xp.pi * n_max, z0, reortho_dt=1, tol=1e-10, solver='RK45')
#print(lyap)

# Poincaré section
sol = hs.integrate(z0, t_eval, solver='BM4', timestep=5e-2, omega=10, diss=5,\
                   extension=True if traj_type == 'gc' else False, check_energy=True)
hs.plot_sol(sol, wrap=False)
