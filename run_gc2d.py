###################################################################################################
##                      Parameters: https://github.com/cchandre/GC2D_intranet                    ##
###################################################################################################

import numpy as xp
import multiprocessing as mp
from gc2d_classes import mock_potential, extract_potential, GC2D
from pathlib import Path
import matplotlib.pyplot as plt

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

traj = {"type": traj_type, "rho": rho, "eta": eta, "CheckEnergy": True}
hs = GC2D(potential, traj, k=5)
z0 = hs.initial_conditions(Ntraj, type="random")

hs.plot_potential(dx=1)

#t_eval = 2 * xp.pi * xp.arange(n_max)
#sol = hs.integrate(z0, t_eval, timestep=0.1, solver='BM4')
#print(sol.dist_copy)

#lyap = hs.compute_lyapunov(2 * xp.pi * n_max, z0, reortho_dt=1, tol=1e-10, solver='RK45')
#print(lyap)

# Plot of the Poincaré section
#hs.plot_sol(sol, wrap=True)
#hs.plot_sol(sol)

# mode = 'omega'

# if mode == 'omega':
#     param_list = xp.logspace(-1, 2, 31)  
# elif mode == 'step':
#     param_list = xp.logspace(-2, 0, 31)[::-1]  
# else:
#     raise ValueError("Mode must be 'omega' or 'step'")

# def run_one(param):
#     step = param if mode == 'step' else 1e-1
#     om = param if mode == 'omega' else 10 
#     sol = hs.integrate(z0, t_eval, timestep=step, omega=om, display=False)
#     print(f"{mode} = {param:.3e}   error = {sol.err / Ntraj}  dist_copy = {sol.dist_copy}")
#     return (param, sol.err / Ntraj, sol.dist_copy)

# if __name__ == '__main__':
#     with mp.Pool(processes=mp.cpu_count()) as pool:
#         results = pool.map(run_one, param_list)

#     sorted_results = sorted(results, key=lambda pair: pair[0])
#     print(sorted_results)
