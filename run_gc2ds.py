###################################################################################################
##                      Parameters: https://github.com/cchandre/GC2D_intranet                    ##
###################################################################################################

import numpy as xp
import multiprocessing as mp
from gc2ds_classes import GC2Ds

# potential
A = 0.6
M = 25

# parameters
Ntraj = 20
n_max = 500

n_data = 200

parameters = {"A": A, "M": M, "CheckEnergy": True, "Lyapunov": False}
hs = GC2Ds(parameters)
z0 = hs.initial_conditions(Ntraj, type="random")

t_eval = 2 * xp.pi * xp.arange(n_max)
#sol = hs.integrate(z0, t_eval, timestep=5e-2, solver='BM4')

#lyap = hs.compute_lyapunov(2 * xp.pi * n_max, z0, reortho_dt=1, tol=1e-10, solver='RK45')
#print(lyap)

# Plot of the Poincaré section
#hs.plot_sol(sol, wrap=True)
#hs.plot_sol(sol)

mode = 'omega'

if mode == 'omega':
    param_list = xp.logspace(-1, 2, n_data)  
elif mode == 'step':
    param_list = xp.logspace(-2, 0, n_data)[::-1]  
else:
    raise ValueError("Mode must be 'omega' or 'step'")

def run_one(param):
    step = param if mode == 'step' else 1e-1
    om = param if mode == 'omega' else 10 
    sol = hs.integrate(z0, t_eval, timestep=step, omega=om, display=False)
    print(f"{mode} = {param:.3e}   error = {sol.err / Ntraj}  dist_copy = {sol.dist_copy}")
    return (param, sol.err / Ntraj, sol.dist_copy)

if __name__ == '__main__':
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(run_one, param_list)

    sorted_results = sorted(results, key=lambda pair: pair[0])
    hs.save_data(sorted_results, info=mode)
