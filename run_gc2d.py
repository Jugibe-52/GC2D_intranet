###################################################################################################
##                      Parameters: https://github.com/cchandre/GC2D_intranet                    ##
###################################################################################################

import numpy as xp
from gc2d_classes import mock_potential, GC2D

# potential
M = 25
A = 0.6
Nx, Ny = 64, 59
potential = mock_potential(A, M, Nx, Ny)

# parameters
rho, eta = 0, 0
Ntraj = 40
t_max = 1500
traj_type = "gc" # 'gc' (guiding centers) or 'fo' (full orbits) 

traj = {"type": traj_type, "rho": rho, "eta": eta}
hs = GC2D(potential, traj)
z0 = hs.initial_conditions(Ntraj, type="random")

t_eval = potential.t_period * xp.arange(t_max)
sol = hs.integrate(z0, t_eval, timestep=1e-1)

# Plot of the Poincaré section
hs.plot_sol(sol, wrap=True)

