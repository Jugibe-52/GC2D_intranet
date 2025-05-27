###################################################################################################
##                      Parameters: https://github.com/cchandre/GC2D_intranet                    ##
###################################################################################################

import numpy as xp
import matplotlib.pyplot as plt
from gc2d_classes import mock_potential, Potential, GC2Ds

# potential
M = 25
A = 0.6
Nx, Ny = 64, 64
potential = mock_potential(A, M, Nx, Ny)

# parameters
rho, eta = 0, 0
Ntraj = 40
t_max = 1500
traj_type = "gc" # 'gc' (guiding centers) or 'fo' (full orbits) 

traj = {"type": traj_type, "rho": rho, "eta": eta}
hs = GC2Ds(potential, traj)
z0 = hs.initial_conditions(Ntraj, type="random")

t_eval = 2 * xp.pi * xp.arange(t_max)
sol = hs.integrate(z0, t_eval, timestep=1e-1)

# Plot of the Poincaré section
x, y = xp.split(sol.y, 2)
plt.plot(x % potential.period, y % potential.period, '.', color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

