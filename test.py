import numpy as xp
import sympy as sp
import matplotlib.pyplot as plt
from pyhamsys import HamSys, solve_ivp_sympext
hs = HamSys()
hamiltonian = lambda q, p, t: p**2 / 2 - sp.cos(q)
hs.compute_vector_field(hamiltonian)
sol = solve_ivp_sympext(hs, (0, 20), xp.asarray([3, 0]), step=1e-1, check_energy=True)
print(f"Error in energy : {sol.err}")
plt.plot(sol.y[0], sol.y[1])
plt.show()