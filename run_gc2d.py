###################################################################################################
##                      Parameters: https://github.com/cchandre/GC2D_intranet                    ##
###################################################################################################

import numpy as np
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", ".matplotlib")

from pyhamsys import solve_ivp_symp, solve_ivp_sympext

from gc2d_classes import GC2D, extract_potential, mock_potential


def main() -> None:
	# potential
	Nx, Ny = 128, 128
	B = 2

	path = Path("/Users/c.chandre/Desktop")
	filename = path / 'PHI.h5'
	indx = [0, 1]
	if filename.exists():
		potential = extract_potential(filename, B=B, indx=indx, nx=Nx, ny=Ny)
	else:
		M = 25
		potential = mock_potential(1 / B, M, Nx, Ny)

	# parameters
	rho, eta = 0, 0
	Ntraj = 20
	n_max = 50
	traj_type = "gc" # 'gc' (guiding centers) or 'fo' (full orbits)

	traj = {"type": traj_type, "rho": rho, "eta": eta}
	hs = GC2D(potential, traj, k=5)
	z0 = hs.initial_conditions(Ntraj, type="random")

	# hs.plot_potential()

	t_eval = 2 * np.pi * np.arange(n_max)

	# lyap = hs.compute_lyapunov(2 * np.pi * n_max, z0, reortho_dt=1, tol=1e-10, solver='RK45')
	# print(lyap)

	# Poincare section
	if traj_type == "gc":
		sol = solve_ivp_sympext(hs, (t_eval.min(), t_eval.max()), z0, step=2e-2, t_eval=t_eval, method='BM4', check_energy=True)
	else:
		sol = solve_ivp_symp(hs.chi, hs.chi_star, (t_eval.min(), t_eval.max()), z0, step=2e-2, t_eval=t_eval, method='BM4')
		sol = hs.rectify_sol(sol, check_energy=True)
	hs.plot_sol(sol, wrap=True)


if __name__ == '__main__':
	main()
