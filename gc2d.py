#
# BSD 2-Clause License
#
# Copyright (c) 2023, Cristel Chandre
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as xp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pyhamsys import solve_ivp_sympext, solve_ivp_symp, OdeSolution
from gc2d_dict import dict_list, Parallelization
from gc2d_classes import GC2Dt, Trajectory, glue_sol, save_data
import multiprocess
import time
import os

def main() -> None:
	if Parallelization == 'all':
		num_cores = multiprocess.cpu_count()
	else:
		num_cores = min(multiprocess.cpu_count(), Parallelization)
	if num_cores >= 2:
		pool = multiprocess.Pool(num_cores)
		pool.map(lambda dict_: run_method(GC2Dt(dict_)), dict_list)
	else:
		for dict_ in dict_list:
			run_method(GC2Dt(dict_))
	plt.show()

def run_method(self):
	if self.PlotResults:
		cs = ['k', 'w'] if self.darkmode else ['w', 'k']
		plt.rc('figure', facecolor=cs[0], titlesize=30, figsize=[8,8])
		plt.rc('text', usetex=True, color=cs[1])
		plt.rc('font', family='serif', size=24)
		plt.rc('axes', facecolor=cs[0], edgecolor=cs[1], labelsize=30, labelcolor=cs[1], titlecolor=cs[1])
		plt.rc('xtick', color=cs[1], labelcolor=cs[1])
		plt.rc('ytick', color=cs[1], labelcolor=cs[1])
	print(f"\033[92m   Integration of {self.__str__()} \033[00m")
	print(f'\033[92m    A = {self.A:.2f}   rho = {self.rho:.2f} \033[00m')
	filestr = f'{type(self).__name__}_A{self.A:.2f}_RHO{self.rho:.4f}'.replace('.', '')
	start = time.time()
	y0 = self.initial_conditions(type=self.init)
	t_eval = 2 * xp.pi * xp.arange(0, self.Tf + 1)

	def _integr(teval:xp.ndarray, y0_:xp.ndarray) -> OdeSolution:
		if self.Method.endswith('gc'):
			return solve_ivp_sympext(self, (teval.min(), teval.max()), y0_, step=self.TimeStep, t_eval=teval, method=self.ode_solver, check_energy=self.CheckEnergy)
		elif self.Method.endswith('fo'):
			sol = solve_ivp_symp(self.chi, self.chi_star, (teval.min(), teval.max()), y0_, step=self.TimeStep, t_eval=teval, method=self.ode_solver)
			return self.rectify_sol(sol, check_energy=self.CheckEnergy)
		
	if not self.TwoStepIntegration:
		sol = _integr(t_eval, y0)
		trapped = Trajectory(sol, 'trapped', self.DictParams)
	else:
		sol = _integr(t_eval[:self.Tmid + 1], y0)
		trapped = Trajectory(sol, 'trapped', self.DictParams)
		untrapped = Trajectory(sol, ['diffusive', 'ballistic'], self.DictParams)
		y0_ = untrapped.sol[:, -1]
		print(f'\033[90m        Continuing with the integration of {untrapped.size} untrapped particles... \033[00m')
		sol = glue_sol(trapped.remove_trapped(sol), _integr(t_eval[self.Tmid:], y0_), check_energy=self.CheckEnergy)
	diffusive = Trajectory(sol, 'diffusive', self.DictParams)
	ballistic = Trajectory(sol, 'ballistic', self.DictParams)

	print(f'\033[90m        Computation finished in {int(time.time() - start)} seconds \033[00m')
	if self.CheckEnergy:
		print(f'\033[90m           with error in energy = {sol.err}')

	data = [trapped, diffusive, ballistic]
	info = 'Trapped / Diffusive / Ballistic'
	save_data(self, data, filestr, info=info)

	if self.Method.startswith('poincare') and self.PlotResults:
		fig, ax = plt.subplots(1, 1)
		ax.set_xlabel('$x$')
		ax.set_ylabel('$y$')
		for traj in [trapped, diffusive, ballistic]:
			if traj.size:
				x, y = (traj.x  % (2 * xp.pi), traj.y  % (2 * xp.pi)) if self.modulo else (traj.x, traj.y)
				ax.plot(x, y, '.', color=traj.color, markersize=3 if self.Method.endswith('_gc') else 1, markeredgecolor='none')
				if self.Method == "poincare_fo":
					xgc, ygc = (traj.xgc  % (2 * xp.pi), traj.ygc  % (2 * xp.pi)) if self.modulo else (traj.xgc, traj.ygc)
					ax.plot(xgc, ygc, '.', color=traj.color, markersize=3, markeredgecolor='none')
		if self.modulo:
			ax.set_xlim(0, 2 * xp.pi)
			ax.set_ylim(0, 2 * xp.pi)
			ax.set_xticks([0, xp.pi, 2 * xp.pi])
			ax.set_yticks([0, xp.pi, 2 * xp.pi])
			ax.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
			ax.set_yticklabels(['0', r'$\pi$', r'$2\pi$'])
		else:
			ax.add_patch(Rectangle((0, 0), 2 * xp.pi, 2 * xp.pi, facecolor='None', edgecolor='g', lw=2))
			ax.set_aspect('equal')
		if self.SaveData:
			fig.savefig(filestr + self.extension, dpi=self.dpi)
			print(f'\033[90m        Figure saved in {filestr}{self.extension} \033[00m')
		plt.pause(0.5)

	if self.Method.startswith('diffusion'):
		vec_data = [self.A, self.rho, trapped.size / self.Ntraj]
		print(f'\033[96m          trap ({trapped.size}) \033[00m')
		for traj in [diffusive, ballistic]:
			if traj.size:
				diff_data, interp_data = traj.compute_data(traj)
				print("\033[96m          {} ({}) : D = ({:.6f}; {:.6f}; {:.6f})  /  interp = ({:.6f}; {:.6f}; {:.6f})".format(traj.type, traj.size, *diff_data, *interp_data))
				vec_data.extend([traj.size / self.Ntraj, *diff_data, *interp_data])
			else:
				vec_data.extend([0, 0, 0, 0, 0, 0, 0])
		file = open(f'{type(self).__name__}_{self.Method}.txt', 'a')
		if os.path.getsize(file.name) == 0:
			file.writelines('%  diffusion laws: r^2 = D t + int   and   r^2 = (a t)^b \n')
			file.writelines('%  A        rho      eta   trapped  diffusive    D       int     R2       a        b        R2      ballistic     D       int      R2      a        b      R2' + '\n')
		file.writelines(' '.join([f'{_:.6f}' for _ in vec_data]) + '\n')
		file.close()

if __name__ == '__main__':
	main()
