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
from scipy.integrate import solve_ivp
from scipy.io import savemat
from pyhamsys import HamSys, solve_ivp_symp, solve_ivp_sympext
import time
from datetime import date

def run_method(case):
	if case.solve_method.startswith('symp'):
		method = f'symplectic integrator ({case.ode_solver})'
	print(f"\033[92m   Integration of {case.__str__()} using {method if case.solve_method.startswith('symp') else 'linear interpolation'} \033[00m")
	t_eval = 2 * xp.pi * xp.arange(0, case.Tf + 1)
	if case.init == 'random':
		y0 = 2 * xp.pi * xp.random.rand(2 * case.Ntraj)
	elif case.init == 'fixed':
		y_vec = xp.linspace(0, 2 * xp.pi, int(xp.sqrt(case.Ntraj)), endpoint=False)
		y_mat = xp.meshgrid(y_vec, y_vec)
		y0 = xp.concatenate((y_mat[0], y_mat[1]), axis=None)
		case.Ntraj = int(xp.sqrt(case.Ntraj))**2
	start = time.time()
	if case.solve_method == 'interp':
		if case.CheckEnergy:
			y0 = xp.concatenate((y0, xp.zeros(case.Ntraj)), axis=None)
		sol = solve_ivp(case.eqn_interp, (0, t_eval.max()), y0, max_step=case.TimeStep, t_eval=t_eval, atol=1, rtol=1)
	elif case.solve_method == 'symp':
		if case.CheckEnergy:
			y0 = xp.concatenate((y0, xp.zeros(case.Ntraj)), axis=None)
		sol = solve_ivp_symp(case.chi, case.chi_star, (0, t_eval.max()), y0, step=case.TimeStep, t_eval=t_eval, method=case.ode_solver)
	elif case.solve_method == 'symp_ext':
		hs = HamSys(ndof=1.5, check_energy=case.CheckEnergy)
		hs.vector_field, hs.vector_field_k = case.eqn_xy, case.eqn_k
		sol = solve_ivp_sympext(hs, (0, t_eval.max()), y0, step=case.TimeStep, t_eval=t_eval, method=case.ode_solver)
	print(f'\033[90m        Computation finished in {int(time.time() - start)} seconds \033[00m')
	if case.CheckEnergy:
		energy = case.compute_energy(sol)
		err_energy = xp.abs(energy - energy[:, 0][:, xp.newaxis])
		print(f'\033[90m           with error in energy = {xp.max(err_energy)}')
	save_data(case, sol.y, 'sol_' + case.solve_method)

def save_data(case, data, filestr, info=[]):
	if case.SaveData:
		y = xp.split(data, 2 + case.CheckEnergy)
		mdic = case.DictParams.copy()
		mdic.update({'x': y[0], 'y': y[1], 'info': info})
		if case.CheckEnergy:
			mdic.update({'k': y[2]})
		mdic.update({'date': date.today().strftime(" %B %d, %Y\n"), 'author': 'cristel.chandre@cnrs.fr'})
		savemat(filestr + '.mat', mdic)
		print(f'\033[90m        Results saved in {filestr}.mat \033[00m')