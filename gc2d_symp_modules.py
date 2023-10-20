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
from scipy.io import savemat
from pyhamsys import HamSys, solve_ivp_sympext
import time
from datetime import date

def run_method(case):
	print(f"\033[92m   Integration of {case.__str__()} \033[00m")
	start = time.time()
	hs = HamSys(ndof=1.5)
	hs.y_dot, hs.k_dot = case.xy_dot, case.k_dot
	hs.hamiltonian = case.potential
	y0 = case.initial_conditions(type=case.init)
	t_eval = 2 * xp.pi * xp.arange(0, case.Tf + 1)
	sol = solve_ivp_sympext(hs, (0, t_eval.max()), y0, step=case.TimeStep, t_eval=t_eval, method=case.ode_solver, check_energy=case.CheckEnergy)
	print(f'\033[90m        Computation finished in {int(time.time() - start)} seconds \033[00m')
	if case.CheckEnergy:
		print(f'\033[90m           with error in energy = {sol.err}')
	save_data(case, sol, 'data')

def save_data(case, sol, filestr:str, info=[]):
	if case.SaveData:
		x, y = xp.split(sol.y, 2)
		mdic = case.DictParams.copy()
		mdic.update({'t': sol.t, 'x': x, 'y': y, 'info': info})
		if case.CheckEnergy:
			mdic.update({'k': sol.k})
		mdic.update({'date': date.today().strftime(" %B %d, %Y\n"), 'author': 'cristel.chandre@cnrs.fr'})
		savemat(filestr + '.mat', mdic)
		print(f'\033[90m        Results saved in {filestr}.mat \033[00m')