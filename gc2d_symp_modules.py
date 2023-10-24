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
from pyhamsys import solve_ivp_sympext, solve_ivp_symp
import time

def run_method(self):
	print(f"\033[92m   Integration of {self.__str__()} \033[00m")
	start = time.time()
	y0 = self.initial_conditions(type=self.init)
	t_eval = 2 * xp.pi * xp.arange(0, self.Tf + 1)
	if self.traj_type == 'gc':
		sol = solve_ivp_sympext(self, (0, t_eval.max()), y0, step=self.TimeStep, t_eval=t_eval, method=self.ode_solver, check_energy=self.CheckEnergy)
	elif self.traj_type == 'fo':
		sol = solve_ivp_symp(self.chi, self.chi_star, (0, t_eval.max()), y0, step=self.TimeStep, t_eval=t_eval, method=self.ode_solver)
		sol = self.rectify_sol(sol, check_energy=self.CheckEnergy)
	print(f'\033[90m        Computation finished in {int(time.time() - start)} seconds \033[00m')
	if self.CheckEnergy:
		print(f'\033[90m           with error in energy = {sol.err}')
	self.save_data(sol)
