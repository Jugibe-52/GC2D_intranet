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
import os
import logging
from typing import Any
os.environ.setdefault("MPLCONFIGDIR", ".matplotlib")
import matplotlib.pyplot as plt
from pyhamsys import solve_ivp_sympext, solve_ivp_symp
import time

from .logging_config import simulation_label

logger = logging.getLogger(__name__)

def run_method(self: Any) -> None:
	logger.info("Starting case: %s", simulation_label(self.DictParams))
	start = time.time()
	y0 = self.initial_conditions(type=self.init)
	logger.info("Initial conditions ready: shape=%s init=%s", y0.shape, self.init)
	t_eval = 2 * xp.pi * xp.arange(0, self.Tf + 1)
	logger.info("Starting integration: solver=%s step=%s samples=%d", self.ode_solver, self.TimeStep, len(t_eval))
	if self.traj_type == 'gc':
		sol = solve_ivp_sympext(self, (0, t_eval.max()), y0, step=self.TimeStep, t_eval=t_eval, method=self.ode_solver, check_energy=self.CheckEnergy)
	elif self.traj_type == 'fo':
		sol = solve_ivp_symp(self.chi, self.chi_star, (0, t_eval.max()), y0, step=self.TimeStep, t_eval=t_eval, method=self.ode_solver)
		sol = self.rectify_sol(sol, check_energy=self.CheckEnergy)
	logger.info("Finished case in %.2f seconds: %s", time.time() - start, simulation_label(self.DictParams))
	if self.CheckEnergy:
		logger.info("Energy error: %s", sol.err)
	self.save_data(sol)
	if getattr(self, 'Method', '').startswith('poincare') and getattr(self, 'PlotResults', False):
		logger.info("Plotting Poincare section: traj=%s modulo=%s", self.traj_type, getattr(self, 'modulo', False))
		fig, ax = plt.subplots(1, 1)
		if self.traj_type == 'gc':
			x, y = xp.split(sol.y, 2)
		else:
			x, y = xp.split(sol.y, 4)[:2]
		if getattr(self, 'modulo', False):
			x, y = x % (2 * xp.pi), y % (2 * xp.pi)
			ax.set_xlim(0, 2 * xp.pi)
			ax.set_ylim(0, 2 * xp.pi)
		ax.plot(x, y, '.', markersize=3 if self.traj_type == 'gc' else 1, markeredgecolor='none')
		ax.set_xlabel('$x$')
		ax.set_ylabel('$y$')
		ax.set_aspect('equal')
		if self.SaveData:
			extension = getattr(self, 'extension', '.png')
			basename = f'{type(self).__name__}_A{self.A:.2f}_RHO{self.rho:.4f}'.replace('.', '')
			filename = f'{basename}{extension}'
			fig.savefig(filename, dpi=getattr(self, 'dpi', 200))
			logger.info("Figure saved in %s", filename)
		if 'agg' not in plt.get_backend().lower():
			plt.pause(0.5)
