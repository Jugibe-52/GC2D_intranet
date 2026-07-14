# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Trajectory-analysis helpers."""

import matplotlib.pyplot as plt
import numpy as xp
from scipy.optimize import curve_fit
from scipy.stats import linregress
from sklearn.metrics import r2_score

from .solution import OdeSolution


def compute_msd(sol:OdeSolution, plot_data:bool=False, output_r2:bool=False):
	x, y = xp.split(sol.y, 2)
	nt = len(sol.t)
	r2 = xp.zeros(nt)
	for _ in range(nt):
		r2[_] = ((x[:, _:] - x[:, :-_ if _ else None])**2 + (y[:, _:] - y[:, :-_ if _ else None])**2).mean()
	t_win, r2_win = sol.t[nt//8:7*nt//8], r2[nt//8:7*nt//8]
	res = linregress(t_win, r2_win)
	diff_data = [res.slope, res.intercept, res.rvalue**2]
	func_fit = lambda t, a, b: (a * t)**b
	popt = curve_fit(func_fit, t_win, r2_win, bounds=((0, 0.25), (xp.inf, 3)))[0]
	r2_fit = func_fit(t_win, *popt)
	interp_data = [*popt, r2_score(r2_win, r2_fit)]
	if plot_data:
		plt.plot(sol.t, r2, ':', color='r', lw=1)
		plt.plot(t_win, r2_win, '-', color='r', lw=2)
		plt.plot(t_win, r2_fit, '-.', color='r', lw=2)
		plt.xlabel('$t$')
		plt.ylabel('$r^2$')
		plt.show()
	if output_r2:
		return sol.t, r2, diff_data, interp_data
	return diff_data, interp_data
