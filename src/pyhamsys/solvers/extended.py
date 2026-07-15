# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Extended-phase-space symplectic solver."""

from functools import lru_cache
import sys
from typing import Union

import numpy as xp
from numpy.typing import ArrayLike

from ..solution import OdeSolution
from ..system import HamSys
from ._common import (
	StepCallback,
	_build_integration_targets,
	_build_output_times,
	_step_count,
	_validate_solver_inputs,
)
from .symplectic import solve_ivp_symp


_COUPLING_BASE = xp.array(
    [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]],
    dtype=float,
)
_COUPLING_COS = xp.array(
    [[1, 0, -1, 0], [0, 1, 0, -1], [-1, 0, 1, 0], [0, -1, 0, 1]],
    dtype=float,
)
_COUPLING_SIN = xp.array(
    [[0, -1, 0, 1], [1, 0, -1, 0], [0, 1, 0, -1], [-1, 0, 1, 0]],
    dtype=float,
)


class _ProgressBar:
	"""Small dependency-free progress bar for long integrations."""

	def __init__(self, total: int, every: int = 100) -> None:
		self.total = max(total, 1)
		self.every = every
		self.steps = 0
		self._closed = False

	def update(self, t: float, _y: xp.ndarray) -> None:
		self.steps += 1
		if self.steps % self.every and self.steps < self.total:
			return
		fraction = min(self.steps / self.total, 1.0)
		width = 30
		filled = int(width * fraction)
		bar = "=" * filled
		if filled < width:
			bar += ">" + " " * (width - filled - 1)
		print(
			f"\rsolve_ivp_sympext [{bar}] {fraction:6.1%} "
			f"({self.steps}/{self.total} steps, t={t:.6g})",
			end="",
			file=sys.stderr,
			flush=True,
		)

	def close(self) -> None:
		if not self._closed:
			print(file=sys.stderr, flush=True)
			self._closed = True


def solve_ivp_sympext(
	hs: HamSys,
	y0: ArrayLike,
	step: float,
	t_span: ArrayLike = (0, 6.283185307179586),
	t_eval: Union[ArrayLike, None] = None,
	method: str = 'BM4',
	omega: float = 10,
	command: Union[StepCallback, None] = None,
	check_energy: bool = False,
	*,
	save_step: Union[float, None] = 6.283185307179586/240,
	progress: bool = False,
) -> OdeSolution:
	"""
	Solve an initial value problem for a Hamiltonian system using an explicit
	symplectic approximation obtained by an extension in phase space (see [1]).

	This function numerically integrates a system of ordinary differential
	equations in canonical coordinates given an initial value:

	dy / dt = {y, H(t, y)}
	y(t0) = y0

	Here t is a 1-D independent variable (time), y(t) is an N-D vector-valued
	function (state), and a Hamiltonian H(t, y) and a canonical Poisson bracket
	{. , .} determine the differential equations. The goal is to find y(t)
	approximately satisfying the differential equations, given an initial value
	y(t0)=y0. The state y(t) is of the form (q(t), p(t)).

	If the Hamiltonian has an explicit time dependence, there is the
	possibility to check energy by computing k(t) where k is a canonically
	conjugate variable to time. Its evolution is given by

	dk / dt = -dH / dt
	k(t0)=0

	Parameters
	----------
	hs : HamSys
		Hamiltonian system containing the Hamiltonian vector field. The
		attributes `y_dot` (for dy / dt) should be specified. If there is an
		explicit time dependence and `check_energy` is True, the attribute
		`k_dot` (for dk / dt) should be specified.
	t_span : 2-member sequence
		Interval of integration (t0, tf). The solver starts with t=t0 and
		integrates until it reaches t=tf. Both t0 and tf must be floats or
		values interpretable by the float conversion function.
	y0 : array_like, shape (2n,)
		Initial state y0=(q0, p0) where q0 are the initial positions and p0 the
		initial momenta.
	step : float
		Maximum internal step size.
	save_step : float or None, optional
		Regular interval between stored states. Mutually exclusive with `t_eval`.
	t_eval : array_like or None, optional
		Times at which to store the computed solution. They must be strictly
		increasing and lie within `t_span`. If None (default), use points
		selected by the solver.
	method : string, optional
        Integration methods are listed on https://pypi.org/project/pyhamsys/
		'BM4' is the default.
	omega : float, optional
		Coupling parameter in the extended phase space (see [1])
	command : function of (t, y) or None, optional
		Function to be run at each step size.
	check_energy : bool, optional
		If True, computes the total energy. Default is False.
	progress : bool, optional
		If True, displays a dynamic progress bar on stderr, refreshed every
		100 internal steps. Default is False.

	Returns
	-------
	Bunch object with the following fields defined:
	t : ndarray, shape (n_points,)
		Time points.
	y : ndarray, shape (2n, n_points)
		Values y(t) = (q(t), p(t)) at `t`.
	k : ndarray, shape (n, n_points)
		Values of k(t) at `t` if `check_energy` is True and if the Hamiltonian
		system has an explicit time dependence.
	err : float
		Error in the computation of the total energy, computed only if
		`check_energy` is True.
	step : float
		Step size used in the computation.

	References
	----------
		[1] Tao, M., 2016, "Explicit symplectic approximation of nonseparable
		Hamiltonians: Algorithm and long time performance",
		Phys. Rev. E 94, 043303
	"""
	check_energy_ = check_energy and hs._time_dependent

	@lru_cache(maxsize=256)
	def _coupling(h:float) -> xp.ndarray:
		coupling: xp.ndarray = (
			_COUPLING_BASE
			+ xp.cos(2 * omega * h) * _COUPLING_COS
			+ xp.sin(2 * omega * h) * _COUPLING_SIN
		) / 2
		return coupling

	def _chi_ext(h:float, t:float, y:xp.ndarray) -> xp.ndarray:
		y_ = hs._split(y, check_energy=check_energy_)
		y_[1] += h * hs.y_dot(t, y_[0])
		if check_energy_:
			y_[-1] += h * hs.k_dot(t, y_[0])
		y_[0] += h * hs.y_dot(t, y_[1])
		if check_energy_:
			y_[-1] += h * hs.k_dot(t, y_[1])
		yr: xp.ndarray = xp.concatenate((y_[0], y_[1]), axis=None)
		yr = xp.einsum('ij,j...->i...', _coupling(h), xp.split(yr, 4)).flatten()
		if not check_energy_:
			return yr
		return xp.concatenate((yr, y_[-1]), axis=None)

	def _chi_ext_star(h:float, t:float, y:xp.ndarray) -> xp.ndarray:
		y_ = hs._split(y, by_var=True, check_energy=check_energy_)
		yr = y_ if not check_energy_ else y_[:-1]
		yr = xp.einsum('ij,j...->i...', _coupling(h), yr).flatten()
		if check_energy_:
			yr = xp.concatenate((yr, y_[-1]), axis=None)
		y_ = hs._split(yr, check_energy=check_energy_)
		y_[0] += h * hs.y_dot(t, y_[1])
		if check_energy_:
			y_[-1] += h * hs.k_dot(t, y_[1])
		y_[1] += h * hs.y_dot(t, y_[0])
		if check_energy_:
			y_[-1] += h * hs.k_dot(t, y_[0])
		return xp.concatenate([_ for _ in y_], axis=None)

	if not hasattr(hs, 'y_dot'):
		raise ValueError("The attribute 'y_dot' must be provided.")
	if check_energy_ and not hasattr(hs, 'k_dot'):
		raise ValueError("In order to check energy for a time-dependent system, the attribute 'k_dot' must be provided.")
	initial_state = xp.asarray(y0)
	y_ = xp.tile(initial_state, 2)
	if check_energy_:
		y_ = xp.concatenate((y_, xp.zeros(len(initial_state)//(2*hs._ndof) )), axis=None)
	progress_bar = None
	progress_command = command
	if progress:
		inputs = _validate_solver_inputs(t_span, y_, step, save_step, t_eval)
		output_times = _build_output_times(inputs)
		targets = _build_integration_targets(output_times, inputs.tf)
		total_steps = sum(
			_step_count(float(target - start), inputs.max_step)
			for start, target in zip(targets[:-1], targets[1:])
			if target != start
		)
		progress_bar = _ProgressBar(total_steps)

		def progress_command(t: float, state: xp.ndarray) -> None:
			progress_bar.update(t, state)
			if command is not None:
				command(t, state)

	try:
		sol = solve_ivp_symp(
			_chi_ext,
			_chi_ext_star,
			t_span,
			y_,
			method=method,
			step=step,
			save_step=save_step,
			t_eval=t_eval,
			command=progress_command,
		)
	finally:
		if progress_bar is not None:
			progress_bar.close()
	y_ = hs._split(sol.y, by_var=True, check_energy=check_energy_)
	sol.y = xp.concatenate(((y_[0] + y_[2]) / 2, (y_[1] + y_[3]) / 2), axis=0)
	if check_energy_:
		sol.k = y_[-1] / 2
	if check_energy:
		sol.err = hs.compute_energy(sol)
	return sol
