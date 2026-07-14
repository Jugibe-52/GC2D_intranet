# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Hamiltonian-system representation and symbolic vector-field creation."""

from functools import partial
from typing import Callable

import numpy as xp
import sympy as sp

from .solution import OdeSolution


class HamSys:
	def __init__(self, ndof:float=1) -> None:
		if str(ndof) != str(int(ndof)) + '.5' * bool(str(ndof).count('.5')):
			raise ValueError('Number of degrees of freedom should be an integer or half an integer.')
		self._ndof = int(ndof)
		self._time_dependent = bool(str(ndof).count('.5'))

	def _split(self, y:xp.ndarray, by_var:bool=False, ext:bool=True, check_energy:bool=False):
		if not check_energy:
			return xp.split(y, 2 + 2 * by_var * ext)
		ndof = self._ndof if not ext else 2 * self._ndof
		np = ndof * len(y) // (2 * ndof + 1)
		if not by_var:
			return [y[:np], y[np:2*np], y[2*np:]]
		return [y[:np//2], y[np//2:np], y[np:3*np//2], y[3*np//2:2*np], y[2*np:]]

	def _create_function(self, t:float, y:xp.ndarray, eqn:Callable) -> xp.ndarray:
		y_ = xp.split(y, 2)
		return xp.asarray(eqn(y_[0], y_[1], t)).flatten()

	def rectify_sol(self, sol:OdeSolution, check_energy:bool=False) -> OdeSolution:
		if not check_energy:
			return sol
		if self._time_dependent:
			vec = self._split(sol.y, ext=False, check_energy=True)
			sol.y = xp.concatenate((vec[0], vec[1]), axis=0)
			sol.k = vec[2]
		sol.err = self.compute_energy(sol)
		return sol

	def compute_vector_field(self, hamiltonian:Callable, output:bool=False) -> None:
		q = sp.symbols('q0:%d'%self._ndof) if self._ndof>=2 else sp.Symbol('q')
		p = sp.symbols('p0:%d'%self._ndof) if self._ndof>=2 else sp.Symbol('p')
		t = sp.Symbol('t')
		energy = sp.lambdify([q, p, t], hamiltonian(q, p, t))
		self.hamiltonian = partial(self._create_function, eqn=energy)
		eqn = sp.simplify(sp.derive_by_array(hamiltonian(q, p, t), [q, p]).doit())
		eqn = sp.flatten([eqn[1], -eqn[0]])
		if output:
			print('y_dot : ', eqn)
		eqn = sp.lambdify([q, p, t], eqn)
		self.y_dot = partial(self._create_function, eqn=eqn)
		eqn_t = -sp.simplify(sp.diff(hamiltonian(q, p, t), t))
		if output and eqn_t!=0:
			print('k_dot : ', eqn_t)
		eqn_t = sp.lambdify([q, p, t], eqn_t)
		self.k_dot = partial(self._create_function, eqn=eqn_t)

	def compute_energy(self, sol:OdeSolution, maxerror:bool=True) -> xp.ndarray:
		if not hasattr(self, 'hamiltonian'):
			raise ValueError("In order to check energy, the attribute 'hamiltonian' must be provided.")
		val_h = self.hamiltonian(sol.t[xp.newaxis], sol.y)
		if self._time_dependent:
			val_h += sol.k
			val_h -= val_h[:, 0][:, xp.newaxis]
		return xp.max(xp.abs(val_h - val_h[:, 0][:, xp.newaxis])) if maxerror else val_h
