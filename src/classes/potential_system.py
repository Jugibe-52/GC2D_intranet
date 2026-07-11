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

import time
import os
from typing import Literal

os.environ.setdefault("MPLCONFIGDIR", ".matplotlib")
import numpy as np
from pyhamsys import HamSys

from contracts import TrajectoryParams
from .potential import Array, Potential, real_imag

class PotentialSystem(Potential, HamSys):
	"""Particle dynamics over an independent, optionally gyroaveraged Potential."""

	def __str__(self) -> str:
		return f'2D Guiding Center ({self.__class__.__name__}) for turbulent potentials'
		
	def __init__(self, potential: Potential, traj: TrajectoryParams) -> None:
		HamSys.__init__(self, ndof=1.5 if traj["type"]=='gc' else 2.5)
		self.traj = traj.copy()
		self.rho = self.traj.get("rho", 0)
		self.eta = self.traj.get("eta", 0)
		if min(potential.kinterp * potential.dx, potential.kinterp * potential.dy) < self.rho:
			raise ValueError(
				f"Interpolation order {potential.kinterp} is too low for rho = {self.rho}. "
				"Increase k or decrease rho."
			)
		mean_value, fluctuations = potential.fields
		fields = (
			None if mean_value is None else mean_value.copy(),
			None if fluctuations is None else [field.copy() for field in fluctuations],
		)
		if self.rho != 0:
			fields = potential.gyroaverage(self.rho, fields)
		Potential.__init__(
			self,
			x=potential.x.copy(),
			y=potential.y.copy(),
			fields=fields,
			freqs=potential.freqs.copy(),
			xy_period=potential.xy_period,
			k=potential.kinterp,
		)
		if self.traj["type"] == 'fo':
			self.v_fo = self.rho / (2 * np.abs(self.eta))
			self.phi_fo = np.sign(self.eta) / self.rho
			self.omlar = 1 / (2 * self.eta)

	def initial_conditions(
		self,
		n_traj: int,
		x: Array | None = None,
		y: Array | None = None,
		type: Literal['random', 'fixed'] = 'fixed',
	) -> Array:
		x, y = self.x if x is None else x, self.y if y is None else y
		if type == 'random':
			np.random.seed(int(time.time()))
			x0 = (x[-1] - x[0]) * np.random.rand(n_traj) + x[0]
			y0 = (y[-1] - y[0]) * np.random.rand(n_traj) + y[0]
			z0 = np.concatenate((x0, y0), axis=None)
		elif type == 'fixed':
			n_traj = int(np.sqrt(n_traj))**2
			x0 = np.linspace(x[0], x[-1], int(np.sqrt(n_traj)), endpoint=False)
			y0 = np.linspace(y[0], y[-1], int(np.sqrt(n_traj)), endpoint=False)
			x0, y0 = np.meshgrid(x0, y0, indexing='ij')
			z0 = np.concatenate((x0.flatten(), y0.flatten()), axis=None)
		else:
			raise ValueError("`type` must be either 'random' or 'fixed'.")
		if self.traj["type"] == 'fo':
			np.random.seed(int(time.time()))
			phi_perp = 2 * np.pi * np.random.rand(n_traj)
			z0 = np.concatenate((z0, np.cos(phi_perp), np.sin(phi_perp)), axis=None)
		return np.asarray(z0)
	
	def get_positions(self, z: Array) -> tuple[Array, Array]:
		x, y = np.split(z if self.traj["type"] == 'gc' else np.split(z, 2)[0], 2)
		return x, y
	
	def get_velocities(self, z: Array) -> tuple[Array, Array] | None:
		if self.traj["type"] == 'gc':
			return None
		vx, vy = np.split(np.split(z, 2)[1], 2)
		return vx, vy

	def hamiltonian(self, t: float, z: Array) -> float | Array:
		x, y = self.get_positions(z)
		phi_t = float(np.sum(self.field_at_time(t, x, y)))
		if self.traj["type"] == 'gc':
			return phi_t
		else:
			velocities = self.get_velocities(z)
			if velocities is None:
				raise RuntimeError("Full-orbit velocities are unavailable.")
			vx, vy = velocities
			return float(np.sum(self.rho / (4 * np.abs(self.eta)) * (vx**2 + vy**2) + phi_t * np.sign(self.eta) / self.rho))

	def y_dot(self, t: float, z: Array, output: Literal['full', 'reduced'] = 'full') -> Array:
		x, y = self.get_positions(z)
		dphidx_t = self.field_at_time(t, x, y, dx=1)
		dphidy_t = self.field_at_time(t, x, y, dy=1)
		if self.traj["type"] == 'gc' or output == 'reduced':
			return np.concatenate((-dphidy_t, dphidx_t), axis=None)
		else:
			velocities = self.get_velocities(z)
			if velocities is None:
				raise RuntimeError("Full-orbit velocities are unavailable.")
			vx, vy = velocities
			return np.concatenate((vx * self.v_fo, vy * self.v_fo, -dphidx_t * self.phi_fo\
						   + vy * self.omlar, -dphidy_t * self.phi_fo - vx * self.omlar), axis=None)

	def y_dot_lyap(self, t: float, z: Array) -> Array:
		if self.traj["type"] == 'fo':
			x, y, vx, vy, *jacobian_parts = np.split(z, 20)
			z = np.concatenate((x, y, vx, vy), axis=None)
			jacobian = np.array(jacobian_parts).reshape((4, 4, -1))
		else:
			x, y, *jacobian_parts = np.split(z, 6)
			z = np.concatenate((x, y), axis=None)
			jacobian = np.array(jacobian_parts).reshape((2, 2, -1))
		z_dot = self.y_dot(t, z)
		d2phidx2_t = self.field_at_time(t, x, y, dx=2)
		d2phidxdy_t = self.field_at_time(t, x, y, dx=1, dy=1)
		d2phidy2_t = self.field_at_time(t, x, y, dy=2)
		A = np.zeros_like(jacobian)
		if self.traj["type"] == 'fo':
			d2phidx2_t *= -self.phi_fo
			d2phidxdy_t *= -self.phi_fo
			d2phidy2_t *= -self.phi_fo
			A[0, 2, :], A[1, 3, :] = self.v_fo * np.ones_like(x), self.v_fo * np.ones_like(x)
			A[2, 3, :], A[3, 2, :] = self.omlar * np.ones_like(x), -self.omlar * np.ones_like(x)
			A[2, 0, :], A[2, 1, :] = d2phidx2_t, d2phidxdy_t
			A[3, 0, :], A[3, 1, :] = d2phidxdy_t, d2phidy2_t
		if self.traj["type"] == 'gc':
			A[0, 0, :], A[0, 1, :] = -d2phidxdy_t, -d2phidy2_t
			A[1, 0, :], A[1, 1, :] = d2phidx2_t, d2phidxdy_t
		J_dot = np.einsum('ijm,jkm->ikm', A, jacobian)
		return np.concatenate((z_dot, J_dot.reshape(-1)), axis=None)

	def k_dot(self, t: float, z: Array) -> float | Array:
		x, y = self.get_positions(z)
		dphidt_t = -float(np.sum(self.field_at_time(t, x, y, dt=1)))
		if self.traj["type"] == 'fo':
			dphidt_t *= -self.phi_fo
		return dphidt_t

	def chi(self, h: float, t: float, z: Array) -> Array:
		x, y, vx, vy = np.split(z, 4)
		exp_ = np.exp(-1j * h * self.omlar)
		x, y = real_imag(x + 1j * y + 1j * self.rho * np.sign(self.eta) * (exp_ - 1) * (vx + 1j * vy)) 
		vx, vy = real_imag(exp_ * (vx + 1j * vy))
		pot = np.split(self.y_dot(t, np.concatenate((x, y), axis=None), output='reduced'), 2)
		vx, vy = real_imag(vx + 1j * vy + h * 1j * (pot[0] + 1j * pot[1]) * self.phi_fo)
		return np.concatenate((x, y, vx, vy), axis=None)
	
	def chi_star(self, h: float, t: float, z: Array) -> Array:
		x, y, vx, vy = np.split(z, 4)
		pot = np.split(self.y_dot(t, np.concatenate((x, y), axis=None), output='reduced'), 2)
		vx, vy = real_imag(vx + 1j * vy + h * 1j * (pot[0] + 1j * pot[1]) * self.phi_fo)
		exp_ = np.exp(-1j * h * self.omlar)
		x, y = real_imag(x + 1j * y + 1j * self.rho * np.sign(self.eta) * (exp_ - 1) * (vx + 1j * vy)) 
		vx, vy = real_imag(exp_ * (vx + 1j * vy))
		return np.concatenate((x, y, vx, vy), axis=None)
	
	def fo2gc(self, z: Array) -> tuple[Array, Array]:
		x, y, vx, vy = np.split(z, 4)
		v = vy + 1j * vx
		theta, rho = np.pi + np.angle(v), self.rho * np.abs(v)
		return x - rho * np.cos(theta), y + rho * np.sin(theta)
