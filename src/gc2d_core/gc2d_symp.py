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
from typing import Any, Literal
os.environ.setdefault("MPLCONFIGDIR", ".matplotlib")
from scipy.special import jv
import multiprocess
from datetime import datetime
from scipy.io import savemat
from .gc2d_symp_modules import run_method
from .config import load_gc2dt_config
from .logging_config import configure_logging
from pyhamsys import OdeSolution, HamSys

logger = logging.getLogger(__name__)

def main() -> None:
	configure_logging()
	config = load_gc2dt_config(config_group="assay", config_version="v_1", version="symplectic_grid")
	dict_list = config.cases()
	parallelization = config.parallelization
	if parallelization == 'all':
		num_cores = multiprocess.cpu_count()
	else:
		num_cores = min(multiprocess.cpu_count(), int(parallelization))
	logger.info("Prepared %d case(s); parallelization=%s; workers=%d", len(dict_list), parallelization, num_cores)
	if num_cores >= 2:
		pool = multiprocess.Pool(num_cores)
		pool.map(lambda dict_: run_method(GC2Dt(dict_)), dict_list)
	else:
		for dict_ in dict_list:
			run_method(GC2Dt(dict_))
	logger.info("All cases finished")

def real_imag(z: xp.ndarray) -> tuple[xp.ndarray, xp.ndarray]:
	return z.real, z.imag

class GC2Dt(HamSys):
	def __repr__(self) -> str:
		return "{self.__class__.__name__}({self.DictParams})".format(self=self)

	def __str__(self) -> str:
		return f'2D Guiding Center ({self.__class__.__name__}) for turbulent potentials'

	def __init__(self, dict_: dict[str, Any]) -> None:
		super().__init__(ndof=1.5 if dict_['traj_type']=='gc' else 2.5)
		for key in dict_:
			setattr(self, key, dict_[key])
		self.DictParams = dict_
		logger.info(
			"Initializing GC2Dt: traj=%s M=%s A=%s rho=%s eta=%s Ntraj=%s Tf=%s",
			self.traj_type,
			self.M,
			self.A,
			self.rho,
			getattr(self, 'eta', 'n/a'),
			self.Ntraj,
			self.Tf,
		)
		xp.random.seed(27)
		self.phases = 2 * xp.pi * xp.random.random((self.M, self.M))
		self.nm = xp.meshgrid(xp.arange(self.M+1), xp.arange(self.M+1), indexing='ij')
		self.phic = xp.zeros((self.M+1, self.M+1), dtype=xp.complex128)
		self.phic[1:, 1:] = self.A / (self.nm[0][1:, 1:]**2 + self.nm[1][1:, 1:]**2)**1.5 * xp.exp(1j * self.phases)
		sqrt_nm = xp.sqrt(self.nm[0]**2 + self.nm[1]**2)
		self.phic[sqrt_nm > self.M] = 0
		if self.traj_type == 'gc':
			flr1_coeff = jv(0, self.rho * sqrt_nm)
			self.phic *= flr1_coeff
			logger.debug("Applied FLR correction for guiding-center trajectory: rho=%s", self.rho)

		self.fft_phi_ = xp.asarray([-self.nm[1] * self.phic, self.nm[0] * self.phic])	
		active_modes = int(xp.count_nonzero(self.phic))
		logger.info("GC2Dt initialized with %d active Fourier modes", active_modes)

	def fft_phi_grid(self, t: float = 0.0, n: int = 64) -> tuple[xp.ndarray, xp.ndarray, xp.ndarray, xp.ndarray]:
		x = xp.linspace(0, 2 * xp.pi, n, endpoint=False)
		y = xp.linspace(0, 2 * xp.pi, n, endpoint=False)
		X, Y = xp.meshgrid(x, y, indexing='ij')
		state = xp.concatenate((X.ravel(), Y.ravel()))
		vx, vy = xp.split(self.y_dot(t, state), 2)
		return X, Y, vx.reshape(n, n), vy.reshape(n, n)

	def plot_fft_phi(
		self,
		t: float = 0.0,
		n: int = 40,
		kind: Literal['quiver', 'stream', 'magnitude'] = 'quiver',
		ax: Any = None,
		show_magnitude: bool = True,
		density: float = 1.5,
		**kwargs: Any,
	) -> tuple[Any, Any]:
		import matplotlib.pyplot as plt

		if kind not in {'quiver', 'stream', 'magnitude'}:
			raise ValueError("`kind` must be 'quiver', 'stream' or 'magnitude'.")
		X, Y, vx, vy = self.fft_phi_grid(t=t, n=n)
		speed = xp.sqrt(vx**2 + vy**2)
		if ax is None:
			fig, ax = plt.subplots(1, 1, figsize=(6, 6))
		else:
			fig = ax.figure
		if show_magnitude or kind == 'magnitude':
			mesh = ax.pcolormesh(X.T, Y.T, speed.T, shading='auto', cmap=kwargs.pop('cmap', 'viridis'))
			fig.colorbar(mesh, ax=ax, label=r'$|\dot{x}, \dot{y}|$')
		if kind == 'quiver':
			default_kwargs = {'pivot': 'mid', 'scale': None}
			default_kwargs.update(kwargs)
			ax.quiver(X.T, Y.T, vx.T, vy.T, **default_kwargs)
		elif kind == 'stream':
			default_kwargs = {'color': 'white' if show_magnitude else None}
			default_kwargs.update(kwargs)
			if default_kwargs['color'] is None:
				default_kwargs.pop('color')
			ax.streamplot(X[:, 0], Y[0, :], vx.T, vy.T, density=density, **default_kwargs)
		ax.set_xlabel('$x$')
		ax.set_ylabel('$y$')
		ax.set_title(r'Field from $\mathrm{fft\_phi\_}$')
		ax.set_aspect('equal')
		ax.set_xlim(0, 2 * xp.pi)
		ax.set_ylim(0, 2 * xp.pi)
		return fig, ax

	def initial_conditions(self, type: str = 'fixed') -> xp.ndarray:
		original_ntraj = self.Ntraj
		logger.info("Generating initial conditions: type=%s traj=%s Ntraj=%s", type, self.traj_type, self.Ntraj)
		if type == 'random':
			y0 = 2 * xp.pi * xp.random.rand(2 * self.Ntraj)
		elif type == 'fixed':
			self.Ntraj = int(xp.sqrt(self.Ntraj))**2
			if self.Ntraj != original_ntraj:
				logger.info("Adjusted Ntraj from %s to square grid size %s for fixed initialization", original_ntraj, self.Ntraj)
			y_vec = xp.linspace(0, 2 * xp.pi, int(xp.sqrt(self.Ntraj)), endpoint=False)
			y_mat = xp.meshgrid(y_vec, y_vec)
			y0 = xp.concatenate((y_mat[0], y_mat[1]), axis=None)
		elif type == 'selected':
			x0 = xp.asarray(self.x0)
			y0_selected = xp.asarray(self.y0)
			if x0.shape != y0_selected.shape:
				raise ValueError("`x0` and `y0` must have the same shape when init='selected'.")
			self.Ntraj = x0.size
			logger.info("Using selected initial conditions with %d trajectories", self.Ntraj)
			y0 = xp.concatenate((x0.ravel(), y0_selected.ravel()), axis=None)
		else:
			raise ValueError("`type` must be 'random', 'fixed' or 'selected'.")
		if self.traj_type == 'fo':
			phi_perp = 2 * xp.pi * xp.random.rand(self.Ntraj)
			y0 = xp.concatenate((y0, xp.cos(phi_perp), xp.sin(phi_perp)), axis=None)
			if self.CheckEnergy:
				y0 = xp.concatenate((y0, xp.zeros(self.Ntraj)), axis=None)
			logger.debug("Added full-orbit velocity variables: CheckEnergy=%s", self.CheckEnergy)
		logger.info("Initial condition vector ready: shape=%s", y0.shape)
		return y0

	def y_dot(self, t: float, y: xp.ndarray) -> xp.ndarray:
		exp_xy = xp.exp(1j * (xp.einsum('ijk,i...->jk...', self.nm, xp.split(y, 2)) - t))
		return (xp.einsum('ijk,jk...->i...', self.fft_phi_, exp_xy).real).reshape(y.shape)
	
	def k_dot(self, t: float, y: xp.ndarray) -> xp.ndarray:
		exp_xy = xp.exp(1j * (xp.einsum('ijk,i...->jk...', self.nm, xp.split(y, 2)) - t))
		return xp.einsum('jk,jk...->...', self.phic, exp_xy).real
	
	def potential(self, t: xp.ndarray, y: xp.ndarray) -> xp.ndarray:
		exp_xy = xp.exp(1j * (xp.einsum('ijk,i...->jk...', self.nm, xp.split(y, 2)) - t))
		return xp.einsum('jk,jk...->...', self.phic, exp_xy).imag
	
	def hamiltonian(self, t: xp.ndarray, y: xp.ndarray) -> xp.ndarray | None:
		if self.traj_type == 'gc':
			return self.potential(t, y)
		elif self.traj_type == 'fo':
			x_, y_, vx, vy = xp.split(y, 4)
			return self.rho / (4 * xp.abs(self.eta)) * (vx**2 + vy**2) + self.potential(t, xp.concatenate((x_, y_), axis=0)) * xp.sign(self.eta) / self.rho
	
	def chi(self, h: float, t: float, y: xp.ndarray) -> xp.ndarray:
		if self.CheckEnergy:
			x_, y_, vx, vy, k = xp.split(y, 5)
		else:
			x_, y_, vx, vy = xp.split(y, 4)
		exp_ = xp.exp(-1j * h / (2 * self.eta))
		x_, y_ = real_imag(x_ + 1j * y_ + 1j * self.rho * xp.sign(self.eta) * (exp_ - 1) * (vx + 1j * vy)) 
		vx, vy = real_imag(exp_ * (vx + 1j * vy))
		pot = xp.split(self.y_dot(t, xp.concatenate((x_, y_), axis=None)), 2)
		vx, vy = real_imag(vx + 1j * vy + h * 1j * (pot[0] + 1j * pot[1]) * xp.sign(self.eta) / self.rho)
		if not self.CheckEnergy:
			return xp.concatenate((x_, y_, vx, vy), axis=None)
		k += h * xp.sign(self.eta) / self.rho * self.k_dot(t, xp.concatenate((x_, y_), axis=None)) 
		return xp.concatenate((x_, y_, vx, vy, k), axis=None)
	
	def chi_star(self, h: float, t: float, y: xp.ndarray) -> xp.ndarray:
		if self.CheckEnergy:
			x_, y_, vx, vy, k = xp.split(y, 5)
		else:
			x_, y_, vx, vy = xp.split(y, 4)
		pot = xp.split(self.y_dot(t, xp.concatenate((x_, y_), axis=None)), 2)
		vx, vy = real_imag(vx + 1j * vy + h * 1j * (pot[0] + 1j * pot[1]) * xp.sign(self.eta) / self.rho)
		if self.CheckEnergy:
			k += h * xp.sign(self.eta) / self.rho * self.k_dot(t, xp.concatenate((x_, y_), axis=None))
		exp_ = xp.exp(-1j * h / (2 * self.eta))
		x_, y_ = real_imag(x_ + 1j * y_ + 1j * self.rho * xp.sign(self.eta) * (exp_ - 1) * (vx + 1j * vy)) 
		vx, vy = real_imag(exp_ * (vx + 1j * vy))
		if not self.CheckEnergy:
			return xp.concatenate((x_, y_, vx, vy), axis=None)
		return xp.concatenate((x_, y_, vx, vy, k), axis=None)
	
	def save_data(self, sol: OdeSolution) -> None:
		if not self.SaveData:
			logger.debug("SaveData disabled; skipping MATLAB export")
			return
		logger.info("Saving simulation data: traj=%s samples=%d", self.traj_type, sol.t.size)
		if self.traj_type == 'gc':
			x, y = xp.split(sol.y, 2)
		elif self.traj_type == 'fo':
			if self.CheckEnergy:
				x, y, vx, vy, _ = xp.split(sol.y, 5)
			else:
				x, y, vx, vy = xp.split(sol.y, 4)
		mdic = self.DictParams.copy()
		mdic.update({'t': sol.t, 'x': x, 'y': y})
		if self.traj_type == 'fo':
			mdic.update({'vx': vx, 'vy': vy})
		if self.CheckEnergy:
			mdic.update({'k': sol.k})
		mdic.update({'date': datetime.now().strftime(" %B %d, %Y\n"), 'author': 'cristel.chandre@cnrs.fr'})
		filename = 'data_' + self.traj_type + '_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.mat'
		savemat(filename, mdic)
		logger.info("Results saved in %s", filename)

if __name__ == '__main__':
	main()
