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
from scipy.special import jv
from scipy.optimize import curve_fit
from scipy.stats import linregress
from sklearn.metrics import r2_score
from scipy.io import savemat
from pyhamsys import OdeSolution, HamSys
from datetime import date


def real_imag(z:xp.ndarray):
	return z.real, z.imag

def glue_sol(sol1:OdeSolution, sol2:OdeSolution, check_energy=False) -> OdeSolution:
	sol2.t = xp.concatenate((sol1.t, sol2.t[1:]), axis=None)
	if hasattr(sol1, 'k'):
		sol2.k = xp.concatenate((sol1.k, sol2.k[1:]), axis=-1)
	if hasattr(sol1, 'err'):
		sol2.err = max([sol1.err, sol2.err])
	sol2.y = xp.concatenate((sol1.y, sol2.y[:, 1:]), axis=-1)
	return sol2

def save_data(self, data, filestr, info=[]):
	if self.SaveData:
		mdic = self.DictParams.copy()
		mdic.update({'data': data, 'info': info})
		mdic.update({'date': date.today().strftime(" %B %d, %Y\n"), 'author': 'cristel.chandre@cnrs.fr'})
		savemat(filestr + '.mat', mdic)
		print(f'\033[90m        Results saved in {filestr}.mat \033[00m')

class GC2Dt(HamSys):
	def __repr__(self) -> str:
		return "{self.__class__.__name__}({self.DictParams})".format(self=self)

	def __str__(self) -> str:
		return f'2D Guiding Center ({self.__class__.__name__}) for turbulent potentials'

	def __init__(self, dict_:dict) -> None:
		super().__init__(ndof=1.5 if dict_['Method'].endswith('gc') else 2.5)
		for key in dict_:
			setattr(self, key, dict_[key])
		self.dim = 2 if self.Method.endswith('gc') else 4
		self.DictParams = dict_
		xp.random.seed(27)
		self.phases = 2 * xp.pi * xp.random.random((self.M, self.M))
		self.nm = xp.meshgrid(xp.arange(self.M+1), xp.arange(self.M+1), indexing='ij')
		self.phic = xp.zeros((self.M+1, self.M+1), dtype=xp.complex128)
		self.phic[1:, 1:] = self.A / (self.nm[0][1:, 1:]**2 + self.nm[1][1:, 1:]**2)**1.5 * xp.exp(1j * self.phases)
		sqrt_nm = xp.sqrt(self.nm[0]**2 + self.nm[1]**2)
		self.phic[sqrt_nm > self.M] = 0
		if self.Method.endswith('gc'):
			flr1_coeff = jv(0, self.rho * sqrt_nm)
			self.phic *= flr1_coeff
		self.fft_phi_ = xp.asarray([-self.nm[1] * self.phic, self.nm[0] * self.phic])	

	def initial_conditions(self, type:str='fixed') -> xp.ndarray:
		if type == 'random':
			y0 = 2 * xp.pi * xp.random.rand(2 * self.Ntraj)
		elif type == 'fixed':
			self.Ntraj = int(xp.sqrt(self.Ntraj))**2
			y_vec = xp.linspace(0, 2 * xp.pi, int(xp.sqrt(self.Ntraj)), endpoint=False)
			y_mat = xp.meshgrid(y_vec, y_vec)
			y0 = xp.concatenate((y_mat[0], y_mat[1]), axis=None)
		elif type == 'selected':
			y0 = xp.concatenate((self.x0, self.y0), axis=None)
		if self.Method.endswith('fo'):
			phi_perp = 2 * xp.pi * xp.random.rand(self.Ntraj)
			y0 = xp.concatenate((y0, xp.cos(phi_perp), xp.sin(phi_perp)), axis=None)
			if self.CheckEnergy:
				y0 = xp.concatenate((y0, xp.zeros(self.Ntraj)), axis=None)
		return y0

	def y_dot(self, t:float, y:xp.ndarray) -> xp.ndarray:
		exp_xy = xp.exp(1j * (xp.einsum('ijk,i...->jk...', self.nm, xp.split(y, 2)) - t))
		return (xp.einsum('ijk,jk...->i...', self.fft_phi_, exp_xy).real).reshape(y.shape)
	
	def k_dot(self, t:float, y:xp.ndarray) -> xp.ndarray:
		exp_xy = xp.exp(1j * (xp.einsum('ijk,i...->jk...', self.nm, xp.split(y, 2)) - t))
		return xp.einsum('jk,jk...->...', self.phic, exp_xy).real
	
	def potential(self, t:xp.ndarray, y:xp.ndarray) -> xp.ndarray:
		exp_xy = xp.exp(1j * (xp.einsum('ijk,i...->jk...', self.nm, xp.split(y, 2)) - t))
		return xp.einsum('jk,jk...->...', self.phic, exp_xy).imag
	
	def hamiltonian(self, t:xp.ndarray, y:xp.ndarray) -> xp.ndarray:
		if self.Method.endswith('gc'):
			return self.potential(t, y)
		elif self.Method.endswith('fo'):
			x_, y_, vx, vy = xp.split(y, 4)
			return self.rho / (4 * xp.abs(self.eta)) * (vx**2 + vy**2) + self.potential(t, xp.concatenate((x_, y_), axis=0)) * xp.sign(self.eta) / self.rho
		
	def fo2gc(self, y, order:int=1) -> xp.ndarray:
		if self.Method.endswith('gc'):
			raise ValueError(f'Already in guiding-center variables')
		x, y, vx, vy = xp.split(y, self.dim)
		v = vy + 1j * vx
		theta, rho = xp.pi + xp.angle(v), self.rho * xp.abs(v)
		x_gc, y_gc = x - rho * xp.cos(theta), y + rho * xp.sin(theta)
		if order <= 1:
			return x_gc, y_gc 
		else:
			raise ValueError(f'fo2gc not available at order {order}')
	
	def chi(self, h:float, t:float, y:xp.ndarray) -> xp.ndarray:
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
	
	def chi_star(self, h:float, t:float, y:xp.ndarray) -> xp.ndarray:
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

class Trajectory(GC2Dt):
	
	type_dict = {'trapped': 0, 'diffusive': 1, 'ballistic': 2}
	color_dict = {'trapped': '#0072bd', 'diffusive': '#EDB120', 'ballistic': '#D95319'}
	omega = lambda t : xp.exp(-1 / (t * (1 - t)))

	def __init__(self, sol:OdeSolution, ttype, dict_) -> None:
		super().__init__(dict_)
		self.type = ['trapped', 'diffusive', 'ballistic'] if ttype == 'all' else ttype
		ntype = [type(self).type_dict[_] for _ in xp.atleast_1d(self.type)]
		x, y = xp.split(sol.y, self.dim)[:2]
		xgc, ygc = x, y if self.Method.endswith('gc') else self.fo2gc(sol.y)
		vec = xp.ones(xgc[:, 0].shape)
		delta = xp.asarray([el.ptp(axis=1) for el in [xgc, ygc]])
		vec[xp.sqrt(xp.sum(delta**2, axis=0)) <= self.threshold] = 0
		untrapped = xp.sqrt(xp.sum(delta, axis=0)) > self.threshold
		vec[xp.all((delta[0] / delta[1] > self.threshold, untrapped), axis=0)] = 2
		vec[xp.all((delta[1] / delta[0] > self.threshold, untrapped), axis=0)] = 2
		indx = xp.any([vec==_ for _ in ntype], axis=0)
		self.t, self.x, self.y, self.xgc, self.ygc  = sol.t, x[indx, :], y[indx, :], xgc[indx, :], ygc[indx,:]
		vec = xp.tile(vec, self.dim)
		self.sol = sol.y[xp.any([vec==_ for _ in ntype], axis=0), :]
		self.size = len(self.x[:, 0])
		self.color = [type(self).color_dict[_] for _ in xp.atleast_1d(self.type)][0]

	def remove_trapped(self, sol:OdeSolution) -> OdeSolution: 
		xgc, ygc = xp.split(sol.y, self.dim)[:2] if self.Method.endswith('gc') else self.fo2gc(sol.y)
		delta = xp.asarray([el.ptp(axis=1) for el in [xgc, ygc]])
		vec = xp.ones(xgc[:, 0].shape)
		vec[xp.sqrt(xp.sum(delta**2, axis=0)) <= self.threshold] = 0
		vec = xp.tile(vec, self.dim)
		sol.y = sol.y[vec!=0, :]
		return sol
	
	def compute_data(self):
		nt = self.x[0, :].size
		r2 = xp.zeros(nt)
		for _ in range(nt):
			r2[_] = ((self.xgc[:, _:] - self.xgc[:, :-_ if _ else None])**2 + (self.ygc[:, _:] - self.ygc[:, :-_ if _ else None])**2).mean()
		t_win, r2_win = self.t[nt//8:7*nt//8], r2[nt//8:7*nt//8]
		res = linregress(t_win, r2_win)
		diff_data = [res.slope, res.intercept, res.rvalue**2]
		func_fit = lambda t, a, b: (a * t)**b
		popt = curve_fit(func_fit, t_win, r2_win, bounds=((0, 0.25), (xp.inf, 3)))[0]
		r2_fit = func_fit(t_win, *popt)
		interp_data = [*popt, r2_score(r2_win, r2_fit)]
		if self.PlotResults:
			fig, ax = plt.subplots(1, 1)
			ax.set_xlabel('$t$')
			ax.set_ylabel('$r^2$')
			color = self.get_color(self.color)[0]
			plt.plot(self.t, r2, ':', color=color, lw=1)
			plt.plot(t_win, r2_win, '-', color=color, lw=2)
			plt.plot(t_win, r2_fit, '-.', color=color, lw=2)
			if self.SaveData:
				filestr = f'{type(self).__name__}_A{self.A:.2f}_RHO{self.rho:.4f}'.replace('.', '')
				fig.savefig(filestr + self.extension, dpi=self.dpi)
				print(f'\033[90m        Figure saved in {filestr}{self.extension} \033[00m')
			plt.pause(0.5)
		return diff_data, interp_data
	
	def compute_rotation(self, h:xp.ufunc) -> xp.ndarray:
		x = h(xp.atleast_2d(self.xgc), xp.atleast_2d(self.ygc))
		nt = x[0, :].size
		omega = type(self).omega(xp.arange(1, nt) / nt)
		return xp.sum(x[:, 1:] * omega[xp.newaxis, :], axis=1) / xp.sum(omega)