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
from scipy.special import jv
import multiprocess
from datetime import datetime
from scipy.io import savemat
from gc2d_symp_modules import run_method
from gc2d_symp_dict import dict_list, Parallelization
from pyhamsys import OdeSolution, HamSys

def main() -> None:
	if Parallelization == 'all':
		num_cores = multiprocess.cpu_count()
	else:
		num_cores = min(multiprocess.cpu_count(), Parallelization)
	if num_cores >= 2:
		pool = multiprocess.Pool(num_cores)
		pool.map(lambda dict_: run_method(GC2Dt(dict_)), dict_list)
	else:
		for dict_ in dict_list:
			run_method(GC2Dt(dict_))

def real_imag(z:xp.ndarray):
	return z.real, z.imag

class GC2Dt(HamSys):
	def __repr__(self) -> str:
		return "{self.__class__.__name__}({self.DictParams})".format(self=self)

	def __str__(self) -> str:
		return f'2D Guiding Center ({self.__class__.__name__}) for turbulent potentials'

	def __init__(self, dict_:dict) -> None:
		super().__init__(ndof=1.5 if dict_['traj_type']=='gc' else 2.5)
		for key in dict_:
			setattr(self, key, dict_[key])
		self.DictParams = dict_
		xp.random.seed(27)
		self.phases = 2 * xp.pi * xp.random.random((self.M, self.M))
		self.nm = xp.meshgrid(xp.arange(self.M+1), xp.arange(self.M+1), indexing='ij')
		self.phic = xp.zeros((self.M+1, self.M+1), dtype=xp.complex128)
		self.phic[1:, 1:] = (self.A / (self.nm[0][1:, 1:]**2 + self.nm[1][1:, 1:]**2)**1.5).astype(xp.complex128) * xp.exp(1j * self.phases)
		sqrt_nm = xp.sqrt(self.nm[0]**2 + self.nm[1]**2)
		self.phic[sqrt_nm > self.M] = 0
		if self.traj_type == 'gc':
			flr1_coeff = jv(0, self.rho * sqrt_nm)
			self.phic *= flr1_coeff
			flr2_coeff = -sqrt_nm * jv(1, self.rho * sqrt_nm) / self.rho
		self.fft_phi_ = xp.asarray([-self.nm[1] * self.phic, self.nm[0] * self.phic])	

	def initial_conditions(self, type:str='fixed') -> xp.ndarray:
		if type == 'random':
			y0 = 2 * xp.pi * xp.random.rand(2 * self.Ntraj)
		elif type == 'fixed':
			self.Ntraj = int(xp.sqrt(self.Ntraj))**2
			y_vec = xp.linspace(0, 2 * xp.pi, int(xp.sqrt(self.Ntraj)), endpoint=False)
			y_mat = xp.meshgrid(y_vec, y_vec)
			y0 = xp.concatenate((y_mat[0], y_mat[1]), axis=None)
		if self.traj_type == 'fo':
			phi_perp = 2 * xp.pi * xp.random.rand(self.Ntraj)
			y0 = xp.concatenate((y0, xp.cos(phi_perp), xp.sin(phi_perp)), axis=None)
			if self.CheckEnergy:
				y0 = xp.concatenate((y0, xp.zeros(self.Ntraj)), axis=None)
		return y0

	def y_dot(self, t:float, y:xp.ndarray) -> xp.ndarray:
		y_ = xp.split(y, 2)
		exp_xy = xp.exp(1j * (self.nm[0][..., xp.newaxis] * y_[0][xp.newaxis, xp.newaxis] + self.nm[1][..., xp.newaxis] * y_[1][xp.newaxis, xp.newaxis] - t))
		return (xp.sum(self.fft_phi_[..., xp.newaxis] * exp_xy[xp.newaxis], (1, 2)).real).reshape(y.shape)
	
	def k_dot(self, t:float, y:xp.ndarray) -> xp.ndarray:
		y_ = xp.split(y, 2)
		exp_xy = xp.exp(1j * (self.nm[0][..., xp.newaxis] * y_[0][xp.newaxis, xp.newaxis] + self.nm[1][..., xp.newaxis] * y_[1][xp.newaxis, xp.newaxis] - t))
		return xp.sum(self.phic[..., xp.newaxis] * exp_xy, (0, 1)).real
	
	def potential(self, t:xp.ndarray, y:xp.ndarray) -> xp.ndarray:
		y_ = xp.split(y, 2)
		exp_xy = xp.exp(1j * (self.nm[0][..., xp.newaxis, xp.newaxis] * y_[0][xp.newaxis, xp.newaxis] + self.nm[1][..., xp.newaxis, xp.newaxis] * y_[1][xp.newaxis, xp.newaxis] - t[xp.newaxis, xp.newaxis]))
		return xp.sum(self.phic[..., xp.newaxis, xp.newaxis] * exp_xy, (0, 1)).imag
	
	def hamiltonian(self, t:xp.ndarray, y:xp.ndarray) -> xp.ndarray:
		if self.traj_type == 'gc':
			return self.potential(t, y)
		elif self.traj_type == 'fo':
			x_, y_, vx, vy = xp.split(y, 4)
			return self.rho / (4 * xp.abs(self.eta)) * (vx**2 + vy**2) + self.potential(t, xp.concatenate((x_, y_), axis=0)) * xp.sign(self.eta) / self.rho
	
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
	
	def save_data(self, sol:OdeSolution) -> None:
		if self.SaveData:
			if self.traj_type == 'gc':
				x, y = xp.split(sol.y, 2)
			elif self.traj_type == 'fo':
				x, y, vx, vy = xp.split(sol.y, 5)
			mdic = self.DictParams.copy()
			mdic.update({'t': sol.t, 'x': x, 'y': y})
			if self.traj_type == 'fo':
				mdic.update({'vx': vx, 'vy': vy})
			if self.CheckEnergy:
				mdic.update({'k': sol.k})
			mdic.update({'date': datetime.now().strftime(" %B %d, %Y\n"), 'author': 'cristel.chandre@cnrs.fr'})
			filename = 'data_' + self.traj_type + '_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.mat'
			savemat(filename, mdic)
			print(f'\033[90m        Results saved in {filename} \033[00m')

if __name__ == '__main__':
	main()
