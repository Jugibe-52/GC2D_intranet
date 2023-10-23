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
from gc2d_symp_modules import run_method
from gc2d_symp_dict import dictparams

def main() -> None:
	run_method(GC2Dt(dictparams))

class GC2Dt:
	def __repr__(self) -> str:
		return "{self.__class__.__name__}({self.DictParams})".format(self=self)

	def __str__(self) -> str:
		return f'2D Guiding Center ({self.__class__.__name__}) for turbulent potentials'

	def __init__(self, dict_: dict) -> None:
		for key in dict_:
			setattr(self, key, dict_[key])
		self.DictParams = dictparams
		xp.random.seed(27)
		self.phases = 2 * xp.pi * xp.random.random((self.M, self.M))
		self.nm = xp.meshgrid(xp.arange(self.M+1), xp.arange(self.M+1), indexing='ij')
		self.phic = xp.zeros((self.M+1, self.M+1), dtype=xp.complex128)
		self.phic[1:, 1:] = (self.A / (self.nm[0][1:, 1:]**2 + self.nm[1][1:, 1:]**2)**1.5).astype(xp.complex128) * xp.exp(1j * self.phases)
		sqrt_nm = xp.sqrt(self.nm[0]**2 + self.nm[1]**2)
		self.phic[sqrt_nm > self.M] = 0
		flr1_coeff = jv(0, self.rho * sqrt_nm)
		self.phic *= flr1_coeff
		self.fft_phi_ = xp.asarray([-self.nm[1] * self.phic, self.nm[0] * self.phic])	

	def initial_conditions(self, type:str='fixed'):
		if type == 'random':
			return 2 * xp.pi * xp.random.rand(2 * self.Ntraj)
		elif type == 'fixed':
			self.Ntraj = int(xp.sqrt(self.Ntraj))**2
			y_vec = xp.linspace(0, 2 * xp.pi, int(xp.sqrt(self.Ntraj)), endpoint=False)
			y_mat = xp.meshgrid(y_vec, y_vec)
			return xp.concatenate((y_mat[0], y_mat[1]), axis=None)

	def xy_dot(self, t:float, y:xp.ndarray) -> xp.ndarray:
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

if __name__ == '__main__':
	main()
