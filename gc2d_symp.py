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
from numpy.fft import fft2, ifft2, fftfreq
from scipy.interpolate import interpn
from gc2d_symp_modules import run_method
from gc2d_symp_dict import dictparams
from pyhamsys import SymplecticIntegrator, solve_ivp_symp, solve_ivp_sympext

def main() -> None:
	run_method(GC2Dt(dictparams))

SCHEMES = ['interp', 'symp', 'symp_ext']

class GC2Dt:
	def __repr__(self) -> str:
		return "{self.__class__.__name__}({self.DictParams})".format(self=self)

	def __str__(self) -> str:
		return f'2D Guiding Center ({self.__class__.__name__}) for the turbulent potential'

	def __init__(self, dict_: dict) -> None:
		for key in dict_:
			setattr(self, key, dict_[key])
		if self.solve_method not in SCHEMES:
			raise ValueError(f"The chosen numerical scheme must be one of {SCHEMES}.")
		self.DictParams = dictparams
		xp.random.seed(27)
		self.phases = 2 * xp.pi * xp.random.random((self.M, self.M))
		self.nm = xp.meshgrid(xp.arange(self.M+1), xp.arange(self.M+1), indexing='ij')
		self.phic = xp.zeros((self.M+1, self.M+1), dtype=xp.complex128)
		self.phic[1:, 1:] = (self.A / (self.nm[0][1:, 1:]**2 + self.nm[1][1:, 1:]**2)**1.5).astype(xp.complex128) * xp.exp(1j * self.phases)
		sqrt_nm = xp.sqrt(self.nm[0]**2 + self.nm[1]**2)
		self.phic[sqrt_nm > self.M] = 0
		if self.solve_method == 'interp':
			self.xy_ = 2 * (xp.linspace(0, 2 * xp.pi, self.N+1, dtype=xp.float64),)
			nminterp = xp.meshgrid(fftfreq(self.N, d=1/self.N), fftfreq(self.N, d=1/self.N), indexing='ij')
			fft_phi = xp.zeros((self.N, self.N), dtype=xp.complex128)
			fft_phi[:self.M+1, :self.M+1] = self.phic
			self.phi = ifft2(fft_phi) * (self.N**2)
			self.pad = lambda psi: xp.pad(psi, ((0, 1),), mode='wrap')
			self.derivs = lambda psi: [self.pad(ifft2(1j * nminterp[_] * fft2(psi))) for _ in range(2)]
			stack = self.derivs(self.phi)
			if self.CheckEnergy:
				stack = (*stack, self.pad(self.phi))
			self.Dphi = xp.moveaxis(xp.stack(stack), 0, -1)
		elif self.solve_method.startswith('symp'):
			self.fft_phi_ = xp.asarray([-self.nm[1] * self.phic, self.nm[0] * self.phic])	
		
	def eqn_interp(self, t:float, y:xp.ndarray) -> xp.ndarray:
		vars = xp.split(y, 2 + self.CheckEnergy)
		r = xp.moveaxis(xp.asarray(vars[0:2]) % (2 * xp.pi), 0, -1)
		fields = xp.moveaxis(interpn(self.xy_, self.Dphi, r), 0, 1)
		dy_gc = xp.concatenate((-(fields[1] * xp.exp(-1j * t)).imag, (fields[0] * xp.exp(-1j * t)).imag), axis=None)
		if self.CheckEnergy:
			dk = (fields[2] * xp.exp(-1j * t)).real
			return xp.concatenate((dy_gc, dk), axis=None)
		return dy_gc
	
	def chi(self, h:float, t:float, y:xp.ndarray) -> xp.ndarray:
		y_ = xp.split(y, 2 + self.CheckEnergy)
		for n in range(1, self.M + 1):
			for m in range(1, self.M + 1):
				cnm = h * (self.phic[n, m] * xp.exp(1j * (n * y_[0] + m * y_[1] - t))).real
				y_[0] -= m * cnm 
				y_[1] += n * cnm
				if self.CheckEnergy:
					y_[2] += cnm
		return xp.concatenate([y_[_] for _ in range(2 + self.CheckEnergy)], axis=None)
	
	def chi_star(self, h:float, t:float, y:xp.ndarray) -> xp.ndarray:
		y_ = xp.split(y, 2 + self.CheckEnergy)
		for n in range(self.M, 0, -1):
			for m in range(self.M, 0, -1):
				cnm = h * (self.phic[n, m] * xp.exp(1j * (n * y_[0] + m * y_[1] - t))).real
				y_[0] -= m * cnm 
				y_[1] += n * cnm
				if self.CheckEnergy:
					y_[2] += cnm
		return xp.concatenate([y_[_] for _ in range(2 + self.CheckEnergy)], axis=None)
	
	def eqn_xy(self, t:float, y:xp.ndarray) -> xp.ndarray:
		y_ = xp.split(y, 2)
		exp_xy = xp.exp(1j * (self.nm[0][..., xp.newaxis] * y_[0][xp.newaxis, xp.newaxis] + self.nm[1][..., xp.newaxis] * y_[1][xp.newaxis, xp.newaxis] - t))
		return (xp.sum(self.fft_phi_[..., xp.newaxis] * exp_xy[xp.newaxis], (1, 2)).real).reshape(y.shape)
	
	def eqn_k(self, t:float, y:xp.ndarray) -> xp.ndarray:
		y_ = xp.split(y, 2)
		exp_xy = xp.exp(1j * (self.nm[0][..., xp.newaxis] * y_[0][xp.newaxis, xp.newaxis] + self.nm[1][..., xp.newaxis] * y_[1][xp.newaxis, xp.newaxis] - t))
		return (xp.sum(self.phic[..., xp.newaxis] * exp_xy, (0, 1)).real).reshape(y.shape)
	
	def compute_energy(self, sol) -> xp.ndarray:
		x, y, k = xp.split(sol.y, 3)
		exp_xy = xp.exp(1j * (self.nm[0][..., xp.newaxis, xp.newaxis] * x[xp.newaxis, xp.newaxis] + self.nm[1][..., xp.newaxis, xp.newaxis] * y[xp.newaxis, xp.newaxis] - sol.t[xp.newaxis, xp.newaxis, xp.newaxis]))
		return k + xp.sum(self.phic[..., xp.newaxis, xp.newaxis] * exp_xy, (0, 1)).imag

if __name__ == '__main__':
	main()
