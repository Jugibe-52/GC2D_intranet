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
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from scipy.special import jv
from scipy.stats import linregress
from scipy.io import savemat
from pyhamsys import OdeSolution, HamSys, solve_ivp_symp, solve_ivp_sympext
from typing import List, Union
from datetime import date
import time

def real_imag(z:xp.ndarray):
	return z.real, z.imag

def glue_sol(sol1:OdeSolution, sol2:OdeSolution) -> OdeSolution:
	sol2.t = xp.concatenate((sol1.t, sol2.t[1:]), axis=None)
	if hasattr(sol1, 'k'):
		sol2.k = xp.concatenate((sol1.k, sol2.k[:, 1:]), axis=-1)
	if hasattr(sol1, 'err'):
		sol2.err = max([sol1.err, sol2.err])
	sol2.y = xp.concatenate((sol1.y, sol2.y[:, 1:]), axis=-1)
	return sol2

def save_data(self, data, filestr:str, info=[]) -> None:
	if self.SaveData:
		mdic = {'x': self.x, 'y': self.y, 'potential': self.potential, 't_period': self.t_period, 'xy_period': self.xy_period}
		mdic.update({'data': data, 'info': info})
		mdic.update({'date': date.today().strftime(" %B %d, %Y\n"), 'author': 'cristel.chandre@cnrs.fr'})
		savemat(filestr + '.mat', mdic)
		print(f'\033[90m        Results saved in {filestr}.mat \033[00m')

def extract_potential(filename):
	import h5py
	with h5py.File(filename, 'r') as f:
		x = xp.array(f['Rcells'][:])
		y = xp.array(f['Zcells'][:])
		freqs = xp.array(f['freqs'][:])
		potential = xp.array(f['PHI_filtered_FT'])
		sum_xy = xp.sum(potential, axis=(1, 2))
		i_omega = xp.flatnonzero(sum_xy)[0]
		omega = freqs[i_omega] * 2 * xp.pi
		values = xp.array(f['PHI_filtered_FT'][i_omega,:,:])
	return Potential(x, y, values, omega=omega)

class Potential:
	def __init__(self, x, y, values, xy_period=None, omega=1):
		self.values = values
		self.nx, self.ny = values.shape
		self.x, self.y = x, y
		self.dx, self.dy = x[1] - x[0], y[1] - y[0]
		self.xy_period = xy_period
		self.omega = omega
		self.t_period = 2 * xp.pi / omega

	def gyroaverage(self, rho):
		fft_potential = fft2(self.values)
		kx, ky = fftfreq(self.nx, d=self.dx), fftfreq(self.ny, d=self.dy)
		kx_, ky_ = xp.meshgrid(kx, ky, indexing='ij')
		return  ifft2(fft_potential * jv(0, 2 * xp.pi * rho * xp.sqrt(kx_**2 + ky_**2)))
	
	def wrap(self, xi, yi):
		xi = ((xp.asarray(xi) - self.x[0]) % self.xy_period) + self.x[0]
		yi = ((xp.asarray(yi) - self.y[0]) % self.xy_period) + self.y[0]
		return xi, yi
	
	def isinside(self, x, y):
		return (x >= self.x[0]) & (x <= self.x[-1]) & (y >= self.y[0]) & (y <= self.y[-1])

def mock_potential(A, M, nx, ny):
    x = xp.linspace(0, 2 * xp.pi, nx, endpoint=False)
    y = xp.linspace(0, 2 * xp.pi, ny, endpoint=False)
    X, Y = xp.meshgrid(x, y, indexing='ij')
    xp.random.seed(27)
    phases = 2 * xp.pi * xp.random.random((M, M))
    nm = xp.meshgrid(xp.arange(M + 1), xp.arange(M + 1), indexing='ij')
    fft_phic = xp.zeros((M + 1, M + 1), dtype=xp.complex128)
    fft_phic[1:, 1:] = A / (nm[0][1:, 1:]**2 + nm[1][1:, 1:]**2)**1.5 * xp.exp(1j * phases)
    fft_phic[xp.sqrt(nm[0]**2 + nm[1]**2) > M] = 0
    exp_xy = xp.exp(1j * (nm[0][:, :, None, None] * X[None, None, :, :] + nm[1][:, :, None, None] * Y[None, None, :, :]))
    return Potential(x, y, xp.einsum('nm,nm...->...', fft_phic, exp_xy), xy_period=2 * xp.pi, omega=1)

class GC2D(HamSys):
	def __str__(self) -> str:
		return f'2D Guiding Center ({self.__class__.__name__}) for turbulent potentials'
		
	def __init__(self, potential, traj, k=3, SaveData=False):
		super().__init__(ndof=1.5 if traj["type"]=='gc' else 2.5)
		self.traj = traj
		self.rho = traj["rho"] if "rho" in traj else 0
		self.eta = traj["eta"] if "eta" in traj else 0
		self.CheckEnergy = traj["CheckEnergy"] if "CheckEnergy" in traj else False
		self.SaveData = SaveData
		if self.rho != 0:
			potential.values = potential.gyroaverage(self.rho)
		self.potential = potential
		x = xp.pad(potential.x, (k, k), mode='linear_ramp',\
			 end_values=(potential.x[0] - k * potential.dx, potential.x[-1] + k * potential.dx))
		y = xp.pad(potential.y, (k, k), mode='linear_ramp', \
			 end_values=(potential.y[0] - k * potential.dy, potential.y[-1] + k * potential.dy))
		if potential.xy_period is not None:
			potential = xp.pad(potential.values, ((k, k), (k, k)), mode='wrap')
		else:
			potential = xp.pad(self.potential.values, ((k, k), (k, k)), mode='constant', constant_values=0)
		self.spline_real = RectBivariateSpline(x, y, potential.real, kx=k, ky=k)
		self.spline_imag = RectBivariateSpline(x, y, potential.imag, kx=k, ky=k) 

	def interpolator(self, xi, yi, dx=0, dy=0):
		if self.potential.xy_period is not None:
			xi, yi = self.potential.wrap(xi, yi)
		return self.spline_real.ev(xi, yi, dx=dx, dy=dy) + 1j * self.spline_imag.ev(xi, yi, dx=dx, dy=dy)

	def initial_conditions(self, n_traj, x=None, y=None, type='fixed'):
		x, y = self.potential.x if x is None else x, self.potential.y if y is None else y
		if type == 'random':
			xp.random.seed(int(time.time()))
			x0 = (x[-1] - x[0]) * xp.random.rand(n_traj) + x[0]
			y0 = (y[-1] - y[0]) * xp.random.rand(n_traj) + y[0]
			z0 = xp.concatenate((x0, y0), axis=None)
		elif type == 'fixed':
			n_traj = int(xp.sqrt(n_traj))**2
			x0 = xp.linspace(x[0], x[-1], int(xp.sqrt(n_traj)), endpoint=False)
			y0 = xp.linspace(y[0], y[-1], int(xp.sqrt(n_traj)), endpoint=False)
			x0, y0 = xp.meshgrid(x0, y0, indexing='ij')
			z0 = xp.concatenate((x0.flatten(), y0.flatten()), axis=None)
		if self.traj["type"] == 'fo':
			xp.random.seed(int(time.time()))
			phi_perp = 2 * xp.pi * xp.random.rand(n_traj)
			z0 = xp.concatenate((z0, xp.cos(phi_perp), xp.sin(phi_perp)), axis=None)
			if self.CheckEnergy:
				z0 = xp.concatenate((z0, xp.zeros(n_traj)), axis=None)
		return z0

	def hamiltonian(self, t, z):
		if self.traj["type"] == 'gc':
			x, y = xp.split(z, 2)
			phi_c = self.interpolator(x, y)
			return (phi_c * xp.exp(-1j * self.potential.omega * t)).imag
		elif self.traj["type"] == 'fo':
			x, y, vx, vy = xp.split(z, 4)
			phi_c = self.interpolator(x, y)
			return self.rho / (4 * xp.abs(self.eta)) * (vx**2 + vy**2)\
				  + (phi_c * xp.exp(-1j * self.potential.omega * t)).imag * xp.sign(self.eta) / self.rho
        
	def y_dot(self, t, z):
		x, y = xp.split(z, 2)
		dv_dx, dv_dy = xp.zeros_like(x, dtype=xp.complex128), xp.zeros_like(y, dtype=xp.complex128)
		if self.potential.xy_period is None:
			ind = self.potential.isinside(x, y)
		else:
			ind = xp.arange(len(x))
		phase = xp.exp(-1j * self.potential.omega * t)
		dv_dx[ind] = self.interpolator(x[ind], y[ind], dx=1, dy=0) * phase
		dv_dy[ind] = self.interpolator(x[ind], y[ind], dx=0, dy=1) * phase
		return  xp.asarray([-dv_dy.imag, dv_dx.imag]).flatten()
    
	def k_dot(self, t, z):
		phi_c = self.interpolator(*xp.split(z, 2))
		return (phi_c * xp.exp(-1j * self.potential.omega * t)).real
        
	def chi(self, h, t, z):
		if self.CheckEnergy:
			x, y, vx, vy, k = xp.split(z, 5)
		else:
			x, y, vx, vy = xp.split(z, 4)
		exp_ = xp.exp(-1j * h / (2 * self.eta))
		x, y = real_imag(x + 1j * y + 1j * self.rho * xp.sign(self.eta) * (exp_ - 1) * (vx + 1j * vy)) 
		vx, vy = real_imag(exp_ * (vx + 1j * vy))
		pot = xp.split(self.y_dot(t, xp.concatenate((x, y), axis=None)), 2)
		vx, vy = real_imag(vx + 1j * vy + h * 1j * (pot[0] + 1j * pot[1]) * xp.sign(self.eta) / self.rho)
		if not self.CheckEnergy:
			return xp.concatenate((x, y, vx, vy), axis=None)
		k += h * xp.sign(self.eta) / self.rho * self.k_dot(t, xp.concatenate((x, y), axis=None)) 
		return xp.concatenate((x, y, vx, vy, k), axis=None)
	
	def chi_star(self, h, t, z):
		if self.CheckEnergy:
			x, y, vx, vy, k = xp.split(z, 5)
		else:
			x, y, vx, vy = xp.split(z, 4)
		pot = xp.split(self.y_dot(t, xp.concatenate((x, y), axis=None)), 2)
		vx, vy = real_imag(vx + 1j * vy + h * 1j * (pot[0] + 1j * pot[1]) * xp.sign(self.eta) / self.rho)
		if self.CheckEnergy:
			k += h * xp.sign(self.eta) / self.rho * self.k_dot(t, xp.concatenate((x, y), axis=None))
		exp_ = xp.exp(-1j * h / (2 * self.eta))
		x, y = real_imag(x + 1j * y + 1j * self.rho * xp.sign(self.eta) * (exp_ - 1) * (vx + 1j * vy)) 
		vx, vy = real_imag(exp_ * (vx + 1j * vy))
		if not self.CheckEnergy:
			return xp.concatenate((x, y, vx, vy), axis=None)
		return xp.concatenate((x, y, vx, vy, k), axis=None)
	
	def fo2gc(self, z):
		x, y, vx, vy = xp.split(z, 4)
		v = vy + 1j * vx
		theta, rho = xp.pi + xp.angle(v), self.rho * xp.abs(v)
		return x - rho * xp.cos(theta), y + rho * xp.sin(theta)
    
	def integrate(self, z0, t_eval, timestep, solver="BM4"):
		print(f"\033[92m   Integration of {self.__str__()} \033[00m")
		start = time.time()
		if self.traj["type"] == 'gc':
			sol = solve_ivp_sympext(self, (t_eval[0], t_eval[-1]), z0, step=timestep, t_eval=t_eval, method=solver, check_energy=self.CheckEnergy)
		elif self.traj["type"] == 'fo':
			sol = solve_ivp_symp(self.chi, self.chi_star, (t_eval[0], t_eval[-1]), z0, step=timestep, t_eval=t_eval, method=solver)
			sol = self.rectify_sol(sol, check_energy=self.CheckEnergy)
		print(f'\033[90m        Computation finished in {int(time.time() - start)} seconds \033[00m')
		if self.CheckEnergy:
			print(f'\033[90m           with error in energy = {sol.err}')
		return sol
	
	def plot_sol(self, sol, wrap=False): 
		x, y = xp.split(sol.y, 2)
		if wrap:
			x, y = self.potential.wrap(x, y)
		plt.plot(x, y, '.', color='blue')
		plt.xlabel('x')
		plt.ylabel('y')
		plt.xlim(xp.amin(self.potential.x), xp.amax(self.potential.x))
		plt.ylim(xp.amin(self.potential.y), xp.amax(self.potential.y))
		plt.show()

class Trajectory(GC2D):
	
	type_dict = {'trapped': 0, 'diffusive': 1, 'ballistic': 2}
	color_dict = {'trapped': '#0072bd', 'diffusive': '#EDB120', 'ballistic': '#D95319'}
	omega = lambda t : xp.exp(-1 / (t * (1 - t)))

	def __init__(self, sol:OdeSolution, ttype:Union[str, List[str]], dict_:dict) -> None:
		super().__init__(dict_)
		self.type = ['trapped', 'diffusive', 'ballistic'] if ttype == 'all' else ttype
		ntype = [type(self).type_dict[_] for _ in xp.atleast_1d(self.type)]
		x, y = xp.split(sol.y, self.dim)[:2]
		xgc, ygc = x, y if self.Method.endswith('gc') else self.fo2gc(sol.y)
		vec = xp.ones(xgc[:, 0].shape)
		delta = xp.asarray([el.ptp(axis=1) for el in [xgc, ygc]])
		vec[xp.sqrt(xp.sum(delta**2, axis=0)) <= self.threshold] = 0
		for _ in range(len(vec)):
			if vec[_]:
				vec[_] = 2 if self.compute_diffdata(sol.t, xgc[_, :], ygc[_, :]) >= self.thresh_b else 1
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
		if self.CheckEnergy:
			sol.k = sol.k[vec!=0, :]
		vec = xp.tile(vec, self.dim)
		sol.y = sol.y[vec!=0, :]
		return sol
	
	def compute_diffdata(self, t, x, y, full_output=False, plot=False, save=False, filename=None, extension='.jpg'):
		nt = t.size
		r2 = xp.zeros(nt)
		for _ in range(nt):
			if x.ndim == 1:
				r2[_] = ((x[_:] - x[:-_ if _ else None])**2 + (y[_:] - y[:-_ if _ else None])**2).mean()
			else:
				r2[_] = ((x[:, _:] - x[:, :-_ if _ else None])**2 + (y[:, _:] - y[:, :-_ if _ else None])**2).mean()
		t_win, r2_win = t[nt//8:7*nt//8], r2[nt//8:7*nt//8]
		res = linregress(xp.log(t_win), xp.log(r2_win))
		if plot:
			fig, ax = plt.subplots(1, 1)
			ax.set_xlabel('ln $t$')
			ax.set_ylabel('ln $r^2$')
			color = self.get_color(self.color)[0]
			plt.plot(xp.log(t), xp.log(r2), ':', color=color, lw=1)
			plt.plot(xp.log(t_win), xp.log(r2_win), '-', color=color, lw=2)
			plt.plot(xp.log(t_win), res.slope * xp.log(t_win) + res.intercept, '-.', color=color, lw=2)
			if save:
				fig.savefig(filename + extension, dpi=self.dpi)
				print(f'\033[90m        Figure saved in {filename + extension} \033[00m')
			plt.pause(0.5)
		if full_output:
			return [res.slope, xp.exp(res.intercept / res.slope), res.rvalue**2]
		return res.slope
	
	def compute_rotation(self, h:xp.ufunc, plot:bool=False, save:bool=False, filename=None, extension='.jpg') -> xp.ndarray:
		x = h(xp.atleast_2d(self.xgc), xp.atleast_2d(self.ygc))
		nt = x[0, :].size
		omega = type(self).omega(xp.arange(1, nt) / nt)
		rotation_numb = xp.sum(x[:, 1:] * omega[xp.newaxis, :], axis=1) / xp.sum(omega)
		if plot:
			fig, ax = plt.subplots(1, 1)
			ax.set_xlabel('$n$')
			ax.set_ylabel('$\omega$')
			ax.plot(rotation_numb, '.', markersize=3)
			if save:
				fig.savefig(filename + extension, dpi=self.dpi)
				print(f'\033[90m        Figure saved in {filename + extension} \033[00m')
			plt.pause(0.5)
		return rotation_numb
