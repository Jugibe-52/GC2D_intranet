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
import matplotlib.colors as mcolors
from scipy.integrate._ivp.ivp import METHODS as IVP_METHODS
from scipy.interpolate import RectBivariateSpline
from scipy.integrate import solve_ivp
from scipy.special import jv
from scipy.stats import linregress
from scipy.io import savemat
from pyhamsys import METHODS, OdeSolution, HamSys, solve_ivp_symp, solve_ivp_sympext
from typing import List, Union
from datetime import date
import time

def real_imag(z:xp.ndarray):
	return z.real, z.imag

def save_data(self, data, filestr:str, info=[]) -> None:
	if self.SaveData:
		mdic = {'x': self.x, 'y': self.y, 'potential': self.potential, 'xy_period': self.xy_period}
		mdic.update({'data': data, 'info': info})
		mdic.update({'date': date.today().strftime(" %B %d, %Y\n"), 'author': 'cristel.chandre@cnrs.fr'})
		savemat(filestr + '.mat', mdic)
		print(f'\033[90m        Results saved in {filestr}.mat \033[00m')

def extract_potential(filename, nx=None, ny=None):
	import h5py
	with h5py.File(filename, 'r') as f:
		x = xp.array(f['Rcells'][:])
		y = xp.array(f['Zcells'][:])
		freqs = xp.array(f['freqs'][:])
		potential = xp.array(f['PHI_filtered_FT'])
		sum_xy = xp.sum(potential, axis=(1, 2))
		i_omega = xp.flatnonzero(sum_xy)[0]
		omega = 2 * xp.pi * freqs[i_omega]
		values = xp.array(f['PHI_filtered_FT'][i_omega,:,:]) / omega
	return Potential(x, y, values, nx=nx, ny=ny)

class Potential:
	def __init__(self, x, y, values, nx=None, ny=None, xy_period=None, tol=1e-10, k=3):
		x, y, values = xp.asarray(x), xp.asarray(y), xp.asarray(values)
		if x.ndim != 1:
			raise ValueError("`x` must be 1-dimensional.")
		if y.ndim != 1:
			raise ValueError("`y` must be 1-dimensional.")
		diff_x, diff_y = xp.diff(x), xp.diff(y)
		if xp.any(diff_x <= 0) or xp.any(diff_y <= 0):
			raise ValueError("Values in `x` or `y` are not properly sorted.")
		if xp.all(xp.abs(diff_x - diff_x[0]) > tol) or xp.all(xp.abs(diff_y - diff_y[0]) > tol):
			raise ValueError("Values in `x` or `y` are not uniformly spaced.")
		self.xmin, self.xmax = x.min(), x.max()
		self.ymin, self.ymax = y.min(), y.max()
		self.x = x if nx is None else xp.linspace(self.xmin, self.xmax, nx)
		self.y = y if ny is None else xp.linspace(self.ymin, self.ymax, ny)
		self.nx, self.ny = self.x.size, self.y.size
		if nx is None and ny is None:
			self.values = values
		else:
			spline_real = RectBivariateSpline(x, y, values.real, kx=k, ky=k)
			spline_imag = RectBivariateSpline(x, y, values.imag, kx=k, ky=k)
			self.values = spline_real(self.x, self.y) + 1j * spline_imag(self.x, self.y)
		self.dx, self.dy = self.x[1] - self.x[0], self.y[1] - self.y[0]
		self.xy_period = xy_period

	def gyroaverage(self, rho):
		fft_potential = fft2(self.values)
		kx, ky = fftfreq(self.nx, d=self.dx), fftfreq(self.ny, d=self.dy)
		kx_, ky_ = xp.meshgrid(kx, ky, indexing='ij')
		return  ifft2(fft_potential * jv(0, 2 * xp.pi * rho * xp.sqrt(kx_**2 + ky_**2)))
	
	def wrap(self, xi, yi):
		xi = ((xp.asarray(xi) - self.xmin) % self.xy_period) + self.xmin
		yi = ((xp.asarray(yi) - self.ymin) % self.xy_period) + self.ymin
		return xi, yi
	
	def isinside(self, x, y):
		return (x > self.xmin) & (x < self.xmax) & (y > self.ymin) & (y < self.ymax)
	
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
    return Potential(x, y, xp.einsum('nm,nm...->...', fft_phic, exp_xy), xy_period=2 * xp.pi)

class GC2D(HamSys):
	def __str__(self) -> str:
		return f'2D Guiding Center ({self.__class__.__name__}) for turbulent potentials'
		
	def __init__(self, potential, traj, k=3, SaveData=False):
		super().__init__(ndof=1.5 if traj["type"]=='gc' else 2.5, btype='pq')
		self.traj = traj
		self.rho = traj["rho"] if "rho" in traj else 0
		self.eta = traj["eta"] if "eta" in traj else 0
		self.CheckEnergy = traj["CheckEnergy"] if "CheckEnergy" in traj else False
		self.Lyapunov = traj["Lyapunov"] if "Lyapunov" in traj else False
		if self.Lyapunov:
			self.CheckEnergy = False
		self.SaveData = SaveData
		if self.rho != 0:
			potential.values = potential.gyroaverage(self.rho)
		self.potential = potential
		x = xp.pad(potential.x, (k, k), mode='linear_ramp',\
			 end_values=(potential.xmin - k * potential.dx, potential.xmax + k * potential.dx))
		y = xp.pad(potential.y, (k, k), mode='linear_ramp', \
			 end_values=(potential.ymin - k * potential.dy, potential.ymax + k * potential.dy))
		if potential.xy_period is not None:
			potential = xp.pad(potential.values, ((k, k), (k, k)), mode='wrap')
		else:
			potential = xp.pad(self.potential.values, ((k, k), (k, k)), mode='constant', constant_values=0)
		self.spline_real = RectBivariateSpline(x, y, potential.real, kx=k, ky=k)
		self.spline_imag = RectBivariateSpline(x, y, potential.imag, kx=k, ky=k) 
		if self.traj["type"] == 'fo':
			self.v_fo = self.rho / (2 * xp.abs(self.eta))
			self.phi_fo = xp.sign(self.eta) / self.rho
			self.omlar = 1 / (2 * self.eta)

	def phic_interp(self, xi, yi, dx=0, dy=0):
		interp_pot = xp.zeros_like(xi, dtype=xp.complex128)
		if self.potential.xy_period is not None:
			xi, yi = self.potential.wrap(xi, yi)
			ind = xp.arange(len(xi))
		else:
			ind = self.potential.isinside(xi, yi)
		interp_pot[ind] = self.spline_real.ev(xi[ind], yi[ind], dx=dx, dy=dy) + \
							1j * self.spline_imag.ev(xi[ind], yi[ind], dx=dx, dy=dy)
		return interp_pot
	
	def plot_potential(self, dx=0, dy=0, nx=512, ny=512):

		def white_centered_cmap(vmin, vmax):
			norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
			cmap = plt.get_cmap('RdBu_r')
			return cmap, norm
		
		xi = xp.linspace(self.potential.xmin, self.potential.xmax, nx)
		yi = xp.linspace(self.potential.ymin, self.potential.ymax, ny)
		X, Y = xp.meshgrid(xi, yi, indexing='ij')
		Z = self.phic_interp(X.flatten(), Y.flatten(), dx=dx, dy=dy).reshape(X.shape)

		vmin_real, vmax_real = Z.real.min(), Z.real.max()
		vmin_imag, vmax_imag = Z.imag.min(), Z.imag.max()

		fig, axs = plt.subplots(1, 2, figsize=(12, 5))
		cmap_real, norm_real = white_centered_cmap(vmin_real, vmax_real)

		c1 = axs[0].pcolormesh(X, Y, Z.real, shading='auto', cmap=cmap_real, norm=norm_real)
		axs[0].set_title(f'Real part of (dx={dx}, dy={dy}) potential')
		axs[0].set_xlabel('x')
		axs[0].set_ylabel('y')
		fig.colorbar(c1, ax=axs[0])

		cmap_imag, norm_imag = white_centered_cmap(vmin_imag, vmax_imag)
		c2 = axs[1].pcolormesh(X, Y, Z.imag, shading='auto', cmap=cmap_imag, norm=norm_imag)
		axs[1].set_title(f'Imaginary part of (dx={dx}, dy={dy}) potential')
		axs[1].set_xlabel('x')
		axs[1].set_ylabel('y')
		fig.colorbar(c2, ax=axs[1])

		plt.tight_layout()
		plt.show()

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
			return xp.sum((self.phic_interp(*xp.split(z, 2)) * xp.exp(-1j * t)).imag, axis=0)
		elif self.traj["type"] == 'fo': 
			x, y, vx, vy = xp.split(z, 4)
			phi_c = self.phic_interp(x, y)
			return xp.sum(self.rho / (4 * xp.abs(self.eta)) * (vx**2 + vy**2)\
				  + (phi_c * xp.exp(-1j * t)).imag * xp.sign(self.eta) / self.rho, axis=0)
        
	def y_dot(self, t, z):
		phase = xp.exp(-1j * t)
		x, y = xp.split(z if self.traj["type"] == 'gc' else xp.split(z, 2)[0], 2)
		dphi_dx = (self.phic_interp(x, y, dx=1) * phase).imag
		dphi_dy = (self.phic_interp(x, y, dy=1) * phase).imag
		if self.traj["type"] == 'gc':
			return xp.concatenate((-dphi_dy, dphi_dx), axis=None)
		elif self.traj["type"] == 'fo':
			vx, vy = xp.split(xp.split(z, 2)[1], 2)
			return xp.concatenate((vx * self.v_fo, vy * self.v_fo, -dphi_dx * self.phi_fo\
						   + vy * self.omlar, -dphi_dy * self.phi_fo - vx * self.omlar), axis=None)

	def y_dot_ext(self, t, z):
		if self.CheckEnergy:
			return xp.concatenate((self.y_dot(t, z[:-1]), self.k_dot(t, z[:-1])), axis=None)
		return self.y_dot(t, z)
	
	def y_dot_lyap(self, t, z):
		if self.traj["type"] == 'fo':
			raise NotImplementedError("Lyapunov exponents are not implemented for full orbits.")
		x, y, J11, J12, J21, J22 = xp.split(z, 6)
		z_dot = self.y_dot(t, xp.concatenate((x, y), axis=None))
		phase = xp.exp(-1j * t)
		d2phi_dx2 = (self.phic_interp(x, y, dx=2) * phase).imag
		d2phi_dxdy = (self.phic_interp(x, y, dx=1, dy=1) * phase).imag
		d2phi_dy2 = (self.phic_interp(x, y, dy=2) * phase).imag
		J11_dot = -J11 * d2phi_dxdy - J21 * d2phi_dy2
		J12_dot = -J12 * d2phi_dxdy - J22 * d2phi_dy2
		J21_dot = J11 * d2phi_dx2 + J21 * d2phi_dxdy
		J22_dot = J12 * d2phi_dx2 + J22 * d2phi_dxdy
		return xp.concatenate((z_dot, J11_dot, J12_dot, J21_dot, J22_dot), axis=None)

	def k_dot(self, t, z):
		x, y = xp.split(z if self.traj["type"] == 'gc' else xp.split(z, 2)[0], 2)
		phi = xp.sum((self.phic_interp(x, y) * xp.exp(-1j * t)).real)
		if self.traj["type"] == 'fo':
			phi *= -self.phi_fo
		return phi

	def chi_fo(self, h, t, z):
		if self.CheckEnergy:
			x, y, vx, vy = xp.split(z[:-1], 4)
			k = z[-1]
		else:
			x, y, vx, vy = xp.split(z, 4)
		exp_ = xp.exp(-1j * h * self.omlar)
		x, y = real_imag(x + 1j * y + 1j * self.rho * xp.sign(self.eta) * (exp_ - 1) * (vx + 1j * vy)) 
		vx, vy = real_imag(exp_ * (vx + 1j * vy))
		pot = xp.split(self.y_dot(t, xp.concatenate((x, y), axis=None)), 2)
		vx, vy = real_imag(vx + 1j * vy + h * 1j * (pot[0] + 1j * pot[1]) * self.phi_fo)
		if not self.CheckEnergy:
			return xp.concatenate((x, y, vx, vy), axis=None)
		k += h * self.phi_fo * self.k_dot(t, xp.concatenate((x, y), axis=None)) 
		return xp.concatenate((x, y, vx, vy, k), axis=None)
	
	def chi_star_fo(self, h, t, z):
		if self.CheckEnergy:
			x, y, vx, vy = xp.split(z[:-1], 5)
			k = z[-1]
		else:
			x, y, vx, vy = xp.split(z, 4)
		pot = xp.split(self.y_dot(t, xp.concatenate((x, y), axis=None)), 2)
		vx, vy = real_imag(vx + 1j * vy + h * 1j * (pot[0] + 1j * pot[1]) * self.phi_fo)
		if self.CheckEnergy:
			k += h * self.phi_fo * self.k_dot(t, xp.concatenate((x, y), axis=None))
		exp_ = xp.exp(-1j * h * self.omlar)
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
    
	def integrate(self, z0, t_eval, timestep, solver="BM4", omega=10, tol=1e-8, display=True):
		if display:
			print(f"\033[92m   Integration of {self.__str__()} \033[00m")
		start = time.time()
		if solver not in METHODS and solver not in IVP_METHODS:
			raise ValueError(f"Solver {solver} is not recognized.")
		if solver in IVP_METHODS and self.CheckEnergy:
			z0 = xp.append(z0, 0)
		if solver in IVP_METHODS:
			sol = solve_ivp(self.y_dot_ext, (t_eval[0], t_eval[-1]), z0, t_eval=t_eval, method=solver, atol=tol, rtol=tol)
			sol = self.rectify_sol(sol, check_energy=self.CheckEnergy)
		else:
			if self.traj["type"] == 'fo':
				sol = solve_ivp_symp(self.chi_fo, self.chi_star_fo, (t_eval[0], t_eval[-1]), z0, step=timestep, t_eval=t_eval, method=solver)
				sol = self.rectify_sol(sol, check_energy=self.CheckEnergy)
			elif self.traj["type"] == 'gc':
				sol = solve_ivp_sympext(self, (t_eval[0], t_eval[-1]), z0, step=timestep, t_eval=t_eval, method=solver, check_energy=self.CheckEnergy, omega=omega)
		if display:
			print(f'\033[90m        Computation finished in {int(time.time() - start)} seconds \033[00m')
			if self.CheckEnergy and hasattr(sol, 'err'):
				print(f'\033[90m           with error in energy = {sol.err}')
		return sol
	
	def compute_lyapunov(self, tf, z0, reortho_dt, tol=1e-8, solver='RK45'):
		if solver not in IVP_METHODS:
			raise ValueError(f"Solver {solver} is not recognized for Lyapunov exponent computation.")
		start = time.time()
		n = len(z0) // 2
		lyap_sum = xp.zeros((2, n), dtype=xp.float64)
		t, z = 0, xp.concatenate((z0, xp.ones(n), xp.zeros(n), xp.zeros(n), xp.ones(n)), axis=None)
		for _ in range(int(tf / reortho_dt)):
			sol = solve_ivp(self.y_dot_lyap, (t, t + reortho_dt), z, method=solver, t_eval=[t + reortho_dt], atol=tol, rtol=tol)
			z, Q = sol.y[:2 * n, -1], xp.moveaxis(sol.y[2 * n:, -1].reshape((2, 2, n)), -1, 0)
			for i in range(n):
				q, r = xp.linalg.qr(Q[i])
				lyap_sum[:, i] += xp.log(xp.abs(xp.diag(r)))
				Q[i] = q
			t += reortho_dt
			z = xp.concatenate((z, xp.moveaxis(Q, 0, -1)), axis=None)
		print(f'\033[90m        Computation finished in {int(time.time() - start)} seconds \033[00m')
		lyap_sort = xp.sort(lyap_sum / tf)
		return lyap_sort
	
	def plot_sol(self, sol, wrap=False): 
		x, y = xp.split(sol.y if self.traj["type"] == 'gc' else xp.split(sol.y, 2)[0], 2)
		if wrap:
			x, y = self.potential.wrap(x, y)
		plt.plot(x, y, '.', color='blue')
		plt.xlabel('x')
		plt.ylabel('y')
		plt.xlim(self.potential.xmin, self.potential.xmax)
		plt.ylim(self.potential.ymin, self.potential.ymax)
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
