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

import numpy as np
from numpy.fft import fft2, ifft2, fftfreq
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import RectBivariateSpline
from scipy.special import jv
from scipy.stats import linregress
from pyhamsys import HamSys
import time

def real_imag(z):
	return z.real, z.imag

def extract_potential(filename, A=1, nx=None, ny=None):
	import h5py
	with h5py.File(filename, 'r') as f:
		x = np.asarray(f['Rcells'][:])
		y = np.asarray(f['Zcells'][:])
		freqs = np.asarray(f['freqs'][:])
		potential = np.asarray(f['PHI_filtered_FT'])
		sum_xy = np.sum(potential, axis=(1, 2))
		nonzero_indices = np.flatnonzero(sum_xy)
		if nonzero_indices.size == 0:
			raise ValueError("No nonzero frequency mode found in potential data.")
		i_omega = nonzero_indices[0]
		omega = 2 * np.pi * freqs[i_omega]
		values = A * potential[i_omega, :, :] / omega
	return Potential(x, y, values, nx=nx, ny=ny)

class Potential:
	def __init__(self, x, y, values, nx=None, ny=None, xy_period=None, k=3):
		x, y, values = np.asarray(x), np.asarray(y), np.asarray(values)
		if x.ndim != 1:
			raise ValueError("`x` must be 1-dimensional.")
		if y.ndim != 1:
			raise ValueError("`y` must be 1-dimensional.")
		if values.shape != (len(x), len(y)):
			raise ValueError("Shape of `values` must match the lengths of `x` and `y`.")
		diff_x, diff_y = np.diff(x), np.diff(y)
		if np.any(diff_x <= 0) or np.any(diff_y <= 0):
			raise ValueError("Values in `x` or `y` are not properly sorted.")
		if not np.allclose(diff_x, diff_x[0]) or not np.allclose(diff_y, diff_y[0]):
			raise ValueError("Values in `x` or `y` are not uniformly spaced.")
		self.xy_period = xy_period
		if nx is not None or ny is not None:
			xi = np.linspace(x.min(), x.max(), nx)
			yi = np.linspace(y.min(), y.max(), ny)
			values_ = self.interp_potential(xi, yi, x, y, values, k=k, xy_period=xy_period)
		else:
			xi, yi, values_ = x, y, values
		self.x, self.y, self.values = xi, yi, values_
		self.dx, self.dy = self.x[1] - self.x[0], self.y[1] - self.y[0]
		self.xmin, self.xmax, self.ymin, self.ymax = self.x.min(), self.x.max(), self.y.min(), self.y.max()
		self.nx, self.ny = self.x.size, self.y.size

	def gyroaverage(self, rho):
		fft_potential = fft2(self.values)
		kx, ky = fftfreq(self.nx, d=self.dx), fftfreq(self.ny, d=self.dy)
		kx_, ky_ = np.meshgrid(kx, ky, indexing='ij')
		return  ifft2(fft_potential * jv(0, 2 * np.pi * rho * np.sqrt(kx_**2 + ky_**2)))
	
	def wrap(self, xi, yi):
		if self.xy_period is None:
			return xi, yi
		xi = ((np.asarray(xi) - self.xmin) % self.xy_period) + self.xmin
		yi = ((np.asarray(yi) - self.ymin) % self.xy_period) + self.ymin
		return xi, yi
	
	def potential_interpolator(self, x, y, values, k=3, xy_period=None):
		kl, kr = k + 1, k + 2 if  xy_period is not None else k + 1
		xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
		dx, dy = x[1] - x[0], y[1] - y[0]
		x_ = np.pad(x, (kl, kr), mode='linear_ramp', end_values=(xmin - kl * dx, xmax + kr * dx))
		y_ = np.pad(y, (kl, kr), mode='linear_ramp', end_values=(ymin - kl * dy, ymax + kr * dy))
		if xy_period:
			values_ = np.pad(values, ((kl, kr), (kl, kr)), mode='wrap')
		else:
			values_ = np.pad(values, ((kl, kr), (kl, kr)), mode='constant', constant_values=0)
		return RectBivariateSpline(x_, y_, values_.real, kx=k, ky=k), \
			RectBivariateSpline(x_, y_, values_.imag, kx=k, ky=k)
	
	def interp_potential(self, xi, yi, x, y, values, k=3, xy_period=None):
		interp_real, interp_imag = self.potential_interpolator(x, y, values, k=k, xy_period=xy_period)
		return interp_real(xi, yi) + 1j * interp_imag(xi, yi)
	
	def isinside(self, x, y):
		return (x > self.xmin) & (x < self.xmax) & (y > self.ymin) & (y < self.ymax)
	
def mock_potential(A, M, nx, ny, seed=27):
    x = np.linspace(0, 2 * np.pi, nx, endpoint=False)
    y = np.linspace(0, 2 * np.pi, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    np.random.seed(seed)
    phases = 2 * np.pi * np.random.random((M, M))
    nm = np.meshgrid(np.arange(M + 1), np.arange(M + 1), indexing='ij')
    fft_phic = np.zeros((M + 1, M + 1), dtype=np.complex128)
    fft_phic[1:, 1:] = A / (nm[0][1:, 1:]**2 + nm[1][1:, 1:]**2)**1.5 * np.exp(1j * phases)
    fft_phic[np.sqrt(nm[0]**2 + nm[1]**2) > M] = 0
    exp_xy = np.exp(1j * (nm[0][:, :, None, None] * X[None, None, :, :] + nm[1][:, :, None, None] * Y[None, None, :, :]))
    return Potential(x, y, np.einsum('nm,nm...->...', fft_phic, exp_xy), xy_period=2 * np.pi)

class GC2D(HamSys):
	def __str__(self) -> str:
		return f'2D Guiding Center ({self.__class__.__name__}) for turbulent potentials'
		
	def __init__(self, potential, traj, k=3):
		super().__init__(ndof=1.5 if traj["type"]=='gc' else 2.5, btype='pq')
		self.traj = traj
		self.rho = traj["rho"] if "rho" in traj else 0
		self.eta = traj["eta"] if "eta" in traj else 0
		if min(k * potential.dx, k * potential.dy) < self.rho:
			raise ValueError(f"Interpolation order {k} is too low for rho = {self.rho}. Increase k or decrease rho.")
		self.CheckEnergy = traj["CheckEnergy"] if "CheckEnergy" in traj else False
		if self.rho != 0:
			potential.values = potential.gyroaverage(self.rho)
		self.potential = potential
		self.spline_real, self.spline_imag = potential.potential_interpolator(potential.x, potential.y, potential.values, k=k, xy_period=potential.xy_period)
		if self.traj["type"] == 'fo':
			self.v_fo = self.rho / (2 * np.abs(self.eta))
			self.phi_fo = np.sign(self.eta) / self.rho
			self.omlar = 1 / (2 * self.eta)

	def phic_interp(self, xi, yi, dx=0, dy=0):
		interp_pot = np.zeros_like(xi, dtype=np.complex128)
		if self.potential.xy_period:
			xi, yi = self.potential.wrap(xi, yi)
			ind = np.arange(len(xi))
		else:
			ind = self.potential.isinside(xi, yi)
		interp_pot[ind] = self.spline_real.ev(xi[ind], yi[ind], dx=dx, dy=dy) + \
							1j * self.spline_imag.ev(xi[ind], yi[ind], dx=dx, dy=dy)
		return interp_pot
	
	def plot_potential(self, dx=0, dy=0, nx=512, ny=512):

		def white_centered_cmap(vmin, vmax):
			norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
			return plt.get_cmap('RdBu_r'), norm
		
		x = np.linspace(self.potential.xmin, self.potential.xmax + self.potential.dx, nx, endpoint=False)
		y = np.linspace(self.potential.ymin, self.potential.ymax + self.potential.dy, ny, endpoint=False)
		Z = self.spline_real(x, y, dx=dx, dy=dy) + 1j * self.spline_imag(x, y, dx=dx, dy=dy)

		vmin_real, vmax_real = Z.real.min(), Z.real.max()
		vmin_imag, vmax_imag = Z.imag.min(), Z.imag.max()

		fig, axs = plt.subplots(1, 2, figsize=(12, 5))
		cmap_real, norm_real = white_centered_cmap(vmin_real, vmax_real)
		c1 = axs[0].pcolormesh(x, y, Z.real.T, shading='auto', cmap=cmap_real, norm=norm_real)
		axs[0].set_title(f'Real part of (dx={dx}, dy={dy}) potential')
		axs[0].set_xlabel('x')
		axs[0].set_ylabel('y')
		fig.colorbar(c1, ax=axs[0])
		cmap_imag, norm_imag = white_centered_cmap(vmin_imag, vmax_imag)
		c2 = axs[1].pcolormesh(x, y, Z.imag.T, shading='auto', cmap=cmap_imag, norm=norm_imag)
		axs[1].set_title(f'Imaginary part of (dx={dx}, dy={dy}) potential')
		axs[1].set_xlabel('x')
		axs[1].set_ylabel('y')
		fig.colorbar(c2, ax=axs[1])
		plt.tight_layout()
		plt.show()

	def initial_conditions(self, n_traj, x=None, y=None, type='fixed'):
		x, y = self.potential.x if x is None else x, self.potential.y if y is None else y
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
		if self.traj["type"] == 'fo':
			np.random.seed(int(time.time()))
			phi_perp = 2 * np.pi * np.random.rand(n_traj)
			z0 = np.concatenate((z0, np.cos(phi_perp), np.sin(phi_perp)), axis=None)
			if self.CheckEnergy:
				z0 = np.concatenate((z0, np.zeros(n_traj)), axis=None)
		return z0

	def hamiltonian(self, t, z):
		if self.traj["type"] == 'gc':
			return np.sum((self.phic_interp(*np.split(z, 2)) * np.exp(-1j * t)).imag, axis=0)
		elif self.traj["type"] == 'fo': 
			x, y, vx, vy = np.split(z, 4)
			phi_c = self.phic_interp(x, y)
			return np.sum(self.rho / (4 * np.abs(self.eta)) * (vx**2 + vy**2)\
				  + (phi_c * np.exp(-1j * t)).imag * np.sign(self.eta) / self.rho, axis=0)
        
	def y_dot(self, t, z):
		phase = np.exp(-1j * t)
		x, y = np.split(z if self.traj["type"] == 'gc' else np.split(z, 2)[0], 2)
		dphi_dx = (self.phic_interp(x, y, dx=1) * phase).imag
		dphi_dy = (self.phic_interp(x, y, dy=1) * phase).imag
		if self.traj["type"] == 'gc':
			return np.concatenate((-dphi_dy, dphi_dx), axis=None)
		elif self.traj["type"] == 'fo':
			vx, vy = np.split(np.split(z, 2)[1], 2)
			return np.concatenate((vx * self.v_fo, vy * self.v_fo, -dphi_dx * self.phi_fo\
						   + vy * self.omlar, -dphi_dy * self.phi_fo - vx * self.omlar), axis=None)
	
	def y_dot_lyap(self, t, z):
		if self.traj["type"] == 'fo':
			raise NotImplementedError("Lyapunov exponents are not implemented for full orbits.")
		x, y, J11, J12, J21, J22 = np.split(z, 6)
		z_dot = self.y_dot(t, np.concatenate((x, y), axis=None))
		phase = np.exp(-1j * t)
		d2phi_dx2 = (self.phic_interp(x, y, dx=2) * phase).imag
		d2phi_dxdy = (self.phic_interp(x, y, dx=1, dy=1) * phase).imag
		d2phi_dy2 = (self.phic_interp(x, y, dy=2) * phase).imag
		J11_dot = -J11 * d2phi_dxdy - J21 * d2phi_dy2
		J12_dot = -J12 * d2phi_dxdy - J22 * d2phi_dy2
		J21_dot = J11 * d2phi_dx2 + J21 * d2phi_dxdy
		J22_dot = J12 * d2phi_dx2 + J22 * d2phi_dxdy
		return np.concatenate((z_dot, J11_dot, J12_dot, J21_dot, J22_dot), axis=None)

	def k_dot(self, t, z):
		x, y = np.split(z if self.traj["type"] == 'gc' else np.split(z, 2)[0], 2)
		phi = np.sum((self.phic_interp(x, y) * np.exp(-1j * t)).real)
		if self.traj["type"] == 'fo':
			phi *= -self.phi_fo
		return phi

	def chi(self, h, t, z):
		if self.CheckEnergy:
			x, y, vx, vy = np.split(z[:-1], 4)
			k = z[-1]
		else:
			x, y, vx, vy = np.split(z, 4)
		exp_ = np.exp(-1j * h * self.omlar)
		x, y = real_imag(x + 1j * y + 1j * self.rho * np.sign(self.eta) * (exp_ - 1) * (vx + 1j * vy)) 
		vx, vy = real_imag(exp_ * (vx + 1j * vy))
		pot = np.split(self.y_dot(t, np.concatenate((x, y), axis=None)), 2)
		vx, vy = real_imag(vx + 1j * vy + h * 1j * (pot[0] + 1j * pot[1]) * self.phi_fo)
		if not self.CheckEnergy:
			return np.concatenate((x, y, vx, vy), axis=None)
		k += h * self.phi_fo * self.k_dot(t, np.concatenate((x, y), axis=None)) 
		return np.concatenate((x, y, vx, vy, k), axis=None)
	
	def chi_star(self, h, t, z):
		if self.CheckEnergy:
			x, y, vx, vy = np.split(z[:-1], 5)
			k = z[-1]
		else:
			x, y, vx, vy = np.split(z, 4)
		pot = np.split(self.y_dot(t, np.concatenate((x, y), axis=None)), 2)
		vx, vy = real_imag(vx + 1j * vy + h * 1j * (pot[0] + 1j * pot[1]) * self.phi_fo)
		if self.CheckEnergy:
			k += h * self.phi_fo * self.k_dot(t, np.concatenate((x, y), axis=None))
		exp_ = np.exp(-1j * h * self.omlar)
		x, y = real_imag(x + 1j * y + 1j * self.rho * np.sign(self.eta) * (exp_ - 1) * (vx + 1j * vy)) 
		vx, vy = real_imag(exp_ * (vx + 1j * vy))
		if not self.CheckEnergy:
			return np.concatenate((x, y, vx, vy), axis=None)
		return np.concatenate((x, y, vx, vy, k), axis=None)
	
	def fo2gc(self, z):
		x, y, vx, vy = np.split(z, 4)
		v = vy + 1j * vx
		theta, rho = np.pi + np.angle(v), self.rho * np.abs(v)
		return x - rho * np.cos(theta), y + rho * np.sin(theta)
	
	def plot_sol(self, sol, wrap=False): 
		x, y = np.split(sol.y if self.traj["type"] == 'gc' else np.split(sol.y, 2)[0], 2)
		if wrap:
			x, y = self.potential.wrap(x, y)
		plt.plot(x.T, y.T, '.')
		plt.xlabel('x')
		plt.ylabel('y')
		plt.xlim(self.potential.xmin, self.potential.xmax)
		plt.ylim(self.potential.ymin, self.potential.ymax)
		plt.show()

class Trajectory(GC2D):
	
	type_dict = {'trapped': 0, 'diffusive': 1, 'ballistic': 2}
	color_dict = {'trapped': '#0072bd', 'diffusive': '#EDB120', 'ballistic': '#D95319'}
	omega = lambda t : np.exp(-1 / (t * (1 - t)))

	def __init__(self, sol, ttype, dict_):
		super().__init__(dict_)
		self.type = ['trapped', 'diffusive', 'ballistic'] if ttype == 'all' else ttype
		ntype = [type(self).type_dict[_] for _ in np.atleast_1d(self.type)]
		x, y = np.split(sol.y, self.dim)[:2]
		xgc, ygc = x, y if self.Method.endswith('gc') else self.fo2gc(sol.y)
		vec = np.ones(xgc[:, 0].shape)
		delta = np.asarray([el.ptp(axis=1) for el in [xgc, ygc]])
		vec[np.sqrt(np.sum(delta**2, axis=0)) <= self.threshold] = 0
		for _ in range(len(vec)):
			if vec[_]:
				vec[_] = 2 if self.compute_diffdata(sol.t, xgc[_, :], ygc[_, :]) >= self.thresh_b else 1
		indx = np.any([vec==_ for _ in ntype], axis=0)
		self.t, self.x, self.y, self.xgc, self.ygc  = sol.t, x[indx, :], y[indx, :], xgc[indx, :], ygc[indx,:]
		vec = np.tile(vec, self.dim)
		self.sol = sol.y[np.any([vec==_ for _ in ntype], axis=0), :]
		self.size = len(self.x[:, 0])
		self.color = [type(self).color_dict[_] for _ in np.atleast_1d(self.type)][0]
	
	def remove_trapped(self, sol): 
		xgc, ygc = np.split(sol.y, self.dim)[:2] if self.Method.endswith('gc') else self.fo2gc(sol.y)
		delta = np.asarray([el.ptp(axis=1) for el in [xgc, ygc]])
		vec = np.ones(xgc[:, 0].shape)
		vec[np.sqrt(np.sum(delta**2, axis=0)) <= self.threshold] = 0
		if self.CheckEnergy:
			sol.k = sol.k[vec!=0, :]
		vec = np.tile(vec, self.dim)
		sol.y = sol.y[vec!=0, :]
		return sol
	
	def compute_diffdata(self, t, x, y, full_output=False, plot=False, save=False, filename=None, extension='.jpg'):
		nt = t.size
		r2 = np.zeros(nt)
		for _ in range(nt):
			if x.ndim == 1:
				r2[_] = ((x[_:] - x[:-_ if _ else None])**2 + (y[_:] - y[:-_ if _ else None])**2).mean()
			else:
				r2[_] = ((x[:, _:] - x[:, :-_ if _ else None])**2 + (y[:, _:] - y[:, :-_ if _ else None])**2).mean()
		t_win, r2_win = t[nt//8:7*nt//8], r2[nt//8:7*nt//8]
		res = linregress(np.log(t_win), np.log(r2_win))
		if plot:
			fig, ax = plt.subplots(1, 1)
			ax.set_xlabel('ln $t$')
			ax.set_ylabel('ln $r^2$')
			color = self.get_color(self.color)[0]
			plt.plot(np.log(t), np.log(r2), ':', color=color, lw=1)
			plt.plot(np.log(t_win), np.log(r2_win), '-', color=color, lw=2)
			plt.plot(np.log(t_win), res.slope * np.log(t_win) + res.intercept, '-.', color=color, lw=2)
			if save:
				fig.savefig(filename + extension, dpi=self.dpi)
				print(f'\033[90m        Figure saved in {filename + extension} \033[00m')
			plt.pause(0.5)
		if full_output:
			return [res.slope, np.exp(res.intercept / res.slope), res.rvalue**2]
		return res.slope
	
	def compute_rotation(self, h, plot=False, save=False, filename=None, extension='.jpg'):
		x = h(np.atleast_2d(self.xgc), np.atleast_2d(self.ygc))
		nt = x[0, :].size
		omega = type(self).omega(np.arange(1, nt) / nt)
		rotation_numb = np.sum(x[:, 1:] * omega[np.newaxis, :], axis=1) / np.sum(omega)
		if plot:
			fig, ax = plt.subplots(1, 1)
			ax.set_xlabel(r'$n$')
			ax.set_ylabel(r'$\omega$')
			ax.plot(rotation_numb, '.', markersize=3)
			if save:
				fig.savefig(filename + extension, dpi=self.dpi)
				print(f'\033[90m        Figure saved in {filename + extension} \033[00m')
			plt.pause(0.5)
		return rotation_numb
