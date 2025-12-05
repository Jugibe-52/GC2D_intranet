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
from pyhamsys import HamSys
import time

def real_imag(z):
	return z.real, z.imag

def extract_potential(filename, B=1, nx=None, ny=None):
	import h5py
	with h5py.File(filename, 'r') as f:
		x = np.asarray(f['Rcells'][:])
		y = np.asarray(f['Zcells'][:])
		freqs = np.atleast_1d(f['freqs'])
		fields = np.asarray(f['fields'][:])
	meanvalue, fluctuations = None, None
	index = np.where(freqs == 0)[0]
	if index.size > 0:
		meanvalue = fields[index[0]].real
		freqs = np.delete(freqs, index)
		fields = np.delete(fields, index, axis=0)
	if freqs.size > 0:
		omega = 2 * np.pi * freqs[0]
		fluctuations = np.asarray(fields, dtype=np.complex128).reshape((-1, len(x), len(y))) / (omega * B)
		freqs = freqs / omega
	if meanvalue is not None and fluctuations is not None:
		meanvalue = meanvalue / (omega * B)
	return Potential(x, y, [meanvalue, fluctuations], freqs, nx=nx, ny=ny)

class Potential:
	def __init__(self, x, y, fields, freqs, nx=None, ny=None, xy_period=None, k=3):
		self.freqs = np.atleast_1d(freqs)
		if x.ndim != 1:
			raise ValueError("`x` must be 1-dimensional.")
		if y.ndim != 1:
			raise ValueError("`y` must be 1-dimensional.")
		diff_x, diff_y = np.diff(x), np.diff(y)
		if np.any(diff_x <= 0) or np.any(diff_y <= 0):
			raise ValueError("Values in `x` or `y` are not properly sorted.")
		if not np.allclose(diff_x, diff_x[0]) or not np.allclose(diff_y, diff_y[0]):
			raise ValueError("Values in `x` or `y` are not uniformly spaced.")
		if not len(fields) == 2:
			raise ValueError("`fields` must be a list of two elements: [meanvalue, fluctuations].")
		if np.asarray(fields[1]).shape != (len(freqs), len(x), len(y)):
			raise ValueError("Shape of `fluctuations`, e.g., `fields[1]`, does not match the lengths of `freqs`, `x`, and `y`.")
		self.xy_period = xy_period
		self.k = k
		if nx is not None or ny is not None:
			xi = np.linspace(x.min(), x.max(), nx)
			yi = np.linspace(y.min(), y.max(), ny)
			interpolators = self.interpolate(x, y, fields)
			fields = self.interp_fields(xi, yi, interpolators)
		else:
			xi, yi = x, y
		self.x, self.y, self.fields = xi, yi, fields
		self.dx, self.dy = self.x[1] - self.x[0], self.y[1] - self.y[0]
		self.xmin, self.xmax, self.ymin, self.ymax = self.x.min(), self.x.max(), self.y.min(), self.y.max()
		self.nx, self.ny = self.x.size, self.y.size

	def gyroaverage(self, rho, fields):
		kx, ky = fftfreq(self.nx, d=self.dx), fftfreq(self.ny, d=self.dy)
		kx_, ky_ = np.meshgrid(kx, ky, indexing='ij')
		meanvalue, fluctuations = 0, 0
		if fields[0] is not None:
			meanvalue = ifft2(fft2(fields[0]) * jv(0, 2 * np.pi * rho * np.sqrt(kx_**2 + ky_**2))).real
		if fields[1] is not None:
			fluctuations = []
			for field in fields[1]:
				gyro_field = ifft2(fft2(field) * jv(0, 2 * np.pi * rho * np.sqrt(kx_**2 + ky_**2))) 
				fluctuations.append(gyro_field)
		return [meanvalue, fluctuations]

	def interpolate(self, x, y, fields):
		kl, kr = self.k + 1, self.k + 2 if  self.xy_period is not None else self.k + 1
		dx, dy = x[1] - x[0], y[1] - y[0]
		xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
		x_ = np.pad(x, (kl, kr), mode='linear_ramp', end_values=(xmin - kl * dx, xmax + kr * dx))
		y_ = np.pad(y, (kl, kr), mode='linear_ramp', end_values=(ymin - kl * dy, ymax + kr * dy))
		kwargs = {'mode': 'wrap'} if self.xy_period else {'mode': 'constant', 'constant_values': 0}
		meanvalue, fluctuations = 0, 0
		if fields[0] is not None:
			fields_ = np.pad(fields[0], ((kl, kr), (kl, kr)), **kwargs)
			meanvalue = RectBivariateSpline(x_, y_, fields_, kx=self.k, ky=self.k)
		if fields[1] is not None:
			fields_ = [np.pad(field, ((kl, kr), (kl, kr)), **kwargs) for field in fields[1]]
			fluctuations = []
			for field in fields_:
				interp_real = RectBivariateSpline(x_, y_, field.real, kx=self.k, ky=self.k)
				interp_imag = RectBivariateSpline(x_, y_, field.imag, kx=self.k, ky=self.k)
				fluctuations.append((interp_real, interp_imag))
		return [meanvalue, fluctuations]

	def interp_fields(self, xi, yi, interpolators):
		meanvalue, fluctuations = 0, 0
		if interpolators[0]:
			meanvalue = interpolators[0](xi, yi)
		if interpolators[1]:
			fluctuations = []
			for (interp_real, interp_imag) in interpolators[1]:
				fluctuations.append(interp_real(xi, yi) + 1j * interp_imag(xi, yi))
		return [meanvalue, fluctuations]
	
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
    return Potential(x, y, [0, [np.einsum('nm,nm...->...', fft_phic, exp_xy)]], freqs=[-1], xy_period=2 * np.pi)

class GC2D(HamSys, Potential):
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
		for key, value in vars(potential).items():
			setattr(self, key, value)
		if self.rho != 0:
			self.fields = self.gyroaverage(self.rho, self.fields)
		self.interpolators = self.interpolate(self.x, self.y, self.fields)
		if self.traj["type"] == 'fo':
			self.v_fo = self.rho / (2 * np.abs(self.eta))
			self.phi_fo = np.sign(self.eta) / self.rho
			self.omlar = 1 / (2 * self.eta)

	def phic_interp(self, xi, yi, dx=0, dy=0):
		xi, yi = self.wrap_or_clip(xi, yi)
		meanvalue, fluctuations = 0, 0
		if self.fields[0]:
			meanvalue = self.interpolators[0].ev(xi, yi, dx=dx, dy=dy)
		if self.fields[1]:
			fluctuations = []
			for (interp_real, interp_imag) in self.interpolators[1]:
				fluctuations.append(interp_real.ev(xi, yi, dx=dx, dy=dy) + 1j * interp_imag.ev(xi, yi, dx=dx, dy=dy))
		return [meanvalue, fluctuations]

	def wrap_or_clip(self, xi, yi):
		if self.xy_period is None:
			xi = np.clip(xi, self.xmin, self.xmax)
			yi = np.clip(yi, self.ymin, self.ymax)
		else:
			xi = ((np.asarray(xi) - self.xmin) % self.xy_period) + self.xmin
			yi = ((np.asarray(yi) - self.ymin) % self.xy_period) + self.ymin
		return xi, yi
	
	def plot_potential(self, dx=0, dy=0, nx=512, ny=512):
		def white_centered_cmap(vmin, vmax):
			if vmin >= 0:
				cmap = plt.get_cmap('Reds')
				norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
				return cmap, norm
			if vmax <= 0:
				cmap = plt.get_cmap('Blues_r')
				norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
				return cmap, norm
			norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
			return plt.get_cmap('RdBu_r'), norm
		x = np.linspace(self.xmin, self.xmax + self.dx, nx, endpoint=False)
		y = np.linspace(self.ymin, self.ymax + self.dy, ny, endpoint=False)
		if self.fields[0] is not None:
			Z = self.interpolators[0](x, y, dx=dx, dy=dy)
			vmin, vmax = Z.min(), Z.max()
			print(vmin, vmax)
			cmap, norm = white_centered_cmap(vmin, vmax)
			plt.figure(figsize=(6, 5))
			c = plt.pcolormesh(x, y, Z.T, shading='auto', cmap=cmap, norm=norm)
			plt.title(f'Potential (dx={dx}, dy={dy}) (freq=0)')
			plt.xlabel('x')
			plt.ylabel('y')
			plt.colorbar(c)
			plt.tight_layout()
		if self.fields[1] is not None:
			for interpolator, freq in zip(self.interpolators[1], self.freqs):
				Zr, Zi = interpolator[0](x, y, dx=dx, dy=dy), interpolator[1](x, y, dx=dx, dy=dy)
				vmin_real, vmax_real = Zr.min(), Zr.max()
				vmin_imag, vmax_imag = Zi.min(), Zi.max()
				fig, axs = plt.subplots(1, 2, figsize=(12, 5))
				cmap_real, norm_real = white_centered_cmap(vmin_real, vmax_real)
				c1 = axs[0].pcolormesh(x, y, Zr.T, shading='auto', cmap=cmap_real, norm=norm_real)
				axs[0].set_title(f'Real part of (dx={dx}, dy={dy}) potential (freq={freq})')
				axs[0].set_xlabel('x')
				axs[0].set_ylabel('y')
				fig.colorbar(c1, ax=axs[0])
				cmap_imag, norm_imag = white_centered_cmap(vmin_imag, vmax_imag)
				c2 = axs[1].pcolormesh(x, y, Zi.T, shading='auto', cmap=cmap_imag, norm=norm_imag)
				axs[1].set_title(f'Imaginary part of (dx={dx}, dy={dy}) potential (freq={freq})')
				axs[1].set_xlabel('x')
				axs[1].set_ylabel('y')
				fig.colorbar(c2, ax=axs[1])
				plt.tight_layout()
		plt.show()

	def initial_conditions(self, n_traj, x=None, y=None, type='fixed'):
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
		if self.traj["type"] == 'fo':
			np.random.seed(int(time.time()))
			phi_perp = 2 * np.pi * np.random.rand(n_traj)
			z0 = np.concatenate((z0, np.cos(phi_perp), np.sin(phi_perp)), axis=None)
			if self.CheckEnergy:
				z0 = np.concatenate((z0, np.zeros(n_traj)), axis=None)
		return z0
	
	def get_positions(self, z):
		return np.split(z if self.traj["type"] == 'gc' else np.split(z, 2)[0], 2)
	
	def get_velocities(self, z):
		return None if self.traj["type"] == 'gc' else np.split(np.split(z, 2)[1], 2)

	def hamiltonian(self, t, z):
		x, y = self.get_positions(z)
		phi_c = self.phic_interp(x, y)
		phi_t = np.sum(phi_c[0])
		for fluct in phi_c[1]:
			phi_t += 2 * np.sum((fluct * np.exp(1j * self.freqs[np.newaxis] * t)).real)
		if self.traj["type"] == 'gc':
			return phi_t
		elif self.traj["type"] == 'fo': 
			vx, vy = self.get_velocities(z)
			return np.sum(self.rho / (4 * np.abs(self.eta)) * (vx**2 + vy**2) + phi_t * np.sign(self.eta) / self.rho)
        
	def y_dot(self, t, z, output='full'):
		x, y = self.get_positions(z)
		dphidx_c, dphidy_c = self.phic_interp(x, y, dx=1), self.phic_interp(x, y, dy=1)
		dphidx_t, dphidy_t = dphidx_c[0], dphidy_c[0]
		for fluct_x, fluct_y, freq in zip(dphidx_c[1], dphidy_c[1], self.freqs):
			phases = 2 * np.exp(1j * freq * t)
			dphidx_t += (fluct_x * phases).real
			dphidy_t += (fluct_y * phases).real
		if self.traj["type"] == 'gc' or output == 'reduced':
			return np.concatenate((-dphidy_t, dphidx_t), axis=None)
		elif self.traj["type"] == 'fo':
			vx, vy = self.get_velocities(z)
			return np.concatenate((vx * self.v_fo, vy * self.v_fo, -dphidx_t * self.phi_fo\
						   + vy * self.omlar, -dphidy_t * self.phi_fo - vx * self.omlar), axis=None)

	def y_dot_lyap(self, t, z):
		if self.traj["type"] == 'fo':
			x, y, vx, vy, *J = np.split(z, 20)
			z = np.concatenate((x, y, vx, vy), axis=None)
			J = np.array(J).reshape((4, 4, -1))
		if self.traj["type"] == 'gc':
			x, y, *J = np.split(z, 6)
			z = np.concatenate((x, y), axis=None)
			J = np.array(J).reshape((2, 2, -1))
		z_dot = self.y_dot(t, z)
		d2phidx2_c = self.phic_interp(x, y, dx=2) 
		d2phidxdy_c = self.phic_interp(x, y, dx=1, dy=1)
		d2phidy2_c = self.phic_interp(x, y, dy=2)
		d2phidx2_t, d2phidxdy_t, d2phidy2_t = d2phidx2_c[0], d2phidxdy_c[0], d2phidy2_c[0]
		for fluct_xx, fluct_xy, fluct_yy, freq in zip(d2phidx2_c[1], d2phidxdy_c[1], d2phidy2_c[1], self.freqs):
			phase = 2 * np.exp(1j * freq * t)
			d2phidx2_t += (fluct_xx * phase).real
			d2phidxdy_t += (fluct_xy * phase).real
			d2phidy2_t += (fluct_yy * phase).real
		A = np.zeros_like(J)
		if self.traj["type"] == 'fo':
			d2phidx2_c *= -self.phi_fo
			d2phidxdy_c *= -self.phi_fo
			d2phidy2_c *= -self.phi_fo
			A[0, 2, :], A[1, 3, :] = self.v_fo * np.ones_like(x), self.v_fo * np.ones_like(x)
			A[2, 3, :], A[3, 2, :] = self.omlar * np.ones_like(x), -self.omlar * np.ones_like(x)
			A[2, 0, :], A[2, 1, :] = d2phidx2_t, d2phidxdy_t
			A[3, 0, :], A[3, 1, :] = d2phidxdy_t, d2phidy2_t
		if self.traj["type"] == 'gc':
			A[0, 0, :], A[0, 1, :] = -d2phidxdy_t, -d2phidy2_t
			A[1, 0, :], A[1, 1, :] = d2phidx2_t, d2phidxdy_t
		J_dot = np.einsum('ijm,jkm->ikm', A, J)
		return np.concatenate((z_dot, J_dot.reshape(-1)), axis=None)

	def k_dot(self, t, z):
		x, y = self.get_positions(z)
		phi_c = self.phic_interp(x, y)
		dphidt_t = 0
		for fluct, freq in zip(phi_c[1], self.freqs):
			dphidt_t += 2 * freq * np.sum((fluct * np.exp(1j * freq * t)).imag)
		if self.traj["type"] == 'fo':
			dphidt_t *= -self.phi_fo
		return dphidt_t

	def chi(self, h, t, z):
		if self.CheckEnergy:
			x, y, vx, vy = np.split(z[:-1], 4)
			k = z[-1]
		else:
			x, y, vx, vy = np.split(z, 4)
		exp_ = np.exp(-1j * h * self.omlar)
		x, y = real_imag(x + 1j * y + 1j * self.rho * np.sign(self.eta) * (exp_ - 1) * (vx + 1j * vy)) 
		vx, vy = real_imag(exp_ * (vx + 1j * vy))
		pot = np.split(self.y_dot(t, np.concatenate((x, y), axis=None), output='reduced'), 2)
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
		pot = np.split(self.y_dot(t, np.concatenate((x, y), axis=None), output='reduced'), 2)
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
		x, y = self.get_positions(sol.y)
		if wrap:
			x, y = self.wrap_or_clip(x, y)
		plt.plot(x.T, y.T, '.')
		plt.xlabel('x')
		plt.ylabel('y')
		if wrap:
			plt.xlim(self.xmin, self.xmax)
			plt.ylim(self.ymin, self.ymax)
		plt.show()
