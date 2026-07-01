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
import os
from numpy.fft import fft2, ifft2, fftfreq
from typing import Any, Sequence
os.environ.setdefault("MPLCONFIGDIR", ".matplotlib")
from scipy.interpolate import RectBivariateSpline
from scipy.special import jv

Array = np.ndarray
FieldList = list[Array | list[Array] | None]
InterpolatorList = list[Any]


def real_imag(z: Array) -> tuple[Array, Array]:
	return z.real, z.imag


class Potential:
	def __init__(
		self,
		x: Array,
		y: Array,
		fields: FieldList,
		freqs: Sequence[float] | Array,
		nx: int | None = None,
		ny: int | None = None,
		xy_period: float | None = None,
		k: int = 3,
	) -> None:
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
		self.xy_period = xy_period
		self.kinterp = k
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

	def gyroaverage(self, rho: float, fields: FieldList) -> FieldList:
		kx, ky = fftfreq(self.nx, d=self.dx), fftfreq(self.ny, d=self.dy)
		kx_, ky_ = np.meshgrid(kx, ky, indexing='ij')
		mean_value, fluctuations = None, None
		if fields[0] is not None:
			mean_value = ifft2(fft2(fields[0]) * jv(0, 2 * np.pi * rho * np.sqrt(kx_**2 + ky_**2))).real
		if fields[1] is not None:
			fluctuations = []
			for field in fields[1]:
				gyro_field = ifft2(fft2(field) * jv(0, 2 * np.pi * rho * np.sqrt(kx_**2 + ky_**2))) 
				fluctuations.append(gyro_field)
		return [mean_value, fluctuations]

	def interpolate(self, x: Array, y: Array, fields: FieldList) -> InterpolatorList:
		kl, kr = self.kinterp + 1, self.kinterp + 2 if  self.xy_period is not None else self.kinterp + 1
		dx, dy = x[1] - x[0], y[1] - y[0]
		xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
		x_ = np.pad(x, (kl, kr), mode='linear_ramp', end_values=(xmin - kl * dx, xmax + kr * dx))
		y_ = np.pad(y, (kl, kr), mode='linear_ramp', end_values=(ymin - kl * dy, ymax + kr * dy))
		kwargs = {'mode': 'wrap'} if self.xy_period else {'mode': 'constant', 'constant_values': 0}
		mean_value, fluctuations = None, None
		if fields[0] is not None:
			fields_ = np.pad(fields[0], ((kl, kr), (kl, kr)), **kwargs)
			mean_value = RectBivariateSpline(x_, y_, fields_, kx=self.kinterp, ky=self.kinterp)
		if fields[1] is not None:
			fields_ = [np.pad(field, ((kl, kr), (kl, kr)), **kwargs) for field in fields[1]]
			fluctuations = []
			for field in fields_:
				interp_real = RectBivariateSpline(x_, y_, field.real, kx=self.kinterp, ky=self.kinterp)
				interp_imag = RectBivariateSpline(x_, y_, field.imag, kx=self.kinterp, ky=self.kinterp)
				fluctuations.append((interp_real, interp_imag))
		return [mean_value, fluctuations]

	def interp_fields(self, xi: Array, yi: Array, interpolators: InterpolatorList) -> FieldList:
		mean_value, fluctuations = None, None
		if interpolators[0]:
			mean_value = interpolators[0](xi, yi)
		if interpolators[1]:
			fluctuations = []
			for (interp_real, interp_imag) in interpolators[1]:
				fluctuations.append(interp_real(xi, yi) + 1j * interp_imag(xi, yi))
		return [mean_value, fluctuations]
