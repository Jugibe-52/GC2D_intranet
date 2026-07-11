import logging
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
from scipy import ndimage

from classes.potential import Array, Potential


logger = logging.getLogger(__name__)


def extract_potential(
	filename: str | Path,
	B: float = 1,
	indx: Sequence[int] | Array | None = None,
	nx: int | None = None,
	ny: int | None = None,
	denoising: bool = False,
	sigma: float = 1,
	k: int = 3,
) -> Potential:
	with h5py.File(filename, 'r') as f:
		x = np.asarray(f['Rcells'][()])
		y = np.asarray(f['Zcells'][()])
		freqs = np.atleast_1d(f['freqs'][()])
		fields = np.atleast_3d(f['fields'][()])
	expected_shape = (len(freqs), len(y), len(x))
	if fields.shape != expected_shape:
		raise ValueError(f"Shape of `fields` in {filename} is {fields.shape}, but expected {expected_shape}.")
	# HDF5 stores every plane as (y, x), while RectBivariateSpline expects
	# the field axes in the same order as its coordinate arguments: (x, y).
	fields = np.transpose(fields, (0, 2, 1))
	mean_value = None
	fluctuation_array = fields
	zero_mask = np.isclose(freqs, 0, atol=1e-5)
	if zero_mask.any():
		idx_zero = np.where(zero_mask)[0]
		if idx_zero.size > 0:
			mean_value = fields[idx_zero[0]].real
		freqs = np.delete(freqs, idx_zero)
		fluctuation_array = np.delete(fields, idx_zero, axis=0)
	if np.any(freqs < 0):
		idx_neg = np.where(freqs < 0)[0]
		freqs = np.delete(freqs, idx_neg)
		fluctuation_array = np.delete(fluctuation_array, idx_neg, axis=0)
	amplitudes = np.ptp(fluctuation_array, axis=(1, 2))
	sort_indices = np.argsort(amplitudes)[::-1]
	freqs = freqs[sort_indices]
	fluctuation_array = fluctuation_array[sort_indices]
	if len(freqs) > 0:
		omega = 2 * np.pi * freqs[0]
		scaling_factor = omega * B
		if fluctuation_array.size > 0:
			fluctuation_array /= scaling_factor
		if mean_value is not None:
			mean_value /= scaling_factor
	if indx is None:
		indx = np.arange(len(freqs) + 1)
	else:
		indx = np.atleast_1d(indx).astype(int)
		if indx.min() < 0 or indx.max() > len(freqs):
			raise ValueError(f"Indices must be in range [0, {len(freqs)}]")
	mean_value = mean_value if 0 in indx else None
	indx = indx[indx != 0] - 1
	freqs = freqs[indx]
	fluctuation_array = fluctuation_array[indx]
	if freqs.size > 0:
		fluctuations: Array | None = fluctuation_array.astype(np.complex128)
	else:
		fluctuations = None
	if denoising and fluctuations is not None:
		fluctuations = np.array([
			ndimage.gaussian_filter(fluct.real, sigma=sigma)
			+ 1j * ndimage.gaussian_filter(fluct.imag, sigma=sigma)
			for fluct in fluctuations
		])
	if denoising and mean_value is not None:
		mean_value = ndimage.gaussian_filter(mean_value, sigma=sigma)
	return Potential(x, y, [mean_value, fluctuations], freqs, nx=nx, ny=ny, k=k)


def mock_potential(A: float, M: int, nx: int, ny: int, seed: int = 27, k: int = 3) -> Potential:
	logger.info(
		"Generating mock potential: A=%g M=%d grid=%dx%d seed=%d interpolation_order=%d",
		A,
		M,
		nx,
		ny,
		seed,
		k,
	)
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
	potential = Potential(
		x,
		y,
		[None, [np.einsum('nm,nm...->...', fft_phic, exp_xy)]],
		freqs=[-1],
		xy_period=2 * np.pi,
		k=k,
	)
	logger.info("Mock potential ready: modes=%d spatial_period=%g", len(potential.freqs), potential.xy_period)
	return potential
