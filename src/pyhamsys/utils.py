# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Numerical and coordinate-transformation utilities."""

import numpy as xp
from scipy.fft import irfft, rfft, rfftfreq


def antiderivative(vec:xp.ndarray, N:int=2**10) -> xp.ndarray:
	nu = rfftfreq(N, d=1/N)
	div = xp.divide(1, 1j * nu, where=nu!=0)
	div[0] = 0
	return irfft(rfft(vec, axis=-1) * div, axis=-1)

def padwrap(vec:xp.ndarray) -> xp.ndarray:
	return xp.concatenate((vec, vec[..., 0][..., xp.newaxis]), axis=-1)

def get_last_elements(a:xp.ndarray, axis:int=0) -> None:
    shape = list(a.shape)
    shape[axis] = 1
    return xp.take(a, -1, axis=axis).reshape(tuple(shape))

def cart2sph(state:xp.ndarray, dim:int=2) -> xp.ndarray:
    if dim == 2:
        x, y, px, py = state
        r, phi = xp.hypot(x, y), xp.arctan2(y, x)
        p_r = (x * px + y * py) / r
        p_phi = x * py - y * px
        return xp.concatenate(([r], [phi], [p_r], [p_phi]), axis=0)
    elif dim == 3:
        x, y, z, px, py, pz = state
        xy, phi = xp.hypot(x, y), xp.arctan2(y, x)
        r, theta = xp.hypot(xy, z), xp.arctan2(xy, z)
        p_r = (x * px + y * py + z * pz) / r
        p_theta = ((x * px + y * py) * z - pz * xy**2) / xy
        p_phi = x * py - y * px
        return xp.concatenate(([r], [theta], [phi], [p_r], [p_theta], [p_phi]), axis=0)

def sph2cart(state:xp.ndarray, dim:int=2) -> xp.ndarray:
    if dim == 2:
        r, phi, p_r, p_phi = state
        x, y = r * xp.cos(phi), r * xp.sin(phi)
        px = p_r * xp.cos(phi) - p_phi * xp.sin(phi) / r
        py = p_r * xp.sin(phi) + p_phi * xp.cos(phi) / r
        return xp.concatenate(([x], [y], [px], [py]), axis=0)
    elif dim == 3:
        r, theta, phi, p_r, p_theta, p_phi = state
        x, y, z = r * xp.sin(theta) * xp.cos(phi), r * xp.sin(theta) * xp.sin(phi), r * xp.cos(theta)
        px = p_r * xp.sin(theta) * xp.cos(phi) + p_theta * xp.cos(theta) * xp.cos(phi) / r - p_phi * xp.sin(phi) / (r * xp.sin(theta))
        py = p_r * xp.sin(theta) * xp.sin(phi) + p_theta * xp.cos(theta) * xp.sin(phi) / r + p_phi * xp.cos(phi) / (r * xp.sin(theta))
        pz = p_r * xp.cos(theta) - p_theta * xp.sin(theta) / r
        return xp.concatenate(([x], [y], [z], [px], [py], [pz]), axis=0)

def rotating(state:xp.ndarray, angle:float, type:str='cartesian', dim:int=2) -> xp.ndarray:
    state_ = state.copy()
    if type == 'spherical':
        state_  = sph2cart(state_)
    if dim == 2:
        x, y, px, py = state_
    elif dim == 3:
        x, y, z, px, py, pz = state_
    xr = x * xp.cos(angle) + y * xp.sin(angle)
    yr = -x * xp.sin(angle) + y * xp.cos(angle)
    pxr = px * xp.cos(angle) + py * xp.sin(angle)
    pyr = -px * xp.sin(angle) + py * xp.cos(angle)
    if dim == 2:
        state_ = xp.concatenate(([xr], [yr], [pxr], [pyr]), axis=0)
    elif dim == 3:
        state_ = xp.concatenate(([xr], [yr], [z], [pxr], [pyr], [pz]), axis=0)
    return cart2sph(state_) if type == 'spherical' else state_

def field_envelope(t:float, te_au:xp.ndarray, envelope:str='sinus') -> float:
    te = xp.cumsum(te_au)
    if envelope == 'sinus':
        return xp.where(t<=0, 0, xp.where(t<=te[0], xp.sin(xp.pi * t / (2 * te[0]))**2, xp.where(t<=te[1], 1, xp.where(t<=te[2], xp.sin(xp.pi * (te[2] - t) / (2 * te_au[2]))**2, 0))))
    elif envelope == 'const':
        return 1
    elif envelope == 'trapez':
        return xp.where(t<=0, 0, xp.where(t<=te[0], t / te[0], xp.where(t<=te[1], 1, xp.where(t<=te[2], (te[2] - t) / te_au[2], 0))))
    else:
	    raise NameError(f'{envelope} envelope not defined')

METHODS = ['Verlet', 'FR', 'Yo# with # any integer', 'Yos6', 'M2', 'M4', 'EFRL', 'PEFRL', 'VEFRL', 'BM4', 'BM6', 'RKN4b', 'RKN6b', 'RKN6a', 'ABA104', 'ABA864', 'ABA1064']
