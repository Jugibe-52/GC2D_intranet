"""Composition coefficients shared by physical and extended ABBA maps."""

from __future__ import annotations

import numpy as np


_CUBE_ROOT_TWO = float(np.cbrt(2.0))
_GAMMA = 1.0 / (2.0 - _CUBE_ROOT_TWO)
_DELTA = -_CUBE_ROOT_TWO / (2.0 - _CUBE_ROOT_TWO)
_ABBA4_COEFFICIENTS = np.asarray((_GAMMA, _DELTA, _GAMMA), dtype=float)

# Yoshida's real symmetric order-six solution. Its negative stages are required
# for the odd composition conditions to cancel while preserving self-adjointness.
_ABBA6_COEFFICIENTS = np.asarray(
	(
		0.78451361047755726382,
		0.23557321335935813368,
		-1.17767998417887100695,
		1.31518632068391121889,
		-1.17767998417887100695,
		0.23557321335935813368,
		0.78451361047755726382,
	),
	dtype=float,
)


__all__: list[str] = []
