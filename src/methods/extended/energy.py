"""Passive normalized energy quadrature from accepted spatial ABBA stages."""
from __future__ import annotations
import numpy as np
from dynamics import HamiltonianSystem
from methods.extended.records import _ABBAStages
from collections.abc import Iterable
from formulations.gc import _EnergyQuadraturePoint
from formulations.state import DoubledFormulation


def _conjugate_momentum_increment_from_stages(
	dynamics: HamiltonianSystem,
	start_time: float,
	duration: float,
	stages: _ABBAStages,
	*,
	particle_count: int,
) -> np.ndarray:
	"""Integrate the projected momentum kappa through one ABBA map."""
	stop_time = start_time + duration

	def momentum_derivative(time: float, state: np.ndarray) -> np.ndarray:
		value = np.asarray(
			dynamics.extended_momentum_derivative(time, state),
			dtype=float,
		)
		if value.shape != (particle_count,) or not np.all(np.isfinite(value)):
			raise ValueError(
				"Energy tracking requires one finite momentum derivative "
				"per particle."
			)
		return value

	# The duplicated GC formulation projects its summed momentum back as
	# kappa=k/2, hence the additional factor one half after the four shears.
	with np.errstate(over="ignore", invalid="ignore"):
		doubled_increment = duration / 2.0 * (
			momentum_derivative(start_time, stages.v_initial)
			+ momentum_derivative(start_time, stages.u_first)
			+ momentum_derivative(stop_time, stages.u_first)
			+ momentum_derivative(stop_time, stages.v_final)
		)
		increment = np.asarray(doubled_increment / 2.0)
	if not np.all(np.isfinite(increment)):
		raise ValueError("The energy-tracking momentum increment became non-finite.")
	return increment


def momentum_increment(
    formulation: 'DoubledFormulation', points: 'Iterable[_EnergyQuadraturePoint]',
) -> np.ndarray:
    """Accumulate physical kappa once from accepted signed shear sources.

    The doubled Hamiltonian uses summed momentum k=2*kappa. Its passive
    quadrature never changes spatial states or nonlinear stopping criteria.
    """
    increment = np.zeros(formulation.particle_count)
    with np.errstate(over="ignore", invalid="ignore"):
        for time, duration, state in points:
            increment += duration * formulation.momentum_rate(time, state)
    increment /= 2
    if not np.all(np.isfinite(increment)):
        raise ValueError('The energy-tracking momentum increment became non-finite.')
    return increment
