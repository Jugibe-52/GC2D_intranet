"""methods.classical.gauss_legendre observation records."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from contracts.methods.observation import IntegrationStep


@dataclass(frozen=True, slots=True)
class GaussLegendre4IntegrationStep(IntegrationStep):
	"""Expose the converged stages of one two-stage Gauss collocation step.

	The inherited state map is the physical fixed-time map, even when the
	integrator also advances the time-conjugate momentum for energy diagnostics.
	The two stage snapshots allow analytic observers to differentiate the ideal
	converged collocation equations without differentiating Newton iterations.
	"""

	first_stage_time: float
	second_stage_time: float
	newton_iterations: int
	residual_evaluations: int
	newton_residual_norm: float
	newton_tolerance: float
	first_stage_state: np.ndarray = field(repr=False, compare=False)
	second_stage_state: np.ndarray = field(repr=False, compare=False)


__all__ = ['GaussLegendre4IntegrationStep']
