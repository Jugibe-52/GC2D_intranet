"""Compatibility result and imports for the physical projection kernels."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .maps.physical import (
    _ABBAStages as _ABBAStages,
    _ResidualEvaluation as _ResidualEvaluation,
    _checked_vector_field_jacobian as _checked_vector_field_jacobian,
    _evaluate_displaced_stages as _evaluate_displaced_stages,
    _differentiate_stages as _differentiate_stages,
)

@dataclass(frozen=True, slots=True)
class _ProjectedStep:
	"""Converged physical state and nonlinear-solve diagnostics for one step."""

	state: np.ndarray
	multiplier: np.ndarray
	stages: _ABBAStages
	iterations: int
	residual_evaluations: int
	residual_norm: float


__all__: list[str] = []
