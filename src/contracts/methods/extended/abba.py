"""methods.extended.abba observation records."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from contracts.methods.observation import ImplicitIntegrationStep


@dataclass(frozen=True, slots=True)
class ABBA2ImplicitIntegrationStep(ImplicitIntegrationStep):
	"""Expose converged ABBA stages without performing diagnostic analysis.

	Analytic tangent diagnostics use these snapshots and ``dynamics`` to evaluate
	the four vector-field Jacobians without importing private solver helpers.
	"""

	multiplier: np.ndarray = field(repr=False, compare=False)
	u_initial: np.ndarray = field(repr=False, compare=False)
	v_initial: np.ndarray = field(repr=False, compare=False)
	u_first: np.ndarray = field(repr=False, compare=False)
	v_final: np.ndarray = field(repr=False, compare=False)
	u_final: np.ndarray = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class UnprojectedABBAIntegrationStep:
	"""Expose one signed ABBA map inside an unprojected composition.

	The two copies remain independent between consecutive entries. Stage arrays
	use the same component-major physical layout as an implicit ABBA snapshot,
	but this record has no multiplier or nonlinear solve of its own.
	"""

	start_time: float
	time: float
	duration: float
	u_initial: np.ndarray = field(repr=False, compare=False)
	v_initial: np.ndarray = field(repr=False, compare=False)
	u_first: np.ndarray = field(repr=False, compare=False)
	v_final: np.ndarray = field(repr=False, compare=False)
	u_final: np.ndarray = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class ABBAImplicitCompositionIntegrationStep(ImplicitIntegrationStep):
	"""Expose one outer projection around a continuous ABBA composition.

	Each signed substep retains both spatial copies and has no independent
	multiplier or nonlinear solve. The complete step owns those quantities.
	"""

	multiplier: np.ndarray = field(repr=False, compare=False)
	composition_coefficients: np.ndarray = field(repr=False, compare=False)
	substeps: tuple[UnprojectedABBAIntegrationStep, ...] = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class ABBA4ImplicitIntegrationStep(ABBAImplicitCompositionIntegrationStep):
	"""Three unprojected ABBA pairs and their one outer projection."""


__all__ = ['ABBA2ImplicitIntegrationStep', 'UnprojectedABBAIntegrationStep', 'ABBAImplicitCompositionIntegrationStep', 'ABBA4ImplicitIntegrationStep']
