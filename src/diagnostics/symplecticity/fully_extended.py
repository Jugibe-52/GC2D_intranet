"""Symplecticity of fully duplicated and fully projected implicit GC maps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynamics import GuidingCenterDynamics
from simulation import FullyExtendedImplicitIntegrationStep, IntegrationStep

from diagnostics.jacobians import central_difference_jacobian

from .observer import (
	gc_fully_duplicated_symplectic_form,
	gc_reduced_time_extended_symplectic_form,
)


_IDENTITY_4 = np.eye(4)
_ANTIDIAGONAL_EMBEDDING = np.vstack((_IDENTITY_4, -_IDENTITY_4))
_COPY_DIFFERENCE = np.hstack((_IDENTITY_4, -_IDENTITY_4))


def _metrics(jacobian: np.ndarray, form: np.ndarray) -> tuple[float, float, float, float]:
	"""Return relative form defect, determinant error, max entry, and condition."""
	jacobian = np.asarray(jacobian, dtype=float)
	if jacobian.shape != form.shape or not np.all(np.isfinite(jacobian)):
		raise ValueError("The analytic Jacobian must be finite and match the form.")
	defect = jacobian.T @ form @ jacobian - form
	return (
		float(np.linalg.norm(defect, ord="fro") / np.linalg.norm(form, ord="fro")),
		abs(float(np.linalg.det(jacobian)) - 1.0),
		float(np.max(np.abs(defect))),
		float(np.linalg.cond(jacobian)),
	)


def _relative_jacobian_audit(
	map_state: object,
	state: np.ndarray,
	analytic: np.ndarray,
	*,
	relative_step: float | None,
) -> float:
	"""Compare one analytic Jacobian with an independent centered difference."""
	if not callable(map_state):
		raise TypeError("The audited state map must be callable.")
	numerical = central_difference_jacobian(
		map_state,
		state,
		relative_step=relative_step,
	)
	scale = max(
		float(np.linalg.norm(analytic, ord="fro")),
		float(np.finfo(float).eps),
	)
	return float(np.linalg.norm(numerical - analytic, ord="fro") / scale)


@dataclass(frozen=True, slots=True)
class GCFullyExtendedSymplecticityRecord:
	"""``R^8`` base and complete reduced ``R^4`` defects for one step."""

	step_index: int
	time: float
	duration: float
	base_map_count: int
	maximum_r8_relative_defect: float
	maximum_r8_determinant_error: float
	maximum_r8_abs_defect: float
	maximum_r8_condition_number: float
	maximum_dpsi_jacobian_audit_error: float
	maximum_dr_jacobian_audit_error: float
	r4_relative_defect: float
	r4_determinant_error: float
	r4_max_abs_defect: float
	r4_condition_number: float
	r4_jacobian_audit_error: float


class GCFullyExtendedSymplecticityObserver:
	"""Differentiate accepted full-state implicit maps in ``R^8`` and ``R^4``."""

	def __init__(
		self,
		dynamics: GuidingCenterDynamics,
		*,
		relative_step: float | None = None,
	) -> None:
		"""Configure independent centered audits of the analytic Jacobians."""
		if not isinstance(dynamics, GuidingCenterDynamics):
			raise TypeError("`dynamics` must be GuidingCenterDynamics.")
		if relative_step is not None and (
			not np.isfinite(float(relative_step)) or float(relative_step) <= 0.0
		):
			raise ValueError("`relative_step` must be positive and finite.")
		self._dynamics = dynamics
		self.relative_step = relative_step
		self._records: list[GCFullyExtendedSymplecticityRecord] = []

	@property
	def records(self) -> tuple[GCFullyExtendedSymplecticityRecord, ...]:
		"""Return immutable scalar records in accepted-step order."""
		return tuple(self._records)

	def __call__(self, record: IntegrationStep) -> None:
		"""Measure one fully extended implicit integration record."""
		if not isinstance(record, FullyExtendedImplicitIntegrationStep):
			raise TypeError(
				"The observer requires FullyExtendedImplicitIntegrationStep records."
			)
		if record.dynamics is not self._dynamics:
			raise ValueError("The observed step belongs to a different GC system.")
		if record.step_index != len(self._records):
			raise ValueError("Symplecticity records must arrive sequentially.")
		if not record.base_maps:
			raise ValueError("A fully extended step must expose at least one base map.")

		r8_form = gc_fully_duplicated_symplectic_form()
		r8_rows: list[tuple[float, ...]] = []
		for base_map in record.base_maps:
			internal_input = np.asarray(base_map.state_before, dtype=float)
			analytic_dpsi = np.asarray(
				base_map.jacobian_state(internal_input),
				dtype=float,
			)
			dpsi_audit = _relative_jacobian_audit(
				base_map.map_state,
				internal_input,
				analytic_dpsi,
				relative_step=self.relative_step,
			)
			physical_start = 0.5 * (internal_input[:4] + internal_input[4:])
			analytic_dr = (
				_COPY_DIFFERENCE
				@ analytic_dpsi
				@ _ANTIDIAGONAL_EMBEDDING
				+ 2.0 * _IDENTITY_4
			)
			if not np.allclose(
				analytic_dr,
				np.asarray(base_map.residual_jacobian, dtype=float),
				rtol=2e-13,
				atol=2e-13,
			):
				raise ValueError("The recorded analytic residual Jacobian is inconsistent.")

			def residual_map(multiplier: np.ndarray) -> np.ndarray:
				value = np.asarray(multiplier, dtype=float)
				mapped = np.asarray(
					base_map.map_state(
						np.concatenate((physical_start + value, physical_start - value))
					),
					dtype=float,
				)
				return np.asarray(mapped[:4] - mapped[4:] + 2.0 * value)

			dr_audit = _relative_jacobian_audit(
				residual_map,
				np.asarray(base_map.projection_multiplier, dtype=float),
				analytic_dr,
				relative_step=self.relative_step,
			)
			r8_rows.append((*_metrics(analytic_dpsi, r8_form), dpsi_audit, dr_audit))
		r8_values = np.asarray(r8_rows, dtype=float)
		r4_jacobian = np.asarray(record.jacobian, dtype=float)
		r4_metrics = _metrics(
			r4_jacobian,
			gc_reduced_time_extended_symplectic_form(),
		)
		r4_audit = _relative_jacobian_audit(
			record.map_state,
			np.asarray(record.state_before, dtype=float),
			r4_jacobian,
			relative_step=self.relative_step,
		)
		self._records.append(
			GCFullyExtendedSymplecticityRecord(
				step_index=record.step_index,
				time=float(record.time),
				duration=float(record.duration),
				base_map_count=len(record.base_maps),
				maximum_r8_relative_defect=float(np.max(r8_values[:, 0])),
				maximum_r8_determinant_error=float(np.max(r8_values[:, 1])),
				maximum_r8_abs_defect=float(np.max(r8_values[:, 2])),
				maximum_r8_condition_number=float(np.max(r8_values[:, 3])),
				maximum_dpsi_jacobian_audit_error=float(np.max(r8_values[:, 4])),
				maximum_dr_jacobian_audit_error=float(np.max(r8_values[:, 5])),
				r4_relative_defect=r4_metrics[0],
				r4_determinant_error=r4_metrics[1],
				r4_max_abs_defect=r4_metrics[2],
				r4_condition_number=r4_metrics[3],
				r4_jacobian_audit_error=r4_audit,
			)
		)


__all__ = [
	"GCFullyExtendedSymplecticityObserver",
	"GCFullyExtendedSymplecticityRecord",
]
