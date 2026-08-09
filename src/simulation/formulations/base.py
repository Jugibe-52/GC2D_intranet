"""Contracts and utilities for direct/adjoint numerical formulations."""

from __future__ import annotations

from typing import Protocol, TypeAlias, runtime_checkable

import numpy as np

from simulation._result import DiagnosticValue
from simulation.problem import InitialValueProblem


Projection: TypeAlias = tuple[np.ndarray, dict[str, DiagnosticValue]]


class PreparedDirectAdjointFormulation(Protocol):
	"""Per-run immutable maps consumed by a composition method."""

	@property
	def dynamics_name(self) -> str:
		"""Stable physical-dynamics label emitted with stage observations."""

	@property
	def initial_internal_state(self) -> np.ndarray:
		"""Return the packed internal initial state for this prepared run."""

	def direct_map(
		self,
		duration: float,
		t: float,
		state: np.ndarray,
	) -> np.ndarray:
		"""Apply one direct map."""

	def adjoint_map(
		self,
		duration: float,
		t: float,
		state: np.ndarray,
	) -> np.ndarray:
		"""Apply one adjoint map."""

	def project(self, internal_history: np.ndarray) -> Projection:
		"""Return the physical history and formulation diagnostics."""


class PreparedStageProjectedFormulation(PreparedDirectAdjointFormulation, Protocol):
	"""Prepared maps with a projection applied after every composition stage."""

	@property
	def supports_stage_projection(self) -> bool:
		"""Whether the prepared formulation permits end-of-stage projection."""

	def project_internal_state(self, state: np.ndarray) -> np.ndarray:
		"""Return the internal state to use at the start of the next stage."""


@runtime_checkable
class DirectAdjointFormulation(Protocol):
	"""Reusable configuration that prepares direct and adjoint maps."""

	def prepare(
		self,
		problem: InitialValueProblem,
		*,
		track_energy: bool,
	) -> PreparedDirectAdjointFormulation:
		"""Create immutable maps bound to one simulation problem."""


class StageProjectedFormulation(DirectAdjointFormulation, Protocol):
	"""Configuration that prepares maps with an end-of-stage projection."""

	def prepare(
		self,
		problem: InitialValueProblem,
		*,
		track_energy: bool,
	) -> PreparedStageProjectedFormulation:
		"""Create maps and their internal end-of-stage projection."""


def generalized_energy_error(
	t: np.ndarray,
	states: np.ndarray,
	momentum: np.ndarray | None,
	hamiltonian: object,
) -> float:
	"""Return maximum drift of physical or extended Hamiltonian."""
	evaluate = getattr(hamiltonian, "hamiltonian", None)
	if not callable(evaluate):
		raise TypeError("Energy diagnostics require HamiltonianSystem.")
	energy = np.asarray(evaluate(t, states), dtype=float)
	if momentum is not None:
		energy = energy + np.asarray(momentum)
	if energy.ndim == 1:
		energy = energy[np.newaxis, :]
	return float(np.max(np.abs(energy - energy[:, :1])))


__all__ = [
	"DirectAdjointFormulation",
	"PreparedDirectAdjointFormulation",
	"PreparedStageProjectedFormulation",
	"StageProjectedFormulation",
]
