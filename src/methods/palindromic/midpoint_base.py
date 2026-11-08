"""Apply a complete palindrome, then average its spatial copies once."""
from __future__ import annotations
from states.initial import InitialConditions

import numpy as np
from contracts.arrays import BackendArray
from contracts.methods.observation import IntegrationStep
from contracts.states.history import DiagnosticValue
from contracts.methods.step import StepInfo, StepResult
from methods.base import FixedStepMethod
from methods.palindromic.configuration import StateExtension, _validate_state_extension
from methods.palindromic.core.midpoint import MidpointResult, midpoint_step
from methods.palindromic.base import PalindromicMethod


class MidpointPalindromicMethod(PalindromicMethod[MidpointResult]):
	"""Apply a complete palindrome, then average its spatial copies once."""

	state_extension: StateExtension

	def __post_init__(self) -> None:
		FixedStepMethod.__post_init__(self)
		self.state_extension = _validate_state_extension(self.state_extension)

	def _preset_metadata(self) -> dict[str, DiagnosticValue]:
		return {}

	def initialize(self, initial_conditions: InitialConditions) -> None:
		self._prepare_maps(initial_conditions)
		self.metadata = {
			'projection_kind': 'arithmetic_mean', 'state_extension': self.state_extension,
			'nonlinear_unknown_dimension': 0, 'vector_field_evaluations_per_step': sum(op.name in ('A', 'B') for op in self.recipe),
			**self._operation_metadata(), **self._preset_metadata(),
		}

	def advance(self, t: float, state: BackendArray, h: float, *, observe: bool = True) -> StepResult[MidpointResult]:
		result: MidpointResult = midpoint_step(self.formulation, self.recipe, t, self._checked_state(state), h)
		return StepResult(result.state, {'copy_separation_norms': result.copy_separation_norm}, result,
		                  result.trace.quadrature)

	def build_observation(self, info: StepInfo, step: StepResult[MidpointResult]) -> IntegrationStep:
		def map_state(candidate: np.ndarray) -> np.ndarray:
			return np.asarray(midpoint_step(self.formulation, self.recipe, info.time, candidate, info.duration).state)
		return IntegrationStep(dynamics_name='', method_name=self.method_name,
			step_index=info.index, start_time=info.time, time=info.end_time, duration=info.duration,
			state_before=info.state_before.copy(), state_after=step.details.state.copy(), map_state=map_state)


__all__ = ['MidpointPalindromicMethod']
