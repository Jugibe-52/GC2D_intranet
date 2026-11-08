"""Bind physical functions to a recipe and temporary doubled workspaces."""
from __future__ import annotations
from states.initial import InitialConditions

import json
from typing import Generic, TypeVar
from contracts.states.history import DiagnosticValue
from contracts.methods.functions import VectorField
from methods.palindromic.maps import GCDoubledMaps, HarmonicCoupling
from methods.base import FixedStepMethod
from methods.palindromic.core.composition import Composition, validate_composition

Detail = TypeVar('Detail')


class PalindromicMethod(FixedStepMethod[Detail], Generic[Detail]):
	"""Bind physical functions to a recipe and temporary doubled workspaces."""

	copies = 2
	recipe: Composition
	composition_name: str = 'custom'
	formulation: GCDoubledMaps

	def _configured_coupling(self) -> HarmonicCoupling | None:
		return None

	def _prepare_maps(self, initial_conditions: InitialConditions, *, jacobian: VectorField | None = None) -> None:
		"""Require explicit C operations and their independent rate together."""
		validate_composition(self.recipe)
		if not isinstance(self.composition_name, str) or not self.composition_name.strip():
			raise ValueError('A composition name must be a nonempty string.')
		coupling: HarmonicCoupling | None = self._configured_coupling()
		if coupling is not None and not isinstance(coupling, HarmonicCoupling):
			raise TypeError('Coupling must be a HarmonicCoupling or None.')
		if any(op.name == 'C' for op in self.recipe) != (coupling is not None):
			raise ValueError('Explicit C operations and a HarmonicCoupling must be configured together.')
		self.formulation = GCDoubledMaps(self.f, initial_conditions.values.size, coordinates=initial_conditions.layout,
			jacobian=jacobian,
			coupling_frequency=None if coupling is None else coupling.frequency)

	def _operation_metadata(self) -> dict[str, DiagnosticValue]:
		coupling: HarmonicCoupling | None = self._configured_coupling()
		return {
			'composition_name': self.composition_name,
			'operation_sequence': ''.join(op.name for op in self.recipe),
			'operation_coefficients': json.dumps([op.coefficient for op in self.recipe], allow_nan=False),
			'operation_count': len(self.recipe),
			'coupling_frequency': 0.0 if coupling is None else coupling.frequency,
		}


__all__ = ['PalindromicMethod']
