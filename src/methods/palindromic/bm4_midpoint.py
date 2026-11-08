"""Uncoupled fourth-order BM4 palindrome with one final arithmetic mean."""
from __future__ import annotations

from dataclasses import dataclass
from contracts.states.history import DiagnosticValue
from methods.palindromic.configuration import StateExtension
from methods.palindromic.core.operations import A, B, T
from methods.palindromic.midpoint_base import MidpointPalindromicMethod

_A1: float = 0.0792036964311957
_A2: float = 0.1303114101821663
_A3: float = 0.2228614958676077
_A4: float = -0.3667132690474257
_A5: float = 0.3246481886897062
_A6: float = 0.1096884778767498


@dataclass(slots=True)
class BM4Midpoint(MidpointPalindromicMethod):
	"""Uncoupled fourth-order BM4 palindrome with one final arithmetic mean."""

	state_extension: StateExtension = 'physical'
	composition_name = 'BM4'
	recipe = (
		A(_A1), B(_A1), T(_A1+_A2), B(_A2), A(_A2),
		A(_A3), B(_A3), T(_A3+_A4), B(_A4), A(_A4),
		A(_A5), B(_A5), T(_A5+_A6), B(_A6), A(_A6),
		A(_A6), B(_A6), T(_A6+_A5), B(_A5), A(_A5),
		A(_A4), B(_A4), T(_A4+_A3), B(_A3), A(_A3),
		A(_A2), B(_A2), T(_A2+_A1), B(_A1), A(_A1),
	)

	def _preset_metadata(self) -> dict[str, DiagnosticValue]:
		return {'coupling_frequency': 0.0, 'composition_stage_count': 12}


__all__ = ['BM4Midpoint']
