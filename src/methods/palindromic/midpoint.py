"""Experimental recipe with final arithmetic projection; no order is inferred."""
from __future__ import annotations

from dataclasses import dataclass, field
from methods.palindromic.maps import HarmonicCoupling
from methods.palindromic.configuration import StateExtension
from methods.palindromic.core.composition import Composition
from methods.palindromic.midpoint_base import MidpointPalindromicMethod


@dataclass(slots=True)
class PalindromicMidpoint(MidpointPalindromicMethod):
	"""Experimental recipe with final arithmetic projection; no order is inferred."""

	recipe: Composition
	composition_name: str = field(default='custom', kw_only=True)
	state_extension: StateExtension = 'physical'
	coupling: HarmonicCoupling | None = field(default=None, kw_only=True)

	def _configured_coupling(self) -> HarmonicCoupling | None:
		return self.coupling


__all__ = ['PalindromicMidpoint']
