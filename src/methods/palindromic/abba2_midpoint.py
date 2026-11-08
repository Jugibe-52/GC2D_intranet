"""Second-order ABBA palindrome followed by one arithmetic mean."""
from __future__ import annotations

from dataclasses import dataclass
from methods.palindromic.configuration import StateExtension
from methods.palindromic.midpoint_base import MidpointPalindromicMethod
from methods.palindromic.abba2_implicit import ABBA2Implicit


@dataclass(slots=True)
class ABBA2Midpoint(MidpointPalindromicMethod):
	"""Second-order ABBA palindrome followed by one arithmetic mean."""

	state_extension: StateExtension = 'physical'
	composition_name = 'ABBA2'
	recipe = ABBA2Implicit.recipe


__all__ = ['ABBA2Midpoint']
