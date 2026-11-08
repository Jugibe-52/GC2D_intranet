"""Experimental recipe with one exterior spatial Hairer projection."""
from __future__ import annotations

from dataclasses import dataclass, field
from methods.palindromic.maps import HarmonicCoupling
from methods.projection import ProjectionChoice
from methods.palindromic.core.composition import Composition
from solvers.nonlinear import AlgebraicSolver, NewtonSolver
from methods.palindromic.projected_base import ProjectedPalindromicMethod


@dataclass(slots=True)
class PalindromicImplicit(ProjectedPalindromicMethod):
	"""Experimental recipe with one exterior spatial Hairer projection."""

	recipe: Composition
	composition_name: str = field(default='custom', kw_only=True)
	projection_formulation: ProjectionChoice = 'reduced_multiplier'
	solver: AlgebraicSolver = field(default_factory=lambda: NewtonSolver(jacobian_method='analytic'), kw_only=True)
	coupling: HarmonicCoupling | None = field(default=None, kw_only=True)

	def _configured_coupling(self) -> HarmonicCoupling | None:
		return self.coupling


__all__ = ['PalindromicImplicit']
