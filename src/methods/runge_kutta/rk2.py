"""Explicit midpoint coefficients for the shared Runge--Kutta algorithm."""

from dataclasses import dataclass, field
from typing import ClassVar
from methods.runge_kutta.explicit import ExplicitRungeKutta
from methods.runge_kutta.tableau import ButcherTableau


@dataclass
class RK2(ExplicitRungeKutta):
	"""Order-two explicit midpoint with two field evaluations per step."""

	tableau: ButcherTableau = field(default=ButcherTableau(
		((0., 0.), (.5, 0.)), (0., 1.), (0., .5)), init=False)
	designed_order: ClassVar[int] = 2


__all__ = ['RK2']
