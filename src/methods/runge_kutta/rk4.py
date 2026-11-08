"""RK4 coefficients for the shared explicit Runge--Kutta algorithm."""

from dataclasses import dataclass, field
from typing import ClassVar
from methods.runge_kutta.explicit import ExplicitRungeKutta
from methods.runge_kutta.tableau import ButcherTableau


@dataclass
class RK4(ExplicitRungeKutta):
	"""Fixed RK4 recipe; stages and quadrature are inherited."""

	tableau: ButcherTableau = field(default=ButcherTableau(
		((0., 0., 0., 0.), (.5, 0., 0., 0.), (0., .5, 0., 0.), (0., 0., 1., 0.)), (1/6, 1/3, 1/3, 1/6), (0., .5, .5, 1.)), init=False)
	designed_order: ClassVar[int] = 4


__all__ = ['RK4']
