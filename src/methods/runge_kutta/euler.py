"""ExplicitEuler coefficients for the shared explicit Runge--Kutta algorithm."""

from dataclasses import dataclass, field
from typing import ClassVar
from methods.runge_kutta.explicit import ExplicitRungeKutta
from methods.runge_kutta.tableau import ButcherTableau


@dataclass
class ExplicitEuler(ExplicitRungeKutta):
	"""Fixed ExplicitEuler recipe; stages and quadrature are inherited."""

	tableau: ButcherTableau = field(default=ButcherTableau(
		((0.,),), (1.,), (0.,)), init=False)
	designed_order: ClassVar[int] = 1


__all__ = ['ExplicitEuler']
