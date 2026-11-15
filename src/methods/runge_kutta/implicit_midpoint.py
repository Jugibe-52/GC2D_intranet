"""One-stage Gauss--Legendre coefficients for the coupled stage solver."""

from dataclasses import dataclass, field
from typing import ClassVar
from methods.runge_kutta.implicit import ImplicitRungeKutta
from methods.runge_kutta.tableau import ButcherTableau


@dataclass
class ImplicitMidpoint(ImplicitRungeKutta):
	"""Order-two symmetric, symplectic implicit midpoint recipe."""

	tableau: ButcherTableau = field(default=ButcherTableau(
		((.5,),), (1.,), (.5,)), init=False)
	designed_order: ClassVar[int] = 2


__all__ = ['ImplicitMidpoint']
