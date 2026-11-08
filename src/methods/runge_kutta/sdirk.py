"""Skvortsov S54b coefficients for sequential diagonal Runge--Kutta stages."""

from dataclasses import dataclass, field
from typing import ClassVar
import numpy as np
from contracts.states.history import DiagnosticValue
from methods.runge_kutta.diagonally_implicit import DiagonallyImplicitRungeKutta
from methods.runge_kutta.tableau import ButcherTableau
from solvers.derivatives import JacobianMethod as SDIRKJacobianMethod

SDIRK_JACOBIAN_METHODS: tuple[SDIRKJacobianMethod, ...] = ('auto', 'analytic', 'finite_difference')

# S54b has a common diagonal 1/4 and its final row equals the output weights.
_A: 'np.ndarray' = np.asarray(
	(
		(1.0 / 4.0, 0.0, 0.0, 0.0, 0.0),
		(-1.0 / 4.0, 1.0 / 4.0, 0.0, 0.0, 0.0),
		(1.0 / 8.0, 1.0 / 8.0, 1.0 / 4.0, 0.0, 0.0),
		(-3.0 / 2.0, 3.0 / 4.0, 3.0 / 2.0, 1.0 / 4.0, 0.0),
		(0.0, 1.0 / 6.0, 2.0 / 3.0, -1.0 / 12.0, 1.0 / 4.0),
	),
	dtype=float,
)
_B: 'np.ndarray' = np.asarray(
	(0.0, 1.0 / 6.0, 2.0 / 3.0, -1.0 / 12.0, 1.0 / 4.0),
	dtype=float,
)
_C: 'np.ndarray' = np.sum(_A, axis=1)

_TABLEAU: ButcherTableau = ButcherTableau(_A, _B, _C)
SDIRK4_TABLEAU_A: np.ndarray = _TABLEAU.A
SDIRK4_TABLEAU_B: np.ndarray = _TABLEAU.B
SDIRK4_TABLEAU_C: np.ndarray = _TABLEAU.C


@dataclass
class SDIRK4(DiagonallyImplicitRungeKutta):
	"""Five-stage, fourth-order S54b recipe with inherited diagonal solves."""

	tableau: ButcherTableau = field(default=_TABLEAU, init=False)
	designed_order: ClassVar[int] = 4
	tableau_name: ClassVar[str] = 'Skvortsov S54b'
	tableau_metadata: ClassVar[tuple[tuple[str, DiagnosticValue], ...]] = (
		('diagonal_coefficient', .25), ('stiffly_accurate', True),
		('symmetric', False), ('symplectic', False))


__all__ = ['SDIRK4_TABLEAU_A', 'SDIRK4_TABLEAU_B', 'SDIRK4_TABLEAU_C',
	'SDIRK_JACOBIAN_METHODS', 'SDIRK4', 'SDIRKJacobianMethod']
