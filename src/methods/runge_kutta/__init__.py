"""Runge--Kutta methods defined by immutable Butcher tableaux."""

from methods.runge_kutta.euler import ExplicitEuler
from methods.runge_kutta.gauss_legendre import (
	GAUSS_JACOBIAN_METHODS,
	GaussJacobianMethod,
	GaussLegendre4,
)
from methods.runge_kutta.rk4 import RK4
from methods.runge_kutta.sdirk import (
	SDIRK4_TABLEAU_A,
	SDIRK4_TABLEAU_B,
	SDIRK4_TABLEAU_C,
	SDIRK_JACOBIAN_METHODS,
	SDIRK4,
	SDIRKJacobianMethod,
)

from methods.runge_kutta.tableau import ButcherTableau
from methods.runge_kutta.base import RungeKutta
from methods.runge_kutta.explicit import ExplicitRungeKutta
from methods.runge_kutta.diagonally_implicit import DiagonallyImplicitRungeKutta
from methods.runge_kutta.implicit import ImplicitRungeKutta

__all__ = [
	"ButcherTableau", "RungeKutta", "ExplicitRungeKutta",
	"DiagonallyImplicitRungeKutta", "ImplicitRungeKutta",
	"ExplicitEuler",
	"GAUSS_JACOBIAN_METHODS",
	"GaussJacobianMethod",
	"GaussLegendre4",
	"RK4",
	"SDIRK4_TABLEAU_A",
	"SDIRK4_TABLEAU_B",
	"SDIRK4_TABLEAU_C",
	"SDIRK_JACOBIAN_METHODS",
	"SDIRK4",
	"SDIRKJacobianMethod",
]
