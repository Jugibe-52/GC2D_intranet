"""Interoperable numerical methods."""

from solvers.nonlinear import NONLINEAR_SOLVERS, NonlinearSolver
from methods.palindromic import (
	PalindromicMethod, PalindromicImplicit, PalindromicMidpoint, Composition, Operation, A, B, T, C,
	BM4Implicit, BM4Midpoint, ABBA2Midpoint, ABBA2Implicit, ABBA4Implicit,
	ABBA4_PROJECTION_PLACEMENTS, ABBA_PROJECTION_FORMULATIONS, ABBA_STATE_EXTENSIONS,
	ProjectionPlacement, ProjectionFormulation, StateExtension,
)
from methods.base import FixedStepMethod, IntegrationMethod
from contracts.methods.base import NumericalMethod
from methods.runge_kutta import (
	ExplicitEuler,
	GAUSS_JACOBIAN_METHODS,
	GaussJacobianMethod,
	GaussLegendre4,
	RK4,
	SDIRK4_TABLEAU_A,
	SDIRK4_TABLEAU_B,
	SDIRK4_TABLEAU_C,
	SDIRK_JACOBIAN_METHODS,
	SDIRK4,
	SDIRKJacobianMethod,
)
from methods.hbvm import HBVM42, HBVMJacobianMethod

from methods.runge_kutta.tableau import ButcherTableau
from methods.runge_kutta.base import RungeKutta
from methods.runge_kutta.explicit import ExplicitRungeKutta
from methods.runge_kutta.diagonally_implicit import DiagonallyImplicitRungeKutta
from methods.runge_kutta.implicit import ImplicitRungeKutta

__all__ = [
	"ButcherTableau", "RungeKutta", "ExplicitRungeKutta",
	"DiagonallyImplicitRungeKutta", "ImplicitRungeKutta",
	'PalindromicMethod', 'PalindromicImplicit', 'PalindromicMidpoint', 'Composition', 'Operation', 'A', 'B', 'T', 'C',
	"ABBA2Midpoint",
	"ABBA2Implicit",
	"ABBA4Implicit",
	"BM4Implicit",
	"BM4Midpoint",
	"ExplicitEuler",
	"GAUSS_JACOBIAN_METHODS",
	"GaussJacobianMethod",
	"GaussLegendre4",
	"HBVM42",
	"HBVMJacobianMethod",
	"ABBA4_PROJECTION_PLACEMENTS",
	"ABBA_PROJECTION_FORMULATIONS",
	"ABBA_STATE_EXTENSIONS",
	"NONLINEAR_SOLVERS",
	"NonlinearSolver",
	"NumericalMethod",
	"IntegrationMethod",
	"FixedStepMethod",
	"ProjectionPlacement",
	"ProjectionFormulation",
	"StateExtension",
	"RK4",
	"SDIRK4_TABLEAU_A",
	"SDIRK4_TABLEAU_B",
	"SDIRK4_TABLEAU_C",
	"SDIRK_JACOBIAN_METHODS",
	"SDIRK4",
	"SDIRKJacobianMethod",
]
