"""Interoperable numerical methods."""

from solvers.nonlinear import NONLINEAR_SOLVERS, NonlinearSolver
from methods.adaptive import DOP853, Radau
from methods.extended import (
	PalindromicMethod, PalindromicImplicit, PalindromicMidpoint, Composition, Operation, A, B, T, C,
	BM4Implicit,
	BM4Midpoint,
	ABBA2Midpoint,
	ABBA2Implicit,
	ABBA4Implicit,
	ABBA6Implicit,
	ABBA4_PROJECTION_PLACEMENTS,
	ABBA_PROJECTION_FORMULATIONS,
	ABBA_STATE_EXTENSIONS,
	ProjectionPlacement,
	ProjectionFormulation,
	StateExtension,
)
from methods.base import IntegrationMethod
from contracts.methods.base import NumericalMethod
from methods.classical import (
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

__all__ = [
	'PalindromicMethod', 'PalindromicImplicit', 'PalindromicMidpoint', 'Composition', 'Operation', 'A', 'B', 'T', 'C',
	"DOP853",
	"Radau",
	"ABBA2Midpoint",
	"ABBA2Implicit",
	"ABBA4Implicit",
	"ABBA6Implicit",
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
