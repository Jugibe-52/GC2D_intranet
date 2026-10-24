"""Public simulation facade over problems, methods, integration and results."""

from formulations.state import SingleFormulation, DoubleFormulation
from contracts.dynamics.coordinates import StateLayout
from dynamics.coordinates import PhysicalCoordinates
from methods.extended.maps import GCDoubledMaps, HarmonicCoupling, gc_coupling_matrix
from methods.splitting import FCSplitMaps
from methods import (
	PalindromicMethod,
	PalindromicImplicit,
	PalindromicMidpoint,
	Composition,
	Operation,
	A,
	B,
	T,
	C,
	DOP853,
	Radau,
	ABBA2Midpoint,
	ABBA2Implicit,
	ABBA4Implicit,
	ABBA6Implicit,
	ABBA4_PROJECTION_PLACEMENTS,
	BM4Implicit,
	BM4Midpoint,
	ExplicitEuler,
	GAUSS_JACOBIAN_METHODS,
	GaussJacobianMethod,
	GaussLegendre4,
	HBVM42,
	HBVMJacobianMethod,
	ABBA_PROJECTION_FORMULATIONS,
	ABBA_STATE_EXTENSIONS,
	NONLINEAR_SOLVERS,
	NonlinearSolver,
	ProjectionPlacement,
	ProjectionFormulation,
	StateExtension,
	RK4,
	SDIRK4_TABLEAU_A,
	SDIRK4_TABLEAU_B,
	SDIRK4_TABLEAU_C,
	SDIRK_JACOBIAN_METHODS,
	SDIRK4,
	SDIRKJacobianMethod,
)
from contracts.methods.base import NumericalMethod
from contracts.methods.adaptive.scipy import AdaptiveIntegrationStep, AdaptiveStepObserver
from contracts.methods.extended.abba import (
	ABBA4ImplicitIntegrationStep,
	ABBA6ImplicitIntegrationStep,
	ABBAImplicitCompositionIntegrationStep,
	ABBA2ImplicitIntegrationStep,
	UnprojectedABBAIntegrationStep,
)
from contracts.methods.classical.gauss_legendre import GaussLegendre4IntegrationStep
from contracts.methods.extended.bm4 import ImplicitBM4IntegrationStep
from contracts.methods.observation import (
	ImplicitIntegrationStep,
	IntegrationStage,
	IntegrationStep,
	StageObserver,
	StepObserver,
)
from problem import InitialValueProblem
from request import (
	DEFAULT_INTEGRATION_STEP,
	DEFAULT_INTEGRATION_STEPS_PER_CYCLE,
	DEFAULT_SAVED_STEPS_PER_CYCLE,
	DEFAULT_SAVE_INTERVAL,
	NORMALIZED_CYCLE_DURATION,
	SimulationRequest,
)
from .runner import (
	simulate,
)
from states.history import States
from states.initial import InitialConditions
from solvers.nonlinear import NewtonSolver, BroydenSolver
from methods.projection import ReducedProjection, SimultaneousProjection

__all__ = [
	"PhysicalCoordinates", "GCDoubledMaps", "FCSplitMaps", "HarmonicCoupling",
	'PalindromicMethod', 'PalindromicImplicit', 'PalindromicMidpoint', 'Composition', 'Operation', 'A', 'B', 'T', 'C',
	"InitialConditions", "NewtonSolver", "BroydenSolver", "ReducedProjection", "SimultaneousProjection",
	"SingleFormulation",
	"DoubleFormulation",
	"DOP853",
	"Radau",
	"AdaptiveIntegrationStep",
	"AdaptiveStepObserver",
	"ABBA2Midpoint",
	"ABBA2Implicit",
	"ABBA4Implicit",
	"ABBA4ImplicitIntegrationStep",
	"ABBA6Implicit",
	"BM4Implicit",
	"BM4Midpoint",
	"ExplicitEuler",
	"GAUSS_JACOBIAN_METHODS",
	"GaussJacobianMethod",
	"GaussLegendre4",
	"HBVM42",
	"HBVMJacobianMethod",
	"DEFAULT_INTEGRATION_STEP",
	"DEFAULT_INTEGRATION_STEPS_PER_CYCLE",
	"DEFAULT_SAVED_STEPS_PER_CYCLE",
	"DEFAULT_SAVE_INTERVAL",
	"ABBA4_PROJECTION_PLACEMENTS",
	"ABBA_PROJECTION_FORMULATIONS",
	"ABBA_STATE_EXTENSIONS",
	"gc_coupling_matrix",
	"GaussLegendre4IntegrationStep",
	"InitialValueProblem",
	"ABBA6ImplicitIntegrationStep",
	"ABBAImplicitCompositionIntegrationStep",
	"ABBA2ImplicitIntegrationStep",
	"ImplicitBM4IntegrationStep",
	"ImplicitIntegrationStep",
	"IntegrationStage",
	"IntegrationStep",
	"NONLINEAR_SOLVERS",
	"NORMALIZED_CYCLE_DURATION",
	"NonlinearSolver",
	"NumericalMethod",
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
	"SimulationRequest",
	"States",
	"StageObserver",
	"StepObserver",
	"StateLayout",
	"UnprojectedABBAIntegrationStep",
	"simulate",
]
