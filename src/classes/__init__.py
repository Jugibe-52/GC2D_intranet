"""Public physical, numerical, simulation, and compatibility APIs."""

from .dynamics import (
	CyclotronSplitSystem,
	DynamicalSystem,
	ExtendedHamiltonianSystem,
	FullCyclotronDynamics,
	GuidingCenterDynamics,
	HamiltonianSystem,
)
from .potential import Potential
from .simulation import (
	BM4Composition,
	DirectAdjointFormulation,
	FCSplitFormulation,
	GCExtendedFormulation,
	InitialConfiguration,
	InitialValueProblem,
	IntegrationStage,
	NumericalMethod,
	RK4,
	SimulationRequest,
	SimulationRunner,
	Solution,
	StageObserver,
	simulate,
)
from .system import SystemFC, SystemGC
from .trajectory import (
	Area,
	FCInitialConfiguration,
	GCInitialConfiguration,
	TrajectoryFC,
	TrajectoryGC,
)

__all__ = [
	"Area",
	"BM4Composition",
	"CyclotronSplitSystem",
	"DirectAdjointFormulation",
	"DynamicalSystem",
	"ExtendedHamiltonianSystem",
	"FCInitialConfiguration",
	"FCSplitFormulation",
	"FullCyclotronDynamics",
	"GCInitialConfiguration",
	"GCExtendedFormulation",
	"GuidingCenterDynamics",
	"HamiltonianSystem",
	"InitialConfiguration",
	"InitialValueProblem",
	"IntegrationStage",
	"NumericalMethod",
	"Potential",
	"RK4",
	"SimulationRequest",
	"SimulationRunner",
	"Solution",
	"StageObserver",
	"SystemFC",
	"SystemGC",
	"TrajectoryFC",
	"TrajectoryGC",
	"simulate",
]
