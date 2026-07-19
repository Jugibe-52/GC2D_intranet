"""Composed systems, numerical solvers and simulation results."""

from .fc import SystemFC
from .fc_solver import solve_symplectic
from .gc import SystemGC
from .gc_solver import solve_extended
from .research import SystemResearch
from .result import SimulationResult
from .solution import Solution
from ._solver_common import step_count as _step_count
from .system import System, create_system

__all__ = [
	"SimulationResult",
	"Solution",
	"System",
	"SystemFC",
	"SystemGC",
	"SystemResearch",
	"_step_count",
	"create_system",
	"solve_extended",
	"solve_symplectic",
]
