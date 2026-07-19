"""Reusable simulation workflows for trajectory notebooks and scripts."""

from typing import Any

__all__ = [
	"extract_potential",
	"fft_phi_grid",
	"integrate_simulation",
	"initialize_guiding_center_square",
	"initialize_trajectory",
	"make_initial_conditions",
	"make_params",
	"make_system",
	"mock_potential",
	"plot_fft_phi",
	"plot_poincare",
	"plot_potential",
	"plot_sol",
	"plot_symplectic_poincare",
	"run_workflow",
	"run_method",
	"save_data",
	"to_symp_params",
]


def __getattr__(name: str) -> Any:
	if name in {"extract_potential", "mock_potential"}:
		from .potentials import extract_potential, mock_potential

		return {"extract_potential": extract_potential, "mock_potential": mock_potential}[name]
	if name in {"fft_phi_grid", "plot_fft_phi", "plot_poincare", "plot_potential", "plot_sol", "plot_symplectic_poincare"}:
		from .plotting import fft_phi_grid, plot_fft_phi, plot_poincare, plot_potential, plot_sol, plot_symplectic_poincare

		return {
			"fft_phi_grid": fft_phi_grid,
			"plot_fft_phi": plot_fft_phi,
			"plot_poincare": plot_poincare,
			"plot_potential": plot_potential,
			"plot_sol": plot_sol,
			"plot_symplectic_poincare": plot_symplectic_poincare,
		}[name]
	if name == "save_data":
		from .export import save_data

		return save_data
	if name == "integrate_simulation":
		from .integration import integrate_simulation

		return integrate_simulation
	if name == "make_initial_conditions":
		from .initial_conditions import make_initial_conditions

		return make_initial_conditions
	if name in {"initialize_guiding_center_square", "initialize_trajectory"}:
		from .trajectory_initialization import initialize_guiding_center_square, initialize_trajectory

		return {
			"initialize_guiding_center_square": initialize_guiding_center_square,
			"initialize_trajectory": initialize_trajectory,
		}[name]
	if name in {"make_params", "make_system", "to_symp_params"}:
		from .params import make_params, make_system, to_symp_params

		return {"make_params": make_params, "make_system": make_system, "to_symp_params": to_symp_params}[name]
	if name == "run_workflow":
		from .workflow import run_workflow

		return run_workflow
	if name == "run_method":
		from .symplectic_legacy import run_method

		return run_method
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
