"""Reusable simulation workflows for GC2D notebooks and scripts."""

__all__ = [
	"extract_potential",
	"fft_phi_grid",
	"integrate_case",
	"make_params",
	"make_system",
	"mock_potential",
	"plot_fft_phi",
	"plot_poincare",
	"plot_potential",
	"plot_sol",
	"plot_symplectic_poincare",
	"run_case",
	"run_method",
	"save_data",
	"to_symp_params",
]


def __getattr__(name: str):
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
	if name == "integrate_case":
		from .integration import integrate_case

		return integrate_case
	if name in {"make_params", "make_system", "to_symp_params"}:
		from .params import make_params, make_system, to_symp_params

		return {"make_params": make_params, "make_system": make_system, "to_symp_params": to_symp_params}[name]
	if name == "run_case":
		from .cases import run_case

		return run_case
	if name == "run_method":
		from .symplectic_legacy import run_method

		return run_method
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
