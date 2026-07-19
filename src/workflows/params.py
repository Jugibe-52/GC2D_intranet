"""Parameter normalization and construction of composed Fourier systems."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import cast

import numpy as np

from classes import FourierPotential, System, create_system, create_trajectory
from config_logging import simulation_label
from contracts import FourierParams, ParameterMap, TrajectoryKind

from .trajectory_initialization import initialize_trajectory

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WorkflowOptions:
	"""Execution and presentation options kept outside the physical entities."""

	parameters: FourierParams
	workflow_method: str
	periods: int
	time_step: float
	solver_method: str
	check_energy: bool
	plot_results: bool
	save_plot: bool
	save_data: bool
	modulo: bool
	show_grid: bool
	darkmode: bool
	extension: str
	dpi: int
	output_dir: str | Path
	output_name: str


def _int_param(value: object, name: str) -> int:
	if isinstance(value, (int, np.integer)):
		return int(value)
	if isinstance(value, (float, np.floating, str)):
		return int(value)
	raise TypeError(f"{name} must be numeric, got {type(value).__name__}.")


def _float_param(value: object, name: str) -> float:
	if isinstance(value, (int, float, np.integer, np.floating, str)):
		return float(value)
	raise TypeError(f"{name} must be numeric, got {type(value).__name__}.")


def _normalize_trajectory_kind(value: object, method: str) -> TrajectoryKind:
	raw_kind = value if value is not None else method.rsplit("_", 1)[-1]
	kind = "fc" if raw_kind == "fo" else raw_kind
	if kind not in {"gc", "fc"}:
		raise ValueError(f"Cannot infer trajectory type from Method={method!r}.")
	return cast(TrajectoryKind, kind)


def to_symp_params(raw_params: Mapping[str, object]) -> FourierParams:
	"""Validate and normalize an untyped Fourier simulation mapping."""
	params: ParameterMap = dict(raw_params)
	workflow_method = params.get("Method", "poincare_gc")
	if not isinstance(workflow_method, str):
		raise TypeError(
			f"Method must be a string, got {type(workflow_method).__name__}."
		)
	trajectory_kind = _normalize_trajectory_kind(
		params.get("traj_type"),
		workflow_method,
	)
	params["traj_type"] = trajectory_kind
	params.setdefault("eta", 0)
	if trajectory_kind == "fc" and _float_param(params["eta"], "eta") == 0:
		raise ValueError("Full-cyclotron integrations require a non-zero eta.")
	params.setdefault("init", "fixed")
	params.setdefault("TimeStep", 0.1 if trajectory_kind == "gc" else 0.005)
	params.setdefault("ode_solver", "BM4")
	params.setdefault("CheckEnergy", True)
	required = ("M", "A", "rho", "eta", "Ntraj", "Tf")
	missing = [key for key in required if key not in params]
	if missing:
		raise ValueError(f"Missing required Fourier parameters: {', '.join(missing)}.")
	params["M"] = _int_param(params["M"], "M")
	params["A"] = _float_param(params["A"], "A")
	params["rho"] = _float_param(params["rho"], "rho")
	params["eta"] = _float_param(params["eta"], "eta")
	params["Ntraj"] = _int_param(params["Ntraj"], "Ntraj")
	params["Tf"] = _int_param(params["Tf"], "Tf")
	params["TimeStep"] = _float_param(params["TimeStep"], "TimeStep")
	params["ode_solver"] = str(params["ode_solver"])
	params["CheckEnergy"] = bool(params["CheckEnergy"])
	initialization = params["init"]
	if initialization not in {"random", "fixed", "selected"}:
		raise ValueError(f"Invalid initial-condition type: {initialization!r}.")
	params["init"] = initialization
	logger.debug(
		"Normalized parameters: %s eta=%s",
		simulation_label(params),
		params.get("eta"),
	)
	return cast(FourierParams, params)


def make_params(
	base: Mapping[str, object],
	**overrides: object,
) -> FourierParams:
	"""Apply notebook overrides and return a normalized Fourier case."""
	params: ParameterMap = dict(base)
	params.update(overrides)
	logger.debug("Preparing notebook parameters with overrides=%s", sorted(overrides))
	if params.get("init") == "selected":
		n_trajectories = _int_param(params.get("Ntraj", 0), "Ntraj")
		if "x0" not in overrides and "x0" in params:
			params["x0"] = np.asarray(params["x0"])[:n_trajectories]
		if "y0" not in overrides and "y0" in params:
			params["y0"] = np.asarray(params["y0"])[:n_trajectories]
	normalized = to_symp_params(params)
	logger.info(
		"Prepared notebook parameters: %s init=%s",
		simulation_label(normalized),
		normalized.get("init"),
	)
	return normalized


def _workflow_options(params: FourierParams) -> WorkflowOptions:
	return WorkflowOptions(
		parameters=cast(FourierParams, dict(params)),
		workflow_method=str(params.get("Method", f"poincare_{params['traj_type']}")),
		periods=int(params["Tf"]),
		time_step=float(params["TimeStep"]),
		solver_method=str(params["ode_solver"]),
		check_energy=bool(params["CheckEnergy"]),
		plot_results=bool(params.get("PlotResults", False)),
		save_plot=bool(params.get("SavePlot", False)),
		save_data=bool(params.get("SaveData", False)),
		modulo=bool(params.get("modulo", False)),
		show_grid=bool(params.get("grid", False)),
		darkmode=bool(params.get("darkmode", False)),
		extension=str(params.get("extension", ".png")),
		dpi=int(params.get("dpi", 200)),
		output_dir=params.get("output_dir", "."),
		output_name=str(params.get("output_name", "notebook")),
	)


def make_system(params: Mapping[str, object]) -> System:
	"""Build FourierPotential and Trajectory independently, then compose them."""
	normalized = to_symp_params(params)
	logger.info("Building system: %s", simulation_label(normalized))
	seed = _int_param(normalized.get("seed", 27), "seed")
	potential = FourierPotential(
		amplitude=normalized["A"],
		modes=normalized["M"],
		seed=seed,
	)
	trajectory = create_trajectory(
		normalized["traj_type"],
		rho=normalized["rho"],
		eta=normalized["eta"],
		n_trajectories=normalized["Ntraj"],
		initialization=normalized["init"],
		x0=normalized.get("x0"),
		y0=normalized.get("y0"),
		seed=seed,
	)
	initialize_trajectory(trajectory, potential.grid)
	system = create_system(potential, trajectory)
	options = _workflow_options(normalized)
	system.options = options
	# SimulationResult reads these presentation flags without coupling the
	# domain classes to workflow configuration.
	system.modulo = options.modulo
	system.show_grid = options.show_grid
	return system


def ensure_system(case: System | Mapping[str, object]) -> System:
	"""Return an existing System or build one from a Fourier case mapping."""
	if isinstance(case, System):
		return case
	return make_system(case)


def get_workflow_options(system: System) -> WorkflowOptions:
	"""Return the options attached by make_system with a useful failure mode."""
	options = getattr(system, "options", None)
	if not isinstance(options, WorkflowOptions):
		raise ValueError(
			"This workflow requires a System built by workflows.params.make_system."
		)
	return options


__all__ = [
	"WorkflowOptions",
	"ensure_system",
	"get_workflow_options",
	"make_params",
	"make_system",
	"to_symp_params",
]
