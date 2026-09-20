"""Generalized-energy convergence studies for GC experiments."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from diagnostics import GCGeneralizedEnergyObserver
from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential
from simulation import (
	BM4Implicit,
	InitialValueProblem,
	SimulationRequest,
	Solution,
	simulate,
)

from ._validation import (
	integer_ratio,
	nonnegative_finite,
	positive_finite,
	positive_integer,
	resolve_rho,
)


def _validated_steps(steps: tuple[float, ...]) -> tuple[float, ...]:
	"""Normalize a non-empty collection of distinct positive step sizes."""
	values = tuple(float(step) for step in steps)
	if not values or any(not np.isfinite(step) or step <= 0 for step in values):
		raise ValueError("`steps` must contain positive finite values.")
	if len(set(values)) != len(values):
		raise ValueError("`steps` must not contain duplicates.")
	return values


@dataclass(frozen=True, slots=True)
class GeneralizedEnergyConfig:
	"""Integration parameters for a generalized-energy convergence study."""

	steps: tuple[float, ...]
	t_span: tuple[float, float]
	output_sample_count: int
	rho: float | None = None
	coupling_frequency: float = 0.0
	newton_absolute_tolerance: float = 1e-13
	newton_relative_tolerance: float = 1e-12
	newton_max_iterations: int = 12
	newton_jacobian_relative_step: float = float(np.cbrt(np.finfo(float).eps))
	progress: bool = False

	def __post_init__(self) -> None:
		"""Validate integration parameters before an expensive comparison."""
		object.__setattr__(self, "steps", _validated_steps(tuple(self.steps)))
		try:
			start, stop = (float(value) for value in self.t_span)
		except (TypeError, ValueError) as exc:
			raise ValueError("`t_span` must contain two finite increasing times.") from exc
		if not np.isfinite(start) or not np.isfinite(stop) or start >= stop:
			raise ValueError("`t_span` must contain two finite increasing times.")
		object.__setattr__(self, "t_span", (start, stop))
		if (
			isinstance(self.output_sample_count, (bool, np.bool_))
			or not isinstance(self.output_sample_count, (int, np.integer))
			or self.output_sample_count < 2
		):
			raise ValueError("`output_sample_count` must be an integer of at least 2.")
		object.__setattr__(self, "output_sample_count", int(self.output_sample_count))
		save_interval = (stop - start) / (self.output_sample_count - 1)
		for step in self.steps:
			integer_ratio(
				save_interval,
				step,
				f"output interval / step for {step:g}",
			)
		if self.rho is not None:
			object.__setattr__(self, "rho", nonnegative_finite(self.rho, "rho"))
		frequency = float(self.coupling_frequency)
		if not np.isfinite(frequency) or frequency < 0:
			raise ValueError("`coupling_frequency` must be finite and non-negative.")
		object.__setattr__(self, "coupling_frequency", frequency)
		for name in (
			"newton_absolute_tolerance",
			"newton_relative_tolerance",
			"newton_jacobian_relative_step",
		):
			object.__setattr__(
				self,
				name,
				positive_finite(getattr(self, name), name),
			)
		object.__setattr__(
			self,
			"newton_max_iterations",
			positive_integer(self.newton_max_iterations, "newton_max_iterations"),
		)


@dataclass(frozen=True, slots=True)
class GeneralizedEnergySummary:
	"""Maximum generalized-energy error for one integration step."""

	step: float
	step_count: int
	absolute_error: float
	max_relative_error: float


@dataclass(frozen=True, slots=True)
class GeneralizedEnergyResult:
	"""Solutions and energy histories produced by a convergence comparison."""

	dynamics: GuidingCenterDynamics
	steps: tuple[float, ...]
	solutions: Mapping[float, Solution]
	generalized_energies: Mapping[float, np.ndarray]
	relative_errors: Mapping[float, np.ndarray]

	def summaries(self) -> tuple[GeneralizedEnergySummary, ...]:
		"""Return maximum absolute and relative errors in configured step order."""
		return tuple(
			GeneralizedEnergySummary(
				step=step,
				step_count=int(
					self.solutions[step].diagnostics["step_count"]
				),
				absolute_error=float(
					self.solutions[step].diagnostics["energy_error"]
				),
				max_relative_error=float(
					np.max(np.abs(self.relative_errors[step]))
				),
			)
			for step in self.steps
		)

	def print_summary(self) -> None:
		"""Print one compact convergence row per integration step."""
		print(
			f"{'step':>10} {'BM4 steps':>12} {'absolute error':>16} "
			f"{'max |relative error|':>20}"
		)
		for row in self.summaries():
			print(
				f"{row.step:10.4g} {row.step_count:12d} "
				f"{row.absolute_error:16.8e} {row.max_relative_error:20.8e}"
			)

	def plot(self) -> tuple[Figure, Axes]:
		"""Plot the relative generalized-energy error for every step size."""
		figure, axes = plt.subplots(figsize=(9, 5), constrained_layout=True)
		for step in self.steps:
			solution = self.solutions[step]
			error = self.relative_errors[step]
			max_error = np.max(np.abs(error))
			axes.plot(
				solution.t,
				error,
				label=(
					rf"$\Delta t={step:g}$"
					rf" — $\max|\varepsilon_K|={max_error:.3e}$"
				),
			)
		axes.axhline(0.0, color="0.5", linestyle="--", linewidth=1)
		axes.set(
			xlabel="$t$",
			ylabel=r"$\varepsilon_K(t)=(K(t)-K(0))/|K(0)|$",
			title=r"Generalized energy conservation $K=H+k$",
		)
		axes.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
		axes.grid(alpha=0.25)
		axes.legend()
		return figure, axes


def run_generalized_energy_comparison(
	potential: Potential,
	configuration: GCInitialConfiguration,
	*,
	config: GeneralizedEnergyConfig,
) -> GeneralizedEnergyResult:
	"""Integrate one centered GC state at several steps and compare ``H + k``."""
	if not isinstance(potential, Potential):
		raise TypeError("`potential` must be a Potential instance.")
	if not isinstance(configuration, GCInitialConfiguration):
		raise TypeError(
			"`configuration` must be a GCInitialConfiguration instance."
		)
	if not isinstance(config, GeneralizedEnergyConfig):
		raise TypeError("`config` must be a GeneralizedEnergyConfig instance.")
	initial_state = configuration.initial_state
	if initial_state is None or configuration.layout.particle_count(initial_state) != 1:
		raise ValueError(
			"The generalized-energy study requires exactly one initial GC state."
		)

	rho = resolve_rho(config.rho, configuration)
	dynamics = GuidingCenterDynamics(potential, rho=rho)
	problem = InitialValueProblem(
		dynamics,
		configuration,
	)
	solutions: dict[float, Solution] = {}
	energies: dict[float, np.ndarray] = {}
	relative_errors: dict[float, np.ndarray] = {}

	for step in config.steps:
		request = SimulationRequest.uniform(
			t_span=config.t_span,
			max_step=step,
			sample_count=config.output_sample_count,
		)
		energy_observer = GCGeneralizedEnergyObserver(
			dynamics,
			initial_time=config.t_span[0],
			initial_state=initial_state,
		)
		method = BM4Implicit(
			coupling_frequency=config.coupling_frequency,
			newton_absolute_tolerance=config.newton_absolute_tolerance,
			newton_relative_tolerance=config.newton_relative_tolerance,
			newton_max_iterations=config.newton_max_iterations,
			newton_jacobian_relative_step=config.newton_jacobian_relative_step,
			newton_jacobian_method="analytic",
			nonlinear_solver="newton",
			progress=config.progress,
			step_observer=energy_observer,
		)
		raw_solution = simulate(problem, method, request)
		records = energy_observer.records
		if len(records) != raw_solution.n_steps + 1:
			raise RuntimeError(
				"Energy records do not match the accepted BM4 step count."
			)
		output_interval = (config.t_span[1] - config.t_span[0]) / (
			config.output_sample_count - 1
		)
		record_stride = integer_ratio(
			output_interval,
			step,
			f"output interval / step for {step:g}",
		)
		selected_records = records[::record_stride]
		if len(selected_records) != raw_solution.t.size:
			raise RuntimeError(
				"Energy records do not align with the requested output grid."
			)
		record_times = np.asarray([record.time for record in selected_records])
		time_tolerance = float(
			64.0
			* np.finfo(float).eps
			* max(
				1.0,
				abs(config.t_span[0]),
				abs(config.t_span[1]),
			)
		)
		if not np.allclose(
			record_times,
			raw_solution.t,
			rtol=0.0,
			atol=time_tolerance,
		):
			raise RuntimeError(
				"Energy-record times do not match the requested output times."
			)
		hamiltonian = np.asarray(
			[record.hamiltonian for record in selected_records],
			dtype=float,
		)
		extended_momentum = np.asarray(
			[record.kappa for record in selected_records],
			dtype=float,
		)
		generalized_energy = np.asarray(
			[record.generalized_energy for record in selected_records],
			dtype=float,
		)
		generalized_energy_error = generalized_energy - generalized_energy[0]
		relative_error = np.asarray(
			[record.relative_error for record in selected_records],
			dtype=float,
		)
		solution = Solution(
			t=raw_solution.t,
			states=raw_solution.states,
			source=raw_solution.source,
			diagnostics={
				**dict(raw_solution.diagnostics),
				"track_energy": True,
				"hamiltonian": hamiltonian[np.newaxis, :],
				"extended_momentum": extended_momentum[np.newaxis, :],
				"extended_momentum_normalization": "kappa_equals_k_over_2",
				"generalized_energy": generalized_energy,
				"generalized_energy_error": generalized_energy_error,
				"energy_error": float(
					np.max(np.abs(generalized_energy_error))
				),
			},
		)

		solutions[step] = solution
		energies[step] = generalized_energy
		relative_errors[step] = relative_error

	return GeneralizedEnergyResult(
		dynamics=dynamics,
		steps=config.steps,
		solutions=MappingProxyType(solutions),
		generalized_energies=MappingProxyType(energies),
		relative_errors=MappingProxyType(relative_errors),
	)


__all__ = [
	"GeneralizedEnergyConfig",
	"GeneralizedEnergyResult",
	"GeneralizedEnergySummary",
	"run_generalized_energy_comparison",
]
