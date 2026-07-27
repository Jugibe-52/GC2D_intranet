"""Guiding-centre system."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from matplotlib.animation import FuncAnimation

from classes.dynamics import GuidingCenterDynamics
from classes.potential import Potential
from classes.simulation.formulations import GCExtendedFormulation
from classes.simulation.methods import BM4Composition
from classes.simulation.problem import InitialValueProblem
from classes.simulation.request import SimulationRequest
from classes.simulation.runner import SimulationRunner
from classes.trajectory import Area, TrajectoryGC

from .observation import StageObserver
from ._visualization import animate_gc_area
from .solution import Solution
from .system import System


class SystemGC(System):
	"""Guiding-centre dynamics over a gyroaveraged potential.

	The trajectory state is component-major ``[x, y]``. Its ``rho`` determines
	the fixed-radius gyroaverage stored as ``effective_potential``; all GC forces,
	energies, and visualizations use that effective field rather than the raw one.
	"""

	trajectory: TrajectoryGC
	dynamics: GuidingCenterDynamics

	def __init__(
		self,
		potential: Potential,
		trajectory: TrajectoryGC,
		*,
		coupling_frequency: float = np.pi / 8,
	) -> None:
		"""Construct a GC system with a configurable copy-coupling frequency.

		``coupling_frequency`` is the non-negative dimensionless angular frequency
		that binds the two extended-phase-space GC copies.  It is numerical, not
		a physical cyclotron frequency, and defaults to ``pi / 8``.
		"""
		if not isinstance(trajectory, TrajectoryGC):
			raise TypeError("SystemGC requires a TrajectoryGC instance.")
		coupling_frequency = float(coupling_frequency)
		if not np.isfinite(coupling_frequency) or coupling_frequency < 0:
			raise ValueError("`coupling_frequency` must be finite and non-negative.")
		super().__init__(potential, trajectory)
		self.dynamics = GuidingCenterDynamics(potential, rho=trajectory.rho)
		# Compatibility property: the formulation is now the real owner of this
		# numerical parameter and receives it when a simulation is assembled.
		self.coupling_frequency = coupling_frequency
		self.effective_potential = self.dynamics.effective_potential

	def vector_field(self, t: float, state: np.ndarray) -> np.ndarray:
		"""Evaluate the GC drift at all positions in a flat ``[x, y]`` state.

		``ex`` and ``ey`` have one value per represented particle, and the packed
		result preserves exactly the input state's component-major shape.
		"""
		return self.dynamics.vector_field(t, state)

	def hamiltonian(
		self,
		t: float | np.ndarray,
		state: np.ndarray,
	) -> np.ndarray:
		"""Evaluate the gyroaveraged Hamiltonian per particle and saved time.

		A scalar ``t`` normally accompanies a flat state; a time vector broadcasts
		against a solution whose trailing axis stores those same saved times.
		"""
		return self.dynamics.hamiltonian(t, state)

	def extended_momentum_derivative(
		self,
		t: float,
		state: np.ndarray,
	) -> np.ndarray:
		"""Evaluate the per-particle derivative of momentum conjugate to time."""
		return self.dynamics.extended_momentum_derivative(t, state)

	def animate_area(
		self,
		solution: Solution,
		*,
		frames: int | None = 120,
		interval: int = 50,
		cmap: str = "RdBu_r",
		repeat: bool = True,
		diagnostic_times: np.ndarray | None = None,
		relative_symplecticity_errors: np.ndarray | None = None,
		relative_copy_separations: np.ndarray | None = None,
		**pcolormesh_kwargs: Any,
	) -> FuncAnimation:
		"""Animate an Area over the effective field and relative diagnostics.

		The relative error is measured against the initial signed area and the
		requested frames are sampled uniformly from the saved solution. ``frames``
		limits displayed samples without reintegration, and ``interval`` is the
		delay between displayed frames in milliseconds. Supplying the three
		projected-diagnostic arrays adds synchronized panels for normalized
		symplecticity error and normalized internal-copy separation.
		"""
		if not isinstance(self.trajectory, Area):
			raise TypeError("`animate_area` requires an Area trajectory.")
		if not isinstance(solution, Solution):
			raise TypeError("`solution` must be a Solution instance.")
		return animate_gc_area(
			self.effective_potential,
			self.trajectory,
			(solution,),
			labels=("trajectory",),
			frames=frames,
			interval=interval,
			cmap=cmap,
			repeat=repeat,
			diagnostic_times=(diagnostic_times,),
			relative_symplecticity_errors=(relative_symplecticity_errors,),
			relative_copy_separations=(relative_copy_separations,),
			pcolormesh_kwargs=pcolormesh_kwargs,
		)

	def animate_area_comparison(
		self,
		solutions: Mapping[str, Solution],
		*,
		diagnostic_times: Mapping[str, np.ndarray] | None = None,
		relative_symplecticity_errors: Mapping[str, np.ndarray] | None = None,
		relative_copy_separations: Mapping[str, np.ndarray] | None = None,
		frames: int | None = 120,
		interval: int = 50,
		cmap: str = "RdBu_r",
		repeat: bool = True,
		**pcolormesh_kwargs: Any,
	) -> FuncAnimation:
		"""Compare Area solutions and projected diagnostics in one animation.

		Mapping keys are display labels and define plotting order. Every solution
		must share saved times. When diagnostics are requested, all three mappings
		must contain exactly the same keys as ``solutions``.
		"""
		if not isinstance(self.trajectory, Area):
			raise TypeError("`animate_area_comparison` requires an Area trajectory.")
		if not isinstance(solutions, Mapping) or len(solutions) < 2:
			raise ValueError("`solutions` must map at least two labels to solutions.")
		labels = tuple(solutions)
		if any(not isinstance(solution, Solution) for solution in solutions.values()):
			raise TypeError("Every compared value must be a Solution instance.")

		diagnostic_mappings = (
			diagnostic_times,
			relative_symplecticity_errors,
			relative_copy_separations,
		)
		if any(mapping is not None for mapping in diagnostic_mappings):
			if any(mapping is None for mapping in diagnostic_mappings):
				raise ValueError("All three diagnostic mappings must be provided.")
			for mapping in diagnostic_mappings:
				assert mapping is not None
				if set(mapping) != set(labels):
					raise ValueError(
						"Diagnostic mappings must have the same keys as `solutions`."
					)

		def ordered(
			mapping: Mapping[str, np.ndarray] | None,
		) -> tuple[np.ndarray | None, ...]:
			"""Align an optional diagnostic mapping with solution insertion order."""
			if mapping is None:
				return tuple(None for _label in labels)
			return tuple(mapping[label] for label in labels)

		return animate_gc_area(
			self.effective_potential,
			self.trajectory,
			tuple(solutions.values()),
			labels=labels,
			frames=frames,
			interval=interval,
			cmap=cmap,
			repeat=repeat,
			diagnostic_times=ordered(diagnostic_times),
			relative_symplecticity_errors=ordered(
				relative_symplecticity_errors
			),
			relative_copy_separations=ordered(relative_copy_separations),
			pcolormesh_kwargs=pcolormesh_kwargs,
		)

	def _integrate(
		self,
		state: np.ndarray,
		*,
		step: float,
		t_span: tuple[float, float],
		n_output_samples: int,
		check_energy: bool,
		progress: bool,
		stage_observer: StageObserver | None,
	) -> Solution:
		"""Assemble the legacy BM4 façade from independent architecture objects."""
		stored_state = self.trajectory.state
		if stored_state is None or not np.array_equal(stored_state, state):
			raise ValueError("The supplied state must match the stored initial state.")
		problem = InitialValueProblem(self.dynamics, self.trajectory)
		request = SimulationRequest.uniform(
			t_span=t_span,
			max_step=step,
			sample_count=n_output_samples,
		)
		method = BM4Composition(
			GCExtendedFormulation(self.coupling_frequency),
			track_energy=check_energy,
			progress=progress,
			stage_observer=stage_observer,
		)
		return SimulationRunner().simulate(problem, method, request)


__all__ = ["SystemGC"]
