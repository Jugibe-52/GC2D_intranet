"""Physical and spatially duplicated states, with optional passive energy tracking.

For one planar particle the four configurations have dimensions 2, 4, 4 and 6.
Energy tracking appends time and normalized momentum blocks of N entries each.
Neither auxiliary coordinate belongs to a spatial projection or a physical solve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from dynamics import HamiltonianSystem
from .._result import DiagnosticValue
from ..problem import InitialValueProblem


@dataclass(frozen=True)
class PhysicalFormulation:
    """Component-major physical state, optionally followed by ``(t, kappa)``.

    Every coordinate block has one entry per particle, including ``t`` and
    ``kappa``. All stored times are prescribed by the integration grid.
    ``kappa`` satisfies
    ``d kappa / dt = -partial_t H`` along the physical trajectory. It has the
    same energy normalization as the physical Hamiltonian, including for FC.
    """

    problem: InitialValueProblem
    initial_time: float
    track_energy: bool = False
    physical_size: int = field(init=False)
    particle_count: int = field(init=False)
    copies: ClassVar[int] = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "physical_size", self.problem.initial_state.size)
        object.__setattr__(self, "particle_count", self.problem.particle_count)
        if self.track_energy and not isinstance(self.problem.dynamics, HamiltonianSystem):
            raise TypeError("Energy tracking requires HamiltonianSystem.")

    @property
    def spatial_size(self) -> int:
        return self.copies * self.physical_size

    @property
    def dimension(self) -> int:
        return self.spatial_size + (2 * self.particle_count if self.track_energy else 0)

    @property
    def initial_state(self) -> np.ndarray:
        return self.pack(self.problem.initial_state, self.initial_time,
                         np.zeros(self.particle_count) if self.track_energy else None)

    def pack(self, physical: np.ndarray, time: float | np.ndarray,
             momentum: np.ndarray | None = None) -> np.ndarray:
        """Embed an accepted state on the spatial diagonal and append auxiliaries.

        Physical arrays have shape ``(physical_size, *sample_axes)``. The time
        and momentum blocks both have shape ``(particle_count, *sample_axes)``.
        Projected methods retain identical accepted spatial copies;
        off-diagonal states exist only inside maps.
        """
        value = np.asarray(physical)
        parts = [value] * self.copies
        if self.track_energy:
            if momentum is None or momentum.shape != (self.particle_count, *value.shape[1:]):
                raise ValueError("Energy momentum must have one component per particle.")
            parts += [np.broadcast_to(time, momentum.shape), momentum]
        return np.concatenate(parts)

    def components(self, state: np.ndarray) -> np.ndarray:
        """View packed states as ``(coordinates, particles, *sample_axes)``.

        With tracking, the last two coordinate rows are always time and
        momentum. This convention also applies to full-cyclotron states.
        """
        value = np.asarray(state)
        coordinates = self.dimension // self.particle_count
        return value.reshape(coordinates, self.particle_count, *value.shape[1:])

    def physical(self, state: np.ndarray) -> np.ndarray:
        """Read the physical representative of an accepted diagonal state."""
        return np.asarray(state[:self.physical_size])

    def time(self, state: np.ndarray) -> np.ndarray | None:
        return self.components(state)[-2] if self.track_energy else None

    def momentum(self, state: np.ndarray) -> np.ndarray | None:
        return self.components(state)[-1] if self.track_energy else None

    def momentum_rate(self, time: float, physical: np.ndarray) -> np.ndarray:
        """Evaluate the passive energy-balance derivative in physical units."""
        dynamics = self.problem.dynamics
        assert isinstance(dynamics, HamiltonianSystem)
        rate = np.asarray(dynamics.extended_momentum_derivative(time, physical), dtype=float)
        if rate.shape != (self.particle_count,) or not np.all(np.isfinite(rate)):
            raise ValueError("The momentum derivative must be finite with one value per particle.")
        return rate

    def finish(self, before: np.ndarray, physical: np.ndarray, end_time: float,
               increment: np.ndarray | None = None) -> np.ndarray:
        """Accept physical coordinates and update only the passive momentum."""
        momentum = self.momentum(before)
        if momentum is not None:
            if increment is None:
                raise ValueError("An energy-tracked step must supply its momentum increment.")
            momentum = momentum + increment
        return self.pack(physical, end_time, momentum)

    def metadata(self) -> dict[str, DiagnosticValue]:
        """Describe actual stored dimensions separately from the observer domain."""
        name = "physical" if self.copies == 1 else "duplicated"
        return {
            "state_formulation": name + ("_with_energy" if self.track_energy else ""),
            "track_energy": self.track_energy,
            "accepted_internal_state_dimension": self.dimension,
            "base_splitting_state_dimension": self.dimension,
            "spatial_state_dimension": self.spatial_size,
            "observer_state_dimension": self.physical_size,
            "observer_state_kind": "physical_map",
            "energy_feedback": False,
        }

    def extract_history(self, times: np.ndarray, history: np.ndarray
                        ) -> tuple[np.ndarray, dict[str, DiagnosticValue]]:
        """Expose physical samples and the common per-particle energy balance."""
        physical = self.physical(history)
        momentum = self.momentum(history)
        if momentum is None:
            return physical, {}
        particle_times = self.time(history)
        assert particle_times is not None
        output_times = np.broadcast_to(times, particle_times.shape)
        # Endpoint arithmetic can differ by round-off. Layout-specific checks
        # and output alignment belong here, leaving the collector generic.
        tolerance = float(16 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(times)))))
        if not np.allclose(particle_times, output_times, rtol=0.0, atol=tolerance):
            raise ValueError("Each particle time must match the output times within round-off.")
        dynamics = self.problem.dynamics
        assert isinstance(dynamics, HamiltonianSystem)
        energy = np.asarray(dynamics.hamiltonian(times, physical), dtype=float)
        if energy.ndim == 1:
            energy = energy[np.newaxis, :]
        if energy.shape != momentum.shape or not np.all(np.isfinite(energy)):
            raise ValueError("Hamiltonian history must be finite with shape (particles, times).")
        balance = energy + momentum
        error = balance - balance[:, :1]
        return physical, {
            "extended_time": output_times.copy(),
            "extended_momentum": momentum,
            "extended_momentum_normalization": "physical_kappa",
            "physical_hamiltonian": energy,
            "hamiltonian": energy,
            "energy_drift": energy - energy[:, :1],
            "generalized_energy": balance,
            "generalized_energy_error": error,
            "energy_error": float(np.max(np.abs(error))),
        }


@dataclass(frozen=True)
class DoubledFormulation(PhysicalFormulation):
    """Two spatial copies and an optional ``(t, kappa)`` pair per particle.

    For N planar particles the dimension is 4N without tracking and 6N with
    tracking. Only the 2N spatial differences enter a diagonal projection.
    """

    copies: ClassVar[int] = 2
