"""Resolve spatial and same-phase returns of one prescribed GC trajectory."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import numpy as np
from scipy.optimize import brentq

from diagnostics.adaptive_trajectory import AdaptiveTrajectoryObserver
from initial_conditions import GCInitialConfiguration
from methods.adaptive.scipy import DOP853, Radau
from contracts.problem import InitialValueProblem
from contracts.request import SimulationRequest
from dynamics import GuidingCenterDynamics
from potential import Potential


@dataclass(frozen=True)
class SingleParticleRecurrenceConfig:
    """Explicit dimensionless integration and return-search controls."""

    final_time: float = 50.0
    search_start: float = 1.0
    samples_per_cycle: int = 200
    rho: float = 0.3
    reference_rtol: float = 1e-10
    reference_atol: float = 1e-12
    reference_max_step: float = 0.025
    refined_rtol: float = 1e-12
    refined_atol: float = 1e-14
    refined_max_step: float = 0.0125
    audit_rtol: float = 1e-13
    audit_atol: float = 1e-15
    audit_max_step: float = 0.00625
    root_xtol: float = 1e-11
    threshold_fractions: tuple[float, ...] = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)

    def __post_init__(self) -> None:
        for name in ("final_time", "search_start", "reference_rtol", "reference_atol",
                     "reference_max_step", "refined_rtol", "refined_atol",
                     "refined_max_step", "audit_rtol", "audit_atol",
                     "audit_max_step", "root_xtol"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not 0 < self.search_start < self.final_time:
            raise ValueError("Require 0 < search_start < final_time.")
        if not float(self.final_time).is_integer():
            raise ValueError("final_time must contain complete unit cycles.")
        if (isinstance(self.samples_per_cycle, bool)
                or int(self.samples_per_cycle) != self.samples_per_cycle
                or self.samples_per_cycle < 2):
            raise ValueError("samples_per_cycle must be an integer >= 2.")
        if not np.isfinite(self.rho) or self.rho < 0:
            raise ValueError("rho must be finite and nonnegative.")
        for suffix in ("rtol", "atol", "max_step"):
            if not (getattr(self, "audit_" + suffix) <= getattr(self, "refined_" + suffix)
                    <= getattr(self, "reference_" + suffix)):
                raise ValueError("Refinement and audit controls must be no looser.")
        if not self.threshold_fractions or any(
                not np.isfinite(v) or not 0 < v < 0.5 for v in self.threshold_fractions):
            raise ValueError("Threshold fractions must lie strictly between 0 and 0.5.")


def minimum_image(displacement: np.ndarray, period: float) -> np.ndarray:
    """Map coordinate differences to [-L/2, L/2); coordinates precede time."""
    if not np.isfinite(period) or period <= 0:
        raise ValueError("period must be finite and positive.")
    return np.asarray((np.asarray(displacement) + period / 2) % period - period / 2)


def distance_to_initial(states: np.ndarray, initial: np.ndarray, period: float) -> np.ndarray:
    """Return planar periodic distances for a state or a (2, samples) history."""
    origin = initial if np.ndim(states) == 1 else initial[:, None]
    return np.asarray(np.linalg.norm(minimum_image(states - origin, period), axis=0))


def locate_returns(dense: Any, dynamics: Any, initial: np.ndarray, period: float,
                   times: np.ndarray, *, root_xtol: float) -> np.ndarray:
    """Bracket strict distance minima and refine delta dot velocity = 0.

    The search mesh detects brackets; it does not set the reported time precision.
    Minimum-image branch cuts are rejected with a local minimum check. A second,
    twice-denser mesh is required by the caller to audit missed close approaches.
    """
    states = dense(times)
    delta = minimum_image(states - initial[:, None], period)
    slopes = np.sum(delta * dynamics.vector_field(times, states), axis=0)

    def stationary(time: float) -> float:
        state = dense(time)
        return float(np.dot(minimum_image(state - initial, period),
                            dynamics.vector_field(time, state)))

    roots: list[float] = []
    for i in np.flatnonzero((slopes[:-1] <= 0) & (slopes[1:] >= 0)):
        left, right = float(times[i]), float(times[i + 1])
        if slopes[i] == 0 and slopes[i + 1] == 0:
            continue  # A flat interval is not an isolated return.
        root = float(brentq(stationary, left, right, xtol=root_xtol,
                            rtol=4 * np.finfo(float).eps))
        probe = min((right - left) * 0.05, 1e-5)
        neighbours = np.asarray([max(times[0], root - probe), root,
                                 min(times[-1], root + probe)])
        distances = distance_to_initial(dense(neighbours), initial, period)
        if (distances[1] <= min(distances[0], distances[2]) + 1e-13 * period
                and (not roots or root - roots[-1] > 10 * root_xtol)):
            roots.append(root)
    return np.asarray(roots)


def threshold_windows(dense: Any, initial: np.ndarray, period: float,
                      times: np.ndarray, minima: np.ndarray, threshold: float,
                      *, root_xtol: float) -> list[dict[str, Any]]:
    """Find visits to a return ball, including visits narrower than mesh spacing.

    Adding resolved minima before bracketing catches short visits around them.
    An interval touching a search boundary is explicitly marked as censored.
    """
    nodes = np.unique(np.r_[times, minima])
    values = distance_to_initial(dense(nodes), initial, period) - threshold

    def residual(time: float) -> float:
        return float(distance_to_initial(dense(time), initial, period) - threshold)

    crossings = []
    for i in range(len(nodes) - 1):
        if values[i] * values[i + 1] < 0:
            crossings.append(brentq(residual, nodes[i], nodes[i + 1], xtol=root_xtol,
                                    rtol=4 * np.finfo(float).eps))
        elif values[i] == 0:
            crossings.append(float(nodes[i]))
    boundaries = np.unique(np.r_[nodes[0], crossings, nodes[-1]])
    windows = []
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
        if residual(float((left + right) / 2)) < 0:
            windows.append({"entry": float(left), "exit": float(right),
                            "duration": float(right - left),
                            "entry_censored": bool(left == nodes[0]),
                            "exit_censored": bool(right == nodes[-1])})
    return windows


def run_single_particle_recurrence(
    potential: Potential, initial: np.ndarray, *, config: SingleParticleRecurrenceConfig,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Integrate one physical initial state with two DOP853 solves and Radau.

    Every solver starts at time zero from the same original state. The measured
    discrepancies are empirical resolution indicators, not rigorous error bounds.
    """
    initial = np.asarray(initial, dtype=float)
    if initial.shape != (2,) or not np.all(np.isfinite(initial)):
        raise ValueError("Exactly one finite planar initial state is required.")
    if not np.allclose(potential.frequencies, 1.0, rtol=0, atol=1e-14):
        raise ValueError("This study requires unit-period forcing for integer-time returns.")
    dynamics = GuidingCenterDynamics(potential, rho=config.rho)
    period = float(potential.grid.period)
    times = np.linspace(0, config.final_time,
                        int(config.final_time * config.samples_per_cycle) + 1)
    search_times = np.unique(np.r_[config.search_start, times[times >= config.search_start]])
    fine_search = np.unique(np.r_[search_times, (search_times[:-1] + search_times[1:]) / 2])
    arrays = {"times": times, "initial_state": initial.copy(),
              "cycle_times": np.arange(1, int(config.final_time) + 1, dtype=float)}
    metadata: dict[str, Any] = {"config": asdict(config), "period": period,
                                "solvers": {}, "thresholds": []}
    solutions = {}
    schedules = (("DOP853", "DOP853", "reference"),
                 ("DOP853_refined", "DOP853", "refined"),
                 ("Radau", "Radau", "audit"))
    for label, method, prefix in schedules:
        options = {key: getattr(config, prefix + "_" + key)
                   for key in ("rtol", "atol", "max_step")}
        print(f"Starting {label}: t=[0, {config.final_time:g}], {options}", flush=True)
        started = perf_counter()
        observer = AdaptiveTrajectoryObserver()
        controls = dict(relative_tolerance=options['rtol'], absolute_tolerance=options['atol'],
                        dense_output=True, step_observer=observer)
        configured = (Radau(**controls, jacobian=lambda t, z: dynamics.particle_vector_field_jacobians(t, z)[0])
                      if method == 'Radau' else DOP853(**controls))
        problem = InitialValueProblem(dynamics, GCInitialConfiguration(initial))
        data = configured.integrate(problem, SimulationRequest(
            t_span=(0.0, config.final_time), max_step=options['max_step'], output_times=times))
        dense = observer.evaluate
        solutions[label] = dense
        arrays[label + ".states"] = data.states.copy()
        arrays[label + ".states"][:, 0] = initial
        arrays[label + ".cycle_states"] = dense(arrays["cycle_times"])
        roots = locate_returns(dense, dynamics, initial, period, search_times,
                               root_xtol=config.root_xtol)
        finer_roots = locate_returns(dense, dynamics, initial, period, fine_search,
                                     root_xtol=config.root_xtol)
        mesh_agrees = roots.size == finer_roots.size
        mesh_shift = float(np.max(np.abs(roots - finer_roots))) if mesh_agrees and roots.size else None
        arrays[label + ".return_times"] = finer_roots
        arrays[label + ".return_states"] = dense(finer_roots) if finer_roots.size else np.empty((2, 0))
        metadata["solvers"][label] = dict(
            method=method, **options, runtime_seconds=perf_counter() - started,
            nfev=int(np.sum(data.diagnostics["function_evaluations"])),
            njev=int(np.sum(data.diagnostics["jacobian_evaluations"])),
            nlu=int(np.sum(data.diagnostics["lu_decompositions"])),
            accepted_steps=int(data.diagnostics["step_count"]), search_count=int(roots.size),
            refined_search_count=int(finer_roots.size), search_counts_agree=mesh_agrees,
            search_maximum_time_shift=mesh_shift,
        )
        print(f"Completed {label} in {metadata['solvers'][label]['runtime_seconds']:.1f} s; "
              f"{finer_roots.size} spatial minima.", flush=True)

    reference = solutions["DOP853_refined"]
    reference_states = arrays["DOP853_refined.states"]
    for label in ("DOP853", "Radau"):
        arrays[label + ".reference_discrepancy"] = np.linalg.norm(
            minimum_image(arrays[label + ".states"] - reference_states, period), axis=0)
    arrays["reference_floor"] = np.maximum(arrays["DOP853.reference_discrepancy"],
                                           arrays["Radau.reference_discrepancy"])
    roots = arrays["DOP853_refined.return_times"]
    # Compare all solvers at each refined event time; never restart at a BM4 point.
    event_rows = []
    for time in roots:
        state = reference(time)
        delta = minimum_image(state - initial, period)
        floor = max(float(np.linalg.norm(minimum_image(solutions[label](time) - state, period)))
                    for label in ("DOP853", "Radau"))
        shifts = {}
        for label in ("DOP853", "Radau"):
            other = arrays[label + ".return_times"]
            shift = float(np.min(np.abs(other - time))) if other.size else None
            shifts[label] = shift if shift is not None and shift < 0.25 else None
        velocity = dynamics.vector_field(time, state)
        phase = float((time + 0.5) % 1.0 - 0.5)
        event_rows.append(dict(time=float(time), distance=float(np.linalg.norm(delta)),
                               cell_fraction=float(np.linalg.norm(delta) / period),
                               dx=float(delta[0]), dy=float(delta[1]),
                               phase_offset_cycles=phase, phase_offset_radians=2 * np.pi * phase,
                               velocity_difference=float(np.linalg.norm(
                                   velocity - dynamics.vector_field(0.0, initial))),
                               stationarity_residual=float(np.dot(delta, velocity)),
                               reference_discrepancy=floor,
                               dop853_time_shift=shifts["DOP853"], radau_time_shift=shifts["Radau"]))
    metadata["returns"] = event_rows
    for fraction in config.threshold_fractions:
        windows = threshold_windows(reference, initial, period, fine_search, roots,
                                    fraction * period, root_xtol=config.root_xtol)
        cycle_distances = distance_to_initial(arrays["DOP853_refined.cycle_states"], initial, period)
        metadata["thresholds"].append(dict(
            fraction=fraction, radius=fraction * period, windows=windows,
            integer_times_inside=arrays["cycle_times"][cycle_distances <= fraction * period].tolist()))
    # Include boundaries so a constrained minimum cannot be mistaken for an interior return.
    candidates = np.r_[config.search_start, roots, config.final_time]
    index = int(np.argmin(distance_to_initial(reference(candidates), initial, period)))
    metadata["closest_spatial_time"] = float(candidates[index])
    metadata["closest_spatial_is_boundary"] = bool(index in (0, len(candidates) - 1))
    arrays["velocity_difference"] = np.linalg.norm(
        dynamics.vector_field(times, reference_states) - dynamics.vector_field(0., initial)[:, None], axis=0)
    if any(not np.all(np.isfinite(value)) for value in arrays.values()):
        raise ValueError("Nonfinite recurrence output.")
    return arrays, metadata
