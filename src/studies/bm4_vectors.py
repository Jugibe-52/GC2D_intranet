"""Aligned BM4 multiplier/displacement study and planar angular diagnostics."""

from dataclasses import dataclass, replace
import numpy as np
from diagnostics.bm4_vectors import BM4VectorObserver
from methods.extended.bm4 import BM4Implicit
from contracts.problem import InitialValueProblem
from contracts.request import SimulationRequest
from solution import Solution
from simulation.runner import simulate


def vector_angles(vectors: np.ndarray, floor: float) -> tuple[np.ndarray, np.ndarray]:
    """Return norms and unwrapped radians, restarting after unresolved vectors.

    Input shape is (step, particle, xy). Directions below the explicit norm
    floor are NaN; unwrapping never bridges such gaps.
    """
    values = np.asarray(vectors, dtype=float)
    if values.ndim != 3 or values.shape[-1] != 2 or not np.all(np.isfinite(values)):
        raise ValueError("Vectors must be finite with shape (step, particle, 2).")
    if not np.isfinite(floor) or floor < 0:
        raise ValueError("The direction floor must be finite and nonnegative.")
    norms = np.linalg.norm(values, axis=-1)
    angles = np.full(norms.shape, np.nan)
    for particle in range(values.shape[1]):
        valid = np.flatnonzero(norms[:, particle] > floor)
        for segment in np.split(valid, np.flatnonzero(np.diff(valid) > 1) + 1):
            if segment.size:
                angles[segment, particle] = np.unwrap(np.arctan2(
                    values[segment, particle, 1], values[segment, particle, 0]
                ))
    return norms, angles


@dataclass(frozen=True)
class BM4VectorStudy:
    """Arrays use (step, particle[, xy]); angles are in radians."""

    solution: Solution
    times: np.ndarray
    mu: np.ndarray
    delta: np.ndarray
    mu_norm: np.ndarray
    delta_norm: np.ndarray
    mu_angle: np.ndarray
    delta_angle: np.ndarray
    angle_difference: np.ndarray
    norm_ratio: np.ndarray


def run_bm4_vector_study(
    problem: InitialValueProblem,
    method: BM4Implicit,
    request: SimulationRequest,
    *,
    direction_floor: float,
) -> BM4VectorStudy:
    """Integrate with a fresh observer and align vectors to the saved endpoints."""
    observer = BM4VectorObserver()
    solution = simulate(problem, replace(method, step_observer=observer), request)
    times = np.asarray(observer.times)
    if times.shape != solution.t[1:].shape or not np.allclose(
        times, solution.t[1:], rtol=0, atol=1e-12
    ):
        raise ValueError("Save every complete integration step for this study.")
    mu = np.asarray(observer.multipliers)
    delta = np.asarray(observer.displacements)
    mu_norm, mu_angle = vector_angles(mu, direction_floor)
    delta_norm, delta_angle = vector_angles(delta, direction_floor)
    # Signed principal rotation from delta to mu; NaNs propagate deliberately.
    difference = np.arctan2(np.sin(mu_angle - delta_angle), np.cos(mu_angle - delta_angle))
    ratio = np.divide(mu_norm, delta_norm, out=np.full_like(mu_norm, np.nan),
                      where=delta_norm > direction_floor)
    x, y = solution.positions()
    np.testing.assert_allclose(delta, np.stack((np.diff(x).T, np.diff(y).T), axis=-1))
    np.testing.assert_allclose(np.max(np.abs(mu), axis=(1, 2)),
                               np.asarray(solution.diagnostics["projection_multiplier_norms"], dtype=float))
    return BM4VectorStudy(solution, times, mu, delta, mu_norm, delta_norm,
                          mu_angle, delta_angle, difference, ratio)
