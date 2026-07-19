# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Extended-phase-space solver used by guiding-centre systems."""

from __future__ import annotations

from functools import lru_cache
import sys
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike

from ._solver_common import (
    StepCallback,
    build_integration_targets,
    build_output_times,
    planned_step_count,
    validate_solver_inputs,
)
from .fc_solver import solve_symplectic
from .solution import OdeSolution


class ExtendedSystem(Protocol):
    """Numerical contract required by :func:`solve_extended`."""

    @property
    def degrees_of_freedom(self) -> int:
        """Number of physical canonical degrees of freedom."""

    @property
    def time_dependent(self) -> bool:
        """Whether the Hamiltonian uses the extended momentum."""

    def vector_field(self, t: float, state: np.ndarray) -> np.ndarray:
        """Return the Hamiltonian vector field at ``(t, state)``."""

    def extended_momentum_derivative(
        self,
        t: float,
        state: np.ndarray,
    ) -> float | np.ndarray:
        """Return the derivative of momentum conjugate to time."""

    def compute_energy(self, solution: OdeSolution) -> np.ndarray | float:
        """Return the requested energy diagnostic for ``solution``."""


_COUPLING_BASE = np.asarray(
    [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]],
    dtype=float,
)
_COUPLING_COS = np.asarray(
    [[1, 0, -1, 0], [0, 1, 0, -1], [-1, 0, 1, 0], [0, -1, 0, 1]],
    dtype=float,
)
_COUPLING_SIN = np.asarray(
    [[0, -1, 0, 1], [1, 0, -1, 0], [0, 1, 0, -1], [-1, 0, 1, 0]],
    dtype=float,
)


class _ProgressBar:
    """Small dependency-free progress bar for long GC integrations."""

    def __init__(self, total: int, every: int = 100) -> None:
        self.total = max(total, 1)
        self.every = every
        self.steps = 0
        self._closed = False

    def update(self, t: float, _state: np.ndarray) -> None:
        self.steps += 1
        if self.steps % self.every and self.steps < self.total:
            return
        fraction = min(self.steps / self.total, 1.0)
        width = 30
        filled = int(width * fraction)
        bar = "=" * filled
        if filled < width:
            bar += ">" + " " * (width - filled - 1)
        print(
            f"\rsolve_extended [{bar}] {fraction:6.1%} "
            f"({self.steps}/{self.total} steps, t={t:.6g})",
            end="",
            file=sys.stderr,
            flush=True,
        )

    def close(self) -> None:
        if not self._closed:
            print(file=sys.stderr, flush=True)
            self._closed = True


def _validated_degrees_of_freedom(system: ExtendedSystem) -> int:
    raw_degrees = system.degrees_of_freedom
    if isinstance(raw_degrees, (bool, np.bool_)):
        raise ValueError("`degrees_of_freedom` must be a positive integer.")
    try:
        numeric_degrees = float(raw_degrees)
    except (TypeError, ValueError) as exc:
        raise ValueError("`degrees_of_freedom` must be a positive integer.") from exc
    if (
        not np.isfinite(numeric_degrees)
        or numeric_degrees < 1
        or not numeric_degrees.is_integer()
    ):
        raise ValueError("`degrees_of_freedom` must be a positive integer.")
    return int(numeric_degrees)


def solve_extended(
    system: ExtendedSystem,
    t_span: ArrayLike,
    y0: ArrayLike,
    step: float,
    t_eval: ArrayLike | None = None,
    method: str = "BM4",
    omega: float = 10,
    command: StepCallback | None = None,
    check_energy: bool = False,
    *,
    save_step: float | None = None,
    n_save_step: int | None = None,
    progress: bool = False,
) -> OdeSolution:
    """Integrate a nonseparable GC Hamiltonian in extended phase space."""
    vector_field = getattr(system, "vector_field", None)
    if not callable(vector_field):
        raise ValueError("The system must provide a callable `vector_field`.")
    if command is not None and not callable(command):
        raise TypeError("`command` must be callable or None.")
    try:
        coupling_frequency = float(omega)
    except (TypeError, ValueError) as exc:
        raise ValueError("`omega` must be finite.") from exc
    if not np.isfinite(coupling_frequency):
        raise ValueError("`omega` must be finite.")

    physical_inputs = validate_solver_inputs(t_span, y0, step)
    if physical_inputs.state.ndim != 1:
        raise ValueError("`solve_extended` requires a one-dimensional initial state.")
    degrees_of_freedom = _validated_degrees_of_freedom(system)
    state_size = physical_inputs.state.size
    state_stride = 2 * degrees_of_freedom
    if state_size % state_stride:
        raise ValueError(
            "The initial state size must be divisible by twice the number of "
            "degrees of freedom."
        )
    trajectory_count = state_size // state_stride

    time_dependent = bool(system.time_dependent)
    track_extended_momentum = bool(check_energy and time_dependent)
    momentum_derivative = getattr(system, "extended_momentum_derivative", None)
    if track_extended_momentum and not callable(momentum_derivative):
        raise ValueError(
            "A time-dependent energy check requires a callable "
            "`extended_momentum_derivative`."
        )
    compute_energy = getattr(system, "compute_energy", None)
    if check_energy and not callable(compute_energy):
        raise ValueError("An energy check requires a callable `compute_energy`.")

    extended_state = np.concatenate(
        (physical_inputs.state, physical_inputs.state),
        axis=0,
    )
    if track_extended_momentum:
        extended_state = np.concatenate(
            (
                extended_state,
                np.zeros(trajectory_count, dtype=extended_state.dtype),
            ),
            axis=0,
        )

    expected_extended_size = 2 * state_size + (
        trajectory_count if track_extended_momentum else 0
    )
    canonical_block_size = state_size // 2

    def split_copies(state: np.ndarray) -> list[np.ndarray]:
        if state.ndim == 0 or state.shape[0] != expected_extended_size:
            raise ValueError("The extended flow changed the state shape.")
        copies = [state[:state_size], state[state_size : 2 * state_size]]
        if track_extended_momentum:
            copies.append(state[2 * state_size :])
        return copies

    def split_variables(state: np.ndarray) -> list[np.ndarray]:
        copies = split_copies(state)
        first, second = copies[:2]
        variables = [
            first[:canonical_block_size],
            first[canonical_block_size:],
            second[:canonical_block_size],
            second[canonical_block_size:],
        ]
        if track_extended_momentum:
            variables.append(copies[-1])
        return variables

    def evaluated_vector_field(t: float, state: np.ndarray) -> np.ndarray:
        derivative = np.asarray(vector_field(t, state))
        if derivative.shape != state.shape:
            raise ValueError("`vector_field` must preserve the physical state shape.")
        return derivative

    def update_extended_momentum(
        momentum: np.ndarray,
        h: float,
        t: float,
        state: np.ndarray,
    ) -> None:
        if not track_extended_momentum:
            return
        assert callable(momentum_derivative)
        momentum += h * np.asarray(momentum_derivative(t, state))

    @lru_cache(maxsize=256)
    def coupling(h: float) -> np.ndarray:
        value = np.asarray((
            _COUPLING_BASE
            + np.cos(2 * coupling_frequency * h) * _COUPLING_COS
            + np.sin(2 * coupling_frequency * h) * _COUPLING_SIN
        ) / 2)
        return value

    def coupled_state(h: float, blocks: list[np.ndarray]) -> np.ndarray:
        value = np.asarray(np.einsum(
            "ij,j...->i...",
            coupling(h),
            np.stack(blocks, axis=0),
        )).flatten()
        return value

    def chi_extended(h: float, t: float, state: np.ndarray) -> np.ndarray:
        parts = split_copies(state)
        first, second = parts[:2]
        momentum = parts[-1] if track_extended_momentum else np.empty(0)
        second += h * evaluated_vector_field(t, first)
        update_extended_momentum(momentum, h, t, first)
        first += h * evaluated_vector_field(t, second)
        update_extended_momentum(momentum, h, t, second)
        result = coupled_state(h, list(np.split(np.concatenate((first, second)), 4)))
        if track_extended_momentum:
            result = np.concatenate((result, momentum), axis=0)
        return result

    def chi_extended_star(h: float, t: float, state: np.ndarray) -> np.ndarray:
        variables = split_variables(state)
        physical_variables = variables[:-1] if track_extended_momentum else variables
        result = coupled_state(h, physical_variables)
        if track_extended_momentum:
            result = np.concatenate((result, variables[-1]), axis=0)
        parts = split_copies(result)
        first, second = parts[:2]
        momentum = parts[-1] if track_extended_momentum else np.empty(0)
        first += h * evaluated_vector_field(t, second)
        update_extended_momentum(momentum, h, t, second)
        second += h * evaluated_vector_field(t, first)
        update_extended_momentum(momentum, h, t, first)
        return np.concatenate(parts, axis=0)

    progress_bar: _ProgressBar | None = None
    progress_command = command
    if progress:
        extended_inputs = validate_solver_inputs(t_span, extended_state, step)
        output_times = build_output_times(
            extended_inputs,
            t_eval,
            save_step,
            n_save_step,
        )
        targets = build_integration_targets(output_times, extended_inputs.tf)
        progress_bar = _ProgressBar(planned_step_count(extended_inputs, targets))

        def progress_command(t: float, state: np.ndarray) -> None:
            assert progress_bar is not None
            progress_bar.update(t, state)
            if command is not None:
                command(t, state)

    try:
        solution = solve_symplectic(
            chi_extended,
            chi_extended_star,
            t_span,
            extended_state,
            step,
            t_eval,
            method,
            progress_command,
            save_step=save_step,
            n_save_step=n_save_step,
        )
    finally:
        if progress_bar is not None:
            progress_bar.close()

    variables = split_variables(solution.y)
    solution.y = np.concatenate(
        (
            (variables[0] + variables[2]) / 2,
            (variables[1] + variables[3]) / 2,
        ),
        axis=0,
    )
    if track_extended_momentum:
        solution.k = variables[-1] / 2
    if check_energy:
        assert callable(compute_energy)
        solution.err = compute_energy(solution)
    return solution


__all__ = ["ExtendedSystem", "solve_extended"]
