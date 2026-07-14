# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Symplectic initial-value problem solvers."""

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Callable, Union

import numpy as xp
from numpy.typing import ArrayLike

from .integrators import SymplecticIntegrator
from .solution import OdeSolution
from .system import HamSys


Flow = Callable[[float, float, xp.ndarray], xp.ndarray]
StepCallback = Callable[[float, xp.ndarray], None]


_COUPLING_BASE = xp.array(
    [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]],
    dtype=float,
)
_COUPLING_COS = xp.array(
    [[1, 0, -1, 0], [0, 1, 0, -1], [-1, 0, 1, 0], [0, -1, 0, 1]],
    dtype=float,
)
_COUPLING_SIN = xp.array(
    [[0, -1, 0, 1], [1, 0, -1, 0], [0, 1, 0, -1], [-1, 0, 1, 0]],
    dtype=float,
)


@dataclass(frozen=True)
class _SolverInputs:
    t0: float
    tf: float
    state: xp.ndarray
    max_step: float
    save_step: Union[float, None]
    t_eval: Union[xp.ndarray, None]


def _validate_solver_inputs(
    t_span: ArrayLike,
    y0: ArrayLike,
    step: float,
    save_step: Union[float, None],
    t_eval: Union[ArrayLike, None],
) -> _SolverInputs:
    """Normalize solver inputs and reject invalid integration domains."""
    try:
        span = xp.asarray(t_span, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("`t_span` must contain exactly two finite numbers.") from exc
    if span.shape != (2,) or not xp.all(xp.isfinite(span)):
        raise ValueError("`t_span` must contain exactly two finite numbers.")
    t0, tf = map(float, span)
    if t0 > tf:
        raise ValueError("Values in `t_span` are not properly sorted.")

    try:
        max_step = float(step)
    except (TypeError, ValueError) as exc:
        raise ValueError("`step` must be a positive finite number.") from exc
    if not xp.isfinite(max_step) or max_step <= 0:
        raise ValueError("`step` must be a positive finite number.")

    state = xp.asarray(y0)
    if state.ndim == 0 or state.size == 0:
        raise ValueError("`y0` must be a non-empty array-like state.")
    if not xp.issubdtype(state.dtype, xp.number):
        raise ValueError("`y0` must contain numeric values.")
    state = xp.asarray(state, dtype=xp.result_type(state.dtype, xp.float64))
    if not xp.all(xp.isfinite(state)):
        raise ValueError("`y0` must contain only finite values.")

    normalized_save_step: Union[float, None] = None
    if save_step is not None:
        if t_eval is not None:
            raise ValueError("`save_step` and `t_eval` are mutually exclusive.")
        try:
            normalized_save_step = float(save_step)
        except (TypeError, ValueError) as exc:
            raise ValueError("`save_step` must be a positive finite number.") from exc
        if not xp.isfinite(normalized_save_step) or normalized_save_step <= 0:
            raise ValueError("`save_step` must be a positive finite number.")

    if t_eval is None:
        return _SolverInputs(
            t0=t0,
            tf=tf,
            state=state,
            max_step=max_step,
            save_step=normalized_save_step,
            t_eval=None,
        )

    try:
        eval_times = xp.asarray(t_eval, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("`t_eval` must be a one-dimensional sequence of finite times.") from exc
    if eval_times.ndim != 1:
        raise ValueError("`t_eval` must be 1-dimensional.")
    if eval_times.size == 0:
        raise ValueError("`t_eval` must contain at least one time.")
    if not xp.all(xp.isfinite(eval_times)):
        raise ValueError("`t_eval` must contain only finite times.")
    if xp.any(eval_times < t0) or xp.any(eval_times > tf):
        raise ValueError("Values in `t_eval` are not within `t_span`.")
    if xp.any(xp.diff(eval_times) <= 0):
        raise ValueError("Values in `t_eval` are not properly sorted.")
    return _SolverInputs(
        t0=t0,
        tf=tf,
        state=state,
        max_step=max_step,
        save_step=None,
        t_eval=eval_times,
    )


def _step_count(duration: float, max_step: float) -> int:
    """Return the fewest steps that do not exceed ``max_step``."""
    ratio = duration / max_step
    # Move by one representable float toward the lower integer to neutralize
    # division roundoff without using a tolerance that grows with ``ratio``.
    return max(1, math.ceil(math.nextafter(ratio, -math.inf)))


def _regular_output_times(t0: float, tf: float, interval: float) -> xp.ndarray:
    """Return a regular grid including both endpoints."""
    if t0 == tf:
        return xp.asarray([t0], dtype=float)

    duration = tf - t0
    regular_count = int(math.floor(duration / interval))
    times = t0 + xp.arange(regular_count + 1, dtype=float) * interval
    endpoint_tolerance = max(
        math.ulp(tf),
        math.ulp(float(times[-1])),
        math.ulp(interval),
    )
    if math.isclose(float(times[-1]), tf, rel_tol=0.0, abs_tol=endpoint_tolerance):
        times[-1] = tf
    elif times[-1] < tf:
        times = xp.append(times, tf)
    else:
        times[-1] = tf
    return times


def _build_output_times(inputs: _SolverInputs) -> xp.ndarray:
    """Create exact output times for automatic or user-supplied sampling."""
    if inputs.t_eval is not None:
        return inputs.t_eval.copy()
    interval = inputs.save_step or inputs.max_step
    return _regular_output_times(inputs.t0, inputs.tf, interval)


def _build_integration_targets(output_times: xp.ndarray, tf: float) -> xp.ndarray:
    """Ensure integration reaches ``tf`` even when it is not sampled."""
    if output_times[-1] == tf:
        return output_times
    return xp.append(output_times, tf)


def _advance_one_step(
    chi: Flow,
    chi_star: Flow,
    t: float,
    y: xp.ndarray,
    stages: tuple[tuple[float, int], ...],
) -> tuple[float, xp.ndarray]:
    """Apply every stage of one symplectic composition step."""
    for stage_step, stage_order in stages:
        if stage_order == 0:
            y = chi(stage_step, t + stage_step, y)
        else:
            y = chi_star(stage_step, t, y)
        t += stage_step
    return t, y


def _integrate_to_target(
    chi: Flow,
    chi_star: Flow,
    integrator: SymplecticIntegrator,
    t: float,
    y: xp.ndarray,
    target: float,
    max_step: float,
    command: Union[StepCallback, None],
) -> tuple[float, xp.ndarray, Union[float, None], int]:
    """Advance to one requested output time without exceeding ``max_step``."""
    duration = target - t
    if duration == 0:
        return target, y, None, 0

    count = _step_count(duration, max_step)
    internal_step = duration / count
    stages = tuple(
        (float(coefficient * internal_step), int(stage_order))
        for coefficient, stage_order in zip(
            integrator.alpha_s, integrator.alpha_o
        )
    )
    segment_start = t
    for index in range(count):
        t, y = _advance_one_step(chi, chi_star, t, y, stages)
        # Prevent accumulated roundoff from changing the integration schedule.
        t = segment_start + (index + 1) * internal_step
        if command is not None:
            command(t, y)
    return target, y, internal_step, count


def solve_ivp_symp(
    chi: Flow,
    chi_star: Flow,
    t_span: ArrayLike,
    y0: ArrayLike,
    step: float,
    t_eval: Union[ArrayLike, None] = None,
    method: str = "BM4",
    command: Union[StepCallback, None] = None,
    *,
    save_step: Union[float, None] = None,
) -> OdeSolution:
    """Solve a Hamiltonian initial-value problem by symplectic splitting.

    ``step`` is the maximum internal step size. ``save_step`` defines a regular
    storage cadence independent from integration accuracy. If ``t_eval`` is
    supplied instead, the solver lands exactly on each requested time and
    stores only those states. ``save_step`` and ``t_eval`` are mutually
    exclusive. Without either one, every internal step is stored.

    ``chi`` and ``chi_star`` receive ``(h, t, y)`` and must return a state with
    the same shape as ``y``. The optional ``command`` callback runs after each
    complete internal step and receives the live state: mutating it changes the
    integration. Integration always reaches the end of ``t_span``, even when
    the last requested sample precedes it.

    The returned :class:`OdeSolution` includes ``requested_step``, ``min_step``,
    ``max_step`` and ``n_steps`` in addition to ``t``, ``y`` and the legacy
    ``step`` attribute. ``step`` is kept as an alias of ``max_step``.
    """
    if not callable(chi) or not callable(chi_star):
        raise TypeError("`chi` and `chi_star` must be callable.")
    if command is not None and not callable(command):
        raise TypeError("`command` must be callable or None.")

    inputs = _validate_solver_inputs(t_span, y0, step, save_step, t_eval)
    output_times = _build_output_times(inputs)
    integration_targets = _build_integration_targets(output_times, inputs.tf)
    integrator = SymplecticIntegrator(method)

    t = inputs.t0
    y = inputs.state.copy()
    result = xp.empty(inputs.state.shape + (len(output_times),), dtype=inputs.state.dtype)
    output_index = 0
    n_steps = 0
    smallest_step = math.inf
    largest_step = 0.0

    for target in integration_targets:
        t, y, internal_step, segment_steps = _integrate_to_target(
            chi,
            chi_star,
            integrator,
            t,
            y,
            float(target),
            inputs.max_step,
            command,
        )
        if internal_step is not None:
            n_steps += segment_steps
            smallest_step = min(smallest_step, internal_step)
            largest_step = max(largest_step, internal_step)

        stored_state = xp.asarray(y)
        if stored_state.shape != inputs.state.shape:
            raise ValueError("`chi` and `chi_star` must preserve the shape of `y0`.")
        if output_index >= len(output_times) or target != output_times[output_index]:
            continue
        if not xp.can_cast(stored_state.dtype, result.dtype, casting="same_kind"):
            raise ValueError(
                "The flow changed the state dtype incompatibly; provide `y0` "
                "with the required dtype."
            )
        result[..., output_index] = stored_state
        output_index += 1

    if n_steps == 0:
        smallest_step = inputs.max_step
        largest_step = inputs.max_step
    return OdeSolution(
        t=output_times,
        y=result,
        step=largest_step,
        requested_step=inputs.max_step,
        min_step=smallest_step,
        max_step=largest_step,
        n_steps=n_steps,
    )

def solve_ivp_sympext(
	hs: HamSys,
	t_span: ArrayLike,
	y0: ArrayLike,
	step: float,
	t_eval: Union[ArrayLike, None] = None,
	method: str = 'BM4',
	omega: float = 10,
	command: Union[StepCallback, None] = None,
	check_energy: bool = False,
	*,
	save_step: Union[float, None] = None,
) -> OdeSolution:
	"""
	Solve an initial value problem for a Hamiltonian system using an explicit
	symplectic approximation obtained by an extension in phase space (see [1]).

	This function numerically integrates a system of ordinary differential
	equations in canonical coordinates given an initial value:

	dy / dt = {y, H(t, y)}
	y(t0) = y0

	Here t is a 1-D independent variable (time), y(t) is an N-D vector-valued
	function (state), and a Hamiltonian H(t, y) and a canonical Poisson bracket
	{. , .} determine the differential equations. The goal is to find y(t)
	approximately satisfying the differential equations, given an initial value
	y(t0)=y0. The state y(t) is of the form (q(t), p(t)).

	If the Hamiltonian has an explicit time dependence, there is the
	possibility to check energy by computing k(t) where k is a canonically
	conjugate variable to time. Its evolution is given by

	dk / dt = -dH / dt
	k(t0)=0

	Parameters
	----------
	hs : HamSys
		Hamiltonian system containing the Hamiltonian vector field. The
		attributes `y_dot` (for dy / dt) should be specified. If there is an
		explicit time dependence and `check_energy` is True, the attribute
		`k_dot` (for dk / dt) should be specified.
	t_span : 2-member sequence
		Interval of integration (t0, tf). The solver starts with t=t0 and
		integrates until it reaches t=tf. Both t0 and tf must be floats or
		values interpretable by the float conversion function.
	y0 : array_like, shape (2n,)
		Initial state y0=(q0, p0) where q0 are the initial positions and p0 the
		initial momenta.
	step : float
		Maximum internal step size.
	save_step : float or None, optional
		Regular interval between stored states. Mutually exclusive with `t_eval`.
	t_eval : array_like or None, optional
		Times at which to store the computed solution. They must be strictly
		increasing and lie within `t_span`. If None (default), use points
		selected by the solver.
	method : string, optional
        Integration methods are listed on https://pypi.org/project/pyhamsys/
		'BM4' is the default.
	omega : float, optional
		Coupling parameter in the extended phase space (see [1])
	command : function of (t, y) or None, optional
		Function to be run at each step size.
	check_energy : bool, optional
		If True, computes the total energy. Default is False.

	Returns
	-------
	Bunch object with the following fields defined:
	t : ndarray, shape (n_points,)
		Time points.
	y : ndarray, shape (2n, n_points)
		Values y(t) = (q(t), p(t)) at `t`.
	k : ndarray, shape (n, n_points)
		Values of k(t) at `t` if `check_energy` is True and if the Hamiltonian
		system has an explicit time dependence.
	err : float
		Error in the computation of the total energy, computed only if
		`check_energy` is True.
	step : float
		Step size used in the computation.

	References
	----------
		[1] Tao, M., 2016, "Explicit symplectic approximation of nonseparable
		Hamiltonians: Algorithm and long time performance",
		Phys. Rev. E 94, 043303
	"""
	check_energy_ = check_energy and hs._time_dependent

	@lru_cache(maxsize=256)
	def _coupling(h:float) -> xp.ndarray:
		coupling: xp.ndarray = (
			_COUPLING_BASE
			+ xp.cos(2 * omega * h) * _COUPLING_COS
			+ xp.sin(2 * omega * h) * _COUPLING_SIN
		) / 2
		return coupling

	def _chi_ext(h:float, t:float, y:xp.ndarray) -> xp.ndarray:
		y_ = hs._split(y, check_energy=check_energy_)
		y_[1] += h * hs.y_dot(t, y_[0])
		if check_energy_:
			y_[-1] += h * hs.k_dot(t, y_[0])
		y_[0] += h * hs.y_dot(t, y_[1])
		if check_energy_:
			y_[-1] += h * hs.k_dot(t, y_[1])
		yr: xp.ndarray = xp.concatenate((y_[0], y_[1]), axis=None)
		yr = xp.einsum('ij,j...->i...', _coupling(h), xp.split(yr, 4)).flatten()
		if not check_energy_:
			return yr
		return xp.concatenate((yr, y_[-1]), axis=None)

	def _chi_ext_star(h:float, t:float, y:xp.ndarray) -> xp.ndarray:
		y_ = hs._split(y, by_var=True, check_energy=check_energy_)
		yr = y_ if not check_energy_ else y_[:-1]
		yr = xp.einsum('ij,j...->i...', _coupling(h), yr).flatten()
		if check_energy_:
			yr = xp.concatenate((yr, y_[-1]), axis=None)
		y_ = hs._split(yr, check_energy=check_energy_)
		y_[0] += h * hs.y_dot(t, y_[1])
		if check_energy_:
			y_[-1] += h * hs.k_dot(t, y_[1])
		y_[1] += h * hs.y_dot(t, y_[0])
		if check_energy_:
			y_[-1] += h * hs.k_dot(t, y_[0])
		return xp.concatenate([_ for _ in y_], axis=None)

	if not hasattr(hs, 'y_dot'):
		raise ValueError("The attribute 'y_dot' must be provided.")
	if check_energy_ and not hasattr(hs, 'k_dot'):
		raise ValueError("In order to check energy for a time-dependent system, the attribute 'k_dot' must be provided.")
	initial_state = xp.asarray(y0)
	y_ = xp.tile(initial_state, 2)
	if check_energy_:
		y_ = xp.concatenate((y_, xp.zeros(len(initial_state)//(2*hs._ndof) )), axis=None)
	sol = solve_ivp_symp(
		_chi_ext,
		_chi_ext_star,
		t_span,
		y_,
		method=method,
		step=step,
		save_step=save_step,
		t_eval=t_eval,
		command=command,
	)
	y_ = hs._split(sol.y, by_var=True, check_energy=check_energy_)
	sol.y = xp.concatenate(((y_[0] + y_[2]) / 2, (y_[1] + y_[3]) / 2), axis=0)
	if check_energy_:
		sol.k = y_[-1] / 2
	if check_energy:
		sol.err = hs.compute_energy(sol)
	return sol
