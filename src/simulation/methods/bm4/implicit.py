"""Implicit BM4 with one Hairer projection around each complete base cycle.

The exported physical state is always the packed physical guiding-centre vector
``z = (x_1, ..., x_p, y_1, ..., y_p)`` in :math:`R^{2p}`, where ``p`` is the
particle count.  A BM4 base cycle temporarily uses two copies ``Y = (u, v)``
in :math:`R^{4p}` because the prepared direct and adjoint maps split the
guiding-centre vector field between those copies.  The accepted internal state stores both copies on the diagonal; optional
energy tracking appends time and normalized momentum blocks, one entry per particle.
The spatial solve uses only the doubled spatial workspace.

For a physical dimension ``m = 2p``, define the diagonal embedding
``E z = (z, z)``, the constraint ``G = [I, -I]``, and its transpose
``N = G.T``.  One projected step finds ``mu in R^m`` from

``r(mu) = G Psi_h(E z + N mu) + 2 mu = 0``,

where ``Psi_h`` is the *complete* twelve-stage BM4 map implemented in
``_core.py``.  The accepted state is the mean of the two corrected output
copies.  There is deliberately no projection between BM4 stages.

Here "implicit" refers only to this multiplier root solve: the twelve prepared
direct/adjoint stages inside ``Psi_h`` are explicit sequential maps.  Newton
may differentiate ``Psi_h`` through the exact ordered product of its
guiding-centre stage Jacobians or through centered differences.  Good Broyden
solves the same reduced equation without changing the numerical method.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import numpy as np

from dynamics import GuidingCenterDynamics

from ...formulations.state import DoubledFormulation
from ...integration import IntegrationMethod, StepInfo, StepResult, NEWTON_ALIASES
from ..._result import DiagnosticValue
from ...formulations.gc import GCDoubledMaps, gc_coupling_matrix
from ...formulations.base import PreparedDirectAdjointFormulation
from ...observation import ImplicitBM4IntegrationStep, IntegrationStage, StepObserver
from ...problem import InitialValueProblem
from ...request import SimulationRequest
from .._nonlinear import (
	NonlinearSolver,
	_solve_broyden,
	_validate_nonlinear_solver,
)
from ._core import _BM4_ORDERS, _BM4_STAGES, _advance_composition


# Strategy used only to assemble Newton's Jacobian of the reduced residual.
NewtonJacobianMethod: TypeAlias = Literal["analytic", "finite_difference"]
NEWTON_JACOBIAN_METHODS: tuple[NewtonJacobianMethod, ...] = (
	"analytic",
	"finite_difference",
)


@dataclass(frozen=True, slots=True)
class _ProjectedBM4Step:
	"""Converged values retained from one reduced Hairer solve.

	``state`` and ``multiplier`` have physical shape ``(m,)``.  ``internal_input``
	and ``mapped`` have doubled shape ``(2*m,)`` and respectively store
	``E z + N mu`` and ``Psi_h(E z + N mu)``.  The doubled values are retained
	only so an optional observer can reconstruct the accepted base cycle without
	re-solving the nonlinear equation.
	"""

	state: np.ndarray
	multiplier: np.ndarray
	internal_input: np.ndarray
	mapped: np.ndarray
	iterations: int
	residual_evaluations: int
	residual_norm: float


def _positive_finite(value: float, name: str) -> float:
	"""Return a positive finite float, rejecting booleans as numeric inputs."""
	if isinstance(value, (bool, np.bool_)):
		raise ValueError(f"`{name}` must be positive and finite.")
	result = float(value)
	if not np.isfinite(result) or result <= 0:
		raise ValueError(f"`{name}` must be positive and finite.")
	return result


def _nonnegative_finite(value: float, name: str) -> float:
	"""Return a non-negative finite float, rejecting boolean values."""
	if isinstance(value, (bool, np.bool_)):
		raise ValueError(f"`{name}` must be non-negative and finite.")
	result = float(value)
	if not np.isfinite(result) or result < 0:
		raise ValueError(f"`{name}` must be non-negative and finite.")
	return result


def _positive_integer(value: int, name: str) -> int:
	"""Return a positive built-in integer without accepting booleans."""
	if (
		isinstance(value, (bool, np.bool_))
		or not isinstance(value, (int, np.integer))
		or value < 1
	):
		raise ValueError(f"`{name}` must be a positive integer.")
	return int(value)


def _bm4_map(
	prepared: GCDoubledMaps,
	t: float,
	internal_state: np.ndarray,
	step: float,
) -> np.ndarray:
	"""Apply the complete unprojected BM4 map to one doubled state.

	``internal_state`` has shape ``(2*m,)`` for a physical state of size ``m``.
	The same signed ``step`` and non-autonomous start time ``t`` are shared by
	all twelve stages.  Stage observation is disabled because nonlinear solvers
	may evaluate this map many times for a single accepted integration step.
	"""
	value = np.asarray(internal_state, dtype=float)
	# The traversal API always accepts diagnostic metadata.  The placeholder
	# index and names below are unobservable because ``stage_observer`` is None.
	result = _advance_composition(
		prepared,
		t,
		value,
		step,
		step_index=0,
		stage_observer=None,
		formulation_name="GCExtendedFormulation",
		method_name="BM4",
	)
	if result.shape != value.shape or not np.all(np.isfinite(result)):
		raise ValueError("The BM4 base map changed shape or became non-finite.")
	return np.asarray(result, dtype=float)


def _central_difference_jacobian(
	map_state: Callable[[np.ndarray], np.ndarray],
	state: np.ndarray,
	*,
	relative_step: float,
) -> np.ndarray:
	"""Differentiate a vector map with scale-aware centered differences.

	For input component ``j``, the perturbation is
	``relative_step * max(1, abs(state[j]))``.  The returned dense matrix has
	shape ``(d, d)``, where ``d = state.size``; here ``d = 2*m`` for the doubled
	BM4 map.
	"""
	value = np.asarray(state, dtype=float)
	dimension = value.size
	jacobian = np.empty((dimension, dimension), dtype=float)
	for column in range(dimension):
		# Scaling each column avoids vanishing perturbations near zero while
		# retaining a relative perturbation for large-magnitude coordinates.
		increment = relative_step * max(1.0, abs(float(value[column])))
		perturbation = np.zeros_like(value)
		perturbation[column] = increment
		forward = np.asarray(map_state(value + perturbation), dtype=float)
		backward = np.asarray(map_state(value - perturbation), dtype=float)
		if forward.shape != value.shape or backward.shape != value.shape:
			raise ValueError("The differentiated BM4 map changed the state shape.")
		jacobian[:, column] = (forward - backward) / (2.0 * increment)
	if not np.all(np.isfinite(jacobian)):
		raise ValueError("The BM4 map Jacobian contains non-finite values.")
	return jacobian


def _projection_matrices(physical_size: int) -> tuple[np.ndarray, np.ndarray]:
	"""Build the matrices that identify the physical diagonal.

	For ``m = physical_size``, ``G = [I, -I]`` has shape ``(m, 2*m)`` and
	measures the difference between both copies.  ``N = G.T`` has shape
	``(2*m, m)`` and maps a multiplier to opposite displacements ``(mu, -mu)``.
	Consequently ``G N = 2 I`` and ``G (z, z) = 0``.
	"""
	identity = np.eye(physical_size)
	constraint = np.concatenate((identity, -identity), axis=1)
	normal = np.concatenate((identity, -identity), axis=0)
	return constraint, normal


def _bm4_evaluation(
	prepared: GCDoubledMaps,
	t: float,
	state: np.ndarray,
	step: float,
	multiplier: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
	"""Return ``(E z + N mu, Psi_h(E z + N mu))``.

	Both ``state`` and ``multiplier`` have physical shape ``(m,)``.  Concatenating
	``state + multiplier`` and ``state - multiplier`` constructs the doubled
	input directly without materializing the diagonal embedding ``E`` or normal
	embedding ``N``.
	"""
	internal_input = np.concatenate((state + multiplier, state - multiplier))
	return internal_input, _bm4_map(prepared, t, internal_input, step)


def _bm4_map_jacobian(
	prepared: GCDoubledMaps,
	t: float,
	internal_input: np.ndarray,
	step: float,
	*,
	relative_step: float,
	method: NewtonJacobianMethod,
) -> np.ndarray:
	"""Return ``D Psi_h`` at a doubled input as a dense ``(2*m, 2*m)`` matrix.

	The analytic path is exact up to floating-point evaluation and is specialized
	to :class:`GuidingCenterDynamics` with exact second spatial derivatives
	(``interpolation_order >= 3``).  The fallback differentiates the same complete
	map, not its individual stages.
	"""
	if method == "analytic":
		return _analytic_bm4_map_jacobian(prepared, t, internal_input, step)
	return _central_difference_jacobian(
		lambda candidate: _bm4_map(prepared, t, candidate, step),
		internal_input,
		relative_step=relative_step,
	)


def _particle_blocks(state: np.ndarray, particle_count: int) -> np.ndarray:
	"""Convert a copy-major doubled vector to per-particle four-vectors.

	The input layout is ``(u_x[0:p], u_y[0:p], v_x[0:p], v_y[0:p])``.  The
	returned array has shape ``(p, 4)`` and rows ``(u_x, u_y, v_x, v_y)`` so
	independent ``4 x 4`` particle Jacobians can be evaluated in a batch.
	"""
	value = np.asarray(state, dtype=float)
	expected_size = 4 * particle_count
	if value.shape != (expected_size,):
		raise ValueError(
			"The analytic BM4 Jacobian requires a doubled two-dimensional GC state."
		)
	return np.column_stack(
		(
			value[:particle_count],
			value[particle_count : 2 * particle_count],
			value[2 * particle_count : 3 * particle_count],
			value[3 * particle_count :],
		)
	)


def _packed_physical(blocks: np.ndarray, offset: int) -> np.ndarray:
	"""Pack one ``(p, 2)`` particle copy as ``(x[0:p], y[0:p])``.

	``offset`` is ``0`` for columns ``(u_x, u_y)`` and ``2`` for columns
	``(v_x, v_y)`` of the array returned by :func:`_particle_blocks`.
	"""
	return np.concatenate((blocks[:, offset], blocks[:, offset + 1]))


def _batched_identity(particle_count: int) -> np.ndarray:
	"""Return ``p`` writable ``4 x 4`` identities for particlewise products."""
	return np.broadcast_to(np.eye(4), (particle_count, 4, 4)).copy()


def _direct_stage_particle_jacobians(
	dynamics: GuidingCenterDynamics,
	state: np.ndarray,
	*,
	duration: float,
	time: float,
	frequency: float,
) -> np.ndarray:
	"""Differentiate one direct GC stage for all particles independently.

	``state`` has shape ``(p, 4)`` with each row ``(u_x, u_y, v_x, v_y)``.
	For signed duration ``s``, the direct map performs

	``v_plus = v + s*f(u, time)``,
	``u_plus = u + s*f(v_plus, time)``,
	``(u_out, v_out) = C(s, frequency) (u_plus, v_plus)``.

	The returned array has shape ``(p, 4, 4)`` and equals the ordered product
	``C @ D(u update) @ D(v update)`` for every particle.
	"""
	particle_count = int(state.shape[0])
	first = _packed_physical(state, 0)
	second = _packed_physical(state, 2)
	# W(u) is the 2 x 2 Jacobian of the physical vector field for each
	# particle.  The second update must use W at the already updated v copy.
	first_jacobians = np.asarray(
		dynamics.particle_vector_field_jacobians(time, first),
		dtype=float,
	)
	updated_second = second + duration * np.asarray(
		dynamics.vector_field(time, first), dtype=float
	)
	second_jacobians = np.asarray(
		dynamics.particle_vector_field_jacobians(time, updated_second),
		dtype=float,
	)
	second_shear = _batched_identity(particle_count)
	second_shear[:, 2:, :2] = duration * first_jacobians
	first_shear = _batched_identity(particle_count)
	first_shear[:, :2, 2:] = duration * second_jacobians
	# The coupling is the final operation of the direct map, hence its
	# Jacobian multiplies both triangular shear factors from the left.
	coupling = gc_coupling_matrix(duration, frequency)
	return np.asarray(coupling @ first_shear @ second_shear)


def _adjoint_stage_particle_jacobians(
	dynamics: GuidingCenterDynamics,
	state: np.ndarray,
	*,
	duration: float,
	time: float,
	frequency: float,
) -> np.ndarray:
	"""Differentiate one adjoint GC stage for all particles independently.

	The adjoint reverses the direct-map operation order: it first applies the
	linear coupling ``C``, then updates ``u`` from the coupled ``v``, and finally
	updates ``v`` from the new ``u``.  The returned ``(p, 4, 4)`` array is
	therefore ``D(v update) @ D(u update) @ C`` for every particle.
	"""
	particle_count = int(state.shape[0])
	coupling = gc_coupling_matrix(duration, frequency)
	# Row vectors are used for the per-particle batch, so multiplying by C.T
	# applies the same coupling represented by a left product on column states.
	coupled = np.asarray(state @ coupling.T)
	first = _packed_physical(coupled, 0)
	second = _packed_physical(coupled, 2)
	second_jacobians = np.asarray(
		dynamics.particle_vector_field_jacobians(time, second),
		dtype=float,
	)
	updated_first = first + duration * np.asarray(
		dynamics.vector_field(time, second), dtype=float
	)
	first_jacobians = np.asarray(
		dynamics.particle_vector_field_jacobians(time, updated_first),
		dtype=float,
	)
	first_shear = _batched_identity(particle_count)
	first_shear[:, :2, 2:] = duration * second_jacobians
	second_shear = _batched_identity(particle_count)
	second_shear[:, 2:, :2] = duration * first_jacobians
	return np.asarray(second_shear @ first_shear @ coupling)


def _packed_particle_jacobians(jacobians: np.ndarray) -> np.ndarray:
	"""Assemble particlewise Jacobians in the global copy-major layout.

	``jacobians`` has shape ``(p, 4, 4)`` in per-particle order.  The result has
	shape ``(4*p, 4*p)`` matching the vector layout
	``(u_x[0:p], u_y[0:p], v_x[0:p], v_y[0:p])``.  Off-particle blocks remain
	zero because the guiding-centre vector field and duplicated coupling act on
	each particle independently.
	"""
	particle_count = int(jacobians.shape[0])
	result = np.zeros((4 * particle_count, 4 * particle_count), dtype=float)
	for particle_index in range(particle_count):
		indices = np.asarray(
			[
				particle_index,
				particle_count + particle_index,
				2 * particle_count + particle_index,
				3 * particle_count + particle_index,
			]
		)
		result[np.ix_(indices, indices)] = jacobians[particle_index]
	return result


def _analytic_bm4_map_jacobian(
	prepared: GCDoubledMaps,
	t: float,
	internal_input: np.ndarray,
	step: float,
) -> np.ndarray:
	"""Accumulate the analytic Jacobian of the complete twelve-stage map.

	The prepared formulation supplies particle count and coupling frequency.
	``particle_jacobian[p]`` starts as ``I_4`` and is left-multiplied by each
	stage factor in execution order, implementing the chain rule
	``J_total = J_12 @ ... @ J_1``.  The final matrix is repacked from
	``(p, 4, 4)`` to the global doubled-state shape ``(4*p, 4*p)``.
	"""
	dynamics = prepared.dynamics
	if not isinstance(dynamics, GuidingCenterDynamics):
		raise TypeError("Analytic implicit-BM4 Jacobians require GC dynamics.")
	particle_count = prepared.particle_count
	frequency = prepared.coupling_frequency
	assert frequency is not None
	# ``current`` is needed because every stage Jacobian is evaluated at that
	# stage's input, not at the original doubled state.
	current = np.asarray(internal_input, dtype=float).copy()
	particle_jacobian = _batched_identity(particle_count)
	current_time = float(t)
	for coefficient, order in zip(_BM4_STAGES, _BM4_ORDERS, strict=True):
		duration = float(coefficient * step)
		blocks = _particle_blocks(current, particle_count)
		if int(order) == 0:
			# A direct non-autonomous stage is evaluated at the end of its
			# signed substep, matching ``_advance_composition``.
			evaluation_time = current_time + duration
			factor = _direct_stage_particle_jacobians(
				dynamics,
				blocks,
				duration=duration,
				time=evaluation_time,
				frequency=frequency,
			)
			current = np.asarray(
				prepared.direct_map(duration, evaluation_time, current),
				dtype=float,
			)
		else:
			# The adjoint counterpart is evaluated at the substep's start.
			evaluation_time = current_time
			factor = _adjoint_stage_particle_jacobians(
				dynamics,
				blocks,
				duration=duration,
				time=evaluation_time,
				frequency=frequency,
			)
			current = np.asarray(
				prepared.adjoint_map(duration, evaluation_time, current),
				dtype=float,
			)
		# Left multiplication preserves the non-commuting chain-rule order.
		particle_jacobian = np.asarray(factor @ particle_jacobian)
		current_time += duration
	if not np.all(np.isfinite(particle_jacobian)):
		raise ValueError("The analytic BM4 map Jacobian contains non-finite values.")
	return _packed_particle_jacobians(particle_jacobian)


def _solve_reduced_projected_bm4_step(
	prepared: GCDoubledMaps,
	t: float,
	state: np.ndarray,
	step: float,
	*,
	absolute_tolerance: float,
	relative_tolerance: float,
	max_iterations: int,
	jacobian_relative_step: float,
	jacobian_method: NewtonJacobianMethod,
	nonlinear_solver: NonlinearSolver = "newton",
) -> _ProjectedBM4Step:
	"""Solve one physical BM4 step through Hairer's reduced multiplier equation.

	For ``m = state.size``, the only nonlinear unknown is ``mu in R^m``.  A
	residual evaluation constructs the doubled input
	``Y_hat = (state + mu, state - mu)``, applies the complete BM4 base map
	``M = Psi_h(Y_hat)``, and evaluates

	``r(mu) = G M + 2*mu``.

	Newton uses ``D r = G (D Psi_h) N + 2 I``.  Good Broyden solves the same
	equation from an inexpensive ``4 I`` initial Jacobian approximation.  Both
	paths accept a root when its infinity norm is at most
	``absolute_tolerance + relative_tolerance * max(1, ||state||_inf)``.

	The returned state is physical shape ``(m,)``; the returned doubled values
	have shape ``(2*m,)`` and support optional post-convergence observation.
	Failure to converge or a singular Newton matrix rejects the entire step.
	"""
	# ``value`` is z_n, the accepted physical state.  No doubled state persists
	# from the preceding integration step.
	value = np.asarray(state, dtype=float)
	if value.ndim != 1 or value.size == 0 or not np.all(np.isfinite(value)):
		raise ValueError("The BM4 physical state must be a finite, non-empty vector.")
	physical_size = value.size
	constraint, normal = _projection_matrices(physical_size)
	# The physical diagonal is the natural initial guess: E*z + N*0 = (z, z).
	multiplier = np.zeros_like(value)
	state_scale = max(1.0, float(np.linalg.norm(value, ord=np.inf)))
	threshold = absolute_tolerance + relative_tolerance * state_scale
	if nonlinear_solver == "broyden":
		def residual_function(
			candidate: np.ndarray,
		) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
			"""Evaluate the reduced projected-BM4 residual at one multiplier."""
			_, mapped = _bm4_evaluation(
				prepared,
				t,
				value,
				step,
				candidate,
			)
			# G*mapped is the copy mismatch after the full BM4 cycle;
			# 2*mu is G*N*mu from the symmetric output correction.
			return constraint @ mapped + 2.0 * candidate, (mapped, candidate)

		# At zero step Psi is the identity, so the exact reduced derivative is
		# G*I*N + 2I = 4I.  It is a natural low-cost initial Broyden matrix.
		result = _solve_broyden(
			residual_function,
			multiplier,
			4.0 * np.eye(physical_size),
			tolerance=threshold,
			max_iterations=max_iterations,
			context=(
				"BM4Implicit at "
				f"t={t:.16g} with step={step:.16g}"
			),
		)
		# The payload caches both values from the converged residual evaluation;
		# accepting the step therefore requires no extra twelve-stage map call.
		mapped, multiplier = result.payload
		# Hairer's output correction is M + N*mu.  Averaging its two copies
		# applies P = [I, I]/2; P*N = 0, so the multiplier cancels exactly.
		corrected = mapped + normal @ multiplier
		projected_state = (
			corrected[:physical_size] + corrected[physical_size:]
		) / 2.0
		return _ProjectedBM4Step(
			state=np.asarray(projected_state),
			multiplier=multiplier.copy(),
			internal_input=np.concatenate((value + multiplier, value - multiplier)),
			mapped=np.asarray(mapped).copy(),
			iterations=result.iterations,
			residual_evaluations=result.residual_evaluations,
			residual_norm=float(np.linalg.norm(result.residual, ord=np.inf)),
		)
	if nonlinear_solver != "newton":
		raise ValueError("Unknown nonlinear solver for implicit BM4.")

	for iteration in range(max_iterations + 1):
		# Each Newton iteration starts with exactly one full twelve-stage map
		# evaluation at the current multiplier.
		internal_input, mapped = _bm4_evaluation(
			prepared,
			t,
			value,
			step,
			multiplier,
		)
		residual = constraint @ mapped + 2.0 * multiplier
		residual_norm = float(np.linalg.norm(residual, ord=np.inf))
		if residual_norm <= threshold:
			# The corrected doubled output lies on the diagonal to nonlinear
			# tolerance; averaging exposes only its physical representative.
			corrected = mapped + normal @ multiplier
			projected_state = (
				corrected[:physical_size] + corrected[physical_size:]
			) / 2.0
			return _ProjectedBM4Step(
				state=np.asarray(projected_state),
				multiplier=multiplier.copy(),
				internal_input=np.asarray(internal_input).copy(),
				mapped=np.asarray(mapped).copy(),
				iterations=iteration,
				residual_evaluations=iteration + 1,
				residual_norm=residual_norm,
			)
		if iteration == max_iterations:
			break
		# J is the dense derivative of the complete doubled map with respect
		# to its doubled input, with shape (2m, 2m).
		base_jacobian = _bm4_map_jacobian(
			prepared,
			t,
			internal_input,
			step,
			relative_step=jacobian_relative_step,
			method=jacobian_method,
		)
		# Chain rule through Y_hat(mu) = E*z + N*mu gives
		# D_mu r = G*J*N + 2I, a reduced m x m matrix.
		residual_jacobian = (
			constraint @ base_jacobian @ normal + 2.0 * np.eye(physical_size)
		)
		try:
			# Solve K*c = r and subtract c; this is equivalent to the usual
			# Newton update K*delta = -r, mu <- mu + delta.
			correction = np.linalg.solve(residual_jacobian, residual)
		except np.linalg.LinAlgError as exc:
			raise RuntimeError(
				"The reduced BM4 projection Jacobian is singular at "
				f"t={t:.16g} with step={step:.16g}."
			) from exc
		multiplier = multiplier - correction

	raise RuntimeError(
		"BM4Implicit did not converge at "
		f"t={t:.16g} with step={step:.16g}: residual norm "
		f"{residual_norm:.3e} exceeds {threshold:.3e} after "
		f"{max_iterations} Newton iterations."
	)


@dataclass(slots=True)
class BM4Implicit(IntegrationMethod[_ProjectedBM4Step]):
	"""Configure physical BM4 with one reduced Hairer projection per cycle.

	The class has no projection-placement or state-extension modes.  Every call
	accepts and returns the physical guiding-centre vector in ``R^(2p)`` and uses
	a doubled ``R^(4p)`` value only while evaluating one complete BM4 cycle,
	where ``p`` is the particle count.

	Parameters
	----------
	coupling_frequency:
		Non-negative harmonic mixing frequency for the two internal GC copies.
		Defaults to zero (no mixing); the reduced Hairer projection remains active.
	newton_absolute_tolerance, newton_relative_tolerance:
		Positive terms in the reduced-residual stopping threshold
		``atol + rtol * max(1, ||z_n||_inf)``.  These controls also apply when
		``nonlinear_solver="broyden"``; their historical names are retained as
		part of the public API.
	newton_max_iterations:
		Maximum number of nonlinear corrections before rejecting a step.
	newton_jacobian_relative_step:
		Positive scale factor for centered differences.  It is used only by
		finite-difference Newton; analytic Newton and Broyden do not consult it.
	newton_jacobian_method:
		``"analytic"`` for the exact ordered GC stage product or
		``"finite_difference"`` for differentiation of the complete base map.
		This selection is consulted only by Newton.  Broyden starts from ``4 I``
		and updates that reduced residual-Jacobian approximation by secants.
	nonlinear_solver:
		``"newton"`` or ``"broyden"`` for the same reduced Hairer equation.
	progress:
		Whether the shared fixed-grid driver reports integration progress.
	step_observer:
		Optional callback receiving one :class:`ImplicitBM4IntegrationStep` per
		accepted main-grid step.  Output-only shadow steps are not observed.
	"""

	# Harmonic coupling of the two temporary physical copies.
	coupling_frequency: float = 0.0
	# State-scaled stopping threshold and common nonlinear correction limit.
	newton_absolute_tolerance: float = 1e-13
	newton_relative_tolerance: float = 1e-12
	newton_max_iterations: int = 12
	# Cube-root epsilon balances truncation and roundoff for centered differences.
	newton_jacobian_relative_step: float = float(np.cbrt(np.finfo(float).eps))
	newton_jacobian_method: NewtonJacobianMethod = "analytic"
	nonlinear_solver: NonlinearSolver = "newton"
	track_energy: bool = False
	# Fixed-grid presentation and optional accepted-step instrumentation.
	progress: bool = False
	step_observer: StepObserver | None = None

	# Resources owned by one run; excluded from constructor options.
	state_formulation: DoubledFormulation = field(init=False, repr=False, compare=False)
	energy_maps: GCDoubledMaps | None = field(init=False, repr=False, compare=False)
	formulation: GCDoubledMaps = field(init=False, repr=False, compare=False)

	def __post_init__(self) -> None:
		"""Validate and normalize all numeric and enumerated solver controls."""
		# Normalize controls before they are copied into individual runs.
		self.coupling_frequency = _nonnegative_finite(self.coupling_frequency, 'coupling_frequency')
		self.newton_absolute_tolerance = _positive_finite(self.newton_absolute_tolerance, 'newton_absolute_tolerance')
		self.newton_relative_tolerance = _positive_finite(self.newton_relative_tolerance, 'newton_relative_tolerance')
		self.newton_max_iterations = _positive_integer(self.newton_max_iterations, 'newton_max_iterations')
		self.newton_jacobian_relative_step = _positive_finite(self.newton_jacobian_relative_step, 'newton_jacobian_relative_step')
		if self.newton_jacobian_method not in NEWTON_JACOBIAN_METHODS:
			raise ValueError(
				"`newton_jacobian_method` must be 'analytic' or 'finite_difference'."
			)
		self.nonlinear_solver = _validate_nonlinear_solver(self.nonlinear_solver)

	def initialize(self, problem: InitialValueProblem, request: SimulationRequest) -> None:
		"""Bind the physical BM4 map, accepted event adapter and output metadata.
		Only two physical copies enter the temporary formulation. Per-run metric
		lists, scheduling and observer dispatch belong to the common coordinator.
		"""
		self.state_formulation = DoubledFormulation(problem, request.t_span[0], self.track_energy)
		self.energy_maps = GCDoubledMaps(problem, self.coupling_frequency, track_energy=True) if self.track_energy else None
		self.formulation = GCDoubledMaps(problem, self.coupling_frequency)
		metadata: dict[str, DiagnosticValue] = {
			"nonlinear_solver": self.nonlinear_solver,
			"nonlinear_absolute_tolerance": self.newton_absolute_tolerance,
			"nonlinear_relative_tolerance": self.newton_relative_tolerance,
			"nonlinear_max_iterations": self.newton_max_iterations,
			"newton_jacobian_relative_step": self.newton_jacobian_relative_step,
			"newton_jacobian_method": self.newton_jacobian_method,
			"coupling_frequency": self.coupling_frequency,
			"projection_solver_formulation": "bm4_implicit_reduced",
			"nonlinear_unknown_dimension": problem.initial_state.size,
			"nonlinear_solves_per_step": 1,
		}
		self.initial_state = self.state_formulation.initial_state
		self.metadata = metadata
		self.diagnostic_aliases = NEWTON_ALIASES

	def _solve(self, t: float, state: np.ndarray, h: float) -> _ProjectedBM4Step:
		"""Solve one complete physical map without observing or accumulating."""
		return _solve_reduced_projected_bm4_step(
			self.formulation, t, state, h,
			absolute_tolerance=self.newton_absolute_tolerance,
			relative_tolerance=self.newton_relative_tolerance,
			max_iterations=self.newton_max_iterations,
			jacobian_relative_step=self.newton_jacobian_relative_step,
			jacobian_method=self.newton_jacobian_method,
			nonlinear_solver=self.nonlinear_solver,
		)

	def advance(self, t: float, state: np.ndarray, h: float) -> StepResult[_ProjectedBM4Step]:
		"""Return one projected state and the already computed solve metrics."""
		physical = self.state_formulation.physical(state)
		result = self._solve(t, physical, h)
		tolerance = self.newton_absolute_tolerance + self.newton_relative_tolerance * max(
			1.0, float(np.linalg.norm(physical, ord=np.inf))
		)
		increment = None
		if self.energy_maps is not None:
			# Replay only the converged spatial stages. No diagnostic coordinate is
			# included in Newton/Broyden, and replay work is not nonlinear work.
			internal = np.concatenate((result.internal_input, np.zeros(self.state_formulation.particle_count)))
			tracked = _advance_composition(self.energy_maps, t, internal, h,
			    step_index=0, stage_observer=None, formulation_name="duplicated_with_energy", method_name=self.method_name)
			increment = tracked[2 * physical.size:] / 2.0
		after = self.state_formulation.finish(state, result.state, t + h, increment)
		return StepResult(after, {
			"nonlinear_iterations": result.iterations,
			"residual_evaluations": result.residual_evaluations,
			"nonlinear_residual_norms": result.residual_norm,
			"nonlinear_tolerances": tolerance,
			"projection_multiplier_norms": float(np.linalg.norm(result.multiplier, ord=np.inf)),
		}, result)

	def build_observation(self, info: StepInfo, step: StepResult[_ProjectedBM4Step]) -> ImplicitBM4IntegrationStep:
		"""Reconstruct converged stage snapshots only for a requested event."""
		result = step.details
		base_stages: list[IntegrationStage] = []
		observed_mapped = _advance_composition(
			self.formulation, info.time, result.internal_input, info.duration,
			step_index=info.index, stage_observer=base_stages.append,
			formulation_name="GCExtendedFormulation", method_name=type(self).__name__,
		)
		if not np.array_equal(observed_mapped, result.mapped):
			raise RuntimeError("The observed BM4 base cycle differs from the converged map.")

		def map_state(candidate: np.ndarray) -> np.ndarray:
			"""Evaluate the same physical map without collecting or observing."""
			return self._solve(info.time, candidate, info.duration).state

		return ImplicitBM4IntegrationStep(
			dynamics_name=self.formulation.dynamics_name, method_name=type(self).__name__,
			step_index=info.index, start_time=info.time,
			time=info.time + info.duration, duration=info.duration,
			state_before=self.state_formulation.physical(info.state_before).copy(), state_after=result.state.copy(),
			map_state=map_state, dynamics=self.formulation.dynamics,
			formulation_name="bm4_implicit_reduced", nonlinear_solver=self.nonlinear_solver,
			newton_iterations=result.iterations, residual_evaluations=result.residual_evaluations,
			newton_residual_norm=result.residual_norm,
			newton_tolerance=float(step.statistics["nonlinear_tolerances"]),
			projection_multiplier_norm=float(step.statistics["projection_multiplier_norms"]),
			coupling_frequency=self.coupling_frequency,
			multiplier=result.multiplier.copy(), base_stages=tuple(base_stages),
		)



__all__ = ["BM4Implicit"]
