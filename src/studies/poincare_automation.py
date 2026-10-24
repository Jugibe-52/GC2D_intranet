"""Fixed 48-particle Poincare studies in physical or independently rescaled units."""

from __future__ import annotations

from contracts.studies.poincare_automation import _NewtonControls

from studies._validation import require_uncoupled_bm4

from dynamics.gc import GuidingCenterDynamics

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from hashlib import file_digest, sha256
from itertools import product
import json
from pathlib import Path
from typing import Any, Iterator, Literal, TYPE_CHECKING, cast

import numpy as np

from execution.options import ExecutionOptions
from problem import InitialValueProblem
from request import SimulationRequest
from contracts.states.history import DiagnosticValue
from execution.execution import Execution

from states.generators import ranked_radial_star
from contracts.methods.base import NumericalMethod
from methods.classical.gauss_legendre import GaussLegendre4
from methods.classical.rk4 import RK4
from methods.extended.bm4 import BM4Implicit, BM4Midpoint
from potential.load import GC2DH5Metadata
from potential.potential import Potential
from simulation.runner import simulate
from states.history import States
from diagnostics.io.persistence import StoredSolution


if TYPE_CHECKING:
	from typing import Mapping


PARTICLE_SET = "ranked-star-40-gap-probes-8-v1"
METHODS = ("BM4Midpoint", "BM4Implicit", "RK4", "GaussLegendre4")
PROBE_FRACTIONS = ((.4407, .5817), (.4417, .5896), (.6073, .5675), (.6030, .5753),
	(.4317, .3856), (.4376, .3814), (.5749, .3911), (.5790, .3918))
PROBE_GAPS = ("northwest", "northwest", "northeast", "northeast",
	"southwest", "southwest", "southeast", "southeast")


def fixed_particle_fractions() -> np.ndarray:
	"""Return owned (48, xy) fractions from the reference viewer's lower cell corner."""
	star: np.ndarray = ranked_radial_star(center=(.5, .5), outer_radius=.425,
		particles=40, arms=8, first_angle=0.)
	assert star is not None
	x: np.ndarray
	y: np.ndarray
	x, y = GuidingCenterDynamics.coordinates.positions(star)
	return np.vstack((np.column_stack((x, y)), np.asarray(PROBE_FRACTIONS)))


def canonical_json(value: Any) -> str:
	"""Serialize resolved JSON data without rounding scientific inputs."""
	return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _number(value: Any, name: str) -> float:
	if isinstance(value, (bool, str)) or not isinstance(value, (int, float, np.number)):
		raise ValueError(f"{name} must be a finite number.")
	number: float = float(value)
	if not np.isfinite(number):
		raise ValueError(f"{name} must be a finite number.")
	return number


def _validated_parameter_values(values: tuple[float, ...], name: str) -> tuple[float, ...]:
	"""Freeze one distinct rho or signed-field sweep without coercing strings."""
	numbers: tuple[float, ...] = tuple(_number(value, name) for value in values)
	if not numbers or len(set(numbers)) != len(numbers):
		raise ValueError(f"{name} must contain unique values and cannot be empty.")
	invalid: bool = any(value < 0 for value in numbers) if name == "rhos" else any(value == 0 for value in numbers)
	if invalid:
		raise ValueError("rho must be nonnegative and B must be nonzero.")
	return numbers


def _validated_methods(values: tuple[str, ...]) -> tuple[str, ...]:
	"""Freeze the ordered, nonempty set of supported study methods."""
	methods: tuple[str, ...] = tuple(values)
	if not methods or len(set(methods)) != len(methods) or any(method not in METHODS for method in methods):
		raise ValueError(f"Select distinct supported methods: {METHODS}.")
	return methods


def _validate_positive_count(value: int, name: str) -> None:
	"""Retain this study's Python-integer-only count policy."""
	if isinstance(value, bool) or not isinstance(value, int) or value < 1:
		raise ValueError(f"{name} must be a positive integer.")


def _positive_number(value: float, name: str) -> float:
	"""Normalize a positive study scale with the strict numeric-input policy."""
	number: float = _number(value, name)
	if number <= 0:
		raise ValueError(f"{name} must be positive.")
	return number


def _validated_coupling(value: float, methods: tuple[str, ...]) -> float:
	"""Check the legacy coupling field against the selected method family."""
	coupling: float = _number(value, "coupling_frequency")
	if coupling < 0 or (coupling != 0 and any(method in ("RK4", "GaussLegendre4") for method in methods)):
		raise ValueError("Coupling must be nonnegative and zero for classical methods.")
	return coupling


def _validated_sigma(value: float) -> float:
	"""Normalize an explicitly supplied nonnegative smoothing width."""
	width: float = _number(value, "sigma")
	if width < 0:
		raise ValueError("sigma must be nonnegative.")
	return width


@dataclass(frozen=True)
class AutomationConfig:
	"""Resolved common controls; requested rho means 2*pi*rho_m/reference_length."""

	rhos: tuple[float, ...]
	methods: tuple[str, ...]
	magnetic_fields: tuple[float, ...] = (1.5,)
	source: str = "data/potential/V1/PHI_2.h5"
	source_index: int = 15
	spatial_rescaling: bool = True
	temporal_rescaling: bool = True
	reference_length: float = .06
	sigma: float | None = None
	interpolation_order: int = 3
	cycles: int = 5000
	steps_per_cycle: int = 50
	coupling_frequency: float = 0.0
	newton_absolute_tolerance: float = 1e-12
	newton_relative_tolerance: float = 1e-11
	newton_max_iterations: int = 40
	newton_jacobian_relative_step: float = float(np.cbrt(np.finfo(float).eps))
	backend: str = "scipy"
	device: str = "cpu"
	device_index: int = 0
	cycles_per_frame: int = 25
	field_grid_size: int = 64
	vector_grid_size: int = 17

	def __post_init__(self) -> None:
		"""Freeze JSON sequences and validate inputs before accessing source data."""
		for name in ("rhos", "magnetic_fields"):
			object.__setattr__(self, name, _validated_parameter_values(getattr(self, name), name))
		methods: tuple[str, ...] = _validated_methods(self.methods)
		object.__setattr__(self, "methods", methods)
		for name in ("source_index", "cycles", "steps_per_cycle", "newton_max_iterations",
				"cycles_per_frame", "field_grid_size", "vector_grid_size", "interpolation_order"):
			_validate_positive_count(getattr(self, name), name)
		if self.interpolation_order not in (3, 5):
			raise ValueError("Use cubic or quintic interpolation for this study.")
		if min(self.field_grid_size, self.vector_grid_size) < 2:
			raise ValueError("Viewer grids require at least two samples per axis.")
		for name in ("spatial_rescaling", "temporal_rescaling"):
			if not isinstance(getattr(self, name), bool):
				raise ValueError(f"{name} must be Boolean.")
		for name in ("reference_length", "newton_absolute_tolerance",
				"newton_relative_tolerance", "newton_jacobian_relative_step"):
			object.__setattr__(self, name, _positive_number(getattr(self, name), name))
		object.__setattr__(self, "coupling_frequency", _validated_coupling(self.coupling_frequency, methods))
		if self.sigma is not None:
			object.__setattr__(self, "sigma", _validated_sigma(self.sigma))
		if not isinstance(self.source, str) or not self.source.strip():
			raise ValueError("source must name a local HDF5 file.")
		self.execution_options()

	def execution_options(self) -> ExecutionOptions:
		return ExecutionOptions(backend=cast(Literal["scipy", "jax"], self.backend),
			device=cast(Literal["cpu", "gpu"], self.device), device_index=self.device_index)

	def cases(self) -> tuple[AutomationCase, ...]:
		return tuple(AutomationCase(rho, B, method)
			for rho, B, method in product(self.rhos, self.magnetic_fields, self.methods))

	def to_dict(self) -> dict[str, Any]:
		return cast(dict[str, Any], json.loads(canonical_json(asdict(self))))


@dataclass(frozen=True)
class AutomationCase:
	"""One member of a resolved Cartesian sweep."""

	rho: float
	B: float
	method: str

	@property
	def key(self) -> str:
		return sha256(canonical_json(asdict(self)).encode()).hexdigest()[:20]


@dataclass(frozen=True)
class AutomationField:
	"""Immutable loaded samples and the fingerprint of their exact source."""

	potential: Potential
	source_path: Path
	source_sha256: str


def load_automation_field(config: AutomationConfig, project_root: Path) -> AutomationField:
	path: Path = Path(config.source).expanduser()
	path = (project_root / path).resolve() if not path.is_absolute() else path.resolve()
	with path.open("rb") as stream:
		digest: str = file_digest(stream, "sha256").hexdigest()
	potential: Potential = Potential.load(path, indx=(config.source_index,),
		spatial_rescaling=config.spatial_rescaling, temporal_rescaling=config.temporal_rescaling,
		sigma=config.sigma, interpolation_order=config.interpolation_order)
	return AutomationField(potential, path, digest)


@dataclass(frozen=True)
class PreparedAutomationCase:
	"""One vectorized problem plus its exact reproducibility record."""

	config: AutomationConfig
	case: AutomationCase
	problem: InitialValueProblem
	request: SimulationRequest
	metadata: dict[str, Any]

	@property
	def dynamics(self) -> GuidingCenterDynamics:
		return cast(GuidingCenterDynamics, self.problem.dynamics)


def prepare_automation_case(config: AutomationConfig, case: AutomationCase,
		field: AutomationField) -> PreparedAutomationCase:
	"""Apply the physical change of variables once, without historical-clock adapters."""
	if case not in config.cases():
		raise ValueError("The case is not part of this configuration.")
	p = field.potential
	m = p.metadata
	if not isinstance(m, GC2DH5Metadata) or len(m.source_frequencies) != 1:
		raise ValueError("A measured single-frequency potential is required.")
	if (m.spatial_rescaling != config.spatial_rescaling or m.temporal_rescaling != config.temporal_rescaling
			or tuple(m.source_field_indices) != (config.source_index,) or m.sigma != config.sigma
			or p.interpolation_order != config.interpolation_order):
		raise ValueError("Loaded field controls do not match the configuration.")
	length = len(m.source_x) * float(m.source_x[1] - m.source_x[0])
	origin: np.ndarray = np.array((m.source_x[0], m.source_y[0]))
	fractions: np.ndarray = fixed_particle_fractions()
	physical_xy = origin + length * fractions
	runtime_origin: np.ndarray = origin if config.spatial_rescaling else np.zeros(2)
	xy = (physical_xy - runtime_origin) / m.spatial_scale
	rho_m = case.rho * config.reference_length / (2*np.pi)
	period = 1 / float(m.source_frequencies[0])
	cycle_duration = period / m.time_scale
	initial: np.ndarray = GuidingCenterDynamics.coordinates.pack_components(xy[:, 0], xy[:, 1])
	dynamics: GuidingCenterDynamics = GuidingCenterDynamics(p, rho=rho_m / m.spatial_scale, B=case.B)
	problem: InitialValueProblem = InitialValueProblem(dynamics, initial, layout=GuidingCenterDynamics.coordinates)
	times = cycle_duration * np.arange(config.cycles + 1, dtype=float)
	request: SimulationRequest = SimulationRequest(t_span=(0., float(times[-1])),
		 output_times=times)
	metadata: dict[str, Any] = dict(schema_version=1, study="poincare_automation", particle_set=PARTICLE_SET,
		config=config.to_dict(), case=asdict(case), source_sha256=field.source_sha256,
		source_path=str(field.source_path), source_field_indices=m.source_field_indices.tolist(),
		source_frequencies=m.source_frequencies.tolist(), source_frequency_unit=m.source_frequency_unit,
		frequency_convention="cycles", time_convention=m.time_convention,
		spatial_scale_m=m.spatial_scale, time_scale_seconds=m.time_scale,
		physical_period_seconds=period, runtime_cycle_duration=cycle_duration,
		cell_bounds_m=[*origin.tolist(), length],
		cell_bounds=[p.grid.x0, p.grid.y0, p.grid.period],
		rho_m=rho_m, rho_runtime=dynamics.rho, B=case.B,
		coupling_runtime=config.coupling_frequency/cycle_duration,
		particle_ids=list(range(1, 49)), initial_fractions=fractions.tolist(),
		initial_positions=xy.tolist(), initial_positions_m=physical_xy.tolist(),
		position_wrapping="none", execution_options=asdict(config.execution_options()),
		expected_step_count=config.cycles*config.steps_per_cycle,
		initial_state_saved=True, samples_per_cycle=1)
	metadata["input_digest"] = sha256(canonical_json(metadata).encode()).hexdigest()
	return PreparedAutomationCase(config, case, problem, request, metadata)


def automation_method(prepared: PreparedAutomationCase) -> NumericalMethod:
	"""Construct a canonical integrator without doubled-copy coupling."""
	c: AutomationConfig
	name: str
	c, name = prepared.config, prepared.case.method
	step: float = float(prepared.metadata["runtime_cycle_duration"]) / c.steps_per_cycle
	# Historical configurations remain readable, but cannot select new coupled runs.
	require_uncoupled_bm4(c.coupling_frequency)
	if name == "RK4":
		return RK4(step=step)
	if name == "BM4Midpoint":
		return BM4Midpoint(step=step)
	kwargs: _NewtonControls = dict(newton_absolute_tolerance=c.newton_absolute_tolerance,
		newton_relative_tolerance=c.newton_relative_tolerance,
		newton_max_iterations=c.newton_max_iterations,
		newton_jacobian_relative_step=c.newton_jacobian_relative_step,
		newton_jacobian_method="analytic")
	# Explicit branches keep each constructor's numerical contract visible.
	if name == "BM4Implicit":
		return BM4Implicit(step=step, **kwargs)
	return GaussLegendre4(step=step, **kwargs)


def validate_automation_solution(solution: States, prepared: PreparedAutomationCase) -> None:
	"""Require complete cycle samples, exact initial identity and method diagnostics."""
	c: AutomationConfig
	d: Mapping[str, DiagnosticValue]
	method: str
	c, d, method = prepared.config, solution.diagnostics, prepared.case.method
	if (solution.states.shape != (96, c.cycles+1) or solution.states.dtype != np.float64
			or not np.isfinite(solution.states).all()
			or not np.array_equal(solution.t, prepared.request.output_times)
			or not np.array_equal(solution.states[:, 0], prepared.problem.initial_state)
			or d.get("step_count") != c.cycles*c.steps_per_cycle
			or d.get("output_interpolation_count") != 0):
		raise ValueError("Incomplete or incompatible 48-particle cycle solution.")
	if c.backend == "jax" and any(d.get(k) != v for k, v in {
		"execution_backend": "jax", "execution_device": c.device, "execution_mode": "device_resident"}.items()):
		raise ValueError("Execution diagnostics differ from the requested backend/device.")
	if method.startswith("BM4") and d.get("coupling_frequency") != prepared.metadata["coupling_runtime"]:
		raise ValueError("BM4 coupling diagnostics differ from the configured runtime rate.")
	if method == "BM4Midpoint" and d.get("projection_kind") != "arithmetic_mean":
		raise ValueError("Expected arithmetic-mean BM4 projection.")
	if method == "BM4Implicit" and d.get("projection_solver_formulation") != "bm4_implicit_reduced":
		raise ValueError("Expected reduced implicit BM4 projection.")
	if method in ("BM4Implicit", "GaussLegendre4"):
		expected: dict[str, object] = dict(nonlinear_solver="newton", nonlinear_solves_per_step=1,
			nonlinear_absolute_tolerance=c.newton_absolute_tolerance,
			nonlinear_relative_tolerance=c.newton_relative_tolerance,
			nonlinear_max_iterations=c.newton_max_iterations, newton_jacobian_method="analytic",
			newton_jacobian_relative_step=c.newton_jacobian_relative_step)
		if any(d.get(k) != v for k, v in expected.items()):
			raise ValueError("Implicit solver diagnostics differ from the configured controls.")
	elif "nonlinear_solver" in d or d.get("nonlinear_unknown_dimension", 0) != 0:
		raise ValueError("Explicit methods must not report nonlinear work.")
	if method == "GaussLegendre4" and (d.get("stage_count") != 2 or d.get("designed_order") != 4):
		raise ValueError("Expected fourth-order two-stage Gauss-Legendre.")
	if method in ("RK4", "GaussLegendre4") and any(k.startswith("projection_") for k in d):
		raise ValueError("Classical methods must not report projection.")


@contextmanager
def automation_execution_context(config: AutomationConfig, execution: Execution | None = None) -> Iterator[None]:
	"""Opt local JAX jobs into float64 temporarily and check the device before work."""
	if config.backend == "jax" and (execution is None or type(execution) is Execution):
		try:
			import jax
		except ImportError as exc:
			raise RuntimeError("JAX is unavailable; install the jax extra or explicitly select scipy.") from exc
		from execution._jax import resolve_device
		with jax.experimental.enable_x64():
			resolve_device(config.execution_options())
			yield
	else:
		yield


def run_automation_case(prepared: PreparedAutomationCase, *,
		execution: Execution | None = None) -> StoredSolution:
	"""Integrate all particles jointly and retain only cycle states/scalar diagnostics."""
	executor: Execution = Execution() if execution is None else execution
	with automation_execution_context(prepared.config, execution):
		solution: States = simulate(prepared.problem, automation_method(prepared), prepared.request,
			execution=executor, options=prepared.config.execution_options())
	validate_automation_solution(solution, prepared)
	diagnostics: dict[str, DiagnosticValue] = {k: v for k, v in solution.diagnostics.items()
		if not isinstance(v, np.ndarray)}
	trimmed: States = States(layout=solution.layout, t=solution.t, states=solution.states, initial_state=solution.initial_state, diagnostics=diagnostics)
	metadata = {**prepared.metadata, "execution_executor": f"{type(executor).__module__}.{type(executor).__qualname__}"}
	return StoredSolution(trimmed, metadata, None)


def validate_saved_automation(stored: StoredSolution, prepared: PreparedAutomationCase) -> None:
	"""Check the full immutable input record, not just its user-facing label."""
	if any(stored.metadata.get(k) != v for k, v in prepared.metadata.items()):
		raise ValueError("Saved case belongs to different scientific inputs or provenance.")
	if not isinstance(stored.metadata.get("execution_executor"), str):
		raise ValueError("Saved case lacks executor provenance.")
	validate_automation_solution(stored.solution, prepared)


__all__ = ["AutomationConfig", "AutomationCase", "AutomationField", "PreparedAutomationCase",
	"PARTICLE_SET", "METHODS", "PROBE_FRACTIONS", "PROBE_GAPS", "fixed_particle_fractions",
	"canonical_json", "load_automation_field", "prepare_automation_case", "automation_method",
	"validate_automation_solution", "run_automation_case", "validate_saved_automation",
	"automation_execution_context"]
