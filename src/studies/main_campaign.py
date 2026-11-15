"""Reproducible 84-particle, eight-method reflected-potential campaign."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import file_digest, sha256
import json
from pathlib import Path
from typing import Any

import numpy as np

from contracts.methods.base import NumericalMethod
from dynamics.gc import GuidingCenterDynamics
from execution.options import ExecutionOptions
from methods.palindromic.abba2_implicit import ABBA2Implicit
from methods.palindromic.abba2_midpoint import ABBA2Midpoint
from methods.palindromic.bm4_implicit import BM4Implicit
from methods.palindromic.bm4_midpoint import BM4Midpoint
from methods.runge_kutta.implicit_midpoint import ImplicitMidpoint
from methods.runge_kutta.rk2 import RK2
from methods.runge_kutta.rk4 import RK4
from methods.runge_kutta.gauss_legendre import GaussLegendre4
from potential.potential import Potential
from potential.load import GC2DH5Metadata
from problem import InitialValueProblem
from request import SimulationRequest
from solvers.nonlinear import NewtonSolver
from states.history import States

METHODS = ('ABBA2Midpoint', 'ABBA2Implicit', 'BM4Midpoint', 'BM4Implicit',
           'ImplicitMidpoint', 'RK2', 'RK4', 'GaussLegendre4')
RHOS = tuple(i / 20 for i in range(11))
SOURCE_SHA256 = '4bbd873cd05992e0a615d66ebbf4d96b38c8b3c36be2ca62932e66af7e6d4ced'
OPTIONS = ExecutionOptions(backend='jax', device='cpu')


@dataclass(frozen=True)
class CampaignConfig:
	"""Integer cycles and aligned recording; the production protocol is 5000/40/10."""
	cycles: int = 5000
	steps_per_cycle: int = 40
	samples_per_cycle: int = 10

	def __post_init__(self) -> None:
		for value in asdict(self).values():
			if type(value) is not int or value <= 0:
				raise ValueError('Campaign counts must be positive integers.')
		if self.steps_per_cycle % self.samples_per_cycle or self.stride % 2:
			raise ValueError('Each saved interval must contain an even number of complete steps.')

	@property
	def stride(self) -> int:
		return self.steps_per_cycle // self.samples_per_cycle


def read_particles(path: Path) -> dict[str, Any]:
	"""Preserve exact physical positions, IDs and colors; never regenerate seeds."""
	record: dict[str, Any] = json.loads(path.read_text())
	particles: list[dict[str, Any]] = record['particles']
	positions: np.ndarray = np.asarray([p['position_m'] for p in particles], dtype=float)
	if ([p['id'] for p in particles] != list(range(1, 85)) or positions.shape != (84, 2)
			or not np.isfinite(positions).all() or record['source_sha256'] != SOURCE_SHA256):
		raise ValueError('Expected the verified original 84-particle record.')
	return dict(particles=particles, families=[{'parent': f['parent'], 'ids': f['ids']}
		for f in record['families']], source_sha256=record['source_sha256'],
		input_sha256=sha256(path.read_bytes()).hexdigest())


def load_field(path: Path) -> Potential:
	"""Load the same measured data with physical-cycle scaling and reflected edges."""
	with path.open('rb') as stream:
		digest: str = file_digest(stream, 'sha256').hexdigest()
	if digest != SOURCE_SHA256:
		raise ValueError('The measured potential fingerprint differs from the agreed source.')
	return Potential.load(path, indx=(15,), sigma=None, spatial_rescaling=True,
		temporal_rescaling=True, interpolation_order=3, spatial_extension='reflected')


def campaign_method(name: str, step: float) -> NumericalMethod:
	"""Keep identical analytic Newton controls across implicit methods."""
	if name not in METHODS:
		raise ValueError(f'Unknown campaign method: {name}.')
	if name == 'RK2':
		return RK2(step=step)
	if name == 'RK4':
		return RK4(step=step)
	if name == 'ABBA2Midpoint':
		return ABBA2Midpoint(step=step)
	if name == 'BM4Midpoint':
		return BM4Midpoint(step=step)
	solver: NewtonSolver = NewtonSolver(absolute_tolerance=1e-12, relative_tolerance=1e-11,
		max_iterations=40, jacobian_method='analytic')
	if name == 'ABBA2Implicit':
		return ABBA2Implicit(step=step, solver=solver)
	if name == 'BM4Implicit':
		return BM4Implicit(step=step, solver=solver)
	if name == 'ImplicitMidpoint':
		return ImplicitMidpoint(step=step, solver=solver)
	return GaussLegendre4(step=step, solver=solver)


def prepare_case(config: CampaignConfig, field: Potential, particles: dict[str, Any],
		rho: float, name: str, *, code_sha256: str) -> tuple[InitialValueProblem, NumericalMethod, dict[str, Any]]:
	"""Build one jointly integrated set of 84 independent guiding centers."""
	if rho not in RHOS or name not in METHODS:
		raise ValueError('The case is outside the agreed sweep.')
	m: Any = field.metadata
	if not isinstance(m, GC2DH5Metadata) or field.spatial_extension != 'reflected':
		raise ValueError('Expected the reflected measured potential.')
	physical: np.ndarray = np.array([p['position_m'] for p in particles['particles']])
	xy: np.ndarray = (physical - np.array((m.source_x[0], m.source_y[0]))) / m.spatial_scale
	initial: np.ndarray = GuidingCenterDynamics.coordinates.pack_components(xy[:, 0], xy[:, 1])
	rho_m: float = rho * .06 / (2 * np.pi)
	dynamics: GuidingCenterDynamics = GuidingCenterDynamics(field, rho=rho_m / m.spatial_scale, B=1.5)
	period: float = 1 / float(field.frequencies[0])
	# Derive outputs from integration indices, so no shadow steps are requested.
	times: np.ndarray = np.arange(config.cycles * config.samples_per_cycle + 1) * config.stride * (period / config.steps_per_cycle)
	request: SimulationRequest = SimulationRequest((0., float(times[-1])), times)
	problem: InitialValueProblem = InitialValueProblem(dynamics, initial,
		layout=GuidingCenterDynamics.coordinates, request=request)
	metadata: dict[str, Any] = dict(study='main_reflected_campaign', schema_version=1,
		config=asdict(config), method=name, rho=rho, rho_m=rho_m,
		rho_runtime=dynamics.rho, magnetic_field=1.5, source_sha256=SOURCE_SHA256,
		particle_record=particles, code_sha256=code_sha256,
		spatial_extension='reflected', spatial_period=field.spatial_period,
		position_wrapping='none', initial_positions=xy.tolist(),
		spatial_scale_m=m.spatial_scale, time_scale_seconds=m.time_scale,
		cycle_duration=period, source_field_indices=[15], sigma=None,
		energy_quadrature='composite_simpson_at_accepted_endpoints',
		energy_units='normalized Hamiltonian; time in physical forcing cycles',
		projection_units='normalized physical coordinates',
		projection_sample_convention='accepted step ending at saved time; no initial multiplier',
		newton_absolute_tolerance=1e-12, newton_relative_tolerance=1e-11, newton_max_iterations=40)
	return problem, campaign_method(name, period / config.steps_per_cycle), metadata


def validate_case(solution: States, metadata: dict[str, Any]) -> None:
	"""Reject missing samples, backend mismatches and incomplete diagnostics."""
	c: CampaignConfig = CampaignConfig(**metadata['config'])
	d: Any = solution.diagnostics
	if (solution.states.shape != (c.cycles * c.samples_per_cycle + 1, 168)
			or solution.states.dtype != np.float64 or not np.isfinite(solution.states).all()
			or d.get('step_count') != c.cycles * c.steps_per_cycle
			or d.get('output_interpolation_count') != 0 or d.get('execution_backend') != 'jax'):
		raise ValueError('Incomplete campaign solution or wrong execution backend.')
	expected: np.ndarray = np.asarray(metadata['initial_positions']).T.ravel()
	if not np.array_equal(solution.states[0], expected):
		raise ValueError('Particle identities or initial states changed.')
	for key in ('physical_hamiltonian', 'hamiltonian_time_derivative', 'energy_integral', 'energy_balance', 'energy_quadrature_difference'):
		values: np.ndarray = np.asarray(d[key])
		if values.shape != (84, solution.t.size) or not np.isfinite(values).all():
			raise ValueError(f'Incomplete diagnostic: {key}.')
	if metadata['method'] in ('ABBA2Implicit', 'BM4Implicit'):
		for key, shape in (('projection_multipliers_sampled', (solution.t.size-1, 2, 84)),
				('projection_particle_norms', (solution.t.size-1, 84)),
				('projection_particle_block_max', (solution.t.size-1, 84))):
			if np.shape(d[key]) != shape or not np.isfinite(d[key]).all():
				raise ValueError(f'Incomplete projection diagnostic: {key}.')


__all__ = ['CampaignConfig', 'METHODS', 'RHOS', 'OPTIONS', 'read_particles',
	'load_field', 'campaign_method', 'prepare_case', 'validate_case']
