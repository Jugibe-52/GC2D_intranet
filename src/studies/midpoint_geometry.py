"""Measure local and accumulated geometry of the actual averaged-copy maps.

States use component-major packing [x_1,...,x_N,y_1,...,y_N]. Tangents retain
independent (N,2,2) particle blocks. All reported coordinates stay unwrapped.
The study reuses the canonical composition and its analytic tangent; it never
estimates derivatives by perturbing the potential or by wrapping trajectories.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import tarfile
from time import perf_counter
from typing import Any, Callable

import numpy as np

from dynamics.coordinates import GCCoordinates
from methods.palindromic.abba2_midpoint import ABBA2Midpoint
from methods.palindromic.bm4_midpoint import BM4Midpoint
from methods.palindromic.core.composition import Composition
from methods.palindromic.core.jacobians import composition_jacobian
from methods.palindromic.core.midpoint import midpoint_step
from methods.palindromic.core.projection import solve_projection
from methods.palindromic.maps import GCDoubledMaps
from methods.projection import ReducedProjection
from solvers.nonlinear import NewtonSolver
from studies.main_campaign import CampaignConfig, load_field, prepare_case

METHODS: tuple[str, ...] = ('ABBA2Midpoint', 'BM4Midpoint')
RECIPES: tuple[Composition, ...] = (ABBA2Midpoint.recipe, BM4Midpoint.recipe)
COUNTS: tuple[int, ...] = (20, 40, 80, 160)
PHASES: tuple[float, ...] = (0., .25, .5, .75)
CAMPAIGN: str = 'main-20261117-rho-div20'
LOCAL_METRICS: tuple[str, ...] = ('det_defect', 'return_norm', 'separation_norm',
	'separation_det', 'identity_residual', 'symplectic_defect', 'return_tangent_norm')
LIFT: np.ndarray = np.concatenate((np.eye(2), np.eye(2)), axis=0)
NORMAL: np.ndarray = np.concatenate((np.eye(2), -np.eye(2)), axis=0)
MEAN: np.ndarray = LIFT.T / 2
DIFFERENCE: np.ndarray = NORMAL.T / 2
J: np.ndarray = np.array([[0., -1.], [1., 0.]])


def midpoint_geometry(maps: GCDoubledMaps, recipe: Composition, time: float,
		state: np.ndarray, step: float, *, reverse: bool = True) -> dict[str, np.ndarray]:
	"""Differentiate the complete physical map and its discarded half-difference.

	The identity det(D Phi)-1 = det(D e) follows from the cross symplectic form
	of the doubled map; it is recorded separately from the directly computed
	determinant so floating-point cancellation is measurable rather than hidden.
	"""
	result: Any = midpoint_step(maps, recipe, time, state, step)
	base: np.ndarray = np.asarray(composition_jacobian(maps, result.trace))
	diagonal_tangent: np.ndarray = base @ LIFT
	tangent: np.ndarray = MEAN @ diagonal_tangent
	separation_tangent: np.ndarray = DIFFERENCE @ diagonal_tangent
	first: np.ndarray
	second: np.ndarray
	first, second = maps.state_formulation.split_copies(result.trace.state)
	separation: np.ndarray = (np.asarray(first) - np.asarray(second)) / 2
	det_defect: np.ndarray = np.linalg.det(tangent) - 1
	separation_det: np.ndarray = np.linalg.det(separation_tangent)
	values: dict[str, np.ndarray] = dict(state=np.asarray(result.state), tangent=tangent,
		separation_tangent=separation_tangent, separation=separation,
		det_defect=det_defect, separation_det=separation_det,
		identity_residual=det_defect-separation_det,
		separation_norm=np.max(np.abs(separation.reshape(2, -1)), axis=0),
		symplectic_defect=np.linalg.norm(np.swapaxes(tangent, -1, -2) @ J @ tangent - J,
			axis=(-1, -2)) / np.linalg.norm(J))
	if reverse:
		back: Any = midpoint_step(maps, recipe, time+step, result.state, -step)
		back_base: np.ndarray = np.asarray(composition_jacobian(maps, back.trace))
		back_tangent: np.ndarray = MEAN @ back_base @ LIFT
		values['return_norm'] = np.max(np.abs((np.asarray(back.state)-state).reshape(2, -1)), axis=0)
		values['return_vector'] = np.asarray(back.state)-state
		values['return_tangent_norm'] = np.linalg.norm(back_tangent @ tangent - np.eye(2), axis=(-1, -2))
	return values


def projected_geometry(maps: GCDoubledMaps, recipe: Composition, time: float,
		state: np.ndarray, step: float, tolerance: float) -> dict[str, np.ndarray]:
	"""Evaluate the real solver return and the implicit-equation tangent.

	The tangent assumes the projection equations hold exactly at the computed
	root. It does not differentiate finite-iteration stopping decisions. Actual
	forward/backward output and residuals quantify those algebraic errors.
	"""
	solver: NewtonSolver = NewtonSolver(absolute_tolerance=tolerance,
		relative_tolerance=tolerance, max_iterations=40, jacobian_method='analytic')
	result: Any = solve_projection(maps, recipe, solver, ReducedProjection(), time, state, step)
	base: np.ndarray = np.asarray(composition_jacobian(maps, result.trace))
	constraint: np.ndarray = NORMAL.T
	kernel: np.ndarray = constraint @ base @ NORMAL + 2*np.eye(2)
	mu_tangent: np.ndarray = -np.linalg.solve(kernel, constraint @ base @ LIFT)
	tangent: np.ndarray = MEAN @ base @ (LIFT + NORMAL @ mu_tangent)
	back: Any = solve_projection(maps, recipe, solver, ReducedProjection(),
		time+step, result.state, -step)
	residual: np.ndarray = np.asarray(result.mapped[:state.size]-result.mapped[state.size:]+2*result.multiplier)
	back_residual: np.ndarray = np.asarray(back.mapped[:state.size]-back.mapped[state.size:]+2*back.multiplier)
	count: int = state.size//2
	return dict(det_defect=np.linalg.det(tangent)-1,
		return_norm=np.max(np.abs((np.asarray(back.state)-state).reshape(2, -1)), axis=0),
		residual_norm=np.max(np.abs(residual.reshape(2, -1)), axis=0),
		backward_residual_norm=np.max(np.abs(back_residual.reshape(2, -1)), axis=0),
		iterations=np.full(count, result.iterations),
		backward_iterations=np.full(count, back.iterations),
		tolerance=np.full(count, result.stats.tolerance), tangent=tangent,
		state=np.asarray(result.state))


def _statistics(values: np.ndarray) -> dict[str, Any]:
	"""Summarize magnitudes while preserving signs in the raw archive."""
	finite: np.ndarray = np.abs(np.asarray(values, dtype=float).ravel())
	finite = finite[np.isfinite(finite)]
	if finite.size == 0:
		return dict(median=None, p95=None, maximum=None, rms=None, finite_count=0)
	return dict(median=float(np.median(finite)), p95=float(np.percentile(finite, 95)),
		maximum=float(np.max(finite)), rms=float(np.sqrt(np.mean(finite**2))), finite_count=int(finite.size))


def _local_grid(maps: GCDoubledMaps, state: np.ndarray, period: float) -> dict[str, np.ndarray]:
	count: int = state.size//2
	shape: tuple[int, ...] = (len(METHODS), len(COUNTS), len(PHASES), count)
	arrays: dict[str, np.ndarray] = {name: np.full(shape, np.nan) for name in LOCAL_METRICS}
	timings: np.ndarray = np.empty(shape[:3])
	for mi, recipe in enumerate(RECIPES):
		for hi, steps in enumerate(COUNTS):
			for pi, phase in enumerate(PHASES):
				started: float = perf_counter()
				values: dict[str, np.ndarray] = midpoint_geometry(maps, recipe, phase*period, state, period/steps)
				timings[mi, hi, pi] = perf_counter()-started
				for name in LOCAL_METRICS:
					arrays[name][mi, hi, pi] = values[name]
	arrays.update(methods=np.asarray(METHODS), steps_per_cycle=np.asarray(COUNTS),
		h=period/np.asarray(COUNTS), phases=np.asarray(PHASES), initial_states=state.reshape(2, -1).T,
		runtime_seconds=timings)
	return arrays


def _finite_difference_audit(dynamics: Any, state: np.ndarray, period: float) -> dict[str, np.ndarray]:
	"""Cross-check three physical particle tangents over four perturbation scales."""
	indices: np.ndarray = np.array([0, 40, 83])
	selected: np.ndarray = state.reshape(2, -1)[:, indices].ravel()
	maps: GCDoubledMaps = GCDoubledMaps(dynamics.vector_field, selected.size,
		coordinates=GCCoordinates(), jacobian=dynamics.particle_vector_field_jacobians,
		coupling_frequency=None)
	deltas: np.ndarray = np.array([1e-4, 1e-5, 1e-6, 1e-7])
	errors: np.ndarray = np.empty((2, deltas.size, indices.size))
	for mi, recipe in enumerate(RECIPES):
		exact: np.ndarray = midpoint_geometry(maps, recipe, period/2, selected, period/40)['tangent']
		for di, delta in enumerate(deltas):
			columns: list[np.ndarray] = []
			for axis in np.eye(selected.size):
				plus: np.ndarray = np.asarray(midpoint_step(maps, recipe, period/2, selected+delta*axis, period/40).state)
				minus: np.ndarray = np.asarray(midpoint_step(maps, recipe, period/2, selected-delta*axis, period/40).state)
				columns.append((plus-minus)/(2*delta))
			dense: np.ndarray = np.column_stack(columns)
			for pi in range(indices.size):
				component_indices: np.ndarray = np.array([pi, pi+indices.size])
				errors[mi, di, pi] = np.linalg.norm(dense[np.ix_(component_indices, component_indices)]-exact[pi], ord=np.inf)
	return dict(methods=np.asarray(METHODS), particle_ids=indices+1, deltas=deltas,
		tangent_infinity_error=errors, phase=np.array(.5), h=np.array(period/40))


def _calibration() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
	"""Use the harmonic oscillator and a smooth nonlinear Hamiltonian control."""
	state: np.ndarray = np.array([.7, -.4, .2, .3, .9, -.6])
	h: np.ndarray = 2.**(-np.arange(7))
	def oscillator(_time: float, values: np.ndarray) -> np.ndarray:
		result: np.ndarray = (J @ values.reshape(2, -1)).ravel()
		return result
	def oscillator_jacobian(_time: float, values: np.ndarray) -> np.ndarray:
		return np.broadcast_to(J, (values.size//2, 2, 2))
	def nonlinear(_time: float, values: np.ndarray) -> np.ndarray:
		x: np.ndarray
		y: np.ndarray
		x, y = values.reshape(2, -1)
		return np.concatenate((-y*(1+.5*x*x), x*(1+.5*y*y)))
	def nonlinear_jacobian(_time: float, values: np.ndarray) -> np.ndarray:
		x: np.ndarray
		y: np.ndarray
		x, y = values.reshape(2, -1)
		return np.stack((np.stack((-x*y, -1-.5*x*x), axis=-1),
			np.stack((1+.5*y*y, x*y), axis=-1)), axis=-2)
	fields: tuple[tuple[Callable[..., np.ndarray], Callable[..., np.ndarray]], ...] = (
		(oscillator, oscillator_jacobian), (nonlinear, nonlinear_jacobian))
	shape: tuple[int, ...] = (2, 2, h.size, state.size//2)
	arrays: dict[str, np.ndarray] = {name: np.full(shape, np.nan) for name in LOCAL_METRICS}
	for fi, (field, jacobian) in enumerate(fields):
		maps: GCDoubledMaps = GCDoubledMaps(field, state.size, coordinates=GCCoordinates(),
			jacobian=jacobian, coupling_frequency=None)
		for mi, recipe in enumerate(RECIPES):
			for hi, step in enumerate(h):
				values: dict[str, np.ndarray] = midpoint_geometry(maps, recipe, 0., state, float(step))
				for name in LOCAL_METRICS:
					arrays[name][fi, mi, hi] = values[name]
	expected_det: np.ndarray = h**6/64
	expected_return: np.ndarray = expected_det[:, None]*np.max(np.abs(state.reshape(2, -1)), axis=0)
	arrays.update(fields=np.array(['oscillator', 'quartic_hamiltonian']), methods=np.asarray(METHODS),
		h=h, initial_states=state.reshape(2, -1).T,
		abba2_expected_det_defect=expected_det, abba2_expected_return_norm=expected_return)
	report: dict[str, Any] = dict(
		oscillator_hamiltonian='(x^2+y^2)/2', nonlinear_hamiltonian='(x^2+y^2)/2+x^2*y^2/4',
		abba2_det_max_absolute_discrepancy=float(np.max(np.abs(arrays['det_defect'][0, 0]-expected_det[:, None]))),
		abba2_return_max_absolute_discrepancy=float(np.max(np.abs(arrays['return_norm'][0, 0]-expected_return))),
		identity_max_absolute_residual=float(np.max(np.abs(arrays['identity_residual']))))
	return arrays, report


def _implicit_controls(maps: GCDoubledMaps, state: np.ndarray, period: float) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
	tolerances: np.ndarray = np.array([1e-8, 1e-11, 1e-14])
	metrics: tuple[str, ...] = ('det_defect', 'return_norm', 'residual_norm',
		'backward_residual_norm', 'iterations', 'backward_iterations', 'tolerance')
	shape: tuple[int, ...] = (2, 3, 4, 4, state.size//2)
	arrays: dict[str, np.ndarray] = {name: np.full(shape, np.nan) for name in metrics}
	timings: np.ndarray = np.empty(shape[:-1])
	failures: list[dict[str, Any]] = []
	for mi, recipe in enumerate(RECIPES):
		for ti, tolerance in enumerate(tolerances):
			for hi, steps in enumerate(COUNTS):
				for pi, phase in enumerate(PHASES):
					started: float = perf_counter()
					try:
						values: dict[str, np.ndarray] = projected_geometry(maps, recipe, phase*period,
							state, period/steps, float(tolerance))
					except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
						timings[mi, ti, hi, pi] = perf_counter()-started
						failures.append(dict(method=METHODS[mi].replace('Midpoint', 'Implicit'),
							tolerance=float(tolerance), steps_per_cycle=steps, phase=phase, error=str(error)))
						continue
					timings[mi, ti, hi, pi] = perf_counter()-started
					for name in metrics:
						arrays[name][mi, ti, hi, pi] = values[name]
	arrays.update(methods=np.array(['ABBA2Implicit', 'BM4Implicit']), tolerances=tolerances,
		steps_per_cycle=np.asarray(COUNTS), h=period/np.asarray(COUNTS), phases=np.asarray(PHASES),
		runtime_seconds=timings)
	return arrays, failures


def _accumulation(maps: GCDoubledMaps, state: np.ndarray, period: float,
		cycles: tuple[int, ...]) -> dict[str, np.ndarray]:
	"""Sum local log determinants and reverse independently from each horizon.

	This avoids multiplying ill-conditioned accumulated tangent matrices.
	The second log sum uses the exact structural identity with det(D e), and
	keeps the directly evaluated Jacobian sum as an independent cross-check.
	"""
	steps: int = max(cycles)*40
	h: float = period/40
	count: int = state.size//2
	arrays: dict[str, np.ndarray] = dict(methods=np.asarray(METHODS), horizons=np.asarray(cycles),
		h=np.asarray(h), times=np.arange(steps+1)*h, particle_ids=np.arange(1, count+1),
		local_det_defect=np.empty((2, steps, count)), local_separation_det=np.empty((2, steps, count)),
		logdet_direct=np.empty((2, steps+1, count)), logdet_structure=np.empty((2, steps+1, count)),
		cycle_states=np.empty((2, steps//40+1, 2*count)),
		return_norm=np.empty((2, len(cycles), count)), returned_states=np.empty((2, len(cycles), 2*count)),
		orientation_flips=np.zeros((2, count), dtype=int), runtime_seconds=np.empty(2))
	for mi, recipe in enumerate(RECIPES):
		started: float = perf_counter()
		current: np.ndarray = state.copy()
		arrays['logdet_direct'][mi, 0] = 0
		arrays['logdet_structure'][mi, 0] = 0
		arrays['cycle_states'][mi, 0] = current
		for index in range(steps):
			values: dict[str, np.ndarray] = midpoint_geometry(maps, recipe, index*h, current, h, reverse=False)
			current = values['state']
			defect: np.ndarray = values['det_defect']
			stable: np.ndarray = values['separation_det']
			arrays['local_det_defect'][mi, index] = defect
			arrays['local_separation_det'][mi, index] = stable
			arrays['orientation_flips'][mi] += (defect <= -1)
			if np.any(defect <= -1) or np.any(stable <= -1):
				raise RuntimeError('An orientation-reversing or singular step invalidates the positive log determinant.')
			arrays['logdet_direct'][mi, index+1] = arrays['logdet_direct'][mi, index]+np.log1p(defect)
			arrays['logdet_structure'][mi, index+1] = arrays['logdet_structure'][mi, index]+np.log1p(stable)
			if (index+1) % 40 == 0:
				arrays['cycle_states'][mi, (index+1)//40] = current
			if (index+1) % 400 == 0:
				print(f'{METHODS[mi]}: forward {(index+1)//40} cycles ({perf_counter()-started:.1f}s)', flush=True)
		for ci, horizon in enumerate(cycles):
			back: np.ndarray = arrays['cycle_states'][mi, horizon].copy()
			for index in range(horizon*40, 0, -1):
				back = np.asarray(midpoint_step(maps, recipe, index*h, back, -h).state)
			arrays['returned_states'][mi, ci] = back
			arrays['return_norm'][mi, ci] = np.max(np.abs((back-state).reshape(2, -1)), axis=0)
		arrays['runtime_seconds'][mi] = perf_counter()-started
		print(f'{METHODS[mi]}: accumulation and returns finished in {arrays["runtime_seconds"][mi]:.1f}s', flush=True)
	return arrays


def _file_hash(path: Path) -> str:
	return sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, data: dict[str, Any]) -> None:
	path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def _rows(arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
	rows: list[dict[str, Any]] = []
	for mi, method in enumerate(arrays['methods']):
		for hi, h in enumerate(arrays['h']):
			row: dict[str, Any] = dict(method=str(method), h=float(h), steps_per_cycle=int(arrays['steps_per_cycle'][hi]))
			for metric in LOCAL_METRICS:
				row[metric] = _statistics(arrays[metric][mi, hi])
			rows.append(row)
	return rows


def _slopes(arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
	"""Record adjacent slopes as observations, without asserting asymptotic order."""
	rows: list[dict[str, Any]] = []
	for mi, method in enumerate(arrays['methods']):
		for hi in range(len(arrays['h'])-1):
			for metric in ('det_defect', 'return_norm', 'separation_norm'):
				coarse: dict[str, Any] = _statistics(arrays[metric][mi, hi])
				fine: dict[str, Any] = _statistics(arrays[metric][mi, hi+1])
				rows.append(dict(method=str(method), coarse_h=float(arrays['h'][hi]),
					fine_h=float(arrays['h'][hi+1]), metric=metric,
					**{key: (float(np.log2(coarse[key]/fine[key])) if fine[key] > 0 else None)
						for key in ('median', 'p95', 'maximum')}))
	return rows


def run_study(project_root: Path, output_directory: Path, *,
		accumulation_cycles: tuple[int, ...] = (10, 100),
		skip_accumulation: bool = False) -> dict[str, Any]:
	"""Run the fixed protocol and write raw arrays, summaries and provenance.

	Persistence is intentionally local here; a caller can archive the complete
	manifest through the project's result-storage workflow without recomputation.
	"""
	if not accumulation_cycles or any(type(c) is not int or c < 1 for c in accumulation_cycles):
		raise ValueError('Accumulation horizons must be positive integer cycle counts.')
	root: Path = project_root.resolve()
	output: Path = output_directory.resolve()
	output.mkdir(parents=True, exist_ok=True)
	started: float = perf_counter()
	campaign_path: Path = root/'notebooks/developements/main_reflected_campaign'/CAMPAIGN/'campaign.json'
	campaign: dict[str, Any] = json.loads(campaign_path.read_text())
	selection_path: Path = root.parent/'TFM_redaccion/datos/campana_referencia.json'
	selection: dict[str, Any] = json.loads(selection_path.read_text())
	field_path: Path = root/'data/potential/V1/PHI_2.h5'
	if (selection['campaign_id'] != CAMPAIGN or selection['representative_rho'] != .015
			or selection['particle_count'] != 84 or selection['source_sha256'] != _file_hash(field_path)
			or [p['id'] for p in campaign['particles']['particles']] != list(range(1, 85))):
		raise ValueError('The selected thesis reference, source or particle identities changed.')
	# Freeze scientific code and literal inputs before any integration. The
	# measured HDF5 is an existing repository input identified by its SHA256.
	with tarfile.open(output/'source.tar.gz', 'w:gz') as archive:
		for source_path in sorted((root/'src').rglob('*.py')):
			archive.add(source_path, arcname=str(source_path.relative_to(root)))
		for script in ('run_midpoint_geometry.py', 'archive_midpoint_geometry.py'):
			script_path: Path = root/'scripts'/script
			if script_path.exists():
				archive.add(script_path, arcname=f'scripts/{script}')
		archive.add(campaign_path, arcname='inputs/campaign.json')
		archive.add(selection_path, arcname='inputs/campana_referencia.json')
		archive.add(root/'tests/test_midpoint_geometry.py', arcname='tests/test_midpoint_geometry.py')
	field: Any = load_field(field_path)
	problem: Any
	metadata: dict[str, Any]
	problem, _, metadata = prepare_case(CampaignConfig(cycles=1), field, campaign['particles'],
		.015, 'ABBA2Midpoint', code_sha256=_file_hash(Path(__file__)))
	state: np.ndarray = problem.initial_state
	period: float = metadata['cycle_duration']
	dynamics: Any = problem.dynamics
	maps: GCDoubledMaps = GCDoubledMaps(dynamics.vector_field, state.size,
		coordinates=GCCoordinates(), jacobian=dynamics.particle_vector_field_jacobians,
		coupling_frequency=None)
	local: dict[str, np.ndarray] = _local_grid(maps, state, period)
	local['particle_ids'] = np.array([p['id'] for p in campaign['particles']['particles']])
	np.savez_compressed(output/'local.npz', **local)
	print('Local 84-particle grid complete', flush=True)
	fd_audit: dict[str, np.ndarray] = _finite_difference_audit(dynamics, state, period)
	np.savez_compressed(output/'fd_audit.npz', **fd_audit)
	# Include both sides, exact joins and corners without mixing these probes
	# with the fixed 84 initial positions in the primary statistics.
	seam_points: np.ndarray = np.array([(edge+offset, ordinate) for edge in (0., 1.)
		for offset in (-1e-8, 0., 1e-8) for ordinate in (.2, .5, .8)] +
		[(abscissa, edge+offset) for edge in (0., 1.) for offset in (-1e-8, 0., 1e-8)
		for abscissa in (.2, .5, .8)] + [(x, y) for x in (0., 1.) for y in (0., 1.)])
	seam_state: np.ndarray = seam_points.T.ravel()
	seam_maps: GCDoubledMaps = GCDoubledMaps(dynamics.vector_field, seam_state.size,
		coordinates=GCCoordinates(), jacobian=dynamics.particle_vector_field_jacobians,
		coupling_frequency=None)
	seams: dict[str, np.ndarray] = _local_grid(seam_maps, seam_state, period)
	np.savez_compressed(output/'seams.npz', **seams)
	calibration: dict[str, np.ndarray]
	calibration_report: dict[str, Any]
	calibration, calibration_report = _calibration()
	np.savez_compressed(output/'calibration.npz', **calibration)
	controls: dict[str, np.ndarray]
	failures: list[dict[str, Any]]
	controls, failures = _implicit_controls(maps, state, period)
	np.savez_compressed(output/'implicit_controls.npz', **controls)
	implicit_rows: list[dict[str, Any]] = []
	for mi, method in enumerate(controls['methods']):
		for ti, tolerance in enumerate(controls['tolerances']):
			implicit_rows.append(dict(method=str(method), absolute_tolerance=float(tolerance),
				relative_tolerance=float(tolerance),
				**{metric: _statistics(controls[metric][mi, ti]) for metric in
					('det_defect', 'return_norm', 'residual_norm')}))
	code_paths: list[Path] = [Path(__file__), root/'src/methods/palindromic/core/midpoint.py',
		root/'src/methods/palindromic/core/composition.py', root/'src/methods/palindromic/core/jacobians.py',
		root/'src/methods/palindromic/maps.py', root/'src/studies/main_campaign.py',
		root/'src/methods/palindromic/abba2_implicit.py', root/'src/methods/palindromic/bm4_midpoint.py']
	summary: dict[str, Any] = dict(schema_version=1, run_id=output.name,
		created_utc=datetime.now(timezone.utc).isoformat(), study='midpoint_geometry',
		provenance=dict(campaign_id=CAMPAIGN, campaign_manifest_sha256=_file_hash(campaign_path),
			reference_selection_sha256=_file_hash(selection_path), source_archive_sha256=_file_hash(output/'source.tar.gz'),
			source_sha256=_file_hash(field_path), reference_campaign_code_sha256=campaign.get('code_sha256'),
			code_files={str(path.relative_to(root)): _file_hash(path) for path in code_paths},
			backend='numpy', precision='float64', numpy_version=np.__version__,
			rho=.015, rho_convention='base_cell_width', magnetic_field=1.5,
			cycle_duration=period, spatial_extension='reflected', spatial_period=2.,
			interpolation_order=3, source_field_indices=[15], sigma=None,
			particle_ids=local['particle_ids'].tolist(), initial_positions=local['initial_states'].tolist(),
			particle_colors=[p['color'] for p in campaign['particles']['particles']],
			position_wrapping='none', phases=list(PHASES), steps_per_cycle=list(COUNTS),
			local_sampling='identical original initial states at each prescribed forcing phase',
			analytic_tangent='canonical retained-stage composition_jacobian, followed by lift/mean',
			implicit_tangent='implicit projection equations differentiated at computed root; not solver stopping algorithm'),
		local_rows=_rows(local), local_slopes=_slopes(local), seam_rows=_rows(seams), calibration=calibration_report,
		finite_difference_audit=dict(particle_ids=fd_audit['particle_ids'].tolist(),
			deltas=fd_audit['deltas'].tolist(), tangent_infinity_error=fd_audit['tangent_infinity_error'].tolist()),
		implicit_control_rows=implicit_rows, implicit_failures=failures,
		status='local_complete_accumulation_pending', timings_seconds=dict(local_calibration_controls=perf_counter()-started))
	_write_json(output/'summary.json', summary)
	print('Local controls and summary saved', flush=True)
	if not skip_accumulation:
		accumulation: dict[str, np.ndarray] = _accumulation(maps, state, period, accumulation_cycles)
		np.savez_compressed(output/'accumulation.npz', **accumulation)
		accumulation_rows: list[dict[str, Any]] = []
		for mi, method in enumerate(METHODS):
			for ci, horizon in enumerate(accumulation_cycles):
				index: int = horizon*40
				logs: np.ndarray = accumulation['logdet_direct'][mi, index]
				stable_logs: np.ndarray = accumulation['logdet_structure'][mi, index]
				accumulation_rows.append(dict(method=method, cycles=horizon,
					logdet=_statistics(logs), logdet_signed_minimum=float(np.min(logs)),
					logdet_signed_maximum=float(np.max(logs)),
					area_ratio_minimum=float(np.exp(np.min(logs))), area_ratio_maximum=float(np.exp(np.max(logs))),
					return_norm=_statistics(accumulation['return_norm'][mi, ci]),
					logdet_identity_discrepancy=float(np.max(np.abs(logs-stable_logs))),
					orientation_flips=int(np.sum(accumulation['orientation_flips'][mi]))))
		summary['accumulation_rows'] = accumulation_rows
		summary['timings_seconds']['accumulation_by_method'] = dict(zip(METHODS, accumulation['runtime_seconds'].tolist(), strict=True))
		summary['status'] = 'complete'
	summary['timings_seconds']['total'] = perf_counter()-started
	_write_json(output/'summary.json', summary)
	manifest: dict[str, Any] = dict(schema_version=1, run_id=output.name,
		status=summary['status'], created_utc=datetime.now(timezone.utc).isoformat(),
		files={path.name: dict(sha256=_file_hash(path), size_bytes=path.stat().st_size)
			for path in sorted(output.iterdir()) if path.is_file() and path.name != 'manifest.json'},
		storage_status='local; remote archival must be verified separately')
	_write_json(output/'manifest.json', manifest)
	return summary


__all__ = ['midpoint_geometry', 'projected_geometry', 'run_study']
