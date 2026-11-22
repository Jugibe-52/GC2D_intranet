"""Accumulate physical-map geometry on a fixed step/horizon grid."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any
import json
import tarfile

import numpy as np

from diagnostics.studies.midpoint_geometry_sweep import digest, write_json, finish_record, matrix_products
from dynamics.coordinates import GCCoordinates
from methods.palindromic.core.composition import Composition
from methods.palindromic.core.midpoint import midpoint_step
from methods.palindromic.maps import GCDoubledMaps
from studies.main_campaign import CampaignConfig, load_field, prepare_case
from studies.midpoint_geometry import METHODS, RECIPES, midpoint_geometry

COUNTS = (20, 40, 80, 160)
HORIZONS = (10, 20, 40, 80)
GROUPS = {
	'periodic_candidates8': [1, 2, 60, 64, 65, 72, 76, 83],
	'chaotic8': [12, 15, 18, 21, 24, 27, 30, 33],
}
RUN_ID = 'midpoint-jacobian-product-20261122-rho0015'
STUDY_PREFIX = 'studies/midpoint_geometry'


def integrate_case(maps: GCDoubledMaps, recipe: Composition, state: np.ndarray,
		period: float, steps_per_cycle: int, horizons: tuple[int, ...]) -> dict[str, np.ndarray]:
	"""Save physical Jacobians, accumulate their product and reverse each horizon.

	States use component-major xy packing; particle diagnostics have shape
	(step, particle). Reversal begins at the forward endpoint clock. All forward
	horizons share one prefix, but each backward trajectory starts independently.
	"""
	if (type(steps_per_cycle) is not int or steps_per_cycle < 1 or not horizons
			or any(type(t) is not int or t < 1 for t in horizons)
			or tuple(sorted(set(horizons))) != horizons
			or not np.isfinite(period) or period <= 0):
		raise ValueError('Use positive step counts, a finite positive period and increasing integer horizons.')
	initial: np.ndarray = np.asarray(state, dtype=float).copy()
	if initial.ndim != 1 or initial.size == 0 or initial.size % 2 or not np.all(np.isfinite(initial)):
		raise ValueError('Expected a finite nonempty component-major xy state.')
	n: int = initial.size // 2
	steps: int = horizons[-1] * steps_per_cycle
	h: float = period / steps_per_cycle
	local: np.ndarray = np.empty((steps, n))
	direct: np.ndarray = np.empty_like(local)
	jacobians: np.ndarray = np.empty((steps, n, 2, 2))
	cycle_states: np.ndarray = np.empty((horizons[-1]+1, initial.size))
	cycle_states[0] = initial
	forward_seconds: np.ndarray = np.empty(len(horizons))
	current: np.ndarray = initial.copy()
	started: float = perf_counter()
	for index in range(steps):
		value: dict[str, np.ndarray] = midpoint_geometry(maps, recipe, index*h, current, h, reverse=False)
		current = value['state']
		local[index] = value['separation_det']
		direct[index] = value['det_defect']
		jacobians[index] = value['tangent']
		if (not np.all(np.isfinite(current)) or not np.all(np.isfinite(local[index]))
				or not np.all(np.isfinite(direct[index])) or not np.all(np.isfinite(jacobians[index]))):
			raise RuntimeError(f'Nonfinite state or Jacobian at step {index+1}.')
		if (index+1) % steps_per_cycle == 0:
			cycle: int = (index+1) // steps_per_cycle
			cycle_states[cycle] = current
			if cycle in horizons:
				forward_seconds[horizons.index(cycle)] = perf_counter()-started
	returned: np.ndarray = np.empty((len(horizons), initial.size))
	backward_seconds: np.ndarray = np.empty(len(horizons))
	for ci, horizon in enumerate(horizons):
		back: np.ndarray = cycle_states[horizon].copy()
		started = perf_counter()
		for index in range(horizon*steps_per_cycle, 0, -1):
			back = np.asarray(midpoint_step(maps, recipe, index*h, back, -h).state)
			if not np.all(np.isfinite(back)):
				raise RuntimeError(f'Nonfinite reverse state at horizon {horizon}, step {index}.')
		returned[ci] = back
		backward_seconds[ci] = perf_counter()-started
	indices: np.ndarray = np.asarray(horizons)*steps_per_cycle
	vectors: np.ndarray = (returned-initial).reshape(len(horizons), 2, n).transpose(0, 2, 1)
	started = perf_counter()
	products: dict[str, np.ndarray] = matrix_products(jacobians, indices)
	product_seconds: np.ndarray = np.asarray(perf_counter()-started)
	return dict(initial_states=initial.reshape(2, n).T, h=np.asarray(h),
		steps_per_cycle=np.asarray(steps_per_cycle), horizons=np.asarray(horizons),
		times=np.arange(steps+1)*h, local_signed_defect=local, local_direct_defect=direct,
		local_jacobians=jacobians, cycle_states=cycle_states, returned_states=returned,
		return_vectors=vectors, return_norm=np.linalg.norm(vectors, axis=-1),
		**products, product_seconds=product_seconds,
		forward_seconds=forward_seconds, backward_seconds=backward_seconds)


def run_sweep(root: Path, output: Path, *, counts: tuple[int, ...] = COUNTS,
		horizons: tuple[int, ...] = HORIZONS) -> dict[str, Any]:
	"""Run the accepted groups and freeze inputs before saving new scientific data."""
	root = root.resolve()
	output = output.resolve()
	if output.exists() and any(output.iterdir()):
		raise FileExistsError('Use a fresh run directory; completed and recoverable records are immutable.')
	if not counts or len(set(counts)) != len(counts) or any(type(c) is not int or c < 1 for c in counts):
		raise ValueError('Step counts must be distinct positive integers.')
	selection_path: Path = root.parent/'TFM_redaccion/datos/campana_referencia.json'
	selection: dict[str, Any] = json.loads(selection_path.read_text())
	campaign_path: Path = root/'notebooks/developements/main_reflected_campaign'/selection['campaign_id']/'campaign.json'
	campaign: dict[str, Any] = json.loads(campaign_path.read_text())
	field_path: Path = root/'data/potential/V1/PHI_2.h5'
	groups_path: Path = root.parent/'TFM_redaccion/datos/evolucion_energetica.json'
	groups_record: dict[str, Any] = json.loads(groups_path.read_text())
	if (selection['campaign_id'] != 'main-20261117-rho-div20'
			or selection['source_sha256'] != digest(field_path)
			or selection['representative_rho'] != .015
			or campaign['code_sha256'] != selection['code_sha256']
			or any(groups_record['groups'][key] != ids for key, ids in GROUPS.items())):
		raise ValueError('Reference field, campaign or group selection does not match the accepted study.')
	field: Any = load_field(field_path)
	problem: Any
	metadata: dict[str, Any]
	problem, _, metadata = prepare_case(CampaignConfig(cycles=1), field, campaign['particles'],
		.015, METHODS[0], code_sha256=digest(Path(__file__)))
	ids: list[int] = [i for group in GROUPS.values() for i in group]
	particles: list[dict[str, Any]] = campaign['particles']['particles']
	if [p['id'] for p in particles] != list(range(1, 85)) or metadata['cycle_duration'] != 1:
		raise ValueError('Unexpected particle order or physical-cycle duration.')
	initial: np.ndarray = problem.initial_state.reshape(2, 84)[:, np.asarray(ids)-1].ravel()
	maps: GCDoubledMaps = GCDoubledMaps(problem.dynamics.vector_field, initial.size,
		coordinates=GCCoordinates(), jacobian=problem.dynamics.particle_vector_field_jacobians,
		coupling_frequency=None)
	output.mkdir(parents=True, exist_ok=True)
	with tarfile.open(output/'source.tar.gz', 'w:gz') as archive:
		for path in sorted((root/'src').rglob('*.py')):
			archive.add(path, arcname=str(path.relative_to(root)))
		for directory in ('scripts', 'tests', 'notebooks/04_08_midpoint_geometry'):
			for path in sorted((root/directory).glob('*midpoint*')):
				if path.is_file():
					archive.add(path, arcname=str(path.relative_to(root)))
		for path in sorted((root/'notebooks/04_08_midpoint_geometry').glob('*.ipynb')):
			archive.add(path, arcname=str(path.relative_to(root)))
		for path in (campaign_path, selection_path, groups_path):
			archive.add(path, arcname='inputs/'+path.name)
	parameters: dict[str, Any] = dict(schema_version=2, run_id=output.name,
		campaign_id=selection['campaign_id'], campaign_sha256=digest(campaign_path),
		source_sha256=digest(field_path), reference_selection_sha256=digest(selection_path),
		group_selection_sha256=digest(groups_path), source_archive_sha256=digest(output/'source.tar.gz'),
		methods=list(METHODS), steps_per_cycle=list(counts), horizons=list(horizons),
		groups=GROUPS, particle_ids=ids, initial_positions=initial.reshape(2, -1).T.tolist(),
		particle_colors=[particles[i-1]['color'] for i in ids], rho=.015, rho_convention='base_cell_width',
		cycle_duration=1., initial_time=0., spatial_extension='reflected', spatial_period=2.,
		magnetic_field=1.5, interpolation_order=3, source_field_indices=[15], sigma=None,
		precision='float64', backend='numpy', numpy_version=np.__version__, position_wrapping='none',
		return_norm='euclidean per particle', area_evaluation='abs(det(A_(N-1) @ ... @ A_0)-1)',
		jacobian_evaluation='analytic physical step tangent including lift and mean',
		product_arithmetic='decimal, exact conversion of float64 local entries',
		product_decimal_digits=80, product_validation_decimal_digits=120,
		timing_scope='forward diagnostics, independent reverse loops, separate matrix products; excludes validation, I/O and network',
		classification='Candidate groups; this experiment does not establish periodicity or chaos.')
	write_json(output/'parameters.json', parameters)
	write_json(output/'status.json', dict(status='running', completed_cases=[]))
	completed: list[str] = []
	for method, recipe in zip(METHODS, RECIPES, strict=True):
		for count in counts:
			name: str = f'{method}-{count}.npz'
			print(f'Computing {method}, h=1/{count}, horizons={horizons}', flush=True)
			arrays: dict[str, np.ndarray] = integrate_case(maps, recipe, initial, 1., count, horizons)
			np.savez_compressed(output/name, **arrays)
			completed.append(name)
			write_json(output/'status.json', dict(status='running', completed_cases=completed))
			print(f'Saved {name}; numerical seconds={arrays["forward_seconds"][-1]+arrays["backward_seconds"].sum():.3f}', flush=True)
	return finish_record(output)


__all__ = ['COUNTS', 'HORIZONS', 'GROUPS', 'RUN_ID', 'STUDY_PREFIX', 'integrate_case', 'run_sweep']
