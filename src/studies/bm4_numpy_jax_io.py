"""Prepare paired cases, recover Modal calls and archive measured results in B2."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import tarfile
from typing import Any

import numpy as np

from diagnostics.io.paths import DEFAULT_RESULTS_BUCKET
from diagnostics.io.persistence import load_solution, save_solution
from diagnostics.io.storage import ArtifactStore
from execution._modal_worker import encode_job
from execution.execution_modal import _write_record
from execution.options import ExecutionOptions
from studies.main_campaign import CampaignConfig, load_field, prepare_case, read_particles

STUDY_PREFIX = 'studies/bm4_numpy_jax'
METHODS = ('BM4Implicit', 'GaussLegendre4', 'RK2')


def prepare_study(root: Path, directory: Path, *, cycles: tuple[int, ...] = (10, 30, 90),
		repetitions: int = 3) -> dict[str, Any]:
	"""Freeze verified inputs and source bytes before any remote submission."""
	if not cycles or any(type(c) is not int or c <= 0 for c in cycles) or len(set(cycles)) != len(cycles):
		raise ValueError('Choose distinct positive integer cycle counts.')
	if type(repetitions) is not int or repetitions < 3:
		raise ValueError('At least three warmed repetitions are required.')
	selection_path: Path = root.parent / 'TFM_redaccion/datos/campana_referencia.json'
	selection: dict[str, Any] = json.loads(selection_path.read_text())
	particles: dict[str, Any] = read_particles(root / 'data/particles/expanded_84.json')
	selection_paths: list[Path] = [root / 'data/particles' / name for name in
		('chaotic_candidates.json', 'periodic_candidates_8.json')]
	reference_particles: dict[int, dict[str, Any]] = {p['id']: p for p in particles['particles']}
	selected: dict[int, dict[str, Any]] = {}
	selection_records: list[dict[str, Any]] = []
	for path in selection_paths:
		group: dict[str, Any] = json.loads(path.read_text())
		ids: list[int] = [p['id'] for p in group['particles']]
		if len(ids) != 8 or len(set(ids)) != 8 or group['source_sha256'] != particles['source_sha256']:
			raise ValueError('Expected eight distinct candidates from the reference potential in each set.')
		for particle in group['particles']:
			if particle != reference_particles.get(particle['id']):
				raise ValueError('A selected particle differs from its reference campaign record.')
			selected[particle['id']] = particle
		selection_records.append(dict(dataset_id=group['dataset_id'], source_file=path.name,
			sha256=sha256(path.read_bytes()).hexdigest(), particle_ids=ids))
	if len(selected) != 16:
		raise ValueError('The selected third and fourth sets must form a 16-particle union.')
	particles['particles'] = [selected[i] for i in sorted(selected)]
	particles['families'] = []
	particles['selection'] = dict(operation='union', source_sets=selection_records)
	particle_count: int = len(selected)
	paths: list[Path] = sorted((root / 'src').rglob('*.py'))
	paths += [root / 'examples/modal_bm4_numpy_jax_app.py', root / 'constraints.txt', root / 'pyproject.toml']
	digest: Any = sha256()
	for path in paths:
		digest.update(path.relative_to(root).as_posix().encode())
		digest.update(path.read_bytes())
	record: dict[str, Any] = dict(schema_version=2, study_prefix=STUDY_PREFIX, run_id=directory.name,
		cycles=list(cycles), methods=list(METHODS), repetitions=repetitions,
		particle_count=particle_count, particle_ids=sorted(selected), particle_selection=particles['selection'],
		rho=selection['representative_rho'], reference_campaign=selection['campaign_id'],
		source_sha256=selection['source_sha256'], code_sha256=digest.hexdigest(),
		bucket=f'{DEFAULT_RESULTS_BUCKET}/{STUDY_PREFIX}/{directory.name}',
		resources=dict(cpu_request=2, cpu_limit=2, memory_request_mib=4096, memory_limit_mib=8192),
		timing_scope='Execution.run: preparation, integration, synchronization and result construction; excludes input decoding, files, network, audits and plots',
		record_step_statistics=False, observers=False, progress=False)
	if (directory / 'study.json').exists():
		if json.loads((directory / 'study.json').read_text()) != record:
			raise ValueError('Study settings or code changed; use a new run identifier.')
		return record
	field: Any = load_field(root / 'data/potential/V1/PHI_2.h5')
	if selection['source_sha256'] != particles['source_sha256']:
		raise ValueError('Reference and particles use different source potentials.')
	directory.mkdir(parents=True, exist_ok=True)
	with tarfile.open(directory / 'source.tar.gz', 'w:gz') as archive:
		for path in paths + selection_paths + [root / 'data/particles/expanded_84.json', selection_path]:
			archive.add(path, arcname=path.name if path == selection_path else path.relative_to(root).as_posix())
	for count in cycles:
		for name in METHODS:
			key: str = f'{name}-{count}'
			problem, method, metadata = prepare_case(CampaignConfig(cycles=count), field, particles,
				record['rho'], name, code_sha256=record['code_sha256'])
			metadata.update(study='bm4_numpy_jax', reference_campaign=record['reference_campaign'],
				particle_count=particle_count, energy_evaluation='disabled', repetitions=repetitions)
			payload: bytes = encode_job(problem, method,
				ExecutionOptions(record_step_statistics=False))
			(directory / f'{key}.payload').write_bytes(payload)
			_write_record(directory / f'{key}.json', metadata)
	_write_record(directory / 'study.json', record)
	return record


def publish_study(directory: Path) -> dict[str, Any]:
	"""Publish frozen files immutably, then reload all trajectory archives."""
	record: dict[str, Any] = json.loads((directory / 'study.json').read_text())
	store: ArtifactStore = ArtifactStore(record['bucket'])
	# Payloads are local trusted-Python recovery files, never published results.
	files: list[Path] = [p for p in directory.rglob('*') if p.is_file()
		and p.suffix in ('.json', '.npz', '.gz') and p.name != 'publication.json']
	for path in sorted(files, key=lambda p: p.name == 'manifest.json'):
		store.put(path, path.relative_to(directory).as_posix())
	verified: list[str] = []
	for count in record['cycles']:
		for name in record['methods']:
			for backend in ('scipy', 'jax'):
				key: str = f'{name}-{count}/{backend}'
				local: Any = load_solution(directory / 'runs' / key)
				remote: Any = load_solution(record['bucket'] + '/runs/' + key)
				np.testing.assert_array_equal(remote.solution.states, local.solution.states)
				if remote.metadata != local.metadata:
					raise ValueError('Remote provenance differs from the local result.')
				verified.append(key)
	publication: dict[str, Any] = dict(status='verified', bucket=record['bucket'], verified_cases=verified)
	_write_record(directory / 'publication.json', publication)
	store.put(directory / 'publication.json', 'publication.json')
	return publication


def run_study(root: Path, directory: Path, *, publish: bool = True) -> dict[str, Any]:
	"""Submit missing jobs once; resume confirmed call IDs without reintegration."""
	import modal
	record: dict[str, Any] = json.loads((directory / 'study.json').read_text())
	keys: list[str] = [f'{name}-{count}' for count in record['cycles'] for name in record['methods']]
	receipts: Path = directory / 'receipts'
	receipts.mkdir(exist_ok=True)
	missing: list[str] = [key for key in keys if not (receipts / f'{key}.json').exists()
		and not (directory / 'runs' / key / 'pair.json').exists()]
	module: Any = None
	if missing:
		# Existing confirmed calls are recoverable even after local edits. Only
		# new submissions require the exact source snapshot prepared for this run.
		with tarfile.open(directory / 'source.tar.gz', 'r:gz') as archive:
			for member in archive.getmembers():
				if member.name.startswith('src/') or member.name.startswith('examples/'):
					stream: Any = archive.extractfile(member)
					if stream is None or (root / member.name).read_bytes() != stream.read():
						raise ValueError('Source changed before submission; prepare a new run identifier.')
		spec: Any = importlib.util.spec_from_file_location('bm4_modal_app', root / 'examples/modal_bm4_numpy_jax_app.py')
		module = importlib.util.module_from_spec(spec)
		spec.loader.exec_module(module)
	context: Any = module.app.run(detach=True) if missing else nullcontext()
	with modal.enable_output(), context:
		for key in missing:
			payload: bytes = (directory / f'{key}.payload').read_bytes()
			path: Path = receipts / f'{key}.json'
			_write_record(path, dict(submission='unconfirmed', input_sha256=sha256(payload).hexdigest()))
			call: Any = module.benchmark.spawn(payload, record['repetitions'])
			_write_record(path, dict(submission='confirmed', call_id=call.object_id,
				input_sha256=sha256(payload).hexdigest()))
			print(f'{key}: submitted {call.object_id}', flush=True)

		def collect(key: str) -> dict[str, Any]:
			local: Path = directory / 'runs' / key
			if (local / 'pair.json').exists():
				return json.loads((local / 'pair.json').read_text())  # type: ignore[no-any-return]
			receipt: dict[str, Any] = json.loads((receipts / f'{key}.json').read_text())
			if receipt.get('submission') != 'confirmed':
				raise RuntimeError('Unconfirmed submission; inspect Modal before resubmitting.')
			result: dict[str, Any] = modal.FunctionCall.from_id(receipt['call_id']).get()
			if result['payload_sha256'] != receipt['input_sha256']:
				raise ValueError('The worker returned a different job.')
			local.mkdir(parents=True, exist_ok=True)
			metadata: dict[str, Any] = json.loads((directory / f'{key}.json').read_text())
			for backend, solution in result.pop('solutions').items():
				if not (local / backend / 'manifest.json').exists():
					save_solution(solution, local / backend,
						metadata={**metadata, 'backend': backend, 'pair': result, 'call_id': receipt['call_id']})
			_write_record(local / 'pair.json', result)
			print(f'{key}: saved both trajectories and timings', flush=True)
			return result

		report: dict[str, Any] = dict(status='running', cases={})
		with ThreadPoolExecutor(max_workers=6) as pool:
			futures: Any = {pool.submit(collect, key): key for key in keys}
			for future in as_completed(futures):
				key = futures[future]
				try:
					future.result()
					report['cases'][key] = 'saved'
				except Exception as exc:
					report['cases'][key] = f'{type(exc).__name__}: {exc}'
				_write_record(directory / 'progress.json', report)
		report['status'] = 'saved' if all(v == 'saved' for v in report['cases'].values()) else 'failed'
		_write_record(directory / 'progress.json', report)
	if report['status'] != 'saved':
		raise RuntimeError(f'Incomplete benchmark; recover existing calls: {report}')
	if publish:
		publish_study(directory)
	return report


__all__ = ['STUDY_PREFIX', 'prepare_study', 'run_study', 'publish_study']
