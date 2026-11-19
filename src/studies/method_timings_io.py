"""Prepare, recover and verify the eight-method Modal timing study."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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
from studies.method_timings import METHODS

STUDY_PREFIX = 'studies/method_timings'


def prepare_study(root: Path, directory: Path, *, cycles: int = 500,
		repetitions: int = 3) -> dict[str, Any]:
	"""Freeze the agreed potential, original particles and executable sources."""
	config: CampaignConfig = CampaignConfig(cycles=cycles)
	if type(repetitions) is not int or repetitions < 3:
		raise ValueError('At least three cached repetitions are required.')
	if (directory / 'study.json').exists():
		raise FileExistsError('Recover this study or choose a new run identifier.')
	selection_path: Path = root.parent / 'TFM_redaccion/datos/campana_referencia.json'
	selection: dict[str, Any] = json.loads(selection_path.read_text())
	if set(selection['methods']) != set(METHODS):
		raise ValueError('The eight reference methods changed.')
	particles: dict[str, Any] = read_particles(root / 'data/particles/expanded_84.json')
	original_path: Path = root / 'data/particles/original_40.json'
	original: dict[str, Any] = json.loads(original_path.read_text())
	if original['particles'] != particles['particles'][:40]:
		raise ValueError('Original particle positions or identities differ from the reference.')
	particles['particles'] = original['particles']
	particles['families'] = []
	particles['original_40_sha256'] = sha256(original_path.read_bytes()).hexdigest()
	paths: list[Path] = sorted((root / 'src').rglob('*.py'))
	paths += [root / 'examples/modal_method_timings_app.py', root / 'constraints.txt', root / 'pyproject.toml']
	digest: Any = sha256()
	for path in paths:
		digest.update(path.relative_to(root).as_posix().encode())
		digest.update(path.read_bytes())
	record: dict[str, Any] = dict(schema_version=1, study_prefix=STUDY_PREFIX, run_id=directory.name,
		cycles=cycles, methods=list(METHODS), repetitions=repetitions, particle_count=40,
		steps_per_cycle=40, samples_per_cycle=10, rho=selection['representative_rho'],
		reference_campaign=selection['campaign_id'], source_sha256=selection['source_sha256'],
		code_sha256=digest.hexdigest(), bucket=f'{DEFAULT_RESULTS_BUCKET}/{STUDY_PREFIX}/{directory.name}',
		resources=dict(cpu_request=2, cpu_limit=2, memory_request_mib=4096, memory_limit_mib=8192),
		timing_scope='Execution.run: preparation, integration, synchronization and host result construction; excludes container startup, input decoding, files, network, audits and plots',
		method_order='Forward in even rounds, reverse in odd rounds; round zero is the first invocation',
		backend='jax', device='cpu', record_step_statistics=False, observers=False, progress=False,
		particle_distances=False, energy_evaluation='disabled', same_container=True)
	field: Any = load_field(root / 'data/potential/V1/PHI_2.h5')
	if selection['source_sha256'] != particles['source_sha256']:
		raise ValueError('Reference and particles use different potentials.')
	directory.mkdir(parents=True, exist_ok=True)
	with tarfile.open(directory / 'source.tar.gz', 'w:gz') as archive:
		for path in paths + [original_path, selection_path]:
			archive.add(path, arcname=path.name if path == selection_path else path.relative_to(root).as_posix())
	for name in METHODS:
		problem, method, metadata = prepare_case(config, field, particles, record['rho'], name,
			code_sha256=record['code_sha256'])
		metadata.update(study='method_timings', reference_campaign=record['reference_campaign'],
			particle_count=40, energy_evaluation='disabled', repetitions=repetitions)
		payload: bytes = encode_job(problem, method, ExecutionOptions(backend='jax', record_step_statistics=False))
		(directory / f'{name}.payload').write_bytes(payload)
		metadata['payload_sha256'] = sha256(payload).hexdigest()
		_write_record(directory / f'{name}.json', metadata)
	_write_record(directory / 'study.json', record)
	return record


def recover_study(bucket: str, directory: Path) -> dict[str, Any]:
	"""Restore verified published files without submitting integrations."""
	store: ArtifactStore = ArtifactStore(bucket)
	directory.mkdir(parents=True, exist_ok=True)
	store.get('manifest.json', directory / 'manifest.json')
	manifest: dict[str, Any] = json.loads((directory / 'manifest.json').read_text())
	for name, expected in manifest['files'].items():
		if Path(name).is_absolute() or '..' in Path(name).parts:
			raise ValueError('Unsafe archive path.')
		path: Path = directory / name
		path.parent.mkdir(parents=True, exist_ok=True)
		store.get(name, path)
		if path.stat().st_size != expected['bytes'] or sha256(path.read_bytes()).hexdigest() != expected['sha256']:
			raise ValueError(f'Artifact verification failed: {name}.')
	return json.loads((directory / 'study.json').read_text())  # type: ignore[no-any-return]


def publish_study(directory: Path) -> dict[str, Any]:
	"""Archive immutably and download every result and provenance file to verify it."""
	import tempfile
	record: dict[str, Any] = json.loads((directory / 'study.json').read_text())
	for name in record['methods']:
		loaded: Any = load_solution(directory / 'runs' / name)
		if loaded.solution.states.shape != (record['cycles'] * 10 + 1, 80):
			raise ValueError('An incomplete solution cannot be published.')
	files: list[Path] = [p for p in directory.rglob('*') if p.is_file()
		and p.suffix in ('.json', '.npz', '.gz')
		and p.relative_to(directory).as_posix() not in ('manifest.json', 'publication.json')]
	manifest: dict[str, Any] = dict(schema_version=1, files={p.relative_to(directory).as_posix():
		{'bytes': p.stat().st_size, 'sha256': sha256(p.read_bytes()).hexdigest()} for p in files})
	_write_record(directory / 'manifest.json', manifest)
	store: ArtifactStore = ArtifactStore(record['bucket'])
	with ThreadPoolExecutor(max_workers=4) as pool:
		list(pool.map(lambda p: store.put(p, p.relative_to(directory).as_posix()), files))
	store.put(directory / 'manifest.json', 'manifest.json')
	with tempfile.TemporaryDirectory(prefix='gc2d-method-timings-verify-') as temporary:
		verified: Path = Path(temporary)
		recover_study(record['bucket'], verified)
		for name in record['methods']:
			local: Any = load_solution(directory / 'runs' / name)
			remote: Any = load_solution(verified / 'runs' / name)
			np.testing.assert_array_equal(local.solution.states, remote.solution.states)
			if local.metadata != remote.metadata:
				raise ValueError('Remote provenance differs from the local result.')
	publication: dict[str, Any] = dict(status='verified', bucket=record['bucket'],
		verified_methods=record['methods'], verified_files=len(files),
		manifest_sha256=sha256((directory / 'manifest.json').read_bytes()).hexdigest())
	_write_record(directory / 'publication.json', publication)
	store.put(directory / 'publication.json', 'publication.json')
	return publication


def run_study(root: Path, directory: Path, *, publish: bool = True) -> dict[str, Any]:
	"""Submit once, preserve the call receipt, and recover without reintegration."""
	import modal
	record: dict[str, Any] = json.loads((directory / 'study.json').read_text())
	result_path: Path = directory / 'timings.json'
	if not result_path.exists():
		receipt_path: Path = directory / 'receipt.json'
		module: Any = None
		if not receipt_path.exists():
			with tarfile.open(directory / 'source.tar.gz', 'r:gz') as archive:
				for member in archive.getmembers():
					if member.name.startswith(('src/', 'examples/')):
						stream: Any = archive.extractfile(member)
						if stream is None or (root / member.name).read_bytes() != stream.read():
							raise ValueError('Sources changed before submission; prepare a new run.')
			spec: Any = importlib.util.spec_from_file_location('timings_modal_app', root / 'examples/modal_method_timings_app.py')
			module = importlib.util.module_from_spec(spec)
			spec.loader.exec_module(module)
		context: Any = module.app.run(detach=True) if module is not None else nullcontext()
		with modal.enable_output(), context:
			if module is not None:
				payloads: dict[str, bytes] = {name: (directory / f'{name}.payload').read_bytes() for name in record['methods']}
				hashes: dict[str, str] = {name: sha256(p).hexdigest() for name, p in payloads.items()}
				_write_record(receipt_path, dict(submission='unconfirmed', payload_sha256=hashes))
				call: Any = module.benchmark.spawn(payloads, record['repetitions'])
				_write_record(receipt_path, dict(submission='confirmed', call_id=call.object_id, payload_sha256=hashes))
				print(f'Submitted all eight methods: {call.object_id}', flush=True)
			receipt: dict[str, Any] = json.loads(receipt_path.read_text())
			if receipt['submission'] != 'confirmed':
				raise RuntimeError('Unconfirmed submission; inspect Modal before any resubmission.')
			result: dict[str, Any] = modal.FunctionCall.from_id(receipt['call_id']).get()
			if result['payload_sha256'] != receipt['payload_sha256']:
				raise ValueError('Worker input fingerprints differ from the submitted study.')
			solutions: dict[str, Any] = result.pop('solutions')
			if set(solutions) != set(record['methods']) or len(result['measurements']) != 8 * (record['repetitions'] + 1):
				raise ValueError('The worker did not complete all methods and repetitions.')
			for name, solution in solutions.items():
				metadata: dict[str, Any] = json.loads((directory / f'{name}.json').read_text())
				if solution.diagnostics['step_count'] != record['cycles'] * record['steps_per_cycle']:
					raise ValueError('Incorrect number of integration steps.')
				if not (directory / 'runs' / name / 'manifest.json').exists():
					save_solution(solution, directory / 'runs' / name,
						metadata={**metadata, 'call_id': receipt['call_id'], 'environment': result['environment']})
			result['call_id'] = receipt['call_id']
			_write_record(result_path, result)
	if publish and not (directory / 'publication.json').exists():
		publish_study(directory)
	return json.loads(result_path.read_text())  # type: ignore[no-any-return]


__all__ = ['STUDY_PREFIX', 'prepare_study', 'recover_study', 'run_study', 'publish_study']
