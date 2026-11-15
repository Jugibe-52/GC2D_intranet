"""Local receipts, recoverable Modal execution and B2 publication for one campaign."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any
from threading import BoundedSemaphore

from diagnostics.io.persistence import load_solution, save_solution
from diagnostics.io.paths import DEFAULT_RESULTS_BUCKET
from diagnostics.io.storage import ArtifactStore
from diagnostics.methods.energy import EnergyObserver
from execution.execution_modal import Execution_Modal, _write_record
from studies.main_campaign import (CampaignConfig, METHODS, RHOS, OPTIONS, read_particles,
	load_field, prepare_case, validate_case)


def code_digest(root: Path) -> str:
	"""Fingerprint importable numerical code, not Git's possibly dirty HEAD alone."""
	digest: Any = sha256()
	paths: list[Path] = [p for p in (root / 'src').rglob('*.py')
		if p.relative_to(root / 'src').parts[0] not in ('visualization', 'studies')]
	paths += [root / 'src/studies/main_campaign.py', root / 'examples/modal_main_campaign_app.py']
	for path in sorted(paths):
		digest.update(path.relative_to(root).as_posix().encode())
		digest.update(path.read_bytes())
	return str(digest.hexdigest())


def prepare_campaign(directory: Path, root: Path, *, config: CampaignConfig,
		particle_path: Path, rhos: tuple[float, ...] = RHOS, methods: tuple[str, ...] = METHODS) -> dict[str, Any]:
	"""Resolve all inputs before submission; existing incompatible records are errors."""
	if not rhos or len(set(rhos)) != len(rhos) or any(r not in RHOS for r in rhos):
		raise ValueError('Select distinct campaign gyro-radii.')
	if not methods or len(set(methods)) != len(methods) or any(m not in METHODS for m in methods):
		raise ValueError('Select distinct campaign methods.')
	particles: dict[str, Any] = read_particles(particle_path)
	field: Any = load_field(root / 'data/potential/V1/PHI_2.h5')
	record: dict[str, Any] = dict(schema_version=1, config=asdict(config), particles=particles,
		rhos=list(rhos), methods=list(methods), code_sha256=code_digest(root),
		source_sha256=particles['source_sha256'],
		bucket=f'{DEFAULT_RESULTS_BUCKET}/developements/main_reflected_campaign/{directory.name}',
		app_name='gc2d-main-reflected')
	# Validate every scientific case before creating an executable campaign.
	for rho in rhos:
		for method in methods:
			prepare_case(config, field, particles, rho, method, code_sha256=record['code_sha256'])
	directory.mkdir(parents=True, exist_ok=True)
	path: Path = directory / 'campaign.json'
	if path.exists() and json.loads(path.read_text()) != record:
		raise ValueError('Campaign inputs/code changed; use a separate campaign directory.')
	_write_record(path, record)
	return record


def run_campaign(directory: Path, root: Path, *, max_workers: int = 4, publish: bool = True) -> dict[str, Any]:
	"""Recover submitted calls; save locally before uploading verified bytes to B2."""
	if not 1 <= max_workers <= 16:
		raise ValueError('Use between one and sixteen workers.')
	record: dict[str, Any] = json.loads((directory / 'campaign.json').read_text())
	if code_digest(root) != record['code_sha256']:
		raise ValueError('Code fingerprint changed since preparation; do not mix implementations.')
	config: CampaignConfig = CampaignConfig(**record['config'])
	field: Any = load_field(root / 'data/potential/V1/PHI_2.h5')
	collection = BoundedSemaphore(1)  # Bound local RAM while remote cases run concurrently.
	report: dict[str, Any] = dict(status='running', cases={}, expected=len(record['rhos'])*len(record['methods']))

	def run_one(rho: float, name: str) -> tuple[str, dict[str, Any]]:
		key: str = f'rho-{rho:.2f}-{name}'
		local: Path = directory / 'runs' / key
		row: dict[str, Any] = dict(rho=rho, method=name, local=str(local), status='running')
		try:
			problem: Any
			method: Any
			metadata: dict[str, Any]
			problem, method, metadata = prepare_case(config, field, record['particles'], rho, name,
				code_sha256=record['code_sha256'])
			executor: Execution_Modal | None = None
			ready: Any = None
			if not (local / 'manifest.json').exists():
				receipts: Path = directory / 'receipts' / key
				paths: list[Path] = sorted(receipts.glob('*.json'), key=lambda p: p.stat().st_mtime_ns)
				if paths:
					executor = Execution_Modal.from_record(paths[-1])
				else:
					from modal.exception import TimeoutError as ModalTimeoutError
					executor = Execution_Modal(app_name=record['app_name'], record_directory=receipts, wait_timeout=0)
					try:
						ready = executor.run(problem, method, options=OPTIONS, observers=(EnergyObserver(),))
					except (ModalTimeoutError, TimeoutError):
						assert executor.last_record_path is not None
						executor = Execution_Modal.from_record(executor.last_record_path)
			with collection:
				if (local / 'manifest.json').exists():
					stored: Any = load_solution(local)
					if stored.metadata != metadata:
						raise ValueError('Saved case provenance differs from the prepared inputs.')
					solution: Any = stored.solution
				else:
					assert executor is not None
					solution = ready if ready is not None else executor.run(problem, method, options=OPTIONS, observers=(EnergyObserver(),))
					validate_case(solution, metadata)
					local.parent.mkdir(parents=True, exist_ok=True)
					save_solution(solution, local, metadata=metadata, potential=field)
				validate_case(solution, metadata)
				row.update({k: v for k, v in solution.diagnostics.items() if k.startswith(('worker_', 'cost_'))})
				row.update(status='saved_local', worker_wall_seconds=solution.diagnostics.get('worker_wall_seconds'),
					call_id=solution.diagnostics.get('execution_call_id'))
				if publish:
					remote: str = record['bucket'] + '/' + key
					store: ArtifactStore = ArtifactStore(remote)
					if store.exists('manifest.json'):
						# Loading verifies all published SHA256 digests and catches mismatched reuse.
						published: Any = load_solution(remote)
						if published.metadata != metadata:
							raise ValueError('Remote case provenance differs from local inputs.')
						validate_case(published.solution, metadata)
					else:
						for path in sorted(local.iterdir()):
							if path.name != 'manifest.json':
								store.put(path, path.name)
						store.put(local / 'manifest.json', 'manifest.json')
						published = load_solution(remote)
						validate_case(published.solution, metadata)
					row.update(status='complete', remote=remote)
			return key, row
		except Exception as exc:
			row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
			return key, row

	with ThreadPoolExecutor(max_workers=max_workers) as pool:
		# Collect short cases first so a long implicit call cannot block every local RAM slot.
		priority: tuple[str, ...] = ('ABBA2Midpoint', 'RK2', 'RK4', 'BM4Midpoint', 'GaussLegendre4', 'ImplicitMidpoint', 'ABBA2Implicit', 'BM4Implicit')
		futures: Any = [pool.submit(run_one, rho, method) for method in priority if method in record['methods'] for rho in record['rhos']]
		for future in as_completed(futures):
			key: str
			row: dict[str, Any]
			key, row = future.result()
			report['cases'][key] = row
			_write_record(directory / 'progress.json', report)
			print(key, row['status'], row.get('worker_wall_seconds', ''), row.get('error', ''), flush=True)
	report['status'] = 'failed' if any(r['status'] == 'failed' for r in report['cases'].values()) else ('complete' if publish else 'saved_local')
	report['cost_estimated_requested_resources_usd'] = sum(r.get('cost_estimated_requested_resources_usd', 0.) for r in report['cases'].values())
	report['cost_estimated_limit_resources_usd'] = sum(r.get('cost_estimated_limit_resources_usd', 0.) for r in report['cases'].values())
	report['cost_billed_usd'] = None
	report['cost_billed_status'] = 'Pending reconciliation with Modal billing; estimates are not invoices.'
	_write_record(directory / 'progress.json', report)
	return report


def publication_cases(directory: Path, record: dict[str, Any], *,
		allow_partial: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
	"""Separate verified results from explicit failures; never hide pending uploads."""
	report: dict[str, Any] = json.loads((directory / 'progress.json').read_text())
	expected: set[str] = {f'rho-{rho:.2f}-{method}' for rho in record['rhos'] for method in record['methods']}
	cases: dict[str, Any] = report['cases']
	if (set(cases) != expected or report['status'] not in ('complete', 'failed')
			or (not allow_partial and (report['status'] != 'complete'
				or any(row['status'] != 'complete' for row in cases.values())))):
		raise ValueError('Finalization requires every case to be complete and verified in B2.')
	available: dict[str, Any] = {}
	missing: dict[str, Any] = {}
	for key, row in cases.items():
		if row['status'] == 'complete':
			available[key] = row
		elif allow_partial and row['status'] == 'failed' and not (directory / 'runs' / key / 'manifest.json').exists():
			missing[key] = dict(status='unavailable', rho=row['rho'], method=row['method'],
				reason=('Nonlinear solver did not converge.' if 'converg' in str(row.get('error', '')).lower()
					else 'Simulation failed before a complete result was saved.'))
		else:
			raise ValueError(f'Case {key} is not verified in B2; recover it before publication.')
	if not available:
		raise ValueError('Publication requires at least one verified result.')
	return available, missing


def finalize_campaign(directory: Path, *, allow_partial: bool = False) -> Path:
	"""Archive resources and export verified cases, with explicit optional failures."""
	record: dict[str, Any] = json.loads((directory / 'campaign.json').read_text())
	report: dict[str, Any] = json.loads((directory / 'progress.json').read_text())
	cases: dict[str, Any]
	missing: dict[str, Any]
	cases, missing = publication_cases(directory, record, allow_partial=allow_partial)
	rows: dict[str, Any] = {key: {k: v for k, v in row.items()
		if k.startswith(('worker_', 'cost_')) or k in ('rho', 'method', 'call_id')}
		for key, row in cases.items()}
	metrics: dict[str, Any] = dict(schema_version=1, code_sha256=record['code_sha256'], cases=rows,
		publication_status='partial' if missing else 'complete', missing_cases=missing,
		expected_cases=len(cases)+len(missing), available_cases=len(cases),
		worker_span_seconds=max(row['worker_finished_unix_seconds'] for row in rows.values())
			-min(row['worker_started_unix_seconds'] for row in rows.values()),
		sum_worker_wall_seconds=sum(row['worker_wall_seconds'] for row in rows.values()),
		cost_estimated_requested_resources_usd=sum(row.get('cost_estimated_requested_resources_usd', 0.) for row in rows.values()),
		cost_estimated_limit_resources_usd=sum(row.get('cost_estimated_limit_resources_usd', 0.) for row in rows.values()),
		cost_billed_usd=None, cost_billed_status=report['cost_billed_status'])
	# Preserve the partial snapshot when a later complete campaign is published.
	path: Path = directory / (f'execution-metrics-partial-{len(rows)}.json' if missing else 'execution-metrics.json')
	_write_record(path, metrics)
	ArtifactStore(record['bucket']).put(path, path.name)
	from visualization.main_campaign import export_campaign
	return export_campaign(directory, allow_partial=allow_partial)


__all__ = ['code_digest', 'prepare_campaign', 'run_campaign', 'publication_cases', 'finalize_campaign']
