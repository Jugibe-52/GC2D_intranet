"""Validate, reduce and archive accumulated midpoint geometry records."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, localcontext
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from diagnostics.io.paths import solution_destination
from diagnostics.io.storage import ArtifactStore


def digest(path: Path) -> str:
	return sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
	path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def load_case(path: Path) -> dict[str, np.ndarray]:
	with np.load(path, allow_pickle=False) as values:
		return {key: values[key] for key in values.files}


def matrix_products(local: np.ndarray, checkpoints: np.ndarray,
		digits: int = 80) -> dict[str, np.ndarray]:
	"""Multiply physical 2-by-2 Jacobians on the left in decimal arithmetic.

	Local float64 entries are converted exactly, without rounding through their
	decimal display. Extra precision controls product/determinant cancellation;
	it cannot remove rounding already present in the local Jacobians.
	"""
	if (local.ndim != 4 or local.shape[-2:] != (2, 2) or not np.all(np.isfinite(local))
			or digits < 30 or checkpoints.ndim != 1 or len(checkpoints) == 0
			or np.any(checkpoints < 1) or np.any(checkpoints > len(local))
			or np.any(np.diff(checkpoints) <= 0)):
		raise ValueError('Expected finite step/particle/2/2 Jacobians and increasing checkpoints.')
	n: int = local.shape[1]
	matrices: np.ndarray = np.empty((len(checkpoints), n, 2, 2), dtype=f'U{digits+32}')
	determinants: np.ndarray = np.empty((len(checkpoints), n), dtype=f'U{digits+32}')
	defects: np.ndarray = np.empty((len(checkpoints), n))
	products: list[tuple[Decimal, Decimal, Decimal, Decimal]] = [
		(Decimal(1), Decimal(0), Decimal(0), Decimal(1)) for _ in range(n)]
	targets: dict[int, int] = {int(step): i for i, step in enumerate(checkpoints)}
	with localcontext() as context:
		context.prec = digits
		for step, jacobians in enumerate(local, start=1):
			for particle, jacobian in enumerate(jacobians):
				a, b, c, d = (Decimal.from_float(float(x)) for x in jacobian.ravel())
				e, f, g, k = products[particle]
				products[particle] = (a*e+b*g, a*f+b*k, c*e+d*g, c*f+d*k)
			if step in targets:
				index: int = targets[step]
				for particle, product in enumerate(products):
					a, b, c, d = product
					determinant: Decimal = a*d-b*c
					matrices[index, particle] = np.asarray([str(x) for x in product]).reshape(2, 2)
					determinants[index, particle] = str(determinant)
					defects[index, particle] = float(abs(determinant-1))
	return dict(cumulative_jacobians_decimal=matrices, cumulative_determinants_decimal=determinants,
		area_defect=defects, product_decimal_digits=np.asarray(digits))


def validate_case(data: dict[str, np.ndarray], count: int, horizons: list[int],
		initial: np.ndarray) -> dict[str, float]:
	"""Rebuild cumulative and endpoint diagnostics from independently saved arrays."""
	n: int = initial.shape[0]
	steps: int = count*max(horizons)
	if (data['local_signed_defect'].shape != (steps, n)
			or data['cycle_states'].shape != (max(horizons)+1, 2*n)
			or data['returned_states'].shape != (len(horizons), 2*n)
			or not np.array_equal(data['initial_states'], initial)
			or data['horizons'].tolist() != horizons or int(data['steps_per_cycle']) != count
			or float(data['h']) != 1/count):
		raise ValueError('Unexpected axes, initial conditions, horizons or step size.')
	for key, values in data.items():
		if values.dtype.kind in 'US':
			if not all(Decimal(str(value)).is_finite() for value in values.ravel()):
				raise ValueError(f'Nonfinite decimal array: {key}')
		elif not np.all(np.isfinite(values)):
			raise ValueError(f'Nonfinite array: {key}')
	if not np.array_equal(data['times'], np.arange(steps+1)/count):
		# Multiplication by h and division can differ by one rounding unit.
		np.testing.assert_allclose(data['times'], np.arange(steps+1)/count, rtol=0, atol=2e-14)
	checkpoints: np.ndarray = np.asarray(horizons)*count
	report: dict[str, float] = dict(identity_max_absolute_residual=float(np.max(
		np.abs(data['local_signed_defect']-data['local_direct_defect']))))
	if 'local_jacobians' in data:
		if data['local_jacobians'].shape != (steps, n, 2, 2):
			raise ValueError('Unexpected local Jacobian axes.')
		np.testing.assert_array_equal(data['local_direct_defect'], np.linalg.det(data['local_jacobians'])-1)
		rebuilt: dict[str, np.ndarray] = matrix_products(data['local_jacobians'], checkpoints, int(data['product_decimal_digits']))
		for key in rebuilt:
			np.testing.assert_array_equal(data[key], rebuilt[key])
		refined: dict[str, np.ndarray] = matrix_products(data['local_jacobians'], checkpoints, int(data['product_decimal_digits'])+40)
		np.testing.assert_allclose(data['area_defect'], refined['area_defect'], rtol=1e-20, atol=1e-30)
		report['product_precision_check_digits'] = float(int(data['product_decimal_digits'])+40)
		report['product_precision_area_max_absolute_difference'] = float(np.max(np.abs(data['area_defect']-refined['area_defect'])))
		with localcontext() as context:
			context.prec = int(data['product_decimal_digits'])+40
			errors: list[Decimal] = [abs(Decimal(str(x))-Decimal(str(y)))/max(Decimal(1), abs(Decimal(str(y))))
				for x, y in zip(data['cumulative_jacobians_decimal'].ravel(), refined['cumulative_jacobians_decimal'].ravel())]
			if max(errors) > Decimal('1e-40'):
				raise ValueError('Accumulated matrix entries do not agree at increased precision.')
			report['product_precision_matrix_max_scaled_difference'] = float(max(errors))
	else:
		# Schema-1 logarithmic records remain readable as historical data.
		if np.any(data['local_signed_defect'] <= -1) or np.any(data['local_direct_defect'] <= -1):
			raise ValueError('The positive-determinant log accumulation is not applicable.')
		logs: np.ndarray = np.cumsum(np.log1p(data['local_signed_defect']), axis=0)
		np.testing.assert_array_equal(data['logdet'][0], np.zeros(n))
		np.testing.assert_array_equal(data['logdet'][1:], logs)
		np.testing.assert_array_equal(data['area_defect'], np.abs(np.expm1(data['logdet'][checkpoints])))
		direct: np.ndarray = np.cumsum(np.log1p(data['local_direct_defect']), axis=0)
		report['logsum_direct_structure_max_absolute_difference'] = float(np.max(np.abs(logs-direct)))
	vectors: np.ndarray = (data['returned_states']-initial.T.ravel()).reshape(len(horizons), 2, n).transpose(0, 2, 1)
	np.testing.assert_array_equal(vectors, data['return_vectors'])
	np.testing.assert_allclose(np.hypot(vectors[..., 0], vectors[..., 1]), data['return_norm'], rtol=3e-16, atol=0)
	return report


def finish_record(directory: Path) -> dict[str, Any]:
	"""Verify all raw cases before creating complete summary and manifest files."""
	p: dict[str, Any] = json.loads((directory/'parameters.json').read_text())
	initial: np.ndarray = np.asarray(p['initial_positions'])
	ids: list[int] = p['particle_ids']
	groups: dict[str, Any] = {name: {'particle_ids': group, 'methods': {}} for name, group in p['groups'].items()}
	checks: dict[str, Any] = dict(status='passed', cases={}, finite=True,
		independent_cumulative_and_euclidean_reductions=True, orientation_flips=0)
	for method in p['methods']:
		for count in p['steps_per_cycle']:
			filename: str = f'{method}-{count}.npz'
			data: dict[str, np.ndarray] = load_case(directory/filename)
			checks['cases'][filename] = validate_case(data, count, p['horizons'], initial)
			checks['orientation_flips'] += int(np.sum(data['local_direct_defect'] < -1))
			for name, group in p['groups'].items():
				indices: list[int] = [ids.index(i) for i in group]
				entry: dict[str, Any] = groups[name]['methods'].setdefault(method, {})
				entry[str(count)] = {
					metric: dict(values=data[metric][:, indices].tolist(),
						maximum=np.max(data[metric][:, indices], axis=1).tolist(),
						median=np.median(data[metric][:, indices], axis=1).tolist(),
						max_particle_ids=[group[i] for i in np.argmax(data[metric][:, indices], axis=1)])
					for metric in ('area_defect', 'return_norm')}
				entry[str(count)]['forward_seconds'] = data['forward_seconds'].tolist()
				entry[str(count)]['backward_seconds'] = data['backward_seconds'].tolist()
				if 'product_seconds' in data:
					entry[str(count)]['product_seconds'] = float(data['product_seconds'])
	summary: dict[str, Any] = dict(schema_version=p['schema_version'], run_id=p['run_id'], status='complete',
		parameters=p, groups=groups, evaluations_per_group=len(p['steps_per_cycle'])*len(p['horizons']),
		evaluation_definition='one (h,T) pair, including both methods and both metrics for eight particles')
	write_json(directory/'summary.json', summary)
	write_json(directory/'checks.json', checks)
	write_json(directory/'status.json', dict(status='complete'))
	names: list[str] = ['parameters.json', 'summary.json', 'checks.json', 'source.tar.gz']
	names += [f'{method}-{count}.npz' for method in p['methods'] for count in p['steps_per_cycle']]
	write_json(directory/'manifest.json', dict(schema_version=1, run_id=p['run_id'], status='complete',
		files={name: dict(sha256=digest(directory/name), bytes=(directory/name).stat().st_size) for name in names}))
	return summary


def verify_record(directory: Path) -> dict[str, Any]:
	manifest: dict[str, Any] = json.loads((directory/'manifest.json').read_text())
	if manifest['status'] != 'complete':
		raise ValueError('Incomplete scientific record.')
	for name, entry in manifest['files'].items():
		if Path(name).name != name or digest(directory/name) != entry['sha256'] or (directory/name).stat().st_size != entry['bytes']:
			raise ValueError(f'Archive mismatch: {name}')
	summary: dict[str, Any] = json.loads((directory/'summary.json').read_text())
	if summary['run_id'] != manifest['run_id'] or summary['status'] != 'complete':
		raise ValueError('Inconsistent run identities or status.')
	return summary


def archive_record(directory: Path) -> dict[str, Any]:
	"""Upload immutable files and verify independent downloads before receipt."""
	summary: dict[str, Any] = verify_record(directory)
	destination: str = solution_destination('studies/midpoint_geometry', summary['run_id'])
	store: ArtifactStore = ArtifactStore(destination)
	manifest: dict[str, Any] = json.loads((directory/'manifest.json').read_text())
	names: list[str] = [*manifest['files'], 'manifest.json']
	for name in names:
		store.put(directory/name, name)
		print(f'Stored {name}', flush=True)
	with tempfile.TemporaryDirectory(prefix='midpoint-sweep-readback-') as temporary:
		for name in names:
			path: Path = Path(temporary)/name
			store.get(name, path)
			if digest(path) != digest(directory/name):
				raise ValueError(f'Read-back mismatch: {name}')
			print(f'Verified {name}', flush=True)
	receipt: dict[str, Any] = dict(status='verified', bucket=destination,
		verified_utc=datetime.now(timezone.utc).isoformat(), verified_files=len(names),
		manifest_sha256=digest(directory/'manifest.json'))
	if (directory/'publication.json').exists():
		previous: dict[str, Any] = json.loads((directory/'publication.json').read_text())
		if previous['manifest_sha256'] == receipt['manifest_sha256'] and previous['bucket'] == destination:
			receipt = previous
	write_json(directory/'publication.json', receipt)
	store.put(directory/'publication.json', 'publication.json')
	return receipt


def recover_record(destination: str, directory: Path) -> dict[str, Any]:
	"""Recover verified saved data without replacing different local contents."""
	store: ArtifactStore = ArtifactStore(destination)
	with tempfile.TemporaryDirectory(prefix='midpoint-sweep-recovery-') as temporary:
		staging: Path = Path(temporary)
		store.get('manifest.json', staging/'manifest.json')
		manifest: dict[str, Any] = json.loads((staging/'manifest.json').read_text())
		for name in manifest['files']:
			if Path(name).name != name:
				raise ValueError('Unsafe archive member name.')
			store.get(name, staging/name)
		summary: dict[str, Any] = verify_record(staging)
		if destination.rstrip('/').split('/')[-1] != summary['run_id']:
			raise ValueError('Destination and run identifier differ.')
		for path in staging.iterdir():
			if (directory/path.name).exists() and digest(directory/path.name) != digest(path):
				raise ValueError(f'Local recovery conflict: {path.name}')
		directory.mkdir(parents=True, exist_ok=True)
		for path in staging.iterdir():
			(directory/path.name).write_bytes(path.read_bytes())
	return summary


__all__ = ['digest', 'write_json', 'load_case', 'matrix_products', 'validate_case', 'finish_record',
	'verify_record', 'archive_record', 'recover_record']
