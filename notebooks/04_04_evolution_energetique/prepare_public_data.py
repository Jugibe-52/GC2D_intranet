"""Build an immutable public subset from the verified online energy campaign."""
from __future__ import annotations

import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
import zipfile

import numpy as np

RUN_ID = 'main-20261117-rho-div20-online-balance-20261120'
STUDY_PREFIX = 'studies/energy_balance_online'
METHODS = ('RK2', 'ImplicitMidpoint', 'RK4', 'GaussLegendre4', 'ABBA2Midpoint', 'ABBA2Implicit', 'BM4Midpoint', 'BM4Implicit')
GROUPS = {'periodic_candidates8': [1, 2, 60, 64, 65, 72, 76, 83], 'chaotic8': [12, 15, 18, 21, 24, 27, 30, 33]}
DIAGNOSTICS = ('hamiltonian', 'delta_h', 'integral_simpson', 'integral_coarse', 'residual', 'quadrature_gap')
ROOT = Path(__file__).resolve().parents[2]
THESIS = ROOT.parent / 'TFM_redaccion'
ARCHIVE = THESIS / 'notebooks/04_04_02_balance_energetico_en_linea/recovery' / RUN_ID


def encoded(value: object) -> bytes:
	return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()


def fingerprint(data: bytes) -> dict:
	return {'bytes': len(data), 'sha256': sha256(data).hexdigest()}


def file_fingerprint(path: Path) -> dict:
	digest = sha256()
	with path.open('rb') as source:
		for block in iter(lambda: source.read(1024 * 1024), b''):
			digest.update(block)
	return {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def require(condition: bool, message: str) -> None:
	if not condition:
		raise ValueError(message)


def metrics(values: np.ndarray) -> dict:
	return {'rms': float(np.sqrt(np.mean(values**2))),
		'maximum_absolute': float(np.abs(values).max()),
		'block_100_cycle_rms': np.sqrt(np.mean(values.reshape(50, 100, 8)**2, axis=(1, 2))).tolist()}


def prepare_bundle(archive: Path, destination: Path, thesis: Path = THESIS) -> str:
	"""Verify every scientific archive, then keep lossless diagnostics for 16 IDs."""
	manifest_bytes = (archive / 'manifest.json').read_bytes()
	original_manifest = json.loads(manifest_bytes)
	publication_bytes = (archive.parents[1] / 'archive_receipt.json').read_bytes()
	publication = json.loads(publication_bytes)
	require(publication['status'] == 'verified' and publication['differences'] == 0
		and publication['manifest'] == original_manifest
		and publication['manifest_sha256'] == sha256(manifest_bytes).hexdigest(), 'Invalid original publication receipt.')
	for name, expected in original_manifest['files'].items():
		require(file_fingerprint(archive / name) == expected, f'Damaged original file: {name}.')
	summary = json.loads((archive / 'summary.json').read_bytes())
	figure_bytes = (thesis / 'figuras/balance_energetico_en_linea_fr.json').read_bytes()
	figure = json.loads(figure_bytes)
	require(len(figure['series']) == 32 and {(row['method'], row['group'], row['quadrature_update_steps']) for row in figure['series']}
		== {(method, group, q) for method in METHODS for group in GROUPS for q in (2, 4)}, 'Incomplete or duplicated figure series.')
	require(len(figure['global_metrics']) == 16 and {(row['method'], row['group']) for row in figure['global_metrics']}
		== {(method, group) for method in METHODS for group in GROUPS}, 'Incomplete or duplicated global RMS records.')
	calculation = archive.parents[1] / 'calculo.py'
	reference = thesis / 'datos/campana_referencia.json'
	require(file_fingerprint(calculation)['sha256'] == summary['calculation_sha256'] == figure['calculation_sha256'], 'Calculation provenance changed.')
	require(file_fingerprint(reference)['sha256'] == summary['reference_sha256'], 'Reference selection changed.')
	require(summary['run_id'] == RUN_ID == figure['run_id'] and summary['groups'] == GROUPS == figure['groups'], 'Wrong run or particle groups.')
	require(sha256((archive / 'summary.json').read_bytes()).hexdigest() == figure['summary_sha256'], 'Figure refers to a different summary.')
	ids = sorted(sum(GROUPS.values(), []))
	columns = np.asarray(ids) - 1
	arrays = {'cycle': np.arange(5001, dtype=np.int64), 'particle_ids': np.asarray(ids, dtype=np.int64)}
	cases = {}
	statistics = {}
	initial_particles = None
	initial_h = None
	for method in METHODS:
		receipt = json.loads((archive / (method + '.json')).read_bytes())
		protocol = dict(run_id=RUN_ID, method=method, rho=.015, cycles=5000, steps_per_cycle=40,
			samples_per_cycle=10, energy_samples_per_cycle=1, particle_count=84, step=.025,
			all_steps_converged=True, all_values_finite=True, energy_feedback=False,
			derivative_history_saved=False, script_sha256=summary['calculation_sha256'],
			reference_sha256=summary['reference_sha256'], result_sha256=original_manifest['files'][method + '.npz']['sha256'])
		for key, expected in protocol.items():
			require(receipt.get(key) == expected, f'{method}: incompatible receipt field {key}.')
		require(figure['input_results_sha256'][method] == receipt['result_sha256'], 'Figure source differs.')
		particles = receipt['initial_particles']
		if initial_particles is None:
			initial_particles = particles
		require(particles == initial_particles, 'The methods do not share initial particles.')
		np.testing.assert_array_equal(receipt['particle_ids'], np.arange(1, 85))
		with np.load(archive / (method + '.npz'), allow_pickle=False) as saved:
			np.testing.assert_array_equal(saved['cycle'], np.arange(5001))
			np.testing.assert_array_equal(saved['particle_ids'], np.arange(1, 85))
			np.testing.assert_allclose(saved['t'], np.arange(50001) / 10, rtol=0, atol=1e-12)
			states = saved['states']
			require(states.shape == (50001, 168) and np.isfinite(states).all(), 'Incomplete original trajectory.')
			data = {key: saved[key] for key in DIAGNOSTICS}
			for key, value in data.items():
				require(value.dtype == np.float64 and value.shape == (5001, 84) and np.isfinite(value).all(), f'Invalid diagnostic {method}/{key}.')
				if key != 'hamiltonian':
					np.testing.assert_array_equal(value[0], np.zeros(84))
			np.testing.assert_array_equal(data['delta_h'], data['hamiltonian'] - data['hamiltonian'][0])
			np.testing.assert_array_equal(data['residual'], data['delta_h'] - data['integral_simpson'])
			np.testing.assert_array_equal(data['quadrature_gap'], data['integral_simpson'] - data['integral_coarse'])
			if initial_h is None:
				initial_h = data['hamiltonian'][0].copy()
			np.testing.assert_array_equal(initial_h, data['hamiltonian'][0])
			for group, group_ids in GROUPS.items():
				selected = np.asarray(group_ids) - 1
				statistics[method, group] = {q: metrics((data['delta_h'] - data[key])[1:, selected])
					for q, key in ((2, 'integral_simpson'), (4, 'integral_coarse'))}
				require(statistics[method, group][2] == receipt['groups'][group]['balance'], 'Original balance statistics differ.')
			for key, value in data.items():
				arrays[method + '/' + key] = np.ascontiguousarray(value[:, columns])
		# Keep scientific fingerprints and numerical settings; omit machine names and private paths.
		cases[method] = {key: receipt[key] for key in ('result_sha256', 'script_sha256', 'reference_sha256',
			'code_sha256', 'source_sha256', 'source_archive_sha256', 'source_files_sha256',
			'temporal_plan_sha256', 'input_bundle_sha256', 'input_bundle_receipt_sha256',
			'all_steps_converged', 'all_values_finite', 'energy_feedback', 'derivative_history_saved',
			'quadrature', 'control', 'time_plan', 'hamiltonian', 'trajectory_agreement', 'versions', 'timing')}
	for series in figure['series']:
		values = statistics[series['method'], series['group']][series['quadrature_update_steps']]
		require(values['block_100_cycle_rms'] == series['values'], 'Recomputed block RMS differs from the thesis figure.')
	for row in figure['global_metrics']:
		values = statistics[row['method'], row['group']]
		require(row['balance_two_steps'] == values[2]['rms'] and row['balance_four_steps'] == values[4]['rms']
			and row['four_over_two'] == values[4]['rms'] / values[2]['rms'], 'Global RMS differs from the thesis figure.')
	study = dict(schema_version=1, study_prefix=STUDY_PREFIX, run_id=RUN_ID, methods=list(METHODS),
		parent_campaign_id=summary['parent_campaign_id'], rho=.015, cycles=5000, step=.025,
		steps_per_cycle=40, samples_per_cycle=10, energy_samples_per_cycle=1,
		integrated_particle_count=84, particle_ids=ids, groups=GROUPS,
		particles=[p for p in initial_particles['particles'] if p['id'] in ids],
		particle_input_sha256=initial_particles['input_sha256'], particle_source_sha256=initial_particles['source_sha256'],
		cases=cases, reference_series=figure['series'], reference_global_metrics=figure['global_metrics'],
		definitions=dict(balance='R(q) = delta_h - I(q), q = 2 or 4; same online-integrated trajectory.',
			block_rms='sqrt(mean(R**2)) over eight particles and 100 whole-cycle endpoints, excluding cycle 0.',
			global_rms='sqrt(mean(R**2)) over eight particles and all 5000 whole-cycle endpoints.',
			limitation='Quadrature sensitivity is not an error bound; small residuals do not certify trajectory accuracy.'))
	buffer = io.BytesIO()
	with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as packed:
		for key, value in sorted(arrays.items()):
			array_buffer = io.BytesIO()
			np.save(array_buffer, value, allow_pickle=False)
			info = zipfile.ZipInfo(key + '.npy')
			info.compress_type = zipfile.ZIP_DEFLATED
			packed.writestr(info, array_buffer.getvalue())
	objects = {'study.json': encoded(study), 'diagnostics.npz': buffer.getvalue()}
	manifest = dict(schema_version=1, study_prefix=STUDY_PREFIX, run_id=RUN_ID,
		files={name: fingerprint(data) for name, data in objects.items()}, source_archives=original_manifest,
		source_manifest_sha256=sha256(manifest_bytes).hexdigest(),
		source_publication_receipt_sha256=sha256(publication_bytes).hexdigest(),
		source_figure_sha256=sha256(figure_bytes).hexdigest(),
		encoding='Lossless float64 diagnostic arrays at cycles 0..5000 for the selected 16 original IDs; original full trajectories remain in the private scientific archive.',
		validation='All original NPZ and JSON hashes, complete trajectories, diagnostic identities, group membership, 32 figure series and 16 global RMS rows verified.')
	objects['manifest.json'] = encoded(manifest)
	if destination.exists():
		if set(p.name for p in destination.iterdir()) != set(objects) or any((destination / name).read_bytes() != data for name, data in objects.items()):
			raise FileExistsError('Choose a new immutable version directory; the destination differs.')
	else:
		destination.mkdir(parents=True)
		for name, data in objects.items():
			(destination / name).write_bytes(data)
	return sha256(objects['manifest.json']).hexdigest()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('destination', type=Path)
	parser.add_argument('--archive', type=Path, default=ARCHIVE)
	args = parser.parse_args()
	print('Manifest SHA-256:', prepare_bundle(args.archive, args.destination))
