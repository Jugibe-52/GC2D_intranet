"""Verify the 21 archived cases and prepare a small, credential-free viewer bundle."""
from __future__ import annotations

import argparse
from hashlib import file_digest, sha256
import json
from pathlib import Path
import sys

import numpy as np

STUDY_PREFIX = 'studies/projection_correction'
RUN_ID = 'main-20261117-rho-div20'
METHODS = ('ABBA2Implicit', 'BM4Implicit')
STATISTICS = ('sampled_median', 'sampled_maximum', 'newton_mean_iterations', 'newton_max_iterations')


def require(condition: bool, message: str) -> None:
	if not condition:
		raise ValueError(message)


def fingerprint(path: Path) -> dict:
	with path.open('rb') as source:
		return {'bytes': path.stat().st_size, 'sha256': file_digest(source, 'sha256').hexdigest()}


def write_json(path: Path, value: dict) -> None:
	path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def prepare(repository: Path, thesis: Path, destination: Path) -> dict:
	"""Recompute reductions from checked archives, without integrating or altering sources."""
	sys.path.insert(0, str(repository / 'src'))
	from studies.main_campaign import campaign_case_key
	ref_path = thesis / 'datos/campana_referencia.json'
	summary_path = thesis / 'datos/correccion_proyeccion.json'
	reference = json.loads(ref_path.read_text())
	saved = json.loads(summary_path.read_text())
	campaign_path = repository / reference['campaign_path']
	campaign = json.loads((campaign_path / 'campaign.json').read_text())
	progress = json.loads((campaign_path / 'progress.json').read_text())['cases']
	require(saved['reference_sha256'] == fingerprint(ref_path)['sha256'], 'Reference hash mismatch.')
	require(saved['script_sha256'] == fingerprint(thesis / 'notebooks/correccion_proyeccion.py')['sha256'], 'Extraction source hash mismatch.')
	require(saved['campaign_sha256'] == fingerprint(campaign_path / 'campaign.json')['sha256'], 'Campaign hash mismatch.')
	for key in ('config', 'source_sha256', 'code_sha256', 'methods', 'bucket'):
		require(campaign[key] == reference[key], f'Campaign field mismatch: {key}.')
	require(campaign['rhos'] == reference['rho_values'] == (np.arange(11) / 400).tolist(), 'Unexpected radii.')
	require(reference['rho_convention'] == 'base_cell_width' and reference['particle_count'] == 84, 'Unexpected coordinates or particle count.')
	require(reference['config'] == dict(cycles=5000, steps_per_cycle=40, samples_per_cycle=10), 'Unexpected sampling.')
	original_rows = {row['case_key']: row for row in saved['cases']}
	rows, proofs = [], []
	for rho in reference['rho_values']:
		for method in METHODS:
			key = campaign_case_key(rho, method)
			row = dict(case_key=key, rho=rho, method=method)
			if key in reference['unavailable_cases']:
				require(method == 'ABBA2Implicit' and rho == 0 and progress[key]['status'] != 'complete', 'Unexpected unavailable case.')
				row.update(status='unavailable', reason='nonlinear_convergence_failure')
				require(row == original_rows[key], f'Unavailable record changed: {key}.')
				rows.append(row)
				continue
			require(progress[key]['status'] == 'complete', f'Incomplete source case: {key}.')
			case = campaign_path / 'runs' / key
			manifest = json.loads((case / 'manifest.json').read_text())
			files = {name: fingerprint(case / name) for name in ('metadata.json', 'solution.npz')}
			for name, actual in files.items():
				require(actual == manifest['files'][name], f'Archive size/SHA-256 mismatch: {key}/{name}.')
				require(actual['sha256'] == original_rows[key]['files_sha256'][name], f'Summary source mismatch: {key}/{name}.')
			metadata = json.loads((case / 'metadata.json').read_text())
			experiment, diagnostics = metadata['experiment'], metadata['diagnostics']
			for name in ('config', 'code_sha256', 'source_sha256'):
				require(experiment[name] == reference[name], f'Experiment field mismatch: {key}/{name}.')
			require(experiment['method'] == method and experiment['rho_runtime'] == experiment['rho'] == rho, 'Method/radius mismatch.')
			require(experiment['schema_version'] == 2 and experiment['projection_units'] == 'normalized physical coordinates', 'Projection convention mismatch.')
			require([p['id'] for p in experiment['particle_record']['particles']] == list(range(1, 85)), 'Particle order mismatch.')
			for name, value in dict(step_count=200000, diagnostic_stride=4, output_interpolation_count=0,
				projection_placement='around_complete_composition', projection_formulation='reduced_multiplier',
				projection_iteration_scope='joint_particle_batch').items():
				require(diagnostics[name]['scalar'] == value, f'Diagnostic convention mismatch: {name}.')
			array_proofs = {}
			with np.load(case / 'solution.npz', allow_pickle=False) as archive:
				def read(name: str, shape: tuple) -> np.ndarray:
					values = archive[diagnostics[name]['array']]
					require(values.shape == shape and np.isfinite(values).all(), f'Invalid diagnostic: {key}/{name}.')
					array_proofs[name] = dict(shape=list(values.shape), dtype=str(values.dtype),
						sha256=sha256(values.tobytes(order='C')).hexdigest())
					return values
				mu = read('projection_multipliers_sampled', (50000, 2, 84))
				infinity = read('projection_particle_norms', (50000, 84))
				blocks = read('projection_particle_block_max', (50000, 84))
				global_norm = read('projection_multiplier_norms', (200000,))
				iterations = read('nonlinear_iterations', (200000,))
				residual = read('nonlinear_residual_norms', (200000,))
				tolerance = read('nonlinear_tolerances', (200000,))
				np.testing.assert_array_equal(infinity, np.max(np.abs(mu), axis=1))
				np.testing.assert_array_equal(infinity.max(axis=1), global_norm[3::4])
				np.testing.assert_array_equal(blocks.max(axis=1), global_norm.reshape(-1, 4).max(axis=1))
				require((blocks >= infinity).all() and (infinity >= 0).all(), 'Invalid infinity-norm block maxima.')
				norm = np.hypot(mu[:, 0, :], mu[:, 1, :])
				require((norm >= infinity).all() and (norm <= np.sqrt(2) * infinity).all(), 'Invalid Euclidean norm bounds.')
				require((tolerance > 0).all() and (residual >= 0).all() and (residual <= tolerance).all(), 'Invalid Newton convergence.')
				require((iterations >= 0).all() and (iterations <= 40).all() and (iterations == np.floor(iterations)).all(), 'Invalid Newton counts.')
				row.update(status='complete', files_sha256={name: proof['sha256'] for name, proof in files.items()},
					sampled_median=float(np.median(norm)), sampled_maximum=float(norm.max()),
					newton_mean_iterations=float(iterations.mean()), newton_max_iterations=int(iterations.max()),
					max_residual_to_tolerance=float((residual / tolerance).max()))
			require(row == original_rows[key], f'Recomputed statistics differ from the thesis summary: {key}.')
			rows.append(row)
			proofs.append(dict(case_key=key, files=files, arrays=array_proofs,
				manifest=fingerprint(case / 'manifest.json'), all_checked_identities_hold=True,
				statistics_equal_thesis_summary=True))
			print(f'Verified {key}: hashes, seven diagnostic arrays and all five statistics.', flush=True)
	require(len(rows) == 22 and len(proofs) == 21, 'Incomplete case inventory.')
	summary = dict(schema_version=1, study_prefix=STUDY_PREFIX, run_id=RUN_ID,
		campaign_id=RUN_ID, config=reference['config'], particle_count=84,
		rho_values=reference['rho_values'], rho_convention='base_cell_width', representative_rho=.015,
		sampled_values_per_case=4200000, particle_steps_per_case=16800000,
		definitions=dict(norm='Euclidean norm hypot(mu_x, mu_y) per particle, in base-cell-width units.',
			sampled_median='Median over 84 particles and the final step of each four-step block; no initial value.',
			sampled_maximum='Maximum over the same 84 particles and 50000 saved correction samples; no initial value.',
			newton='Iterations and residual for the joint solve of all 84 particles at every one of 200000 steps.'),
		limitations='The all-step Euclidean maximum cannot be reconstructed: multiplier components were saved every four steps, while intermediate block maxima use the infinity norm.',
		cases=rows)
	series = [dict(method=method, statistic=stat, rho=reference['rho_values'],
		values=[row.get(stat) for row in rows if row['method'] == method]) for method in METHODS for stat in STATISTICS]
	plot_proofs = {}
	for name in ('correccion_proyeccion_fr', 'iteraciones_newton_fr'):
		path = thesis / 'figuras' / (name + '.json')
		plot = json.loads(path.read_text())
		require(plot['summary_sha256'] == fingerprint(summary_path)['sha256'] and plot['reference_sha256'] == saved['reference_sha256'], 'Plot source-data mismatch.')
		for curve in plot['series']:
			require(curve in series, 'Plot series differs from verified reductions.')
		plot_proofs[name] = dict(json=fingerprint(path), recorded_plot_source_sha256=plot['plot_source_sha256'],
			series=plot['series'])
	provenance = dict(schema_version=1, study_prefix=STUDY_PREFIX, run_id=RUN_ID,
		audit='All 21 metadata/solution archive byte counts and SHA-256 fingerprints were checked, seven arrays per case were checked, reductions were recomputed and exactly matched the retained thesis statistics.',
		archive_solution_bytes=sum(p['files']['solution.npz']['bytes'] for p in proofs),
		array_digest_convention='SHA-256 of array.tobytes(order="C"); dtype and shape recorded separately.',
		sources=dict(reference=fingerprint(ref_path), summary=fingerprint(summary_path),
			campaign=fingerprint(campaign_path / 'campaign.json'), extraction_script=fingerprint(thesis / 'notebooks/correccion_proyeccion.py'),
			current_plot_script=fingerprint(thesis / 'notebooks/correccion_proyeccion_graficas.py')),
		plots=plot_proofs, cases=proofs)
	destination.mkdir(parents=True, exist_ok=True)
	write_json(destination / 'summary.json', summary)
	write_json(destination / 'provenance.json', provenance)
	manifest = dict(schema_version=1, study_prefix=STUDY_PREFIX, run_id=RUN_ID,
		files={name: fingerprint(destination / name) for name in ('summary.json', 'provenance.json')})
	write_json(destination / 'manifest.json', manifest)
	print(json.dumps({name: fingerprint(destination / name) for name in (*manifest['files'], 'manifest.json')}, indent=2))
	return manifest


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--thesis', type=Path, required=True)
	parser.add_argument('--destination', type=Path, required=True)
	args = parser.parse_args()
	prepare(Path(__file__).resolve().parents[2], args.thesis.resolve(), args.destination.resolve())
