"""Prepare a verified public NumPy/SciPy–JAX dataset without network transfers."""
from __future__ import annotations

import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
import tarfile
import zipfile

import numpy as np

RUN_ID = 'methods-20261119-union16-rho0015-c10-30-90'
METHODS = ('BM4Implicit', 'GaussLegendre4', 'RK2')
CYCLES = (10, 30, 90)


def fingerprint(data):
    return dict(bytes=len(data), sha256=sha256(data).hexdigest())


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def prepare_bundle(archive: Path, destination: Path) -> str:
    """Verify original archives and retain complete paired states, times and timings."""
    study_bytes = (archive / 'study.json').read_bytes()
    study = json.loads(study_bytes)
    publication = json.loads((archive / 'publication.json').read_text())
    expected_cases = [f'{method}-{cycles}/{backend}' for cycles in CYCLES
        for method in METHODS for backend in ('scipy', 'jax')]
    if (study['run_id'] != RUN_ID or study['study_prefix'] != 'studies/bm4_numpy_jax'
            or study['methods'] != list(METHODS) or study['cycles'] != list(CYCLES)
            or study['particle_count'] != 16 or study['rho'] != .015
            or study['reference_campaign'] != 'main-20261117-rho-div20'
            or publication['status'] != 'verified' or publication['bucket'] != study['bucket']
            or publication['verified_cases'] != expected_cases):
        raise ValueError('The selected complete reference publication is required.')
    derived = json.loads((archive / 'derived-publication.json').read_text())
    summary_bytes = (archive / 'summary.json').read_bytes()
    if derived['status'] != 'verified' or sha256(summary_bytes).hexdigest() != derived['files_sha256']['summary.json']:
        raise ValueError('The saved reference summary is not verified.')
    reference_rows = {(r['method'], r['cycles']): r for r in json.loads(summary_bytes)['rows']}
    particles = {}
    with tarfile.open(archive / 'source.tar.gz') as source:
        for group, filename in [('chaotic', 'chaotic_candidates.json'), ('periodic', 'periodic_candidates_8.json')]:
            raw = source.extractfile('data/particles/' + filename).read()
            frozen = next(s for s in study['particle_selection']['source_sets'] if s['source_file'] == filename)
            if sha256(raw).hexdigest() != frozen['sha256']:
                raise ValueError('The frozen particle selection differs from the recorded input.')
            particles[group] = dict(record=json.loads(raw), source_file=filename, source_sha256=sha256(raw).hexdigest())
    trajectory_files = {}
    cases = {}
    sources = {}
    for cycles in CYCLES:
        for method in METHODS:
            key = f'{method}-{cycles}'
            pair_bytes = (archive / 'runs' / key / 'pair.json').read_bytes()
            pair = json.loads(pair_bytes)
            receipt = json.loads((archive / 'receipts' / (key + '.json')).read_text())
            if (receipt['submission'] != 'confirmed' or receipt['input_sha256'] != pair['payload_sha256']
                    or sha256((archive / (key + '.payload')).read_bytes()).hexdigest() != receipt['input_sha256']):
                raise ValueError('A paired result has an unconfirmed or different input.')
            expected_order = [(backend, repetition) for repetition in range(4)
                for backend in (('scipy', 'jax') if repetition % 2 == 0 else ('jax', 'scipy'))]
            if [(m['backend'], m['repetition']) for m in pair['measurements']] != expected_order or pair['environment'].get('x64') is not True:
                raise ValueError('The original measurement order or float64 provenance is invalid.')
            for measurement in pair['measurements']:
                first = measurement['repetition'] == 0
                cache = (int(first), int(not first)) if measurement['backend'] == 'jax' else (0, 0)
                if (measurement['phase'] != ('first' if first else 'warm')
                        or (measurement['integration_cache_misses'], measurement['integration_cache_hits']) != cache
                        or not np.isfinite([measurement['seconds'], measurement['process_seconds']]).all()
                        or measurement['seconds'] <= 0 or measurement['process_seconds'] < 0):
                    raise ValueError('The original measurement or compilation-cache audit is invalid.')
            cases[key] = dict(pair=pair, call_id=receipt['call_id'])
            arrays = {}
            for backend in ('scipy', 'jax'):
                path = archive / 'runs' / key / backend
                original_manifest_bytes = (path / 'manifest.json').read_bytes()
                original_manifest = json.loads(original_manifest_bytes)
                for name, expected in original_manifest['files'].items():
                    if fingerprint((path / name).read_bytes()) != expected:
                        raise ValueError(f'Damaged original archive: {key}/{backend}/{name}')
                metadata = json.loads((path / 'metadata.json').read_text())
                original = metadata['experiment']
                if (metadata['schema_version'] != 2 or original['pair'] != pair
                        or original['call_id'] != receipt['call_id'] or original['backend'] != backend
                        or original['method'] != method or original['rho'] != study['rho']
                        or original['source_sha256'] != study['source_sha256']
                        or original['code_sha256'] != study['code_sha256']
                        or original['config'] != dict(cycles=cycles, samples_per_cycle=10, steps_per_cycle=40)
                        or metadata['diagnostics']['step_count']['scalar'] != 40 * cycles):
                    raise ValueError('The archived numerical configuration differs from the study.')
                record = original['particle_record']['particles']
                if [p['id'] for p in record] != study['particle_ids']:
                    raise ValueError('Unexpected particle-column order.')
                for group in particles.values():
                    for particle in group['record']['particles']:
                        if particle != record[study['particle_ids'].index(particle['id'])]:
                            raise ValueError('A frozen particle differs from its archived input.')
                with np.load(path / 'solution.npz', allow_pickle=False) as saved:
                    arrays[backend] = {name: saved[name].copy() for name in ('t', 'states')}
                for name, value in arrays[backend].items():
                    buffer = io.BytesIO()
                    np.save(buffer, value, allow_pickle=False)
                    trajectory_files[f'{key}/{backend}/{name}.npy'] = buffer.getvalue()
                sources[f'{key}/{backend}'] = dict(manifest_sha256=sha256(original_manifest_bytes).hexdigest(),
                    files=original_manifest['files'], call_id=receipt['call_id'])
            left, right = arrays['scipy'], arrays['jax']
            if left['states'].shape != (cycles * 10 + 1, 32) or right['states'].shape != left['states'].shape:
                raise ValueError('Incomplete trajectory.')
            np.testing.assert_array_equal(left['t'], right['t'])
            np.testing.assert_array_equal(left['states'][0], right['states'][0])
            delta = left['states'] - right['states']
            distances = np.hypot(delta[:, :16], delta[:, 16:])
            reference = reference_rows[method, cycles]
            if float(distances.max()) != reference['maximum_distance']:
                raise ValueError('The original all-particle distance does not match its saved summary.')
            for group, data in particles.items():
                columns = [study['particle_ids'].index(p['id']) for p in data['record']['particles']]
                if float(distances[:, columns].max()) != reference['group_distances'][group]['maximum_distance']:
                    raise ValueError('A group distance differs from the figure-source summary.')
            for backend in ('scipy', 'jax'):
                warm = [m['seconds'] for m in pair['measurements'] if m['backend'] == backend and m['phase'] == 'warm']
                if warm != reference['backends'][backend]['warm_seconds'] or float(np.median(warm)) != reference['backends'][backend]['median_seconds']:
                    raise ValueError('Individual times differ from the figure-source summary.')
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as packed:
        for name, data in sorted(trajectory_files.items()):
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_DEFLATED
            packed.writestr(info, data)
    objects = {'study.json': study_bytes, 'timings.json': encoded(cases),
        'particles.json': encoded(particles), 'trajectories.npz': buffer.getvalue()}
    manifest = dict(schema_version=1, study_prefix=study['study_prefix'], run_id=RUN_ID,
        files={name: fingerprint(data) for name, data in objects.items()},
        source_publication=publication, source_archives=sources,
        source_summary_sha256=sha256(summary_bytes).hexdigest(),
        source_tar_sha256=sha256((archive / 'source.tar.gz').read_bytes()).hexdigest(),
        trajectory_encoding='Lossless float64 t and states arrays from schema-2 original NPZ archives; time rows, 16 x columns then 16 y columns.',
        original_audit='The private publication receipt records read-back verification of all 18 trajectories. This bundle preserves all saved paired states and individual times; repeated trajectories were not retained.')
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
    parser.add_argument('--archive', type=Path, default=Path(__file__).resolve().parents[2] / 'outputs/studies/bm4_numpy_jax' / RUN_ID)
    args = parser.parse_args()
    print('Manifest SHA-256:', prepare_bundle(args.archive, args.destination))
