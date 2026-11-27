"""Verify the projection study's inputs against the existing public frozen kit."""
from hashlib import file_digest, sha256
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT.parent / 'TFM_redaccion/datos/campana_referencia.json'
PUBLIC = ROOT / 'outputs/colab-public/04_05_correction_projection/calculation/v1'
BASE = ROOT / 'outputs/colab-public/04_04_evolution_energetique/calculation/v1'
METHODS = ('ABBA2Implicit', 'BM4Implicit')


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def prepare(destination=PUBLIC):
    """Keep source and potential byte-identical; audit the shared time plan."""
    ref = json.loads(REFERENCE.read_text())
    sys.path.insert(0, str(ROOT / 'src'))
    from studies.main_campaign import campaign_case_key
    campaign = ROOT / ref['campaign_path']
    base = json.loads((BASE / 'manifest.json').read_text())
    source = campaign / 'source.tar.gz'
    canonical = ROOT.parent / 'TFM_redaccion/notebooks/04_04_02_balance_energetico_en_linea/inputs/ABBA2Implicit.inputs.npz'
    with np.load(canonical, allow_pickle=False) as data:
        expected = {key: data[key].copy() for key in data.files}
    experiment = json.loads(canonical.with_suffix('.json').read_text())['metadata']['experiment']
    particle_record = experiment['particle_record']
    assert [p['id'] for p in particle_record['particles']] == list(range(1, 85))
    fingerprints = {key: sha256(value.tobytes()).hexdigest() for key, value in expected.items()}
    cases = []
    for rho in ref['rho_values']:
        for method in METHODS:
            key = campaign_case_key(rho, method)
            item = dict(case_key=key, rho=rho, method=method)
            if key in ref['unavailable_cases']:
                cases.append(dict(item, reference_status='unavailable', reason='nonlinear_convergence_failure'))
                continue
            case = campaign / 'runs' / key
            manifest = json.loads((case / 'manifest.json').read_text())
            hashes = {}
            for name in ('metadata.json', 'solution.npz'):
                hashes[name] = digest(case / name)
                assert hashes[name] == manifest['files'][name]['sha256']
            metadata = json.loads((case / 'metadata.json').read_text())
            exp, diag = metadata['experiment'], metadata['diagnostics']
            assert exp['method'] == method and exp['rho_runtime'] == rho
            assert exp['code_sha256'] == ref['code_sha256'] and exp['source_sha256'] == ref['source_sha256']
            assert exp['config'] == ref['config']
            with np.load(case / 'solution.npz', allow_pickle=False) as values:
                for name, expected_value in expected.items():
                    source_name = name if name in ('initial_state', 't') else diag[name]['array']
                    np.testing.assert_array_equal(values[source_name], expected_value)
            cases.append(dict(item, reference_status='complete', archived_files_sha256=hashes))
    protocol = dict(schema_version=1, study_prefix='studies/projection_correction',
        reference_run_id=ref['campaign_id'], methods=list(METHODS), rho_values=ref['rho_values'],
        config=ref['config'], particle_count=84, samples_per_cycle=10,
        source_code_sha256=ref['code_sha256'], source_sha256=ref['source_sha256'],
        source_tar_sha256=digest(source), reference_sha256=digest(REFERENCE),
        base_kit_sha256=base['sha256'], base_manifest_sha256=digest(BASE / 'manifest.json'),
        base_manifest_url='https://f003.backblazeb2.com/file/gc2d-colab-public/04_04_evolution_energetique/calculation/v1/manifest.json',
        particle_record=particle_record, initial_positions=experiment['initial_positions'],
        physical_parameters={key: experiment[key] for key in ('magnetic_field', 'source_field_indices',
            'spatial_extension', 'spatial_period', 'spatial_scale_m', 'time_scale_seconds', 'cycle_duration',
            'newton_absolute_tolerance', 'newton_relative_tolerance', 'newton_max_iterations')},
        shared_input_method='ABBA2Implicit', shared_input_bundle_sha256=digest(canonical),
        shared_array_sha256=fingerprints,
        temporal_plan_provenance='Initial states, requested output times and complete accepted-step clocks are bitwise equal in all 21 available projection archives. The unavailable rho=0 ABBA2 case uses the same planned input arrays for its new attempt.',
        cases=cases)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / 'protocol.json'
    target.write_text(json.dumps(protocol, sort_keys=True, indent=2) + '\n')
    return protocol


if __name__ == '__main__':
    record = prepare()
    print(f"Verified {sum(c['reference_status'] == 'complete' for c in record['cases'])} archived input plans; no new public objects are required.")
