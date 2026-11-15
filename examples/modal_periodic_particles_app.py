"""Independent fixed-step JAX shooting jobs; results stay local, without B2."""
from pathlib import Path
import json
import sys
import modal

ROOT = Path(__file__).resolve().parents[1]
image = (modal.Image.debian_slim(python_version=f'{sys.version_info.major}.{sys.version_info.minor}')
    .add_local_file(ROOT / 'constraints.txt', '/tmp/gc2d-constraints.txt', copy=True)
    .pip_install_from_pyproject(str(ROOT / 'pyproject.toml'), optional_dependencies=['jax'],
                               extra_options='-c /tmp/gc2d-constraints.txt')
    .env({'JAX_ENABLE_X64': 'true', 'JAX_PLATFORMS': 'cpu'})
    .add_local_python_source('contracts', 'diagnostics', 'dynamics', 'execution', 'formulations',
                            'states', 'methods', 'potential', 'solvers', 'problem', 'request', 'simulation', 'studies')
    .add_local_file(ROOT / 'data/potential/V1/PHI_2.h5', '/data/PHI_2.h5'))
app = modal.App('gc2d-periodic-particles')


@app.function(image=image, cpu=(2., 2.), memory=(4096, 8192), timeout=3600,
              retries=0, max_containers=6, scaledown_window=30)
def search(cycles: int, candidates: list[dict]) -> dict:
    from studies.main_campaign import load_field
    from studies.periodic_particles import search_period
    from diagnostics.studies.campaign_resources import start_measurement, finish_measurement
    started = start_measurement()
    result = search_period(load_field(Path('/data/PHI_2.h5')), cycles, candidates)
    result['resources'] = finish_measurement(started)
    return result


@app.function(image=image, cpu=(2., 2.), memory=(4096, 8192), timeout=3600,
              retries=0, max_containers=6, scaledown_window=30)
def verify(candidate: dict, prior: dict | None = None) -> dict:
    from studies.main_campaign import load_field
    from studies.periodic_particles import verify_orbit
    from diagnostics.studies.campaign_resources import start_measurement, finish_measurement
    started = start_measurement()
    result = verify_orbit(load_field(Path('/data/PHI_2.h5')), candidate, prior)
    result['resources'] = finish_measurement(started)
    return result


@app.local_entrypoint()
def main(periods: str = '5,6,7,8,9,10,11,12', mode: str = 'search', candidate_index: int = 0, attempt: int = 1) -> None:
    if mode not in ('search', 'verify'):
        raise ValueError('Choose search or verify.')
    output = ROOT / 'build/periodic_search_rho030'
    candidates = json.loads((output / 'candidates.json').read_text())
    jobs = []
    for cycles in map(int, periods.split(',')):
        label = f'search-{cycles}' if mode == 'search' else f'verify-{cycles}-{candidate_index}'
        if mode == 'verify' and attempt > 1:
            label += f'-attempt{attempt}'
        path = output / f'{label}.json'
        if path.exists():
            continue
        seeds = [c for c in candidates if c['period'] == cycles][:8]
        receipt = output / f'{label}.receipt.json'
        if receipt.exists():
            call = modal.FunctionCall.from_id(json.loads(receipt.read_text())['call_id'])
        else:
            if mode == 'search':
                call = search.spawn(cycles, seeds)
            else:
                found = json.loads((output / f'search-{cycles}.json').read_text())
                prior = None
                if attempt > 1:
                    prior_label = f'verify-{cycles}-{candidate_index}'
                    if attempt > 2:
                        prior_label += f'-attempt{attempt-1}'
                    prior_path = output / f'{prior_label}.json'
                    if prior_path.exists():
                        prior = json.loads(prior_path.read_text())
                    else:
                        prior_receipt = json.loads((output / f'{prior_label}.receipt.json').read_text())
                        prior = modal.FunctionCall.from_id(prior_receipt['call_id']).get()
                    if prior['accepted']:
                        raise ValueError('This candidate is already verified; do not repeat it.')
                call = verify.spawn(found['orbits'][candidate_index], prior)
            receipt.write_text(json.dumps(dict(call_id=call.object_id, cycles=cycles, seeds=seeds), indent=2)+'\n')
        jobs.append((cycles, path, call))
    for cycles, path, call in jobs:
        result = call.get()
        path.write_text(json.dumps(result, indent=2)+'\n')
        summary = f'{len(result["orbits"])} distinct roots' if mode == 'search' else f'accepted={result["accepted"]}: {result["reason"]}'
        print(f'Period {cycles}: {summary}; {result["resources"]["worker_wall_seconds"]:.1f} s', flush=True)
