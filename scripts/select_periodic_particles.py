"""Prepare reproducible seeds or install eight verified independent particle records."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from studies.main_campaign import SOURCE_SHA256, load_field
from diagnostics.studies.periodic_particles import distinct_orbit, orbit_metrics

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = ROOT / 'notebooks/developements/main_reflected_campaign/main-20261115/runs/rho-0.30-GaussLegendre4'


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(reference: Path, output: Path) -> None:
    manifest = json.loads((reference / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        path = reference / name
        if path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
            raise ValueError(f'Archive verification failed: {name}')
    metadata = json.loads((reference / 'metadata.json').read_text())['experiment']
    if (metadata['source_sha256'] != SOURCE_SHA256 or metadata['rho'] != .3
            or metadata['method'] != 'GaussLegendre4' or metadata['spatial_extension'] != 'reflected'):
        raise ValueError('The candidate reference differs from the agreed reflected rho=0.30 study.')
    stride = metadata['config']['samples_per_cycle']
    size = len(metadata['particle_record']['particles'])
    with np.load(reference / 'solution.npz') as archive:
        times = archive['t'][::stride]
        xy = archive['states'][::stride].reshape(-1, 2, size).transpose(0, 2, 1)
    np.testing.assert_allclose(times, np.arange(len(times)) * metadata['cycle_duration'], atol=1e-10, rtol=0)
    candidates = []
    for cycles in range(1, 33):
        distance = np.linalg.norm(xy[cycles:] - xy[:-cycles], axis=2)
        eligible = np.all((xy[:-cycles] > .03) & (xy[:-cycles] < .97), axis=2)
        eligible &= np.linalg.norm(xy[1:len(xy)-cycles+1] - xy[:-cycles], axis=2) > .002
        distance[~eligible] = np.inf
        seeds = []
        for index in np.argsort(distance, axis=None)[:3000]:
            cycle, particle = np.unravel_index(index, distance.shape)
            if not np.isfinite(distance[cycle, particle]):
                break
            point = xy[cycle, particle]
            if any(np.linalg.norm(point - seed['xy']) < .012 for seed in seeds):
                continue
            seeds.append(dict(period=cycles, xy=point.tolist(), distance=float(distance[cycle, particle]),
                              particle=int(particle + 1), cycle=int(cycle)))
            if len(seeds) == 8:
                break
        candidates.extend(seeds)
    output.mkdir(exist_ok=True, parents=True)
    (output / 'candidates.json').write_text(json.dumps(candidates, indent=2) + '\n')
    (output / 'reference.json').write_text(json.dumps(dict(path=str(reference.relative_to(ROOT)), manifest=manifest), indent=2) + '\n')
    print(f'Verified reference; saved {len(candidates)} candidate seeds.')


def finalize(output: Path, periods: list[int]) -> None:
    if len(periods) != 8 or len(set(periods)) != 8:
        raise ValueError('Choose eight distinct verified periods.')
    results = []
    for cycles in periods:
        attempts = [json.loads(p.read_text()) for p in sorted(output.glob(f'verify-{cycles}-*.json'))
                    if 'receipt' not in p.name]
        accepted = [result for result in attempts if result['accepted']]
        if not accepted:
            raise ValueError(f'Period {cycles} has no accepted verification.')
        result = max(accepted, key=lambda item: item['refinement'][-1]['steps_per_cycle'])
        orbit = np.asarray(result['returns'])
        metrics = orbit_metrics(orbit)
        if (not result['accepted'] or orbit.shape != (cycles + 1, 2)
                or result['period_cycles'] != cycles or metrics['closure'] > 1e-7
                or result['fine_gauss_closure'] > 1e-7 or result['independent_rk4_closure'] > 1e-7
                or max(result['repetition_closures']) > 1e-6
                or result['refinement'][-1]['initial_position_change'] > 1e-7
                or metrics['earlier_return_min'] <= 1e-4 or metrics['cycle_excursion'] <= .002):
            raise ValueError(f'Period {cycles} did not pass the recorded numerical checks.')
        np.testing.assert_array_equal(orbit[0], result['xy'])
        if not all(distinct_orbit(orbit, np.asarray(other['returns'])) for other in results):
            raise ValueError('The selected records contain phase duplicates.')
        results.append(result)

    field = load_field(ROOT / 'data/potential/V1/PHI_2.h5')
    metadata = field.metadata
    path = ROOT / 'data/particles/periodic_8.json'
    record = json.loads(path.read_text())
    historical_origin = np.asarray(record['cell_bounds_m'][:2])
    historical_width = record['cell_bounds_m'][2]
    origin = np.array((metadata.source_x[0], metadata.source_y[0]))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#17becf']
    particles = []
    for index, result in enumerate(results):
        position = origin + metadata.spatial_scale * np.asarray(result['xy'])
        seed = result['candidate']['candidate']
        particles.append(dict(id=85+index, kind='periodic', parent=None, color=colors[index],
            position_m=position.tolist(), cell_fraction=((position-historical_origin)/historical_width).tolist(),
            period_cycles=result['period_cycles'], period_seconds=result['period_cycles'] * metadata.time_scale / field.frequencies[0],
            selection_rho=.3, runtime_initial_position=result['xy'],
            provenance=dict(reference_particle_id=seed['particle'], reference_cycle=seed['cycle'],
                            reference_return_distance=seed['distance'], initial_position_refined=True),
            verification=dict(method='GaussLegendre4',
                steps_per_cycle=result['refinement'][-1]['steps_per_cycle'],
                refinement=[{k:v for k,v in row.items() if k != 'newton_trace'} for row in result['refinement']],
                check_steps_per_cycle=result['check_steps_per_cycle'],
                fine_gauss_closure=result['fine_gauss_closure'], independent_rk4_closure=result['independent_rk4_closure'],
                repetition_count=result['repetition_count'], repetition_closures=result['repetition_closures'],
                earlier_return_min=result['metrics']['earlier_return_min'],
                cycle_excursion=result['metrics']['cycle_excursion'], cycle_returns=result['returns'])))
    task_results = []
    task_receipts = []
    for result_path in sorted(output.glob('*.json')):
        if result_path.name.startswith(('search-', 'verify-')) and 'receipt' not in result_path.name:
            item = json.loads(result_path.read_text())
            task_results.append(dict(task=result_path.stem, **item['resources']))
            receipt = output / f'{result_path.stem}.receipt.json'
            saved_call = json.loads(receipt.read_text())
            inputs = (dict(search_seeds=saved_call['seeds']) if result_path.name.startswith('search-')
                      else dict(candidate=item['candidate'], continued_attempt='attempt' in result_path.stem))
            task_receipts.append(dict(task=result_path.stem, call_id=saved_call['call_id'],
                                     inputs=inputs, result_sha256=digest(result_path)))
    # Persist costs as estimates for measured worker intervals, never as invoices.
    resources = dict(tasks=task_results, task_count=len(task_results),
        sum_worker_wall_seconds=sum(t['worker_wall_seconds'] for t in task_results),
        estimated_requested_resources_usd=sum(t['cost_estimated_requested_resources_usd'] for t in task_results),
        estimated_limit_resources_usd=sum(t['cost_estimated_limit_resources_usd'] for t in task_results),
        currency='USD', billed_cost_status='unavailable',
        scope='Measured completed worker intervals, including search and continued attempts; excludes container startup, idle time, transfer and work lost to preemption.')
    reference = json.loads((output / 'reference.json').read_text())
    sources = ['src/studies/periodic_particles.py', 'src/diagnostics/studies/periodic_particles.py',
               'examples/modal_periodic_particles_app.py', 'scripts/select_periodic_particles.py']
    record.update(title='8 periodic initial conditions', status='verified_numerically', particle_count=8,
        description='Eight distinct nonstationary zero-winding periodic orbits, numerically verified at rho=0.30 for the reflected potential.',
        particles=particles,
        provenance=dict(reference=reference, modal_tasks=task_receipts,
                        reproduction_source_sha256={name:digest(ROOT/name) for name in sources},
                        created_utc=datetime.now(timezone.utc).isoformat()),
        verification=dict(status='passed', rho=.3, spatial_extension='reflected', spatial_period=2.,
            distance_convention='Unwrapped Euclidean closure in normalized base-cell coordinates; zero winding.',
            temporal_phase=0., minimum_period_test='All integer forcing-cycle returns before the claimed period.',
            distinctness_test='All pairs of cycle returns, allowing phase shifts and period-two spatial copies; separation > 1e-4.',
            closure_tolerance=1e-7, initial_position_refinement_tolerance=1e-7,
            repeated_closure_tolerance=1e-6, minimum_earlier_return_distance=1e-4,
            finite_difference_step=2e-6, shooting_tolerance=2e-11,
            gauss_solver_absolute_tolerance=1e-12, gauss_solver_relative_tolerance=1e-11,
            gauss_solver_max_iterations=40, backend='jax', precision='float64', executor='modal',
            qualification='Numerical verification at the stated finite resolutions; not a rigorous existence proof.'),
        execution=resources)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print('Installed eight verified periodic initial conditions:', periods)
    print('Total measured worker seconds:', resources['sum_worker_wall_seconds'])
    print('Estimated compute USD (requested resources):', resources['estimated_requested_resources_usd'])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'finalize'))
    parser.add_argument('--reference', type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument('--output', type=Path, default=ROOT/'build/periodic_search_rho030')
    parser.add_argument('--periods', default='5,6,7,8,9,10,11,12')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.reference, args.output)
    else:
        finalize(args.output, [int(p) for p in args.periods.split(',')])


if __name__ == '__main__':
    main()
