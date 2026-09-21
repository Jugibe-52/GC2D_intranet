"""Validate downloaded artifacts without running the integrator."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import study_io

RUN = 'aws_48p_5000c_20s_16proc_spot_20260920'
study_io.ROOT = ROOT if (ROOT / 'resultados' / RUN / 'COMPLETE.json').exists() else ROOT / 'aws' / 'download_staging'
directory, meta, arrays, positions, initial = study_io.load_calculation(RUN)
eq = np.testing.assert_array_equal
for key, value in {'particle_count': 48, 'cycles': 5000, 'steps_per_cycle': 20,
                   'complete_steps': 100000, 'process_count': 16,
                   'reference_computed': False, 'newton_history_recorded': True,
                   'method': 'BM4Implicit', 'arithmetic': 'float64'}.items():
    assert meta[key] == value, key
launch = json.loads((ROOT / 'aws' / 'launch_record.json').read_text())
assert meta['checkpoint_driver_sha256'] == launch['source_sha256']['checkpoint_calculation.py']
assert meta['diagnostic_instrumentation_sha256'] == launch['source_sha256']['newton_diagnostics.py']
assert meta['snapshot_sha256'] == study_io.digest(ROOT / 'assets' / 'gc2d_snapshot.zip')
for key, value in arrays.items():
    if value.dtype.kind in 'if':
        assert np.isfinite(value).all(), key
assert arrays['states'].shape == (96, 100001)
assert arrays['cycle_positions'].shape == (5001, 48, 2)
eq(arrays['times'], np.arange(100001) * 0.05)
eq(arrays['cycle_times'], np.arange(5001))
eq(arrays['cycle_positions'][:, :, 0], arrays['states'][:48, ::20].T)
eq(arrays['cycle_positions'][:, :, 1], arrays['states'][48:, ::20].T)
eq(arrays['initial_positions'], arrays['cycle_positions'][0])
eq(arrays['particle_ids'], np.arange(1, 49))
assert len(set(arrays['colors_hex'])) == 48
assert len(initial) == 48 and len(positions) == 240000
eq(positions['cycle'], np.repeat(np.arange(1, 5001), 48))
eq(positions['particle'], np.tile(np.arange(1, 49), 5000))
eq(positions['color'], np.tile(arrays['colors_hex'], 5000))
eq(initial['color'], arrays['colors_hex'])
eq(positions[['x_unwrapped', 'y_unwrapped']], arrays['cycle_positions'][1:].reshape(-1, 2))
eq(positions[['x_wrapped', 'y_wrapped']], arrays['cycle_positions_wrapped'][1:].reshape(-1, 2))
eq(initial[['x_unwrapped', 'y_unwrapped']], arrays['initial_positions'])
eq(initial['initial_radius_over_L'], np.linspace(0, .49, 48))

assert len(meta['workers']) == 16
for w, worker in enumerate(meta['workers']):
    assert worker['worker'] == w + 1
    assert worker['particle_ids'] == [w + 1, w + 17, w + 33]
    assert worker['resumed_steps'] + worker['new_steps'] == 100000
    assert worker['checkpoint_chunks'] == 200
    assert worker['snapshot_sha256'] == meta['snapshot_sha256']

for key in ['nonlinear_iterations', 'nonlinear_residuals', 'nonlinear_tolerances',
            'projection_multiplier_norms']:
    assert arrays[key].shape == (16, 100000), key
offsets = arrays['newton_history_offsets']
assert offsets.shape == (16, 100001)
eq(np.diff(offsets, axis=1), arrays['nonlinear_iterations'] + 1)
assert offsets[0, 0] == 0
eq(offsets[1:, 0], offsets[:-1, -1])
nr = arrays['newton_history_residuals']
mu = arrays['newton_history_mu_norms']
assert len(nr) == len(mu) == offsets[-1, -1]
ends = offsets[:, 1:].reshape(-1)
starts = offsets[:, :-1].reshape(-1)
corrections = arrays['nonlinear_iterations'].reshape(-1)
tolerances = arrays['nonlinear_tolerances'].reshape(-1)
eq(nr[ends - 1], arrays['nonlinear_residuals'].reshape(-1))
eq(mu[ends - 1], arrays['projection_multiplier_norms'].reshape(-1))
assert (nr[ends - 1] <= tolerances).all()
eq(mu[starts], np.zeros(len(starts)))

step_rows = 0
for frame in pd.read_csv(directory / 'newton_steps.csv.gz', chunksize=100000,
                         float_precision='round_trip'):
    row = np.arange(step_rows, step_rows + len(frame))
    w, s = row // 100000, row % 100000
    eq(frame['worker'], w + 1)
    eq(frame['step'], s + 1)
    eq(frame['cycle'], s // 20 + 1)
    eq(frame['time_normalized'], arrays['times'][s + 1])
    eq(frame['newton_corrections'], corrections[row])
    eq(frame['residual_evaluations'], ends[row] - starts[row])
    eq(frame['initial_residual_inf'], nr[starts[row]])
    eq(frame['final_residual_inf'], nr[ends[row] - 1])
    eq(frame['mu_inf'], mu[ends[row] - 1])
    eq(frame['tolerance'], tolerances[row])
    eq(frame['residual_over_tolerance'], nr[ends[row] - 1] / tolerances[row])
    assert frame['converged'].all()
    step_rows += len(frame)
assert step_rows == 1600000

iteration_rows = 0
for frame in pd.read_csv(directory / 'newton_iterations.csv.gz', chunksize=100000,
                         float_precision='round_trip'):
    row = np.arange(iteration_rows, iteration_rows + len(frame))
    global_step = np.searchsorted(ends, row, side='right')
    iteration = row - starts[global_step]
    eq(frame['worker'], global_step // 100000 + 1)
    eq(frame['step'], global_step % 100000 + 1)
    eq(frame['time_normalized'], arrays['times'][global_step % 100000 + 1])
    eq(frame['newton_iteration'], iteration)
    eq(frame['residual_inf'], nr[row])
    eq(frame['mu_inf'], mu[row])
    eq(frame['tolerance'], tolerances[global_step])
    eq(frame['residual_over_tolerance'], nr[row] / tolerances[global_step])
    eq(frame['accepted'], iteration == corrections[global_step])
    iteration_rows += len(frame)
assert iteration_rows == len(nr)

for category, notebook in [('resultados', 'calculo_executed.ipynb'),
                           ('figuras', 'visualizacion_executed.ipynb')]:
    folder = study_io.ROOT / category / RUN
    status = json.loads((folder / 'execution_status.json').read_text())
    assert status['status'] == 'success' and status['run_id'] == RUN
    nb = json.loads((folder / notebook).read_text())
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and ''.join(cell['source']).strip():
            assert cell['execution_count'] is not None
        assert not any(out.get('output_type') == 'error' for out in cell.get('outputs', []))

report = {
    'checked_utc': datetime.now(timezone.utc).isoformat(), 'run_id': RUN,
    'internal_validation': 'passed', 'numeric_sha256_files_verified': 7,
    'return_position_rows': len(positions), 'initial_particles': len(initial),
    'workers': 16, 'complete_steps_per_worker': 100000,
    'newton_step_rows': step_rows, 'newton_iteration_rows': iteration_rows,
    'all_newton_steps_converged': True,
    'maximum_residual_over_tolerance': float(np.max(nr[ends - 1] / tolerances)),
    'mu_norm_definition': meta['mu_norm_definition'],
    'maximum_mu_inf': float(mu.max()), 'executed_notebooks_successful': 2,
    'checkpoint_chunks_reported': sum(w['checkpoint_chunks'] for w in meta['workers']),
    'external_archive_checksum_verified': False,
    'remaining': 'Compare archive checksum with S3 sidecar and cloud status; cancel persistent Spot request.'
}
study_io.atomic_json(ROOT / 'aws' / 'validation_download_internal.json', report)
print(json.dumps(report, indent=2))
