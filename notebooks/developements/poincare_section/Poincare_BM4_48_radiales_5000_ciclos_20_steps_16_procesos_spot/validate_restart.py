"""Check a resumed smoke run against the unmodified public BM4 integrator."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import json
import numpy as np
import pandas as pd
from study_io import ROOT, load_calculation, atomic_json, utc_now, load_snapshot
from parallel_calculation import calculate_group


def main():
    folder, meta, a, positions, initial = load_calculation('checkpoint_resume_test')
    assert meta['validation_run'] and (meta['particle_count'], meta['process_count'], meta['cycles']) == (48, 16, 2)
    assert all(w['resumed_steps'] == 10 and w['new_steps'] == 30 for w in meta['workers'])
    assert len(positions) == 96 and len(set(a['colors_hex'])) == 48
    np.testing.assert_array_equal(np.diff(a['newton_history_offsets'], axis=1), a['nonlinear_iterations'] + 1)
    settings = {k: meta[k] for k in ('rho','coupling_frequency','newton_atol','newton_rtol',
                                   'newton_max_iterations','jacobian_relative_step','t_span','step')}
    settings.update(n_steps=40, record_newton_history=False)
    load_snapshot()
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn')) as pool:
        jobs = [pool.submit(calculate_group, np.arange(w,48,16), a['initial_positions'], settings) for w in range(16)]
        for w, job in enumerate(jobs):
            result = job.result()
            ids = result['indices']
            expected = np.stack((a['states'][ids].T, a['states'][48+ids].T), axis=-1)
            np.testing.assert_array_equal(result['xy'], expected)
            np.testing.assert_array_equal(result['times'], a['times'])
            for original, saved in [('nonlinear_iterations','nonlinear_iterations'),
                                    ('nonlinear_residual_norms','nonlinear_residuals'),
                                    ('nonlinear_tolerances','nonlinear_tolerances'),
                                    ('projection_multiplier_norms','projection_multiplier_norms')]:
                np.testing.assert_array_equal(result['diagnostics'][original], a[saved][w])
    steps = pd.read_csv(folder/'newton_steps.csv.gz')
    history = pd.read_csv(folder/'newton_iterations.csv.gz')
    assert len(steps) == 640 and len(history) == len(a['newton_history_residuals'])
    report = {'verified_utc':utc_now(), 'resumed_steps_per_worker':10, 'workers':16,
              'particles':48, 'total_steps_per_worker':40, 'baseline':'unmodified public simulate BM4Implicit',
              'all_states_and_diagnostics_bitwise_equal':True, 'cycle_rows':len(positions),
              'newton_step_rows':len(steps), 'newton_history_rows':len(history)}
    atomic_json(ROOT/'aws/validation_restart.json',report)
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    main()
