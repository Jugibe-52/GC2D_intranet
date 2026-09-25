"""Compose a reproducible fixed-grid Poincare experiment from frozen source."""
import importlib.metadata
import json
import os
import platform
import numpy as np
import pandas as pd
from study_io import ROOT, digest, load_snapshot, begin_calculation, publish_calculation
from parallel_calculation import simulate_parallel


def compute_and_save(parameters, run_id, validation_run=False):
    """Integrate only the requested original particle IDs and commit all products."""
    p = dict(parameters)
    cfg = json.loads((ROOT / 'experiment.json').read_text())
    if not validation_run:
        for key in ('method', 'particle_ids', 'cycles', 'steps_per_cycle', 'process_count', 'checkpoint_steps'):
            assert p[key] == cfg[key], f'Notebook/deployment mismatch: {key}'
    potential, provenance, snapshot = load_snapshot()
    from dynamics import GuidingCenterDynamics
    source = json.loads((ROOT / 'assets' / 'original_particles.json').read_text())
    ids = np.asarray(p['particle_ids'], dtype=int)
    assert len(set(ids)) == len(ids) and np.all(np.diff(ids) > 0)
    original = {r['particle']: r for r in source['particles']}
    selected = [original[int(i)] for i in ids]
    xy0 = np.array([[r['x'], r['y']] for r in selected])
    colors = np.array([r['rgba'] for r in selected])
    hexcolors = [r['color'] for r in selected]
    radii = np.array([r['radius_over_L'] for r in selected])
    assert len(set(hexcolors)) == len(ids)
    assert p['radial_angle_rad'] == source['radial_angle_rad'] == 0.0
    np.testing.assert_array_equal(radii, np.asarray(p['radial_fractions']))
    n, cycles, spc, workers = len(ids), p['cycles'], p['steps_per_cycle'], p['process_count']
    assert workers >= 1 and n >= workers and cycles > 0 and spc > 0
    implicit = p['method'] == 'BM4Implicit'
    assert p['method'] in ('BM4Implicit','BM4Midpoint','RK4')
    steps = cycles * spc
    h = p['cycle_duration'] / spc
    t0, tf = p['t0'], p['t0'] + cycles * p['cycle_duration']
    assert t0 == 0 and p['cycle_duration'] == 1
    np.testing.assert_allclose(potential.frequencies * p['cycle_duration'], [1.], rtol=0, atol=1e-14)
    field = GuidingCenterDynamics(potential, rho=p['rho'])
    initial_state = np.concatenate((xy0[:, 0], xy0[:, 1]))
    for phase in (0, .173, .637):
        np.testing.assert_allclose(field.vector_field(phase, initial_state),
                                   field.vector_field(phase+1, initial_state), rtol=1e-12, atol=1e-12)
    versions = {'python': platform.python_version(), **{name: importlib.metadata.version(name)
                for name in ('numpy','scipy','h5py','matplotlib','pandas','threadpoolctl')}}
    params = {**p, 't_span':[t0,tf], 'step':h, 'particle_count':n, 'snapshot_sha256':snapshot,
              'versions':versions, 'validation_run':validation_run, 'reference_computed':False,
              'original_particles_sha256':digest(ROOT/'assets/original_particles.json')}
    output = begin_calculation(run_id, params)
    settings = dict(rho=p['rho'], coupling_frequency=p['coupling_frequency'],
                    newton_atol=p['newton_atol'], newton_rtol=p['newton_rtol'],
                    newton_max_iterations=p['newton_max_iterations'],
                    jacobian_relative_step=p['jacobian_relative_step'], t_span=[t0,tf],
                    step=h, n_steps=steps, method=p['method'], particle_ids=ids.tolist(),
                    record_newton_history=implicit, checkpoint_steps=p['checkpoint_steps'],
                    checkpoint_directory=str(output/'checkpoints'),
                    test_stop_after_chunks=int(os.environ.get('POINCARE_TEST_STOP_AFTER_CHUNKS','0')) if validation_run else 0)
    solution = simulate_parallel(xy0, settings, processes=workers)
    states, times = solution.states, solution.t
    assert states.shape == (2*n, steps+1) and np.isfinite(states).all()
    np.testing.assert_array_equal(states[:, 0], initial_state)
    cycle_xy = np.stack((states[:n, ::spc].T, states[n:, ::spc].T), axis=-1)
    cycle_times = times[::spc]
    np.testing.assert_allclose(cycle_times, np.arange(cycles+1), rtol=0, atol=1e-12)
    L = potential.grid.period
    origin = np.array([potential.grid.xmin, potential.grid.ymin])
    wrapped = (cycle_xy-origin) % L + origin
    time_scale = provenance['characteristic_period_s']
    length_scale = provenance['characteristic_length_m'] / (2*np.pi)
    physical = wrapped*length_scale + np.array(provenance['source_origin_m'])
    tables = pd.DataFrame({'cycle':np.repeat(np.arange(cycles+1),n),
        'time_normalized':np.repeat(cycle_times,n),'time_s':np.repeat(cycle_times*time_scale,n),
        'particle':np.tile(ids,cycles+1),'color':np.tile(hexcolors,cycles+1),
        'initial_radius_over_L':np.tile(radii,cycles+1),
        'x_unwrapped':cycle_xy[:,:,0].ravel(),'y_unwrapped':cycle_xy[:,:,1].ravel(),
        'x_wrapped':wrapped[:,:,0].ravel(),'y_wrapped':wrapped[:,:,1].ravel(),
        'x_over_L':wrapped[:,:,0].ravel()/L,'y_over_L':wrapped[:,:,1].ravel()/L,
        'R_wrapped_m':physical[:,:,0].ravel(),'Z_wrapped_m':physical[:,:,1].ravel()})
    arrays = dict(times=times,states=states,cycle_times=cycle_times,cycle_positions=cycle_xy,
                  cycle_positions_wrapped=wrapped, initial_positions=xy0,particle_ids=ids,
                  colors_rgba=colors, colors_hex=np.array(hexcolors))
    diag = solution.diagnostics
    meta = {**params, 'run_id':run_id,'schema_version':1,'complete_steps':steps,
            'arithmetic':'float64', 'trajectory_accuracy_certified':False,
            'colours':dict(zip(map(str,ids),hexcolors)), 'field_provenance':provenance,
            'time_scale_s':time_scale,'length_scale_m':length_scale,
            'runtime_seconds':solution.parallel_wall_seconds,
            'runtime_scope':'current attempt, including checkpoint loading and merge',
            'workers':solution.workers, 'simultaneous_integration_seconds':solution.simultaneous_integration_seconds,
            'partition':'round-robin over selected particle IDs',
            'original_ids_and_colors_preserved':True, 'initial_grid_is_new':False,
            'newton_history_recorded':implicit, 'blas_threads_per_process':1,
            'checkpoint_driver_sha256':digest(ROOT/('rk4_checkpoints.py' if p['method']=='RK4' else 'checkpoint_calculation.py')),
            'diagnostic_instrumentation_sha256':digest(ROOT/'newton_diagnostics.py') if implicit else None}
    if implicit:
        for src,dst in [('nonlinear_iterations','nonlinear_iterations'),
                        ('nonlinear_residual_norms','nonlinear_residuals'),
                        ('nonlinear_tolerances','nonlinear_tolerances'),
                        ('projection_multiplier_norms','projection_multiplier_norms')]:
            arrays[dst] = diag[src]
        for key in ('newton_history_offsets','newton_history_residuals','newton_history_mu_norms'):
            arrays[key] = diag[key]
        assert np.all(arrays['nonlinear_residuals'] <= arrays['nonlinear_tolerances'])
        meta.update(nonlinear_solver='newton',newton_jacobian='analytic',
                    mu_norm_definition='Infinity norm of the reduced implicit multiplier per particle group',
                    projection='one reduced Hairer projection per complete BM4 step',
                    maximum_residual_to_tolerance=float(np.max(arrays['nonlinear_residuals']/arrays['nonlinear_tolerances'])))
    elif p['method']=='BM4Midpoint':
        arrays['copy_separation_norms'] = diag['copy_separation_norms']
        meta.update(nonlinear_solver=None,newton_jacobian=None,mu_norm_definition=None,
                    projection='arithmetic mean of two BM4 copies after each complete step',
                    midpoint_diagnostic='Infinity norm of copy separation before arithmetic projection; not an implicit multiplier')
    else:
        meta.update(nonlinear_solver=None,newton_jacobian=None,mu_norm_definition=None,projection=None,field_evaluations_per_step=4,checkpoint_clock='Public RK4 uniform grid within fixed chunks; restart repeats the same chunk schedule',rk4_checkpoint_driver_sha256=digest(ROOT/'rk4_checkpoints.py'))
    publish_calculation(output,meta,arrays,tables[tables.cycle>0],tables[tables.cycle==0])
    print(f'Complete: {p["method"]}, {n} particles, {cycles} cycles, {steps} steps per particle.',flush=True)
    print(f'Original particle IDs: {ids.tolist()}',flush=True)
    return output
