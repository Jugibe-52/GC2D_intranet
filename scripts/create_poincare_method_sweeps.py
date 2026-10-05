"""Create matched method-specific Poincare notebooks and an explicit batch spec."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import textwrap

import nbformat as nbf

from studies.poincare_rho_sweep import RhoStarConfig

ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    'BM4Implicit': 'bm4_implicit',
    'RK4': 'rk4',
    'GaussLegendre4': 'gauss_legendre',
}
SOURCE_SHA256 = '4bbd873cd05992e0a615d66ebbf4d96b38c8b3c36be2ca62932e66af7e6d4ced'


def code(source: str):
    """Create a readable code cell without saved execution output."""
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip() + '\n')


def write_notebook(path: Path, cells: list) -> None:
    """Protect existing scientific working notebooks from replacement."""
    if path.exists():
        raise FileExistsError(f'Notebook already exists: {path}')
    notebook = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {
        'display_name': 'GC2D codigo', 'language': 'python', 'name': 'gc2d-codigo',
    }})
    nbf.validate(notebook)
    nbf.write(notebook, path)


def create(rho_values: tuple[float, ...], *, run_version: str = 'v1') -> Path:
    """Materialize three explicit experiments and a combined parallel manifest."""
    if not rho_values or len(set(rho_values)) != len(rho_values):
        raise ValueError('Rho values must be nonempty and unique.')
    # Six decimals keep distinct input radii from sharing an archive prefix.
    run_ids = [f'rho_{rho:.6f}'.replace('.', '_') + '_' + run_version for rho in rho_values]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError('Rho values collide at six decimal places.')
    jobs = []
    for method, slug in METHODS.items():
        experiment = f'developements/poincare_section/{slug}_star_40_rho_0_30_folded_radial_potential'
        folder = ROOT / 'notebooks' / experiment
        folder.mkdir(parents=True, exist_ok=True)
        config = RhoStarConfig(
            rho_hat=rho_values[0], method=method, particles=40, arms=8,
            outer_radius_fraction=.85, first_angle=0., magnetic_field=1.5,
            characteristic_length=.06, source_selection=(0, 1), interpolation_order=3,
            cycles=5000, steps_per_cycle=50, coupling_frequency=0.,
            spatial_normalization='none', hamiltonian_convention='radial',
        )
        for rho, run_id in zip(rho_values, run_ids):
            job_config = {**asdict(config), 'rho_hat': rho}
            jobs.append(dict(job_id=slug + '__' + run_id, experiment_path=experiment,
                             run_id=run_id, config=job_config))
        shared = f'''
            from pathlib import Path
            from diagnostics.paths import find_project_root, solution_destination

            ROOT = find_project_root(Path.cwd())
            EXPERIMENT_PATH = {experiment!r}
            METHOD = {method!r}
            RHO_VALUES = {rho_values!r}
            RUN_IDS = {dict(zip(rho_values, run_ids))!r}
            STORAGE = 'bucket'
            SOURCE = ROOT / 'data/potential/V1/PHI_2.h5'
            EXPECTED_SOURCE_SHA256 = {SOURCE_SHA256!r}
        '''
        calculation = [nbf.v4.new_markdown_cell(f'''# {method}: radial Poincare star, parallel rho sweep

This experiment reproduces the original BM4Midpoint star with **{method}**.
There are {len(rho_values)} independent rho runs. The combined batch launches all
three methods concurrently through `examples/modal_poincare_methods_app.py`.
Each remote integration uses JAX CPU float64, two cores, a 4 GiB memory limit,
a two-hour timeout, and no configured retries. Account quotas govern actual
concurrency; the batch report measures overlapping remote execution intervals.

Forty particles have distinct radial distances from 0 to 85% of the cell half-width,
assigned alternately to eight arms, with one center particle and first arm toward +R.
There is no random sampling. Each integration covers 5,000 forcing cycles using
50 complete steps per cycle: 250,000 steps and 5,001 saved states, including cycle 0.
Coordinates remain in meters and time is tau=t/T0. The physical gyro-radius is
rho*0.06/(2*pi) meters. The radial Hamiltonian is (T0/B)*Phi/(2*pi), with B=1.5 T,
HDF5 fields (0, 1), and cubic interpolation. Folding into the periodic cell occurs
only after integration. All methods use identical physical inputs and output times.

BM4Implicit and two-stage GaussLegendre4 use analytic Jacobians and identical
Newton tolerances: absolute 1e-12, relative 1e-11, at most 40 corrections,
and Jacobian relative step cbrt(float64 epsilon). BM4Implicit uses its reduced
implicit projection with coupling frequency zero. RK4 is classical explicit RK4.
This is a Poincare study; it does not certify trajectory accuracy or implement a
separate long-time accuracy/reference comparison.

Results are NPZ/JSON archives in the configured bucket, under this experiment's
complete relative path and run identifier. Existing matching archives are reused;
confirmed Modal receipts recover existing calls without duplicate submission.
A failed upload is a failure. The visualisation notebook loads these archives
independently and never repeats trajectory integration.
'''), code(shared + '''
            from dataclasses import replace
            import numpy as np
            import matplotlib.pyplot as plt
            from IPython.display import Markdown, display
            from studies.poincare_rho_sweep import RhoStarConfig, prepare_rho_star
            from studies.poincare_rho_batch import RhoBatchJob, run_modal_rho_batch
            from visualization.poincare_rho_sweep import initial_star_table, plot_initial_rho_star
        '''), code(f'''
            BASE_CONFIG = RhoStarConfig(
                rho_hat=RHO_VALUES[0], method=METHOD, particles=40, arms=8,
                outer_radius_fraction=0.85, first_angle=0.0,
                magnetic_field=1.5, characteristic_length=0.06,
                source_selection=(0, 1), interpolation_order=3,
                cycles=5000, steps_per_cycle=50, coupling_frequency=0.0,
                spatial_normalization='none', hamiltonian_convention='radial',
                newton_absolute_tolerance=1e-12, newton_relative_tolerance=1e-11,
                newton_max_iterations=40, newton_jacobian_method='analytic',
                newton_jacobian_relative_step=float(np.cbrt(np.finfo(float).eps)),
            )
            APP_NAME = 'gc2d-poincare-methods'
            MAX_WORKERS = 33
            RECORD_DIRECTORY = ROOT / 'logs/execution_modal/poincare_method_sweeps'
            PROGRESS_PATH = ROOT / 'logs/poincare_method_sweeps' / (METHOD + '_progress.json')
            prepared = prepare_rho_star(SOURCE, BASE_CONFIG)
            assert prepared.metadata['source_sha256'] == EXPECTED_SOURCE_SHA256
            assert prepared.request.output_times.size == 5001
            assert prepared.metadata['expected_step_count'] == 250000
            display(Markdown(initial_star_table(prepared)))
            plot_initial_rho_star(prepared)
            plt.show()
            jobs = [RhoBatchJob(
                job_id={slug!r} + '__' + RUN_IDS[rho], experiment_path=EXPERIMENT_PATH,
                run_id=RUN_IDS[rho], config=replace(BASE_CONFIG, rho_hat=rho),
            ) for rho in RHO_VALUES]
            print('Independent integrations:', len(jobs))
        '''), code('''
            report = run_modal_rho_batch(
                SOURCE, jobs, record_directory=RECORD_DIRECTORY, progress_path=PROGRESS_PATH,
                max_workers=MAX_WORKERS, app_name=APP_NAME, storage=STORAGE,
                project_root=ROOT, save_folded_returns=True,
                expected_source_sha256=EXPECTED_SOURCE_SHA256,
            )
            display(report)
            assert all(row['status'] == 'completed' for row in report['jobs']), 'Some runs failed; inspect the progress report.'
        ''')]
        visualisation = [nbf.v4.new_markdown_cell(f'''# {method}: saved folded Poincare returns

Load the same bucket archives as calculation.ipynb. No integration is performed.
The left panel shows the effective gyroaveraged Hamiltonian and its physical GC
vector field, transformed consistently to cell fractions. The phase control uses
51 samples over one forcing period and verifies matching endpoints. Potential
and arrow scales remain fixed across phases. The right panel shows saved returns;
hollow rings identify selected particles at Start cycle, including cycle zero.
Changing rho updates both panels. Colors remain attached to radial particle IDs.
'''), code(shared + '''
            import numpy as np
            from IPython.display import IFrame, display
            from visualization.poincare_rho_sweep import load_available_rho_runs, export_rho_sweep
            from visualization.poincare_probe import display_notebook_viewer_link
            from studies.poincare_rho_sweep import folded_rho_positions

            destinations = {rho: solution_destination(EXPERIMENT_PATH, RUN_IDS[rho],
                storage=STORAGE, project_root=ROOT) for rho in RHO_VALUES}
            runs = load_available_rho_runs(destinations)
            for rho, saved in runs.items():
                assert saved.metadata['method'] == METHOD
                assert saved.metadata['source_sha256'] == EXPECTED_SOURCE_SHA256
                assert saved.metadata['execution_options']['backend'] == 'jax'
                assert saved.metadata['execution_options']['device'] == 'cpu'
                assert saved.metadata['execution_executor'] == 'modal'
                assert saved.metadata['config']['hamiltonian_convention'] == 'radial'
                assert saved.solution.states.shape == (80, 5001)
                assert saved.solution.states.dtype == np.float64
                assert saved.solution.diagnostics['step_count'] == 250000
                np.testing.assert_array_equal(saved.solution.t, np.arange(5001, dtype=float))
                wrapped, fractions = folded_rho_positions(saved.solution, saved.metadata)
                np.testing.assert_array_equal(saved.solution.diagnostics['cycle_positions_wrapped'], wrapped)
                np.testing.assert_array_equal(saved.solution.diagnostics['cycle_positions_cell_fraction'], fractions)
                assert np.all((fractions >= 0.0) & (fractions < 1.0))
            print('Verified completed rho values:', tuple(runs))
        '''), code('''
            OUTPUT_HTML = ROOT / 'notebooks' / EXPERIMENT_PATH / 'poincare_folded.html'
            export_rho_sweep(runs, OUTPUT_HTML, rho_values=RHO_VALUES, selected_rho=0.30,
                cycles_per_frame=25, fold_to_cell=True, colormap='turbo',
                color_range=(0.025, 0.975), source=SOURCE,
                field_grid_size=64, vector_grid_size=17, phase_steps=50)
            viewer_url = display_notebook_viewer_link(OUTPUT_HTML,
                notebook_name=Path(EXPERIMENT_PATH).name + '__visualisation')
            display(IFrame(viewer_url, width='100%', height=1100))
        ''')]
        write_notebook(folder / 'calculation.ipynb', calculation)
        write_notebook(folder / 'visualisation.ipynb', visualisation)
    spec = ROOT / 'logs/poincare_method_sweeps/batch_spec.json'
    spec.parent.mkdir(parents=True, exist_ok=True)
    if spec.exists():
        raise FileExistsError(f'Batch specification already exists: {spec}')
    spec.write_text(json.dumps(dict(source=str(ROOT / 'data/potential/V1/PHI_2.h5'),
        expected_source_sha256=SOURCE_SHA256, app_name='gc2d-poincare-methods',
        max_workers=33, storage='bucket', jobs=jobs), indent=2) + '\n')
    return spec


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rho-values', type=float, nargs='+', required=True)
    parser.add_argument('--run-version', default='v1')
    args = parser.parse_args()
    print(create(tuple(args.rho_values), run_version=args.run_version))
