"""Create four matched gap-probe notebook pairs and their 44-job manifest."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import textwrap

import nbformat as nbf

from studies.poincare_rho_sweep import RhoStarConfig


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = '4bbd873cd05992e0a615d66ebbf4d96b38c8b3c36be2ca62932e66af7e6d4ced'
METHODS = {
    'BM4Midpoint': 'bm4_midpoint', 'BM4Implicit': 'bm4_implicit',
    'RK4': 'rk4', 'GaussLegendre4': 'gauss_legendre',
}
RHO_VALUES = tuple(index / 20 for index in range(11))
MARKER = 'gc2d_gap_probe_loader'


def code(source: str):
    """Create a clean code cell with no stored execution artifacts."""
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip() + '\n')


def write_notebook(path: Path, cells: list) -> None:
    """Create each scientific pair once, retaining local ignored status."""
    if path.exists():
        raise FileExistsError(f'Notebook already exists: {path}')
    notebook = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {
        'display_name': 'GC2D codigo', 'language': 'python', 'name': 'gc2d-codigo',
    }})
    nbf.validate(notebook)
    nbf.write(notebook, path)


def add_probe_loading(path: Path, *, method: str | None = None,
                      rho_values_expression: str = 'RHO_VALUES') -> None:
    """Extend only viewer loading/export cells, preserving original assertions."""
    notebook = nbf.read(path, as_version=4)
    if any(cell.get('metadata', {}).get(MARKER) for cell in notebook.cells):
        raise FileExistsError(f'The viewer already includes the probe loader: {path}')
    if method is None:
        loader = code('''
            from studies.poincare_gap_probes import validate_saved_gap_probes

            PROBE_DESTINATIONS = {
                method: {
                    rho: solution_destination(experiment + '/gap_probes',
                        'rho_' + f'{rho:.6f}'.replace('.', '_') + '_v1',
                        storage=STORAGE, project_root=ROOT)
                    for rho in RHO_VALUES
                }
                for method, (experiment, _) in SOURCE_EXPERIMENTS.items()
            }
            probe_runs_by_method = load_rho_method_runs(PROBE_DESTINATIONS)
            for method, probe_runs in probe_runs_by_method.items():
                assert set(probe_runs) == set(RHO_VALUES), f'Missing gap-probe rho values for {method}.'
                for rho, saved in probe_runs.items():
                    validate_saved_gap_probes(saved, background=runs_by_method[method][rho])
                print(f'{method}: 8 additional probes at {len(probe_runs)} rho values loaded')
        ''')
        target = 'export_rho_method_comparison('
        # Required positional arguments precede the optional keyword argument.
        def amend(source: str) -> str:
            return source.replace('runs_by_method, OUTPUT_HTML, source=SOURCE,',
                                  'runs_by_method, OUTPUT_HTML, probe_runs_by_method=probe_runs_by_method, source=SOURCE,')
    else:
        loader = code(f'''
            from studies.poincare_gap_probes import validate_saved_gap_probes

            PROBE_METHOD = {method!r}
            PROBE_EXPERIMENT_PATH = EXPERIMENT_PATH + '/gap_probes'
            PROBE_RHO_VALUES = {rho_values_expression}
            probe_destinations = {{rho: solution_destination(PROBE_EXPERIMENT_PATH,
                'rho_' + f'{{rho:.6f}}'.replace('.', '_') + '_v1',
                storage=STORAGE, project_root=ROOT) for rho in PROBE_RHO_VALUES}}
            probe_runs = load_available_rho_runs(probe_destinations)
            assert set(probe_runs) == set(PROBE_RHO_VALUES), 'Some gap-probe rho values are missing.'
            for rho, saved in probe_runs.items():
                assert saved.metadata['method'] == PROBE_METHOD
                validate_saved_gap_probes(saved, background=runs[rho])
            print('Additional gap probes loaded:', 8, 'at each of', len(probe_runs), 'rho values')
        ''')
        target = 'export_rho_sweep(runs, OUTPUT_HTML,'
        def amend(source: str) -> str:
            return source.replace(target, target + ' probe_runs=probe_runs,')
    loader.metadata[MARKER] = True
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == 'code' and target in cell.source:
            modified = amend(cell.source)
            if modified == cell.source:
                raise ValueError(f'Cannot identify the export arguments in {path}.')
            cell.source = modified
            cell.outputs, cell.execution_count = [], None
            notebook.cells.insert(index, loader)
            break
    else:
        raise ValueError(f'Cannot identify the export cell in {path}.')
    notebook.cells[0].source += '''

Eight separately calculated gap probes (particle IDs 41–48) are loaded from each
method's `gap_probes/` experiment. The original 40 particles, saved states and
colors are retained. Two seeds per northwest, northeast, southwest and southeast
gap were selected jointly from all four methods at rho=0.30, using cycles 0–5000;
the identical eight initial cell fractions are reused at every rho. These locations
need not remain gaps for other rho values. The independent probe archives use
the same field, method, physical controls, integration interval and cycle sampling.
'''
    nbf.validate(notebook)
    nbf.write(notebook, path)


def create() -> Path:
    """Materialize reproducible seeds, paired notebooks and an explicit job list."""
    selection_path = ROOT / 'logs/poincare_gap_probes/gap_seed_selection.json'
    selection = json.loads(selection_path.read_text())
    rows = selection['probes']
    seeds = {
        'fractions': [row['initial_cell_fraction'] for row in rows],
        'particle_ids': [row['particle_id'] for row in rows],
        'gap_names': [row['region'] for row in rows],
        'selection_rho': selection['selection_rho'],
    }
    if seeds['particle_ids'] != list(range(41, 49)) or len(rows) != 8:
        raise ValueError('The selected probes must preserve particle IDs 41 through 48.')
    seed_table = '| Particle ID | Gap | Initial (R-R0)/L | Initial (Z-Z0)/L |\n|---|---|---:|---:|\n'
    seed_table += '\n'.join(f"| {r['particle_id']} | {r['region']} | {r['initial_cell_fraction'][0]:.4f} | {r['initial_cell_fraction'][1]:.4f} |" for r in rows)
    run_ids = {rho: f'rho_{rho:.6f}'.replace('.', '_') + '_v1' for rho in RHO_VALUES}
    jobs = []
    for method, slug in METHODS.items():
        original = f'developements/poincare_section/{slug}_star_40_rho_0_30_folded_radial_potential'
        experiment = original + '/gap_probes'
        folder = ROOT / 'notebooks' / experiment
        folder.mkdir(parents=True, exist_ok=True)
        config = RhoStarConfig(rho_hat=0., method=method, particles=8, arms=8,
            outer_radius_fraction=.85, first_angle=0., magnetic_field=1.5,
            characteristic_length=.06, source_selection=(0, 1), interpolation_order=3,
            cycles=5000, steps_per_cycle=50, coupling_frequency=0.,
            spatial_normalization='none', hamiltonian_convention='radial',
            newton_absolute_tolerance=1e-12, newton_relative_tolerance=1e-11,
            newton_max_iterations=40, newton_jacobian_method='analytic')
        jobs.extend(dict(job_id=slug + '_gap_probes__' + run_ids[rho],
                         experiment_path=experiment, run_id=run_ids[rho],
                         config={**asdict(config), 'rho_hat': rho}) for rho in RHO_VALUES)
        shared = f'''
            from pathlib import Path
            import numpy as np
            from diagnostics.paths import find_project_root, solution_destination
            from studies.poincare_gap_probes import GapProbeSeeds

            ROOT = find_project_root(Path.cwd())
            EXPERIMENT_PATH = {experiment!r}
            ORIGINAL_EXPERIMENT_PATH = {original!r}
            METHOD = {method!r}
            RHO_VALUES = {RHO_VALUES!r}
            RUN_IDS = {run_ids!r}
            STORAGE = 'bucket'
            SOURCE = ROOT / 'data/potential/V1/PHI_2.h5'
            EXPECTED_SOURCE_SHA256 = {SOURCE_SHA256!r}
            # Explicit coordinates are cell fractions, measured from the lower corner.
            PROBE_SEEDS = GapProbeSeeds(
                fractions={tuple(tuple(r['initial_cell_fraction']) for r in rows)!r},
                particle_ids={tuple(seeds['particle_ids'])!r},
                gap_names={tuple(seeds['gap_names'])!r}, selection_rho=0.30,
            )
        '''
        intro = f'''# {method}: eight independently calculated Poincare gap probes

These eight additional particles extend the saved 40-particle star. The original
forty trajectories are reused. Particle IDs 41–48 remain stable in all methods
and rho values; there is no random sampling or mutual particle interaction.

The initial cell fractions below were selected from all four methods at rho=0.30,
using every saved cycle from 0 through 5000. The selection maximizes minimum
periodic distance to saved returns within four screenshot-calibrated regions;
the second point lies in the same peak-connected gap component with a minimum
separation. The complete criterion, grid spacing, region bounds, measured
clearances and original archive references are in `gap_seed_selection.json`.
The same eight coordinates are reused from rho=0.00 through 0.50 in steps of 0.05;
they need not occupy gaps at other rho values. Gap labels describe seed locations,
not constraints on subsequent motion.

{seed_table}

Physical coordinates are X=X0+L*fraction in meters, with L from the verified HDF5
periodic cell. The physical gyro-radius is rho*0.06/(2*pi) meters. The magnetic
field is 1.5 T, source selection is (0,1), and interpolation is cubic. Time is
tau=t/T0 and the radial Hamiltonian is (T0/B)*Phi/(2*pi). Integrations cover
5000 forcing cycles with 50 complete steps per cycle (250000 steps), retaining
5001 unwrapped states and optional folded coordinates including cycle 0.
BM4 coupling is zero. Implicit methods use analytic Jacobians with Newton absolute
tolerance 1e-12, relative tolerance 1e-11, at most 40 corrections and Jacobian
relative step cbrt(float64 epsilon). RK4 and BM4Midpoint do not use Newton solves.
This Poincare experiment does not compute an independent accuracy reference.
'''
        calculation = [nbf.v4.new_markdown_cell(intro + '''

Each rho is an independent JAX CPU float64 integration on Modal, using two CPU
cores, a 4 GiB memory limit, a two-hour timeout and no configured retries.
This notebook submits this method's 11 jobs; the combined batch manifest submits
all 44 jobs in parallel, subject to account capacity. Existing matching archives
and confirmed call receipts are reused. Failed bucket uploads remain failures.
Results are NPZ/JSON archives below this experiment's complete notebook-relative
bucket prefix. Run visualisation.ipynb independently to load saved results.
'''), code(shared + '''
            from dataclasses import replace
            from IPython.display import display
            from studies.poincare_rho_sweep import RhoStarConfig
            from studies.poincare_gap_probes import prepare_gap_probes
            from studies.poincare_rho_batch import RhoBatchJob, run_modal_rho_batch
        '''), code(f'''
            BASE_CONFIG = RhoStarConfig(
                rho_hat=0.30, method=METHOD, particles=8, arms=8,
                # Star-only geometry fields remain matched; PROBE_SEEDS defines all positions.
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
            MAX_WORKERS = 44
            RECORD_DIRECTORY = ROOT / 'logs/execution_modal/poincare_gap_probes'
            PROGRESS_PATH = ROOT / 'logs/poincare_gap_probes' / (METHOD + '_progress.json')
            prepared = prepare_gap_probes(SOURCE, BASE_CONFIG, seeds=PROBE_SEEDS)
            assert prepared.metadata['source_sha256'] == EXPECTED_SOURCE_SHA256
            assert prepared.problem.particle_count == 8
            assert prepared.request.output_times.size == 5001
            assert prepared.metadata['expected_step_count'] == 250000
            np.testing.assert_array_equal(prepared.metadata['particle_id'], np.arange(41, 49))
            display({{'initial_positions_m': prepared.metadata['initial_positions_m'],
                     'particle_ids': PROBE_SEEDS.particle_ids, 'gap_names': PROBE_SEEDS.gap_names}})
            jobs = [RhoBatchJob(
                job_id={slug!r} + '_gap_probes__' + RUN_IDS[rho],
                experiment_path=EXPERIMENT_PATH, run_id=RUN_IDS[rho],
                config=replace(BASE_CONFIG, rho_hat=rho), probe_seeds=PROBE_SEEDS,
            ) for rho in RHO_VALUES]
            print('Independent probe integrations:', len(jobs))
        '''), code('''
            report = run_modal_rho_batch(
                SOURCE, jobs, record_directory=RECORD_DIRECTORY, progress_path=PROGRESS_PATH,
                max_workers=MAX_WORKERS, app_name=APP_NAME, storage=STORAGE,
                project_root=ROOT, save_folded_returns=True,
                expected_source_sha256=EXPECTED_SOURCE_SHA256,
            )
            display(report)
            assert all(row['status'] == 'completed' for row in report['jobs']), 'Some runs failed; inspect their confirmed receipts.'
        ''')]
        precision = 2 if method == 'BM4Midpoint' else 6
        visualisation = [nbf.v4.new_markdown_cell(intro + '''

This notebook loads saved original-star and probe archives from the same bucket;
it never repeats integration. The left panel shows the effective gyroaveraged
Hamiltonian and its consistently transformed guiding-center field, with 51
independent phase samples and identical endpoints. The right panel contains all
48 particle IDs, preserving the original 40 colors. Start-cycle rings include
cycle zero. Rho updates both saved returns and the field; scales stay fixed
across phase samples. The two seeds in each gap can be selected together.
'''), code(shared + f'''
            from IPython.display import IFrame, display
            from studies.poincare_gap_probes import validate_saved_gap_probes
            from visualization.poincare_rho_sweep import load_available_rho_runs, export_rho_sweep
            from visualization.poincare_probe import display_notebook_viewer_link

            ORIGINAL_RUN_PRECISION = {precision}
            original_destinations = {{rho: solution_destination(ORIGINAL_EXPERIMENT_PATH,
                'rho_' + f'{{rho:.{{ORIGINAL_RUN_PRECISION}}f}}'.replace('.', '_') + '_v1',
                storage=STORAGE, project_root=ROOT) for rho in RHO_VALUES}}
            probe_destinations = {{rho: solution_destination(EXPERIMENT_PATH, RUN_IDS[rho],
                storage=STORAGE, project_root=ROOT) for rho in RHO_VALUES}}
            runs = load_available_rho_runs(original_destinations)
            probe_runs = load_available_rho_runs(probe_destinations)
            assert set(runs) == set(probe_runs) == set(RHO_VALUES), 'Some requested archives are missing.'
            for rho in RHO_VALUES:
                original, probes = runs[rho], probe_runs[rho]
                assert original.solution.states.shape == (80, 5001)
                assert probes.solution.states.shape == (16, 5001)
                assert original.metadata['source_sha256'] == EXPECTED_SOURCE_SHA256
                assert original.metadata['method'] == METHOD
                assert original.metadata['config']['hamiltonian_convention'] == 'radial'
                for saved in (original, probes):
                    assert saved.solution.states.dtype == np.float64
                    assert saved.solution.diagnostics['step_count'] == 250000
                    assert saved.metadata['execution_options']['backend'] == 'jax'
                    assert saved.metadata['execution_options']['device'] == 'cpu'
                    assert saved.metadata['execution_executor'] == 'modal'
                    np.testing.assert_array_equal(saved.solution.t, np.arange(5001, dtype=float))
                validated = validate_saved_gap_probes(probes, background=original)
                np.testing.assert_array_equal(validated.fractions, PROBE_SEEDS.fractions)
                np.testing.assert_array_equal(validated.particle_ids, PROBE_SEEDS.particle_ids)
            print('Verified original stars and eight gap probes at every rho.')
        '''), code('''
            OUTPUT_HTML = ROOT / 'notebooks' / EXPERIMENT_PATH / 'poincare_gap_probes.html'
            export_rho_sweep(runs, OUTPUT_HTML, probe_runs=probe_runs,
                rho_values=RHO_VALUES, selected_rho=0.30, cycles_per_frame=25,
                fold_to_cell=True, colormap='turbo', color_range=(0.025, 0.975),
                source=SOURCE, field_grid_size=64, vector_grid_size=17, phase_steps=50)
            viewer_url = display_notebook_viewer_link(OUTPUT_HTML,
                notebook_name=Path(ORIGINAL_EXPERIMENT_PATH).name + '__gap_probes')
            display(IFrame(viewer_url, width='100%', height=1100))
        ''')]
        write_notebook(folder / 'calculation.ipynb', calculation)
        write_notebook(folder / 'visualisation.ipynb', visualisation)
        (folder / 'gap_seed_selection.json').write_text(json.dumps(selection, indent=2) + '\n')
        (folder / 'README.md').write_text(f'''# {method} gap probes

`calculation.ipynb` calculates eight independent seed particles at eleven rho values
using JAX CPU float64 on Modal. `visualisation.ipynb` loads those archives and the
original forty particles, then exports an HTML viewer with an active local URL.
Original particle IDs, trajectories and colors are preserved; added IDs are 41–48.

Seeds were selected jointly from the four saved methods at rho=0.30 and reused at
all rho values. `gap_seed_selection.json` records the numerical selection rule,
seed coordinates, region bounds, clearances and original data references. It does
not assert that every seed lies in a gap for every other gyro-radius.

Bucket prefix: `gc2d_data:gc2d-notebooks-data/{experiment}/`.
Each run uses `rho_<six-decimal-value>_v1`, for example `rho_0_300000_v1`.
The original star's run identifiers remain unchanged. Results contain NPZ arrays,
JSON metadata and a checksummed manifest. Failed uploads are reported as failures.

The combined 44-job specification is `logs/poincare_gap_probes/batch_spec.json`
relative to the project root; confirmed receipts live under
`logs/execution_modal/poincare_gap_probes/`. Reopening matching calculations reuses
saved archives or confirmed remote calls. Visualization performs no integrations.
''')
        add_probe_loading(folder.parent / 'visualisation.ipynb', method=method)
        if method == 'BM4Midpoint':
            for viewer in sorted(folder.parent.glob('visualisation_rho_*.ipynb')):
                add_probe_loading(viewer, method=method, rho_values_expression='(RHO,)')
    joint = ROOT / 'notebooks/developements/poincare_section/four_methods_star_40_rho_sweep_folded_radial_potential'
    add_probe_loading(joint / 'visualisation.ipynb')
    with (joint / 'README.md').open('a') as stream:
        stream.write('''

The viewer also loads each source experiment's `gap_probes/` archives: eight
additional particles with shared IDs 41–48 and identical initial cell fractions
across all four methods and eleven rho values. Two probes were selected in each
of four gaps using the combined rho=0.30 saved returns. The full selection record
and matched calculation/visualisation notebooks live in every `gap_probes/`
directory. The original 40-particle archives, IDs and colors remain unchanged.
Probe results are loaded independently; generating this viewer does not recompute
either the original forty trajectories or the eight probes.
''')
    spec = ROOT / 'logs/poincare_gap_probes/batch_spec.json'
    if spec.exists():
        raise FileExistsError(f'Batch specification already exists: {spec}')
    spec.write_text(json.dumps(dict(source=str(ROOT / 'data/potential/V1/PHI_2.h5'),
        expected_source_sha256=SOURCE_SHA256, app_name='gc2d-poincare-methods',
        max_workers=44, storage='bucket',
        record_directory=str(ROOT / 'logs/execution_modal/poincare_gap_probes'),
        probe_seeds=seeds, jobs=jobs), indent=2) + '\n')
    return spec


if __name__ == '__main__':
    print(create())
