"""Create eleven explicit calculation notebooks and one read-only rho viewer."""

from pathlib import Path
import argparse
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = 'developements/poincare_section/bm4_midpoint_star_40_rho_sweep'
KERNEL = {'display_name': 'GC2D codigo', 'language': 'python', 'name': 'gc2d-codigo'}


def code(source: str):
    """Keep generated cells readable and independently editable."""
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip() + '\n')


def write(folder: Path, name: str, cells: list) -> None:
    """Write a notebook only when it does not already exist."""
    path = folder / name
    if path.exists():
        raise FileExistsError(f'Refusing to replace an existing notebook: {path}')
    notebook = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': KERNEL})
    nbf.validate(notebook)
    nbf.write(notebook, path)


def main(spatial_normalization: str = 'none') -> None:
    """Materialize the requested rho grid without launching any calculation."""
    normalized = spatial_normalization == 'characteristic_length'
    experiment = EXPERIMENT + ('_normalized_space' if normalized else '')
    folder = ROOT / 'notebooks' / experiment
    folder.mkdir(parents=True, exist_ok=True)
    spatial_notes = ('''Positions use **q = 2*pi*(X-X0)/lambda**, with lambda = 0.06 m and X0
the lower corner of the measured cell. No periodic wrapping is applied. Time remains
tau = t/T0. The runtime gyro-radius is the usual dimensionless rho; its physical value
is rho_m = rho*lambda/(2*pi), so rho=0.30 remains 0.002864789 m.
The normalized stream function is (2*pi/lambda)^2 * (T0/B)*Phi. The square factor
preserves the same physical dynamics under this coordinate change, at unchanged time.
Equivalently this stream is 2*pi times the canonical H5 loader's Phi_hat.
''' if normalized else '''Positions remain in **meters** without spatial normalization or wrapping. Time alone is
normalized: tau = t/T0. The input rho is the project's usual dimensionless gyro-radius;
the actual gyroaverage uses rho_m = rho * 0.06 / (2*pi). For rho=0.30 this is 0.002864789 m.
The dimensional GC stream function is (T0/B)*Phi.
''')
    viewer_notes = ('''Both panels use dimensionless coordinates q = 2*pi*(X-X0)/lambda with lambda = 0.06 m.
These are the coordinates actually integrated and saved by the normalized experiment.
The stream function is scaled consistently, so the physical dynamics and tau=t/T0
are preserved. The left panel shows the initial star; the right shows cycle returns.
''' if normalized else '''Both panels use meters. The left panel shows the initial star; the right panel shows
once-per-cycle returns. No spatial division by cell width is applied.
''')
    for index in range(11):
        rho = index / 20
        name = f'calculation_rho_{rho:.2f}.ipynb'.replace('.', '_', 1)
        # Replace only the decimal separator, retaining the .ipynb suffix.
        cells = [nbf.v4.new_markdown_cell(f'''# BM4Midpoint star — rho = {rho:.2f}

40 particles have 40 distinct radii from 0% to 85% of **R = half the physical cell width**.
Their IDs increase with distance; arm = (ID - 1) modulo 8, so each arm has five
assigned particles. Particle 1 is the single shared center. The first arm points along +R.
Colors follow the same radial order.

Integrate **5,000 forcing cycles, 50 complete BM4Midpoint steps per cycle** (250,000 steps),
with **coupling_frequency = 0**. Save the initial state and the 5,000 integer-cycle returns.
Internal steps are neither stored as observations nor interpolated into extra samples.

{spatial_notes}
This is a Poincare study, not an
independent trajectory-accuracy certification.

Use the **GC2D codigo** kernel. Deploy `examples/modal_poincare_rho_app.py` once with the
project's `.venv/bin/modal deploy` command. The authenticated Modal worker uses JAX CPU
float64, two CPU cores, at most 4 GiB memory, a two-hour limit and no configured retries.
Only running this notebook submits its rho. The other ten calculation files stay idle.
Results are stored in the configured bucket. Reopening a completed run loads its archive;
an interrupted job resumes from its confirmed Modal receipt without resubmitting.
'''), code(f'''
            from pathlib import Path
            import os
            import numpy as np
            import matplotlib.pyplot as plt
            from IPython.display import Markdown, display

            from contracts.execution_options import ExecutionOptions
            from diagnostics.paths import find_project_root, solution_destination
            from studies.poincare_rho_sweep import (
                RhoStarConfig, prepare_rho_star, modal_rho_executor, run_and_save_rho_star,
            )
            from visualization.poincare_rho_sweep import plot_initial_rho_star, initial_star_table

            ROOT = find_project_root(Path.cwd())
            EXPERIMENT_PATH = {experiment!r}
            RHO = {rho:.2f}
            RUN_ID = 'rho_' + f'{{RHO:.2f}}'.replace('.', '_') + '_v1'
            STORAGE = 'bucket'
            RCLONE_CONFIG = Path.home() / '.config/gc2d-notebooks-backup/rclone.conf'
            os.environ.setdefault('RCLONE_CONFIG', str(RCLONE_CONFIG))
            DESTINATION = solution_destination(EXPERIMENT_PATH, RUN_ID, storage=STORAGE, project_root=ROOT)
            SOURCE = ROOT / 'data/potential/V1/PHI_2.h5'
            EXPECTED_SOURCE_SHA256 = '4bbd873cd05992e0a615d66ebbf4d96b38c8b3c36be2ca62932e66af7e6d4ced'
            CONFIG = RhoStarConfig(
                rho_hat=RHO, particles=40, arms=8,
                outer_radius_fraction=0.85, first_angle=0.0,
                magnetic_field=1.5, characteristic_length=0.06,
                source_selection=(0, 1), interpolation_order=3,
                cycles=5000, steps_per_cycle=50, coupling_frequency=0.0,
                spatial_normalization={spatial_normalization!r},
            )
            OPTIONS = ExecutionOptions(backend='jax', device='cpu')
            APP_NAME = 'gc2d-poincare-rho-sweep'
            RECORD_DIRECTORY = ROOT / 'logs/execution_modal' / Path(EXPERIMENT_PATH).name / RUN_ID
            RESUME_RECORD = None  # A unique confirmed receipt is recovered automatically.
            print('Result destination:', DESTINATION)
        '''), code('''
            prepared = prepare_rho_star(SOURCE, CONFIG)
            assert prepared.metadata['source_sha256'] == EXPECTED_SOURCE_SHA256
            assert prepared.problem.particle_count == 40
            assert prepared.request.output_times.size == 5001
            assert prepared.metadata['expected_step_count'] == 250000
            print('rho in meters:', prepared.metadata['rho_m'])
            print('Cell bounds in meters:', prepared.metadata['cell_bounds_m'])
            print('Runtime spatial units:', prepared.metadata['space_unit'])
            print('Runtime cell bounds:', prepared.metadata['cell_bounds'])
            print('Runtime gyro-radius:', prepared.metadata['rho_runtime'])
            print('Physical seconds per forcing cycle:', prepared.metadata['time_unit_seconds'])
            display(Markdown(initial_star_table(prepared)))
            plot_initial_rho_star(prepared)
            plt.show()
        '''), code('''
            executor = modal_rho_executor(RECORD_DIRECTORY, app_name=APP_NAME, resume_record=RESUME_RECORD)
            saved = run_and_save_rho_star(prepared, DESTINATION, executor=executor, options=OPTIONS)
            print('Complete archive saved and verified structurally:', DESTINATION)
            print('Modal call:', saved.metadata.get('modal_call_id'))
            print('Saved cycle samples:', saved.solution.t.size - 1)
            print('Worker wall seconds:', saved.solution.diagnostics.get('worker_wall_seconds'))
            print('Open visualisation.ipynb to refresh the rho selector and export the offline web viewer.')
        ''')]
        write(folder, name, cells)
    write(folder, 'visualisation.ipynb', [nbf.v4.new_markdown_cell(f'''# {'Normalized' if normalized else 'Dimensional'} Poincare star — rho selector

This notebook **only reads completed results**. It never submits or repeats an integration.
Run both cells to verify the available archives and export a self-contained offline HTML viewer.
The 11 rho choices span 0.00–0.50 in steps of 0.05; incomplete choices remain disabled.
Initially only rho=0.30 is calculated. Rerun this notebook after completing another rho.

{viewer_notes}
Saved float64 coordinates are unwrapped; browser copies use float32 in the same units.
Play/Pause, cycle interval sliders, accumulation, zoom, pan, point size, particle
checkboxes, Show all, Hide all, Invert and arm selection follow the reference viewer.
Particle IDs and colors increase with initial radius. Time labels count forcing cycles.
'''), code(f'''
        from pathlib import Path
        import os
        from IPython.display import IFrame, FileLink, display
        from diagnostics.paths import find_project_root, solution_destination
        from visualization.poincare_rho_sweep import load_available_rho_runs, export_rho_sweep

        ROOT = find_project_root(Path.cwd())
        EXPERIMENT_PATH = {experiment!r}
        STORAGE = 'bucket'
        RCLONE_CONFIG = Path.home() / '.config/gc2d-notebooks-backup/rclone.conf'
        os.environ.setdefault('RCLONE_CONFIG', str(RCLONE_CONFIG))
        RHO_VALUES = tuple(index / 20 for index in range(11))
        RUN_VERSION = 'v1'
        SELECTED_RHO = 0.30
        CYCLES_PER_FRAME = 25
        OUTPUT_HTML = ROOT / 'notebooks' / EXPERIMENT_PATH / 'poincare_rho_sweep.html'
        destinations = {{
            rho: solution_destination(EXPERIMENT_PATH,
                'rho_' + f'{{rho:.2f}}'.replace('.', '_') + '_' + RUN_VERSION,
                storage=STORAGE, project_root=ROOT)
            for rho in RHO_VALUES
        }}
        runs = load_available_rho_runs(destinations)
        print('Completed rho values:', ', '.join(f'{{rho:.2f}}' for rho in runs))
    '''), code('''
        export_rho_sweep(runs, OUTPUT_HTML, rho_values=RHO_VALUES,
                         selected_rho=SELECTED_RHO, cycles_per_frame=CYCLES_PER_FRAME)
        print('Offline web viewer:', OUTPUT_HTML)
        display(FileLink(str(OUTPUT_HTML)))
        display(IFrame('poincare_rho_sweep.html', width='100%', height=1100))
    ''')])
    print(f'Created 11 calculations and 1 visualization in {folder}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spatial-normalization', choices=('none', 'characteristic_length'), default='none')
    main(parser.parse_args().spatial_normalization)
