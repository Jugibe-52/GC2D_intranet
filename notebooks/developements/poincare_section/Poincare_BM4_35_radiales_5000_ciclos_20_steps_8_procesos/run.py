#!/usr/bin/env python3
"""Execute calculation or visualization notebooks on a local or AWS Linux host."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback

from study_io import ROOT, atomic_json, load_calculation, new_run_id, utc_now, validate_run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('calculate', 'visualize'))
    parser.add_argument('--run-id', help='Unique calculation ID, or saved ID to visualize.')
    parser.add_argument('--smoke', action='store_true', help='Calculation only: two cycles, all 35 particles, 20 steps/cycle.')
    parser.add_argument('--timeout', type=int, default=18000, help='Timeout per notebook cell, in seconds.')
    parser.add_argument('--dry-run', action='store_true', help='Print paths and overrides without executing or creating files.')
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive.')
    if args.smoke and args.stage != 'calculate':
        parser.error('--smoke applies only to calculate.')
    if args.stage == 'visualize' and args.run_id is None:
        pointer = ROOT / 'resultados' / 'latest_success.json'
        if not pointer.is_file():
            parser.error('No completed default calculation. Specify --run-id.')
        args.run_id = json.loads(pointer.read_text())['run_id']
    run_id = validate_run_id(args.run_id or new_run_id())
    source = ROOT / ('calculo.ipynb' if args.stage == 'calculate' else 'visualizacion.ipynb')
    directory = ROOT / ('resultados' if args.stage == 'calculate' else 'figuras') / run_id
    if args.dry_run:
        print(json.dumps({'stage': args.stage, 'source': str(source), 'run_id': run_id,
                          'output': str(directory), 'validation_run': args.smoke,
                          'cycles_override': 2 if args.smoke else None,
                          'python': sys.executable, 'cell_timeout_s': args.timeout}, indent=2))
        return 0

    # All scientific parameters remain in the notebooks. Only smoke validation
    # changes the cycle count in an in-memory copy of the calculation notebook.
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager

    if args.stage == 'calculate':
        directory.mkdir(parents=True, exist_ok=False)
    else:
        load_calculation(run_id)  # Fail before touching figure outputs on incomplete data.
        directory.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.read(source, as_version=4)
    if args.smoke:
        parameter_index = next(i for i, cell in enumerate(notebook.cells)
                               if 'parameters' in cell.metadata.get('tags', []))
        notebook.cells.insert(parameter_index + 1, nbformat.v4.new_code_cell(
            '# Temporary validation overrides: original notebook remains unchanged.\n'
            'N_CYCLES = 2\nVALIDATION_RUN = True', metadata={'tags': ['validation-overrides']}))

    status_path = directory / 'execution_status.json'
    executed_path = directory / (source.stem + '_executed.ipynb')
    record = {'stage': args.stage, 'run_id': run_id, 'started_utc': utc_now(),
              'status': 'running', 'validation_run': args.smoke, 'python': sys.executable}
    atomic_json(status_path, record)
    with (directory / 'execution.log').open('w', encoding='utf-8', buffering=1) as log:
        def write(message):
            log.write(message)
            print(message, end='', flush=True)

        class LoggedNotebookClient(NotebookClient):
            """Mirror kernel stream output to the cloud log while a cell runs."""

            def process_message(self, msg, cell, cell_index):
                if msg['msg_type'] == 'stream':
                    write(msg['content']['text'])
                return super().process_message(msg, cell, cell_index)

        def on_cell_start(cell, cell_index, **kwargs):
            if cell.cell_type == 'code':
                label = ','.join(cell.metadata.get('tags', [])) or str(cell_index)
                write(f'\n[{utc_now()}] Cell {cell_index}: {label}\n')

        def on_cell_executed(**kwargs):
            nbformat.write(notebook, executed_path)

        manager = KernelManager(kernel_name='python3')
        # Run in the exact interpreter used for this command, independent of
        # whichever Jupyter kernelspec happens to be installed on the host.
        manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
        backend = 'Agg' if args.stage == 'calculate' else 'module://matplotlib_inline.backend_inline'
        (ROOT / '.cache' / 'matplotlib').mkdir(parents=True, exist_ok=True)
        (ROOT / '.cache' / 'ipython').mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ, POINCARE_RUN_ID=run_id, MPLBACKEND=backend,
                           MPLCONFIGDIR=str(ROOT / '.cache' / 'matplotlib'),
                           IPYTHONDIR=str(ROOT / '.cache' / 'ipython'),
                           OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        client = LoggedNotebookClient(
            notebook, km=manager, timeout=args.timeout, startup_timeout=60,
            resources={'metadata': {'path': str(ROOT)}},
            on_cell_start=on_cell_start, on_cell_executed=on_cell_executed,
        )
        try:
            client.execute(env=environment, cleanup_kc=True)
            if args.stage == 'calculate':
                load_calculation(run_id)  # Require complete numeric products, not only an exited kernel.
            record.update(status='success', finished_utc=utc_now())
            write(f'\nSuccess: {directory}\n')
            return 0
        except BaseException as error:
            record.update(status='failed', finished_utc=utc_now(),
                          error=f'{type(error).__name__}: {error}')
            write(traceback.format_exc())
            return 130 if isinstance(error, KeyboardInterrupt) else 1
        finally:
            nbformat.write(notebook, executed_path)
            atomic_json(status_path, record)


if __name__ == '__main__':
    raise SystemExit(main())
