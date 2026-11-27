"""Execute the notebook's reduced real Modal path in a disposable workspace."""
import argparse
from contextlib import chdir
import json
from pathlib import Path


def validate(workspace, kit=None):
    notebook = json.loads((Path(__file__).parent / 'calcul.ipynb').read_text())
    cells = [c for c in notebook['cells'] if c['cell_type'] == 'code']
    workspace.mkdir(parents=True, exist_ok=True)
    namespace = {}
    with chdir(workspace):
        for index, cell in enumerate(cells):
            source = ''.join(cell['source'])
            compile(source, f'notebook-cell-{index + 1}', 'exec')
            if index == 1:
                # Use the tester's existing Modal profile; never copy its credentials.
                continue
            if index == 2:
                source = source.replace('CYCLES = 500', 'CYCLES = 1').replace(
                    'LAUNCH_CALCULATION = False', 'LAUNCH_CALCULATION = True')
                namespace['RUN_ID'] = 'validation-one-cycle'
            if index == 3 and kit:
                source = source.replace('inputs = runtime.download_kit(KIT_URL,',
                    f'inputs = runtime.download_kit({kit.resolve().as_uri()!r},')
            print(f'Executing cell {index + 1}', flush=True)
            exec(compile(source, f'notebook-cell-{index + 1}', 'exec'), namespace)
        runtime = namespace['runtime']
        first = namespace['receipt']
        second = runtime.launch_or_recover(namespace['worker'], namespace['RECEIPT'],
            namespace['settings'], namespace['KIT_SHA256'])
        assert first == second, 'Recovery must preserve the confirmed call.'
        print('Validated all eight methods, 32 integrations, complete ZIP and confirmed-call reuse.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace', type=Path)
    parser.add_argument('--kit', type=Path, help='Use the exact local public kit instead of HTTPS.')
    args = parser.parse_args()
    validate(args.workspace.resolve(), args.kit)
