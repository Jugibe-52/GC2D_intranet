"""Validate notebook code or run a small frozen-source numerical smoke check."""
import argparse
from contextlib import chdir
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parent


def validate_local(workspace, kit):
    """Run 24 real integrations in a fresh interpreter; never contact Modal."""
    import calculation_runtime as runtime
    from build_calculation import KIT_URL
    notebook = json.loads((ROOT / 'calcul.ipynb').read_text())
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), f'notebook-cell-{index}', 'exec')
    digest = notebook['metadata']['gc2d']['kit_sha256']
    workspace.mkdir(parents=True, exist_ok=True)
    inputs = runtime.download_kit(kit.resolve().as_uri() if kit else KIT_URL, digest, workspace / 'kit')
    script = '''
import json, sys
from pathlib import Path
helper, workspace = map(Path, sys.argv[1:])
sys.path.insert(0, str(helper))
sys.path.insert(0, str(workspace / 'kit/source/src'))
from calculation_runtime import calculate_local, verify_results, check_settings
settings = check_settings([1], 3, 'validation-one-cycle')
data = calculate_local(workspace / 'kit', **settings)
inputs = json.loads((workspace / 'kit/inputs.json').read_text())
study, timings = verify_results(data, settings, inputs)
(workspace / 'validation-one-cycle.zip').write_bytes(data)
print('Verified three paired cases, 24 integrations, six complete trajectories and ZIP provenance.', flush=True)
'''
    environment = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
        JAX_ENABLE_X64='true', JAX_PLATFORMS='cpu', JAX_ENABLE_COMPILATION_CACHE='false')
    subprocess.run([sys.executable, '-I', '-c', script, str(ROOT), str(workspace)], env=environment,
        check=True, timeout=600)
    data = (workspace / 'validation-one-cycle.zip').read_bytes()
    settings = runtime.check_settings([1], 3, 'validation-one-cycle')
    runtime.verify_results(data, settings, inputs)
    # Exercise integrity and semantic rejection with the real returned artifact.
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    damaged = dict(files)
    damaged['timings.json'] += b' '
    for kind in ('checksum', 'cache', 'trajectory'):
        changed = dict(damaged if kind == 'checksum' else files)
        if kind == 'cache':
            timings = json.loads(changed['timings.json'])
            first = next(iter(timings['cases']))
            timings['cases'][first]['measurements'][1]['integration_cache_misses'] = 0
            changed['timings.json'] = json.dumps(timings).encode()
        if kind == 'trajectory':
            import numpy as np
            name = 'runs/RK2-1/jax/solution.npz'
            with np.load(io.BytesIO(changed[name]), allow_pickle=False) as arrays:
                values = {key: arrays[key].copy() for key in arrays.files}
            values['states'][1, 0] += 1
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **values)
            changed[name] = buffer.getvalue()
        if kind != 'checksum':
            manifest = json.loads(changed['manifest.json'])
            manifest['files'] = {name: dict(bytes=len(value), sha256=sha256(value).hexdigest())
                for name, value in changed.items() if name != 'manifest.json'}
            changed['manifest.json'] = json.dumps(manifest).encode()
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
            for name, value in changed.items():
                archive.writestr(name, value)
        try:
            runtime.verify_results(buffer.getvalue(), settings, inputs)
        except (ValueError, AssertionError):
            pass
        else:
            raise AssertionError(f'Failed to reject corrupted {kind}.')
    print('Rejected modified ZIP bytes, false cache audit and altered trajectory arrays.', flush=True)


def validate_modal(workspace, kit):
    """Explicit opt-in to run the actual notebook path on the tester's Modal account."""
    notebook = json.loads((ROOT / 'calcul.ipynb').read_text())
    cells = [c for c in notebook['cells'] if c['cell_type'] == 'code']
    workspace.mkdir(parents=True, exist_ok=True)
    namespace = {'RUN_ID': 'validation-one-cycle'}
    with chdir(workspace):
        for index, cell in enumerate(cells):
            source = ''.join(cell['source'])
            if index == 1:
                continue  # Reuse the tester's Modal profile without copying credentials.
            if index == 2:
                source = source.replace('CYCLES = [10, 30, 90]', 'CYCLES = [1]').replace(
                    'LAUNCH_CALCULATION = False', 'LAUNCH_CALCULATION = True')
            if index == 3 and kit:
                source = source.replace('inputs = runtime.download_kit(KIT_URL,',
                    f'inputs = runtime.download_kit({kit.resolve().as_uri()!r},')
            print(f'Executing cell {index + 1}', flush=True)
            exec(compile(source, f'notebook-cell-{index + 1}', 'exec'), namespace)
        second = namespace['runtime'].launch_or_recover(namespace['worker'], namespace['RECEIPT'],
            namespace['settings'], namespace['KIT_SHA256'])
        assert second == namespace['receipt'], 'The confirmed call must be reused.'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace', type=Path)
    parser.add_argument('--kit', type=Path)
    parser.add_argument('--modal', action='store_true', help='Explicitly submit the reduced run to Modal.')
    args = parser.parse_args()
    (validate_modal if args.modal else validate_local)(args.workspace.resolve(), args.kit)
