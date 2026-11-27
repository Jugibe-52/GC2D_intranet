"""Run reduced integrations and an independent checkpoint/recovery audit locally."""
import argparse
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import calculation_runtime as runtime

ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT.parents[1] / 'outputs/colab-public/04_04_evolution_energetique/calculation/v1'


def validate(workspace, kit):
    import numpy as np
    workspace.mkdir(parents=True, exist_ok=True)
    record = json.loads((PUBLIC / 'manifest.json').read_text())
    inputs = runtime.prepare_kit(kit.read_bytes(), record['sha256'], workspace / 'kit')
    settings = runtime.check_settings(2, 1, 'validation-two-cycles')
    environment = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
        JAX_ENABLE_X64='true', JAX_PLATFORMS='cpu')
    script = '''
from pathlib import Path
import sys
helper, workspace = map(Path, sys.argv[1:])
sys.path.insert(0, str(helper))
from calculation_runtime import calculate_local
output = calculate_local(workspace / 'kit', cycles=2, block_cycles=1,
    run_id='validation-two-cycles', output_dir=workspace / 'results')
(workspace / 'validation-two-cycles.zip').write_bytes(output)
'''
    subprocess.run([sys.executable, '-I', '-c', script, str(ROOT), str(workspace)],
        env=environment, check=True, timeout=900)
    data = (workspace / 'validation-two-cycles.zip').read_bytes()
    runtime.verify_results(data, settings, inputs)
    # Pause after one block, resume in a fresh interpreter, compare every array
    # with the uninterrupted run. Exact clocks and saved accumulators are reused.
    command = [sys.executable, str(workspace / 'kit' / inputs['script']), 'RK2', '--cycles', '2',
        '--block-cycles', '1', '--input-dir', str(workspace / 'kit/inputs'),
        '--output-dir', str(workspace / 'checkpoint-check')]
    subprocess.run(command + ['--max-blocks', '1'], env=environment, check=True, capture_output=True)
    if (workspace / 'checkpoint-check/RK2.json').exists():
        raise AssertionError('A partial run must not publish a final receipt.')
    subprocess.run(command, env=environment, check=True, capture_output=True)
    with np.load(workspace / 'results/RK2.npz') as original, np.load(workspace / 'checkpoint-check/RK2.npz') as resumed:
        for key in original.files: np.testing.assert_array_equal(original[key], resumed[key])
    # Original independent quadrature tests retain the hash-bound producer.
    original_tests = ROOT.parents[2] / 'TFM_redaccion/notebooks/04_04_02_balance_energetico_en_linea/test_balance.py'
    subprocess.run([sys.executable, str(original_tests)], env=environment, check=True)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    for kind in ('checksum', 'residual'):
        changed = dict(files)
        if kind == 'checksum': changed['study.json'] += b' '
        else:
            name = 'results/RK2.npz'
            with np.load(io.BytesIO(changed[name]), allow_pickle=False) as a:
                arrays = {key: a[key].copy() for key in a.files}
            arrays['residual'][1, 0] += 1
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **arrays)
            changed[name] = buffer.getvalue()
            receipt = json.loads(changed['results/RK2.json'])
            receipt['result_sha256'] = sha256(changed[name]).hexdigest()
            changed['results/RK2.json'] = json.dumps(receipt).encode()
            manifest = {'files': {name: dict(bytes=len(value), sha256=sha256(value).hexdigest())
                for name, value in changed.items() if name != 'manifest.json'}}
            changed['manifest.json'] = json.dumps(manifest).encode()
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as archive:
            for name, value in changed.items(): archive.writestr(name, value)
        try: runtime.verify_results(buffer.getvalue(), settings, inputs)
        except (ValueError, AssertionError): pass
        else: raise AssertionError('Corruption was not rejected: ' + kind)
    print('Verified eight methods × two cycles, 84 particles, ZIP integrity, residual identities, checkpoint equality and six independent Simpson tests.')


def validate_public_inputs(workspace):
    """Validate anonymous downloads without submitting or rerunning integrations."""
    from build_calculation import KIT_URL
    notebook = json.loads((ROOT / 'calcul.ipynb').read_text())
    pins = notebook['metadata']['gc2d']
    inputs = runtime.download_kit(KIT_URL, pins['manifest_sha256'], pins['kit_sha256'], workspace / 'kit')
    print(f"Anonymous HTTPS download, three-part assembly and {len(inputs['files'])} scientific inputs verified.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace', type=Path)
    parser.add_argument('--kit', type=Path, default=PUBLIC / 'calculation-kit.tar.xz')
    parser.add_argument('--public-inputs-only', action='store_true', help='Check public inputs without any numerical calculation.')
    arguments = parser.parse_args()
    if arguments.public_inputs_only:
        validate_public_inputs(arguments.workspace.resolve())
    else:
        validate(arguments.workspace.resolve(), arguments.kit.resolve())
