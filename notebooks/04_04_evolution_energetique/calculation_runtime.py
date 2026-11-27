"""Verified inputs, isolated historical integrations and reader-owned ZIP results."""
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile
import tempfile
from urllib.request import urlopen
from urllib.parse import urljoin
import zipfile

METHODS = ('ABBA2Midpoint', 'ABBA2Implicit', 'BM4Midpoint', 'BM4Implicit',
    'ImplicitMidpoint', 'RK2', 'RK4', 'GaussLegendre4')
REFERENCE_RUN_ID = 'main-20261117-rho-div20-online-balance-20261120'
GROUPS = {'periodic_candidates8': [1, 2, 60, 64, 65, 72, 76, 83],
    'chaotic8': [12, 15, 18, 21, 24, 27, 30, 33]}


def check_settings(cycles, block_cycles, run_id):
    if type(cycles) is not int or not 1 <= cycles <= 5000:
        raise ValueError('Choose an integer duration from 1 to 5000 cycles.')
    if type(block_cycles) is not int or not 1 <= block_cycles <= cycles:
        raise ValueError('Choose an integer block length between 1 and the cycle count.')
    if not isinstance(run_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', run_id):
        raise ValueError('Use a short run identifier containing letters, numbers, - or _.')
    return dict(cycles=cycles, block_cycles=block_cycles, run_id=run_id)


def safe_name(name):
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and '..' not in path.parts and '\\' not in name and str(path) == name


def unpack_tar(data, destination):
    """Reject links, traversal and duplicate entries before writing any file."""
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:*') as archive:
        members, seen = archive.getmembers(), set()
        for member in members:
            target = destination / member.name
            if (not member.isfile() or not safe_name(member.name) or member.name in seen
                    or not target.resolve().is_relative_to(destination.resolve()) or target.is_symlink()):
                raise ValueError('Unsafe or duplicated archive entry.')
            seen.add(member.name)
        for member in members:
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.extractfile(member).read())



def verify_inputs(kit):
    manifest = json.loads((kit / 'inputs.json').read_text())
    if manifest['reference_run_id'] != REFERENCE_RUN_ID or manifest['methods'] != list(METHODS):
        raise ValueError('Unexpected reproduction-kit identity.')
    expected = {manifest['script'], manifest['reference'], manifest['source'],
        'GC2D_codigo/data/potential/V1/PHI_2.h5'} | {
        f'inputs/{method}.inputs{extension}' for method in METHODS for extension in ('.json', '.npz')}
    if set(manifest['files']) != expected:
        raise ValueError('Unexpected reproduction-kit contents.')
    for name, entry in manifest['files'].items():
        if not safe_name(name):
            raise ValueError('Unsafe input name.')
        data = (kit / name).read_bytes()
        if len(data) != entry['bytes'] or sha256(data).hexdigest() != entry['sha256']:
            raise ValueError(f'Damaged input: {name}')
    for method in METHODS:
        record = json.loads((kit / f'inputs/{method}.inputs.json').read_text())
        proof = manifest['provenance'][method]
        if (proof['public_input_receipt_sha256'] != manifest['files'][f'inputs/{method}.inputs.json']['sha256']
                or proof['input_bundle_sha256'] != manifest['files'][f'inputs/{method}.inputs.npz']['sha256']
                or record['source_archive_sha256'] != manifest['source_tar_sha256']
                or record['reference_sha256'] != manifest['files'][manifest['reference']]['sha256']):
            raise ValueError('Original input provenance mismatch.')
    return manifest


def prepare_kit(data, expected_sha256, destination):
    if sha256(data).hexdigest() != expected_sha256:
        raise ValueError('Reproduction-kit checksum mismatch; nothing was executed.')
    unpack_tar(data, destination)
    return verify_inputs(destination)


def download_kit(url, manifest_sha256, expected_sha256, destination):
    with urlopen(url, timeout=180) as response:
        metadata_bytes = response.read()
    if sha256(metadata_bytes).hexdigest() != manifest_sha256:
        raise ValueError('Public kit manifest checksum mismatch.')
    metadata = json.loads(metadata_bytes)
    if metadata['sha256'] != expected_sha256 or metadata['reference_run_id'] != REFERENCE_RUN_ID:
        raise ValueError('Unexpected public calculation kit.')
    chunks = []
    for index, part in enumerate(metadata['parts'], 1):
        if part['file'] != f'calculation-kit.tar.xz.part{index:03d}':
            raise ValueError('Unexpected kit part name.')
        with urlopen(urljoin(url, part['file']), timeout=300) as response:
            data = response.read()
        if len(data) != part['bytes'] or sha256(data).hexdigest() != part['sha256']:
            raise ValueError('Public kit part checksum mismatch.')
        chunks.append(data)
    data = b''.join(chunks)
    if len(data) != metadata['bytes']:
        raise ValueError('The assembled kit has the wrong length.')
    return prepare_kit(data, expected_sha256, destination)


def write_json(path, record):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def launch_or_recover(worker, receipt_path, settings, kit_sha256, resume_call_id=''):
    """Record an ambiguous submission before spawn so it cannot silently repeat."""
    requested = {**settings, 'kit_sha256': kit_sha256}
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt['settings'] != requested:
            raise ValueError('This run identifier has different settings; choose a new one.')
        if receipt['submission'] != 'confirmed':
            if not resume_call_id:
                raise RuntimeError('Submission was interrupted. Inspect your Modal dashboard and supply its call ID to recover.')
            if not re.fullmatch(r'fc-[A-Za-z0-9]+', resume_call_id):
                raise ValueError('Expected a Modal call identifier beginning with fc-.')
            receipt = dict(submission='confirmed', call_id=resume_call_id, settings=requested)
            write_json(receipt_path, receipt)
            return receipt
        if resume_call_id and resume_call_id != receipt['call_id']:
            raise ValueError('The saved call differs from RESUME_CALL_ID.')
        return receipt
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    if resume_call_id:
        if not re.fullmatch(r'fc-[A-Za-z0-9]+', resume_call_id):
            raise ValueError('Expected a Modal call identifier beginning with fc-.')
        receipt = dict(submission='confirmed', call_id=resume_call_id, settings=requested)
        write_json(receipt_path, receipt)
        return receipt
    import modal
    with modal.enable_output(), worker.app.run(detach=True):
        write_json(receipt_path, dict(submission='unconfirmed', settings=requested))
        call = worker.calculate.spawn(**settings)
        receipt = dict(submission='confirmed', call_id=call.object_id, settings=requested)
        write_json(receipt_path, receipt)
    return receipt



def residual_summary(arrays):
    import numpy as np
    values = {}
    for name, ids in GROUPS.items():
        columns = np.asarray(ids) - 1
        fine = arrays['residual'][1:, columns]
        coarse = (arrays['delta_h'] - arrays['integral_coarse'])[1:, columns]
        rms2, rms4 = float(np.sqrt(np.mean(fine ** 2))), float(np.sqrt(np.mean(coarse ** 2)))
        values[name] = dict(rms_simpson_2=rms2, rms_simpson_4=rms4,
            ratio_4_over_2=rms4 / rms2 if rms2 else None)
    return values


def verify_arrays(arrays, cycles):
    """Check the physical residual identities independently from the producer."""
    import numpy as np
    expected = {'t', 'states', 'cycle', 'particle_ids', 'hamiltonian', 'delta_h',
        'integral_simpson', 'residual', 'integral_coarse', 'quadrature_gap'}
    if set(arrays) != expected:
        raise ValueError('Unexpected numerical result arrays.')
    np.testing.assert_array_equal(arrays['cycle'], np.arange(cycles + 1))
    np.testing.assert_array_equal(arrays['particle_ids'], np.arange(1, 85))
    np.testing.assert_allclose(arrays['t'], np.arange(cycles * 10 + 1) / 10, rtol=0, atol=1e-12)
    shapes = {'t': (cycles * 10 + 1,), 'states': (cycles * 10 + 1, 168),
        'cycle': (cycles + 1,), 'particle_ids': (84,)}
    for key, array in arrays.items():
        if array.shape != shapes.get(key, (cycles + 1, 84)) or not np.isfinite(array).all():
            raise ValueError(f'Incomplete or non-finite result: {key}')
        if key not in ('cycle', 'particle_ids') and array.dtype != np.float64:
            raise ValueError('The numerical protocol requires float64 arrays.')
    np.testing.assert_array_equal(arrays['delta_h'], arrays['hamiltonian'] - arrays['hamiltonian'][0])
    np.testing.assert_array_equal(arrays['residual'], arrays['delta_h'] - arrays['integral_simpson'])
    np.testing.assert_array_equal(arrays['quadrature_gap'], arrays['integral_simpson'] - arrays['integral_coarse'])
    for key in ('delta_h', 'residual', 'integral_simpson', 'integral_coarse', 'quadrature_gap'):
        np.testing.assert_array_equal(arrays[key][0], np.zeros(84))
    return residual_summary(arrays)


def verify_results(data, settings, original_inputs=None):
    import numpy as np
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(not safe_name(n) for n in names):
            raise ValueError('Unsafe result archive.')
        manifest = json.loads(archive.read('manifest.json'))
        expected = {'manifest.json', 'study.json', 'inputs.json', 'source.tar.gz', 'calculo.py', 'reference.json'}
        expected |= {f'results/{method}{ext}' for method in METHODS for ext in ('.json', '.npz')}
        if set(names) != expected or set(manifest['files']) != expected - {'manifest.json'}:
            raise ValueError('The result manifest does not cover the expected ZIP.')
        for name, entry in manifest['files'].items():
            value = archive.read(name)
            if len(value) != entry['bytes'] or sha256(value).hexdigest() != entry['sha256']:
                raise ValueError(f'Damaged result: {name}')
        study = json.loads(archive.read('study.json'))
        if (any(study[k] != v for k, v in settings.items()) or study['methods'] != list(METHODS)
                or study['reference_run_id'] != REFERENCE_RUN_ID or study['particle_count'] != 84
                or study['rho'] != .015 or study['steps_per_cycle'] != 40):
            raise ValueError('The recovered call used different study settings.')
        inputs = json.loads(archive.read('inputs.json'))
        if original_inputs is not None and inputs != original_inputs:
            raise ValueError('The recovered call used different original inputs.')
        for name, key in [('source.tar.gz', inputs['source']), ('calculo.py', inputs['script']), ('reference.json', inputs['reference'])]:
            if sha256(archive.read(name)).hexdigest() != inputs['files'][key]['sha256']:
                raise ValueError('The recovered historical sources differ.')
        summary = {}
        for method in METHODS:
            receipt = json.loads(archive.read(f'results/{method}.json'))
            if (receipt['method'] != method or receipt['cycles'] != settings['cycles']
                    or receipt['run_id'] != settings['run_id'] or receipt['reference_run_id'] != REFERENCE_RUN_ID
                    or receipt['rho'] != .015 or receipt['step'] != .025 or receipt['particle_count'] != 84
                    or receipt['steps_per_cycle'] != 40 or receipt['samples_per_cycle'] != 10
                    or receipt['reference_sha256'] != inputs['files'][inputs['reference']]['sha256']
                    or receipt['source_sha256'] != inputs['source_sha256']
                    or receipt['code_sha256'] != inputs['source_code_sha256']
                    or receipt['script_sha256'] != inputs['script_sha256']
                    or receipt['result_sha256'] != manifest['files'][f'results/{method}.npz']['sha256']
                    or receipt['source_archive_sha256'] != inputs['source_tar_sha256']
                    or receipt['input_bundle_sha256'] != inputs['provenance'][method]['input_bundle_sha256']
                    or receipt['input_bundle_receipt_sha256'] != inputs['provenance'][method]['public_input_receipt_sha256']
                    or not receipt['all_steps_converged'] or not receipt['all_values_finite']
                    or receipt['energy_feedback'] or receipt['derivative_history_saved']):
                raise ValueError(f'Result provenance or convergence mismatch: {method}')
            with np.load(io.BytesIO(archive.read(f'results/{method}.npz')), allow_pickle=False) as arrays:
                summary[method] = verify_arrays(arrays, settings['cycles'])
            if receipt['groups'] != summary[method]:
                raise ValueError('The displayed residual summary differs from the saved arrays.')
        return study, summary


def calculate_local(kit, cycles=5000, block_cycles=100, run_id='reader-run', output_dir=None):
    """Run each unchanged historical program in a fresh interpreter, then audit."""
    import numpy as np
    settings = check_settings(cycles, block_cycles, run_id)
    inputs = verify_inputs(kit)
    with tempfile.TemporaryDirectory(prefix='gc2d-online-results-') as directory:
        output = Path(output_dir) if output_dir is not None else Path(directory) / 'results'
        output.mkdir(parents=True, exist_ok=True)
        for method in METHODS:
            command = [sys.executable, str(kit / inputs['script']), method, '--cycles', str(cycles),
                '--block-cycles', str(block_cycles), '--input-dir', str(kit / 'inputs'), '--output-dir', str(output)]
            process = subprocess.run(command, capture_output=True, text=True)
            if process.returncode:
                raise RuntimeError(f'{method} did not finish (exit {process.returncode}). No result ZIP was published.')
            print(f'{method}: {cycles} cycles completed and ready for verification.', flush=True)
        study = dict(**settings, reference_run_id=REFERENCE_RUN_ID, methods=list(METHODS), rho=.015,
            particle_count=84, steps_per_cycle=40, samples_per_cycle=10, energy_samples_per_cycle=1,
            timing_scope='JAX integration, compilation and online diagnostics; excludes input restoration, checks, checkpoints, compression and transfers',
            trajectory_comparison_performed=False, quadrature_accuracy_certified=False)
        files = {'study.json': (json.dumps(study, sort_keys=True, indent=2) + '\n').encode(),
            'inputs.json': (kit / 'inputs.json').read_bytes(),
            'source.tar.gz': (kit / inputs['source']).read_bytes(),
            'calculo.py': (kit / inputs['script']).read_bytes(),
            'reference.json': (kit / inputs['reference']).read_bytes()}
        for method in METHODS:
            receipt_path = output / (method + '.json')
            receipt = json.loads(receipt_path.read_text())
            with np.load(output / (method + '.npz'), allow_pickle=False) as arrays:
                groups = verify_arrays(arrays, cycles)
            receipt['historical_format_receipt_sha256'] = sha256(receipt_path.read_bytes()).hexdigest()
            receipt['groups'] = groups
            # The legacy script embeds its fixed reference run ID; keep it as
            # provenance and make the new reader run explicit in the public ZIP.
            receipt['reference_run_id'] = receipt.pop('run_id')
            receipt['run_id'] = run_id
            receipt.pop('bucket', None)
            files[f'results/{method}.json'] = (json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
            files[f'results/{method}.npz'] = (output / (method + '.npz')).read_bytes()
        manifest = dict(schema_version=1, files={name: dict(bytes=len(value), sha256=sha256(value).hexdigest())
            for name, value in files.items()})
        files['manifest.json'] = (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode()
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
            for name, value in files.items(): archive.writestr(name, value)
        result = buffer.getvalue()
        verify_results(result, settings, inputs)
        return result
