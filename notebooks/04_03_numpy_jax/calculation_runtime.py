"""Portable input verification, submission recovery and paired-result checks."""
from hashlib import sha256
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import tarfile
import tempfile
from urllib.request import urlopen
import zipfile

METHODS = ('BM4Implicit', 'GaussLegendre4', 'RK2')
BACKENDS = ('scipy', 'jax')
REFERENCE_CYCLES = (10, 30, 90)
REFERENCE_RUN_ID = 'methods-20261119-union16-rho0015-c10-30-90'


def check_settings(cycles, repetitions, run_id):
    if (not isinstance(cycles, (tuple, list)) or not 1 <= len(cycles) <= 3
            or any(type(c) is not int or not 1 <= c <= 90 for c in cycles)
            or len(set(cycles)) != len(cycles)):
        raise ValueError('Choose one to three distinct integer durations from 1 to 90 cycles.')
    if type(repetitions) is not int or not 3 <= repetitions <= 10:
        raise ValueError('Choose 3 to 10 warmed repetitions.')
    if not isinstance(run_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', run_id):
        raise ValueError('Use a short run identifier containing letters, numbers, - or _.')
    return dict(cycles=list(cycles), repetitions=repetitions, run_id=run_id)


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
    cases = [f'{m}-{c}' for c in REFERENCE_CYCLES for m in METHODS]
    expected = {'source.tar.gz', 'study.json'} | {case + suffix for case in cases for suffix in ('.json', '.payload')}
    if set(manifest['files']) != expected or manifest['reference_run_id'] != REFERENCE_RUN_ID:
        raise ValueError('Unexpected reproduction-kit contents.')
    for name, entry in manifest['files'].items():
        data = (kit / name).read_bytes()
        if len(data) != entry['bytes'] or sha256(data).hexdigest() != entry['sha256']:
            raise ValueError(f'Damaged input: {name}')
    for case in cases:
        if manifest['provenance'][case]['public_payload_sha256'] != manifest['files'][case + '.payload']['sha256']:
            raise ValueError('Input provenance mismatch.')
    return manifest


def prepare_kit(data, expected_sha256, destination):
    if sha256(data).hexdigest() != expected_sha256:
        raise ValueError('Reproduction-kit checksum mismatch; nothing was executed.')
    unpack_tar(data, destination)
    manifest = verify_inputs(destination)
    unpack_tar((destination / 'source.tar.gz').read_bytes(), destination / 'source')
    return manifest


def download_kit(url, expected_sha256, destination):
    with urlopen(url, timeout=120) as response:
        return prepare_kit(response.read(), expected_sha256, destination)


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
            raise RuntimeError('Submission was interrupted. Inspect your Modal dashboard before starting another run.')
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


def verify_results(data, settings, original_inputs=None):
    """Check every ZIP entry and paired timing audit before offering a download."""
    import numpy as np
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(not safe_name(n) for n in names):
            raise ValueError('Unsafe result archive.')
        manifest = json.loads(archive.read('manifest.json'))
        if set(names) != set(manifest['files']) | {'manifest.json'}:
            raise ValueError('Result manifest does not cover the complete ZIP.')
        for name, entry in manifest['files'].items():
            value = archive.read(name)
            if len(value) != entry['bytes'] or sha256(value).hexdigest() != entry['sha256']:
                raise ValueError(f'Damaged result: {name}')
        study = json.loads(archive.read('study.json'))
        if any(study[k] != v for k, v in settings.items()):
            raise ValueError('The recovered call used different settings.')
        if (study['methods'] != list(METHODS) or study['particle_count'] != 16
                or study['rho'] != .015 or study['reference_run_id'] != REFERENCE_RUN_ID):
            raise ValueError('Wrong methods, particles or reference study.')
        inputs = json.loads(archive.read('inputs.json'))
        if original_inputs is not None and inputs != original_inputs:
            raise ValueError('The recovered call used different original inputs.')
        if sha256(archive.read('source.tar.gz')).hexdigest() != inputs['files']['source.tar.gz']['sha256']:
            raise ValueError('Different frozen sources in returned archive.')
        timings = json.loads(archive.read('timings.json'))
        expected_cases = {f'{m}-{c}' for c in settings['cycles'] for m in METHODS}
        if set(timings['cases']) != expected_cases:
            raise ValueError('Missing or unexpected paired cases.')
        expected_files = {'study.json', 'inputs.json', 'source.tar.gz', 'timings.json', 'manifest.json'}
        for case, pair in timings['cases'].items():
            method, count_string = case.rsplit('-', 1)
            count = int(count_string)
            template = f'{method}-{count if count in REFERENCE_CYCLES else 10}'
            original_sha = inputs['files'][template + '.payload']['sha256']
            if pair['original_payload_sha256'] != original_sha or (count in REFERENCE_CYCLES and pair['payload_sha256'] != original_sha):
                raise ValueError('Different original numerical payload.')
            if pair['reference_payload_sha256'] != inputs['provenance'][template]['payload_sha256']:
                raise ValueError('Different historical input provenance.')
            if not pair['environment']['x64'] or not pair['same_container'] or not pair['repeated_states_identical']:
                raise ValueError('Worker audits failed.')
            rows = pair['measurements']
            expected = [(b, r) for r in range(settings['repetitions'] + 1)
                for b in (BACKENDS if r % 2 == 0 else BACKENDS[::-1])]
            if [(r['backend'], r['repetition']) for r in rows] != expected:
                raise ValueError('Incomplete, duplicated or incorrectly ordered measurements.')
            for row in rows:
                first = row['repetition'] == 0
                cache = (1, 0) if first else (0, 1)
                if row['backend'] == 'scipy':
                    cache = (0, 0)
                if (row['integration_cache_misses'], row['integration_cache_hits']) != cache:
                    raise ValueError('Invalid compilation-cache audit.')
                if row['phase'] != ('first' if first else 'warm') or not math.isfinite(row['seconds']) or row['seconds'] <= 0:
                    raise ValueError('Invalid timing measurement.')
                if not math.isfinite(row['process_seconds']) or row['process_seconds'] < 0:
                    raise ValueError('Invalid process time.')
            for metric in ('maximum_distance', 'rms_distance', 'final_maximum_distance', 'maximum_distance_first_10_cycles'):
                if not math.isfinite(pair['agreement'][metric]) or pair['agreement'][metric] < 0:
                    raise ValueError('Invalid backend discrepancy.')
            pair_name = f'runs/{case}/pair.json'
            if json.loads(archive.read(pair_name)) != pair:
                raise ValueError('Inconsistent paired timing records.')
            expected_files.add(pair_name)
            states, times = {}, {}
            for backend in BACKENDS:
                base = f'runs/{case}/{backend}/'
                expected_files.update(base + f for f in ('solution.npz', 'metadata.json', 'manifest.json'))
                metadata = json.loads(archive.read(base + 'metadata.json'))
                experiment, diagnostics = metadata['experiment'], metadata['diagnostics']
                if (experiment['method'] != method or experiment['backend'] != backend
                        or experiment['run_id'] != settings['run_id'] or experiment['config']['cycles'] != count
                        or diagnostics['step_count']['scalar'] != count * 40
                        or diagnostics['output_interpolation_count']['scalar'] != 0):
                    raise ValueError('Incorrect trajectory metadata or step count.')
                with np.load(io.BytesIO(archive.read(base + 'solution.npz')), allow_pickle=False) as arrays:
                    states[backend], times[backend] = arrays['states'].copy(), arrays['t'].copy()
                if (states[backend].shape != (count * 10 + 1, 32) or states[backend].dtype != np.float64
                        or not np.isfinite(states[backend]).all()
                        or times[backend].shape != (count * 10 + 1,)
                        or not np.allclose(times[backend], np.linspace(0, count, count * 10 + 1), rtol=0, atol=2e-12)):
                    raise ValueError('Incomplete or invalid numerical trajectory.')
            np.testing.assert_array_equal(times['scipy'], times['jax'])
            np.testing.assert_array_equal(states['scipy'][0], states['jax'][0])
            delta = states['scipy'] - states['jax']
            distance = np.hypot(delta[:, :16], delta[:, 16:])
            actual = dict(maximum_distance=float(distance.max()), rms_distance=float(np.sqrt(np.mean(distance**2))),
                final_maximum_distance=float(distance[-1].max()), maximum_distance_first_10_cycles=float(distance[:101].max()))
            if any(not np.isclose(pair['agreement'][k], v, rtol=1e-13, atol=0) for k, v in actual.items()):
                raise ValueError('Backend agreement does not match the saved trajectories.')
        if set(names) != expected_files:
            raise ValueError('Incomplete trajectories or unexpected result files.')
        return study, timings


def calculate_local(kit, cycles=(10, 30, 90), repetitions=3, run_id='reader-run'):
    """Run all selected pairs; also allow a disposable local smoke validation."""
    from diagnostics.io.persistence import save_solution, load_solution
    from execution._modal_worker import decode_job, encode_job
    from request import SimulationRequest
    from studies.bm4_numpy_jax import benchmark_pair
    import numpy as np

    settings = check_settings(cycles, repetitions, run_id)
    inputs = verify_inputs(kit)
    study = json.loads((kit / 'study.json').read_text())
    study.pop('bucket', None)
    study.update(settings, reference_run_id=inputs['reference_run_id'],
        storage='ZIP download to the reader computer',
        reference_configuration=list(cycles) == list(REFERENCE_CYCLES) and repetitions == 3,
        execution_layout='All cases sequential; both backends share one container for each case; caches reset before every case',
        original_input_files=inputs['files'])
    timings = dict(schema_version=1, cases={})
    with tempfile.TemporaryDirectory(prefix='gc2d-paired-results-') as directory:
        output = Path(directory)
        for count in cycles:
            for method in METHODS:
                case = f'{method}-{count}'
                template = f'{method}-{count if count in REFERENCE_CYCLES else 10}'
                original_payload = (kit / (template + '.payload')).read_bytes()
                payload = original_payload
                if count not in REFERENCE_CYCLES:
                    problem, numerical_method, options, observers = decode_job(payload)
                    request = SimulationRequest.uniform(t_span=(0., float(count)), sample_count=count * 10 + 1)
                    payload = encode_job(problem.with_request(request), numerical_method, options, observers)
                # The original function resets both JAX and integration caches,
                # alternates library order and asserts bitwise repeatability.
                pair = benchmark_pair(payload, repetitions)
                solutions = pair.pop('solutions')
                pair.update(original_payload_sha256=sha256(original_payload).hexdigest(),
                    reference_payload_sha256=inputs['provenance'][template]['payload_sha256'],
                    same_container=True, repeated_states_identical=True)
                timings['cases'][case] = pair
                metadata = json.loads((kit / (template + '.json')).read_text())
                metadata['config']['cycles'] = count
                metadata.update(repetitions=repetitions, run_id=run_id, pair=pair,
                    payload_sha256=sha256(payload).hexdigest(), environment=pair['environment'],
                    reference_payload_sha256=pair['reference_payload_sha256'],
                    original_payload_sha256=sha256(original_payload).hexdigest())
                for backend, solution in solutions.items():
                    if (solution.states.shape != (count * 10 + 1, 32)
                            or solution.diagnostics['step_count'] != count * 40
                            or solution.states.dtype != np.float64 or not np.isfinite(solution.states).all()
                            or solution.t[0] != 0 or solution.t[-1] != count):
                        raise ValueError(f'Incomplete trajectory: {case}/{backend}')
                    target = output / 'runs' / case / backend
                    save_solution(solution, target, metadata={**metadata, 'backend': backend})
                    restored = load_solution(target)
                    np.testing.assert_array_equal(restored.solution.t, solution.t)
                    np.testing.assert_array_equal(restored.solution.states, solution.states)
                (output / 'runs' / case / 'pair.json').write_text(json.dumps(pair, indent=2, allow_nan=False) + '\n')
        for name, value in [('study.json', study), ('timings.json', timings), ('inputs.json', inputs)]:
            (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
        (output / 'source.tar.gz').write_bytes((kit / 'source.tar.gz').read_bytes())
        files = {p.relative_to(output).as_posix(): p.read_bytes() for p in sorted(output.rglob('*')) if p.is_file()}
        manifest = dict(schema_version=1,
            files={name: dict(bytes=len(data), sha256=sha256(data).hexdigest()) for name, data in files.items()})
        files['manifest.json'] = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
            for name, data in files.items():
                archive.writestr(name, data)
        return buffer.getvalue()
