"""Frozen projection kernels, durable Modal receipts and checked reader ZIPs."""
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
import time
from urllib.request import urlopen
from urllib.parse import urljoin
import zipfile

BASE_METHODS = ('ABBA2Midpoint', 'ABBA2Implicit', 'BM4Midpoint', 'BM4Implicit',
    'ImplicitMidpoint', 'RK2', 'RK4', 'GaussLegendre4')
BASE_REFERENCE_RUN_ID = 'main-20261117-rho-div20-online-balance-20261120'
GROUPS = {'periodic_candidates8': [1, 2, 60, 64, 65, 72, 76, 83],
    'chaotic8': [12, 15, 18, 21, 24, 27, 30, 33]}


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



def verify_base_inputs(kit):
    manifest = json.loads((kit / 'inputs.json').read_text())
    if manifest['reference_run_id'] != BASE_REFERENCE_RUN_ID or manifest['methods'] != list(BASE_METHODS):
        raise ValueError('Unexpected reproduction-kit identity.')
    expected = {manifest['script'], manifest['reference'], manifest['source'],
        'GC2D_codigo/data/potential/V1/PHI_2.h5'} | {
        f'inputs/{method}.inputs{extension}' for method in BASE_METHODS for extension in ('.json', '.npz')}
    if set(manifest['files']) != expected:
        raise ValueError('Unexpected reproduction-kit contents.')
    for name, entry in manifest['files'].items():
        if not safe_name(name):
            raise ValueError('Unsafe input name.')
        data = (kit / name).read_bytes()
        if len(data) != entry['bytes'] or sha256(data).hexdigest() != entry['sha256']:
            raise ValueError(f'Damaged input: {name}')
    for method in BASE_METHODS:
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
    return verify_base_inputs(destination)


def download_kit(url, manifest_sha256, expected_sha256, destination):
    with urlopen(url, timeout=180) as response:
        metadata_bytes = response.read()
    if sha256(metadata_bytes).hexdigest() != manifest_sha256:
        raise ValueError('Public kit manifest checksum mismatch.')
    metadata = json.loads(metadata_bytes)
    if metadata['sha256'] != expected_sha256 or metadata['reference_run_id'] != BASE_REFERENCE_RUN_ID:
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


METHODS = ('ABBA2Implicit', 'BM4Implicit')
REFERENCE_RUN_ID = 'main-20261117-rho-div20'
STUDY_PREFIX = 'studies/projection_correction'
SCALARS = ('nonlinear_iterations', 'nonlinear_residual_norms',
    'nonlinear_tolerances', 'projection_multiplier_norms')


def check_settings(cycles, block_cycles, run_id, rho_indices):
    if type(cycles) is not int or not 1 <= cycles <= 5000:
        raise ValueError('Choose an integer duration from 1 to 5000 cycles.')
    if type(block_cycles) is not int or not 1 <= block_cycles <= cycles:
        raise ValueError('Choose a block length between 1 and the cycle count.')
    if not isinstance(run_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', run_id):
        raise ValueError('Use a short run identifier containing letters, numbers, - or _.')
    if (not isinstance(rho_indices, list) or not rho_indices or len(set(rho_indices)) != len(rho_indices)
            or any(type(i) is not int or not 0 <= i <= 10 for i in rho_indices)):
        raise ValueError('Radius indices must be distinct integers between 0 and 10.')
    return dict(cycles=cycles, block_cycles=block_cycles, run_id=run_id, rho_indices=sorted(rho_indices))


def verify_protocol(kit, expected_sha256):
    import numpy as np
    data = (kit / 'projection-protocol.json').read_bytes()
    if sha256(data).hexdigest() != expected_sha256:
        raise ValueError('Projection protocol checksum mismatch.')
    protocol = json.loads(data)
    base = verify_base_inputs(kit)
    if (protocol['reference_run_id'] != REFERENCE_RUN_ID or protocol['study_prefix'] != STUDY_PREFIX
            or protocol['methods'] != list(METHODS) or protocol['rho_values'] != [i / 400 for i in range(11)]
            or protocol['config'] != dict(cycles=5000, steps_per_cycle=40, samples_per_cycle=10)
            or protocol['source_tar_sha256'] != base['source_tar_sha256']
            or protocol['source_sha256'] != base['source_sha256']
            or protocol['source_code_sha256'] != base['source_code_sha256']
            or protocol['reference_sha256'] != base['files'][base['reference']]['sha256']):
        raise ValueError('The shared frozen kit does not match the projection protocol.')
    path = kit / ('inputs/' + protocol['shared_input_method'] + '.inputs.npz')
    if sha256(path.read_bytes()).hexdigest() != protocol['shared_input_bundle_sha256']:
        raise ValueError('The common initial state or time plan changed.')
    with np.load(path, allow_pickle=False) as arrays:
        for name, digest in protocol['shared_array_sha256'].items():
            if sha256(arrays[name].tobytes()).hexdigest() != digest:
                raise ValueError('Shared array differs: ' + name)
        np.testing.assert_array_equal(arrays['initial_state'], np.asarray(protocol['initial_positions']).T.ravel())
    record = json.loads(path.with_suffix('.json').read_text())['metadata']['experiment']
    if (protocol['particle_record'] != record['particle_record']
            or [p['id'] for p in protocol['particle_record']['particles']] != list(range(1, 85))
            or any(record[key] != value for key, value in protocol['physical_parameters'].items())):
        raise ValueError('Particle identities, colors or physical parameters differ.')
    return protocol, base


def selected_cases(protocol, settings):
    return [case for case in protocol['cases'] if case['rho'] in [i / 400 for i in settings['rho_indices']]]


def launch_or_recover(worker, directory, settings, protocol, protocol_sha256,
        allow_launch=False, resume_calls=None):
    """Persist every call before waiting; never replace an ambiguous submission."""
    resume_calls = resume_calls or {}
    cases = selected_cases(protocol, settings)
    if set(resume_calls) - {case['case_key'] for case in cases}:
        raise ValueError('A recovery call identifies a case outside this selection.')
    directory.mkdir(parents=True, exist_ok=True)
    receipts, pending = {}, []
    for case in cases:
        key = case['case_key']
        path = directory / (key + '.receipt.json')
        requested = dict(settings=settings, case=case, protocol_sha256=protocol_sha256,
            runtime_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
        previous = json.loads(path.read_text()) if path.exists() else None
        call_id = resume_calls.get(key, '')
        if call_id and not re.fullmatch(r'fc-[A-Za-z0-9]+', call_id):
            raise ValueError('A Modal call identifier must begin with fc-.')
        if previous is not None and previous['requested'] != requested:
            raise ValueError('This run identifier has different inputs or code; choose a new identifier.')
        if previous is not None and previous['submission'] == 'confirmed':
            if call_id and call_id != previous['call_id']:
                raise ValueError('Recovery call differs from the saved receipt: ' + key)
            receipts[key] = previous
        elif call_id:
            receipts[key] = dict(submission='confirmed', call_id=call_id, requested=requested)
            write_json(path, receipts[key])
        elif previous is not None:
            raise RuntimeError('Ambiguous submission for ' + key + '; inspect Modal and supply its call ID. No replacement was launched.')
        elif not allow_launch:
            raise RuntimeError('Enable launch explicitly or provide the missing recovery call: ' + key)
        else:
            pending.append((case, path, requested))
    if pending:
        import modal
        with modal.enable_output(), worker.app.run(detach=True):
            for case, path, requested in pending:
                write_json(path, dict(submission='unconfirmed', requested=requested))
                call = worker.calculate.spawn(case=case, settings=settings, protocol_sha256=protocol_sha256)
                receipt = dict(submission='confirmed', call_id=call.object_id, requested=requested)
                write_json(path, receipt)
                receipts[case['case_key']] = receipt
    write_json(directory / 'receipts.json', receipts)
    return receipts


def summarize_arrays(arrays, cycles):
    """Recompute Euclidean sampled statistics and all-step joint Newton work."""
    import numpy as np
    steps, samples = cycles * 40, cycles * 10
    expected = {'t', 'states', 'projection_multipliers_sampled', *SCALARS,
        'step_start_times', 'step_sizes', 'step_times', 'particle_ids'}
    if set(arrays) != expected:
        raise ValueError('Unexpected projection result arrays.')
    shapes = dict(t=(samples + 1,), states=(samples + 1, 168),
        projection_multipliers_sampled=(samples, 2, 84), particle_ids=(84,))
    for key, value in arrays.items():
        if value.shape != shapes.get(key, (steps,)) or not np.isfinite(value).all():
            raise ValueError('Incomplete or non-finite array: ' + key)
        if key not in ('particle_ids', 'nonlinear_iterations') and value.dtype != np.float64:
            raise ValueError('The numerical protocol requires float64: ' + key)
    np.testing.assert_array_equal(arrays['particle_ids'], np.arange(1, 85))
    np.testing.assert_array_equal(arrays['t'][1:], arrays['step_times'][3::4])
    np.testing.assert_array_equal(arrays['step_start_times'][1:], arrays['step_times'][:-1])
    np.testing.assert_allclose(arrays['t'], np.arange(samples + 1) / 10, rtol=0, atol=1e-12)
    np.testing.assert_allclose(arrays['step_sizes'], .025, rtol=0, atol=2e-12)
    mu = arrays['projection_multipliers_sampled']
    infinity = np.max(np.abs(mu), axis=1)
    np.testing.assert_array_equal(infinity.max(axis=1), arrays['projection_multiplier_norms'][3::4])
    iterations, residual, tolerance = (arrays[key] for key in SCALARS[:3])
    if ((iterations < 0).any() or (iterations > 40).any() or (iterations != np.floor(iterations)).any()
            or (tolerance <= 0).any() or (residual < 0).any() or (residual > tolerance).any()):
        raise ValueError('Invalid Newton convergence history.')
    euclidean = np.hypot(mu[:, 0], mu[:, 1])
    return dict(sampled_median=float(np.median(euclidean)), sampled_maximum=float(euclidean.max()),
        newton_mean_iterations=float(iterations.mean()), newton_max_iterations=int(iterations.max()),
        max_residual_to_tolerance=float((residual / tolerance).max()))


def pack_files(files):
    manifest = dict(schema_version=1, files={name: dict(bytes=len(value), sha256=sha256(value).hexdigest())
        for name, value in files.items()})
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, value in files.items():
            archive.writestr(name, value)
        archive.writestr('manifest.json', json.dumps(manifest, sort_keys=True, indent=2))
    return buffer.getvalue()


def verify_case_result(data, case, settings, protocol, protocol_sha256):
    import numpy as np
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(not safe_name(n) for n in names):
            raise ValueError('Unsafe result archive.')
        manifest = json.loads(archive.read('manifest.json'))
        if set(manifest['files']) != set(names) - {'manifest.json'}:
            raise ValueError('Incomplete result checksums.')
        for name, entry in manifest['files'].items():
            value = archive.read(name)
            if len(value) != entry['bytes'] or sha256(value).hexdigest() != entry['sha256']:
                raise ValueError('Damaged result: ' + name)
        receipt = json.loads(archive.read('result.json'))
        if (receipt['case'] != case or receipt['settings'] != settings
                or receipt['reference_run_id'] != REFERENCE_RUN_ID
                or receipt['study_prefix'] != STUDY_PREFIX or receipt['protocol_sha256'] != protocol_sha256
                or json.loads(archive.read('protocol.json')) != protocol
                or receipt['source_code_sha256'] != protocol['source_code_sha256']
                or receipt['source_sha256'] != protocol['source_sha256']
                or receipt['source_archive_sha256'] != protocol['source_tar_sha256']
                or receipt['projection_units'] != 'normalized physical coordinates'
                or receipt['projection_iteration_scope'] != 'joint_particle_batch'
                or receipt['projection_formulation'] != 'reduced_multiplier'
                or receipt['projection_placement'] != 'around_complete_composition'):
            raise ValueError('Different case, settings or provenance in the recovered result.')
        if sha256(archive.read('calculation_runtime.py')).hexdigest() != sha256(Path(__file__).read_bytes()).hexdigest():
            raise ValueError('The numerical driver differs from this notebook.')
        if not all(math.isfinite(t) and t >= 0 for t in receipt['block_numerical_seconds']):
            raise ValueError('Invalid numerical timings.')
        if (not math.isfinite(receipt['numerical_seconds']) or receipt['numerical_seconds'] < 0
                or receipt['numerical_seconds'] != sum(receipt['block_numerical_seconds'])):
            raise ValueError('Invalid total numerical timing.')
        common = {'manifest.json', 'result.json', 'protocol.json', 'calculation_runtime.py'}
        if receipt['status'] == 'complete':
            if set(names) != common | {'arrays.npz'}:
                raise ValueError('Unexpected completed-case payload.')
            with np.load(io.BytesIO(archive.read('arrays.npz')), allow_pickle=False) as arrays:
                summary = summarize_arrays(arrays, settings['cycles'])
                initial_hash = sha256(arrays['states'][0].tobytes()).hexdigest()
                if initial_hash != protocol['shared_array_sha256']['initial_state']:
                    raise ValueError('The physical initial conditions changed.')
                if settings['cycles'] == 5000:
                    for name in ('t', 'step_start_times', 'step_sizes', 'step_times'):
                        if sha256(arrays[name].tobytes()).hexdigest() != protocol['shared_array_sha256'][name]:
                            raise ValueError('The exact archived time plan changed: ' + name)
            if len(receipt['block_numerical_seconds']) != math.ceil(settings['cycles'] / settings['block_cycles']):
                raise ValueError('Incomplete block timings.')
            if (set(receipt['summary']) != set(summary)
                    or receipt['summary']['newton_max_iterations'] != summary['newton_max_iterations']
                    or any(not math.isclose(receipt['summary'][key], value, rel_tol=2e-13, abs_tol=0.)
                        for key, value in summary.items())):
                raise ValueError('Summary differs from the saved histories.')
        elif receipt['status'] == 'failed':
            if (set(names) != common or receipt['reason'] not in ('nonlinear_convergence_failure', 'non_finite_numerical_state')
                    or type(receipt['first_failed_step']) is not int
                    or not 1 <= receipt['first_failed_step'] <= settings['cycles'] * 40):
                raise ValueError('Invalid numerical failure record.')
            if len(receipt['block_numerical_seconds']) != math.ceil(receipt['first_failed_step'] / (40 * settings['block_cycles'])):
                raise ValueError('Failed-case block timings do not identify its stopping block.')
        else:
            raise ValueError('Unknown result status.')
        return receipt


def calculate_case(kit, case, settings, protocol_sha256):
    """Integrate one case in a fresh process with unchanged historical kernels."""
    settings = check_settings(**settings)
    protocol, inputs = verify_protocol(kit, protocol_sha256)
    if case not in selected_cases(protocol, settings):
        raise ValueError('Unexpected case selection.')
    with tempfile.TemporaryDirectory(prefix='gc2d-projection-') as directory:
        root = Path(directory)
        spec = root / 'request.json'
        write_json(spec, dict(case=case, settings=settings, protocol_sha256=protocol_sha256))
        result = root / 'case.zip'
        command = [sys.executable, str(Path(__file__).resolve()), '--integrate', str(kit), str(spec), str(result)]
        subprocess.run(command, check=True)
        data = result.read_bytes()
        verify_case_result(data, case, settings, protocol, protocol_sha256)
        return data


def integrate_case(kit, case, settings, protocol_sha256):
    """Run blocks on the exact archived clock; observe without feeding back."""
    import numpy as np
    protocol, inputs = verify_protocol(kit, protocol_sha256)
    with tempfile.TemporaryDirectory(prefix='gc2d-frozen-') as directory:
        frozen = Path(directory)
        unpack_tar((kit / inputs['source']).read_bytes(), frozen)
        digest = sha256()
        paths = [p for p in (frozen / 'src').rglob('*.py') if p.relative_to(frozen / 'src').parts[0] not in ('visualization', 'studies')]
        paths += [frozen / 'src/studies/main_campaign.py', frozen / 'examples/modal_main_campaign_app.py']
        for path in sorted(paths):
            digest.update(path.relative_to(frozen).as_posix().encode())
            digest.update(path.read_bytes())
        if digest.hexdigest() != protocol['source_code_sha256']:
            raise ValueError('Frozen numerical-source fingerprint differs.')
        sys.path.insert(0, str(frozen / 'src'))
        import jax
        jax.config.update('jax_enable_x64', True)
        import jax.numpy as jnp
        from studies.main_campaign import load_field, campaign_method, campaign_case_key
        from dynamics.gc import GuidingCenterDynamics
        from states.initial import InitialConditions
        from execution._jax import prepare_dynamics
        from execution._jax_step import prepare_step
        if campaign_case_key(case['rho'], case['method']) != case['case_key']:
            raise ValueError('Frozen campaign case-key convention changed.')
        cycles = settings['cycles']
        with np.load(kit / 'inputs/ABBA2Implicit.inputs.npz', allow_pickle=False) as source:
            initial_state = source['initial_state'].copy()
            times = source['t'][:cycles * 10 + 1].copy()
            starts = source['step_start_times'][:cycles * 40].copy()
            sizes = source['step_sizes'][:cycles * 40].copy()
            ends = source['step_times'][:cycles * 40].copy()
        dynamics = prepare_dynamics(GuidingCenterDynamics(
            load_field(kit / 'GC2D_codigo/data/potential/V1/PHI_2.h5'), rho=case['rho'], B=1.5))
        initial = InitialConditions(initial_state, layout=GuidingCenterDynamics.coordinates)
        method = campaign_method(case['method'], .025)
        run = method.new_run(dynamics.vector_field, initial, jacobian=dynamics.particle_vector_field_jacobians)
        step = prepare_step(run, jax.devices('cpu')[0], initial_conditions=initial)

        def advance(state, clock):
            after, statistics, success = step(clock[0], state, clock[1])
            values = (after, statistics['projection_multipliers'].reshape(2, 84),
                *(statistics[name] for name in SCALARS), jnp.all(success))
            return after, values

        compiled = jax.jit(lambda state, clock: jax.lax.scan(advance, state, clock))
        state = initial_state
        parts, durations = [], []
        failure = None
        for start in range(0, cycles * 40, settings['block_cycles'] * 40):
            stop = min(cycles * 40, start + settings['block_cycles'] * 40)
            started = time.perf_counter()
            state, result = jax.device_get(compiled(state, (starts[start:stop], sizes[start:stop])))
            durations.append(time.perf_counter() - started)
            history, multipliers, iterations, residual, tolerance, infinity, success = result
            finite = np.isfinite(history).all(axis=1) & np.isfinite(multipliers).all(axis=(1, 2))
            finite &= np.isfinite(residual) & np.isfinite(tolerance) & np.isfinite(infinity)
            accepted = success & finite & (residual <= tolerance) & (tolerance > 0)
            if not accepted.all():
                index = int(np.flatnonzero(~accepted)[0])
                failure = dict(status='failed', first_failed_step=start + index + 1,
                    reason='nonlinear_convergence_failure' if finite[index] else 'non_finite_numerical_state')
                break
            parts.append((history[3::4], multipliers[3::4], iterations, residual, tolerance, infinity))
            print(case['case_key'], stop // 40, '/', cycles, 'cycles', flush=True)
        receipt = dict(schema_version=1, study_prefix=STUDY_PREFIX, reference_run_id=REFERENCE_RUN_ID,
            case=case, settings=settings, protocol_sha256=protocol_sha256,
            source_code_sha256=protocol['source_code_sha256'], source_sha256=protocol['source_sha256'],
            source_archive_sha256=protocol['source_tar_sha256'],
            projection_iteration_scope='joint_particle_batch', projection_formulation='reduced_multiplier',
            projection_placement='around_complete_composition', projection_units='normalized physical coordinates',
            block_numerical_seconds=durations, numerical_seconds=sum(durations),
            timing_scope='JAX compilation, fixed-step integration and passive projection diagnostics; excludes input restoration, verification, packaging and network transfers',
            dependency_versions={name: __import__(name).__version__ for name in ('numpy', 'scipy', 'jax', 'jaxlib', 'h5py')},
            trajectory_comparison_performed=False)
        files = {'protocol.json': (kit / 'projection-protocol.json').read_bytes(),
            'calculation_runtime.py': Path(__file__).read_bytes()}
        if failure:
            receipt.update(failure)
        else:
            merged = [np.concatenate([p[i] for p in parts]) for i in range(6)]
            arrays = dict(t=times, states=np.vstack((initial_state, merged[0])),
                projection_multipliers_sampled=merged[1], particle_ids=np.arange(1, 85),
                step_start_times=starts, step_sizes=sizes, step_times=ends,
                **dict(zip(SCALARS, merged[2:])))
            receipt.update(status='complete', summary=summarize_arrays(arrays, cycles))
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **arrays)
            files['arrays.npz'] = buffer.getvalue()
        files['result.json'] = (json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
        return pack_files(files)


def combine_results(directory, settings, protocol, protocol_sha256):
    """Keep individual checked ZIPs and a reusable small summary in one archive."""
    rows, case_files = [], []
    for case in selected_cases(protocol, settings):
        path = directory / (case['case_key'] + '.zip')
        result = verify_case_result(path.read_bytes(), case, settings, protocol, protocol_sha256)
        row = dict(case_key=case['case_key'], rho=case['rho'], method=case['method'],
            status=result['status'], reference_status=case['reference_status'],
            numerical_seconds=result['numerical_seconds'])
        if result['status'] == 'complete':
            row.update(result['summary'])
        else:
            row.update(reason=result['reason'], first_failed_step=result['first_failed_step'])
        rows.append(row)
        case_files.append(path)
    summary = dict(schema_version=1, study_prefix=STUDY_PREFIX, reference_run_id=REFERENCE_RUN_ID,
        run_id=settings['run_id'], config=dict(cycles=settings['cycles'], steps_per_cycle=40, samples_per_cycle=10),
        particle_count=84, methods=list(METHODS), rho_values=[i / 400 for i in settings['rho_indices']],
        representative_rho=.015, cases=rows,
        limitations='Euclidean multiplier statistics use the last step of each four-step block, without the initial state; Newton statistics use every joint-batch solve. New runs are distinct from the historical reference.')
    write_json(directory / 'summary.json', summary)
    write_json(directory / 'study.json', dict(**settings, protocol_sha256=protocol_sha256,
        study_prefix=STUDY_PREFIX, reference_run_id=REFERENCE_RUN_ID))
    files = [*case_files, directory / 'summary.json', directory / 'study.json']
    files += sorted(directory.glob('*.receipt.json'))
    if (directory / 'receipts.json').exists():
        files.append(directory / 'receipts.json')
    manifest = dict(schema_version=1, files={p.name: dict(bytes=p.stat().st_size,
        sha256=sha256(p.read_bytes()).hexdigest()) for p in files})
    destination = directory / (settings['run_id'] + '.zip')
    temporary = destination.with_suffix('.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path in files:
            archive.write(path, path.name)
        archive.writestr('manifest.json', json.dumps(manifest, sort_keys=True, indent=2))
    temporary.replace(destination)
    with zipfile.ZipFile(destination) as archive:
        for name, record in manifest['files'].items():
            with archive.open(name) as stream:
                digest = sha256()
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            if digest.hexdigest() != record['sha256'] or archive.getinfo(name).file_size != record['bytes']:
                raise ValueError('Combined ZIP verification failed: ' + name)
    return destination, summary


if __name__ == '__main__':
    if len(sys.argv) != 5 or sys.argv[1] != '--integrate':
        raise SystemExit('Use --integrate KIT REQUEST_JSON RESULT_ZIP.')
    kit, spec, result = map(Path, sys.argv[2:])
    request = json.loads(spec.read_text())
    result.write_bytes(integrate_case(kit, **request))
