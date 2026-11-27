"""Portable, standard-library-only preparation, recovery and result checks."""
from hashlib import sha256
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import tarfile
from urllib.request import urlopen
import zipfile

METHODS = ('RK2', 'ImplicitMidpoint', 'RK4', 'GaussLegendre4',
    'ABBA2Midpoint', 'ABBA2Implicit', 'BM4Midpoint', 'BM4Implicit')


def check_settings(cycles, repetitions, run_id):
    if type(cycles) is not int or not 1 <= cycles <= 500:
        raise ValueError('Choose an integer duration from 1 to 500 cycles.')
    if type(repetitions) is not int or not 3 <= repetitions <= 10:
        raise ValueError('Choose 3 to 10 cached repetitions.')
    if not isinstance(run_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', run_id):
        raise ValueError('Use a short run identifier containing letters, numbers, - or _.')
    return dict(cycles=cycles, repetitions=repetitions, run_id=run_id)


def unpack_tar(data, destination):
    """Reject traversal, links and duplicates before writing any trusted bundle."""
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:*') as archive:
        members = archive.getmembers()
        seen = set()
        for member in members:
            path = PurePosixPath(member.name)
            if (not member.isfile() or path.is_absolute() or '..' in path.parts
                    or '\\' in member.name or member.name in seen):
                raise ValueError('Unsafe or duplicated archive entry.')
            target = destination / member.name
            if target.resolve().is_relative_to(destination.resolve()) is False or target.is_symlink():
                raise ValueError('Unsafe extraction destination.')
            seen.add(member.name)
        for member in members:
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.extractfile(member).read())


def verify_inputs(kit):
    manifest = json.loads((kit / 'inputs.json').read_text())
    expected = {'source.tar.gz', 'study.json'} | {m + s for m in METHODS for s in ('.json', '.payload')}
    if set(manifest['files']) != expected:
        raise ValueError('Unexpected reproduction-kit contents.')
    for name, entry in manifest['files'].items():
        data = (kit / name).read_bytes()
        if len(data) != entry['bytes'] or sha256(data).hexdigest() != entry['sha256']:
            raise ValueError(f'Damaged input: {name}')
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
        data = response.read()
    return prepare_kit(data, expected_sha256, destination)


def write_json(path, record):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def launch_or_recover(worker, receipt_path, settings, kit_sha256, resume_call_id=''):
    """A pending receipt prevents duplicate jobs after an ambiguous submission."""
    import modal
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
        receipt = dict(submission='confirmed', call_id=resume_call_id, settings=requested)
        write_json(receipt_path, receipt)
        return receipt
    with modal.enable_output(), worker.app.run(detach=True):
        write_json(receipt_path, dict(submission='unconfirmed', settings=requested))
        call = worker.calculate.spawn(**settings)
        receipt = dict(submission='confirmed', call_id=call.object_id, settings=requested)
        write_json(receipt_path, receipt)
    return receipt


def verify_results(data, settings):
    """Check the returned archive before saving or offering it for download."""
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts for n in names):
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
        if study['methods'] != list(METHODS) or study['particle_count'] != 40:
            raise ValueError('Wrong methods or particles.')
        timings = json.loads(archive.read('timings.json'))
        expected = {(m, r) for m in METHODS for r in range(settings['repetitions'] + 1)}
        rows = timings['measurements']
        if len(rows) != len(expected) or {(r['method'], r['repetition']) for r in rows} != expected:
            raise ValueError('Incomplete or duplicated timing measurements.')
        if not timings['same_container'] or not timings['repeated_states_identical']:
            raise ValueError('Worker audits failed.')
        for row in rows:
            if row['step_count'] != settings['cycles'] * 40:
                raise ValueError('Incorrect step count.')
            first = row['repetition'] == 0
            if (row['integration_cache_misses'], row['integration_cache_hits']) != ((1, 0) if first else (0, 1)):
                raise ValueError('Invalid compilation-cache audit.')
            order = METHODS if row['repetition'] % 2 == 0 else METHODS[::-1]
            if row['position'] != order.index(row['method']) or row['phase'] != ('first' if first else 'warm'):
                raise ValueError('Incorrect repetition order or phase.')
            if not math.isfinite(row['seconds']) or row['seconds'] <= 0:
                raise ValueError('Invalid elapsed time.')
        for method in METHODS:
            if not any(n.startswith(f'runs/{method}/') and n.endswith('.npz') for n in names):
                raise ValueError('A trajectory archive is missing.')
        return study, timings
