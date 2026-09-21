"""Portable snapshot, run-directory and result integrity helpers for the notebooks."""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys
import zipfile


ROOT = Path(__file__).resolve().parent


def utc_now():
    """Return an unambiguous timestamp for execution records."""
    return datetime.now(timezone.utc).isoformat()


def new_run_id():
    return datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%S_%fZ')


def validate_run_id(value):
    """Keep every run inside its package results directory."""
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}', value):
        raise ValueError('Run ID must contain 1-96 letters, numbers, underscores or hyphens.')
    return value


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, data):
    """Publish a complete JSON file using a same-filesystem rename."""
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def load_snapshot():
    """Verify the bundled field and source before importing the exact frozen code."""
    manifest = json.loads((ROOT / 'assets' / 'snapshot.json').read_text())
    archive_path = ROOT / 'assets' / 'gc2d_snapshot.zip'
    expected = manifest['sha256']
    if digest(archive_path) != expected:
        raise ValueError('The numerical snapshot checksum does not match.')
    runtime = ROOT / '.runtime' / expected
    runtime.mkdir(parents=True, exist_ok=True)
    for package in ('potential', 'dynamics', 'initial_conditions', 'simulation'):
        loaded = sys.modules.get(package)
        if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(runtime):
            raise RuntimeError('Restart the kernel before loading the frozen project implementation.')
    # Serialize extraction, and never rewrite a verified file being imported
    # by another worker. The parent materializes this cache before spawning.
    with (ROOT / '.runtime' / 'snapshot.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                relative = Path(member.filename)
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError('Invalid snapshot member.')
                target = runtime / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                content = archive.read(member)
                if not target.exists() or target.read_bytes() != content:
                    temporary = target.with_name('.' + target.name + '.tmp')
                    temporary.write_bytes(content)
                    temporary.replace(target)
        fcntl.flock(lock, fcntl.LOCK_UN)
    if str(runtime) not in sys.path:
        sys.path.insert(0, str(runtime))
    importlib.invalidate_caches()
    import numpy as np
    from potential import Grid, Potential
    provenance = json.loads((runtime / 'provenance.json').read_text())
    with np.load(runtime / 'field.npz', allow_pickle=False) as saved:
        potential = Potential(Grid(**provenance['grid']), mean=saved['mean'],
                              modes=saved['modes'], frequencies=saved['frequencies'],
                              interpolation_order=provenance['interpolation_order'])
    return potential, provenance, expected


def begin_calculation(run_id, parameters):
    """Reserve a new calculation without overwriting an earlier attempt."""
    run_id = validate_run_id(run_id)
    directory = ROOT / 'resultados' / run_id
    directory.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects an interactive notebook from concurrent reuse.
    with (directory / 'run_parameters.json').open('x', encoding='utf-8') as stream:
        json.dump({'created_utc': utc_now(), **parameters}, stream, indent=2, allow_nan=False)
    return directory


def publish_calculation(directory, metadata, arrays, positions, initial_positions):
    """Save numeric products, then publish a checksum manifest as the commit marker."""
    import numpy as np
    directory = Path(directory)
    for name, frame in [('positions_after_each_cycle.csv', positions),
                        ('initial_positions.csv', initial_positions)]:
        temporary = directory / ('.' + name + '.tmp')
        frame.to_csv(temporary, index=False, float_format='%.17g')
        temporary.replace(directory / name)
    with (directory / '.bm4_trajectory.npz.tmp').open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    (directory / '.bm4_trajectory.npz.tmp').replace(directory / 'bm4_trajectory.npz')
    atomic_json(directory / 'metadata.json', metadata)
    from newton_diagnostics import write_diagnostic_tables
    diagnostic_files = write_diagnostic_tables(directory, metadata, arrays)
    files = ('bm4_trajectory.npz', 'metadata.json', 'positions_after_each_cycle.csv',
             'initial_positions.csv', 'run_parameters.json') + diagnostic_files
    atomic_json(directory / 'COMPLETE.json', {
        'schema_version': 1, 'completed_utc': utc_now(), 'run_id': directory.name,
        'sha256': {name: digest(directory / name) for name in files},
    })
    if not metadata.get('validation_run', False):
        atomic_json(ROOT / 'resultados' / 'latest_success.json', {'run_id': directory.name})


def load_calculation(run_id=None):
    """Read completed artifacts only, without importing or running the integrator."""
    import numpy as np
    import pandas as pd
    if run_id is None:
        pointer = ROOT / 'resultados' / 'latest_success.json'
        if not pointer.is_file():
            raise FileNotFoundError('No completed calculation. Run calculo.ipynb or select RUN_ID explicitly.')
        run_id = json.loads(pointer.read_text())['run_id']
    directory = ROOT / 'resultados' / validate_run_id(run_id)
    marker = directory / 'COMPLETE.json'
    if not marker.is_file():
        raise FileNotFoundError(f'No complete result manifest: {marker}')
    manifest = json.loads(marker.read_text())
    expected_files = {'bm4_trajectory.npz', 'metadata.json', 'positions_after_each_cycle.csv',
                      'initial_positions.csv', 'run_parameters.json',
                      'newton_steps.csv.gz', 'newton_iterations.csv.gz'}
    if (manifest.get('schema_version') != 1 or manifest.get('run_id') != run_id
            or set(manifest.get('sha256', {})) != expected_files):
        raise ValueError('Unsupported or incomplete result manifest.')
    for name, expected in manifest['sha256'].items():
        if digest(directory / name) != expected:
            raise ValueError(f'Result checksum mismatch: {name}')
    metadata = json.loads((directory / 'metadata.json').read_text())
    with np.load(directory / 'bm4_trajectory.npz', allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    positions = pd.read_csv(directory / 'positions_after_each_cycle.csv', float_precision='round_trip')
    initial_positions = pd.read_csv(directory / 'initial_positions.csv', float_precision='round_trip')
    return directory, metadata, arrays, positions, initial_positions
