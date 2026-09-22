"""Checksum-verified persistence for separately calculated probe particles."""

import hashlib
import json
from pathlib import Path

import numpy as np


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_poincare_probe(directory: Path, *, expected_contract: dict | None = None) -> tuple[dict, dict]:
    """Read complete products and optionally require identical inputs and source."""
    directory = Path(directory)
    complete = json.loads((directory / 'COMPLETE.json').read_text())
    for name in ('trajectory.npz', 'metadata.json'):
        if _digest(directory / name) != complete['sha256'][name]:
            raise ValueError(f'Probe product checksum mismatch: {name}')
    metadata = json.loads((directory / 'metadata.json').read_text())
    if expected_contract is not None and metadata['contract'] != json.loads(json.dumps(expected_contract)):
        raise ValueError('Saved probe uses different settings or source. Choose a new run directory.')
    with np.load(directory / 'trajectory.npz', allow_pickle=False) as arrays:
        result = {name: arrays[name].copy() for name in ('times', 'xy')}
        if 'copy_separation_norms' in arrays:
            result['copy_separation_norms'] = arrays['copy_separation_norms'].copy()
    result.update(runtime_seconds=metadata['runtime_seconds'], step_count=metadata['step_count'])
    if 'method_diagnostics' in metadata:
        result['method_diagnostics'] = metadata['method_diagnostics']
    return result, metadata


def save_poincare_probe(directory: Path, result: dict, *, contract: dict) -> Path:
    """Commit float64 trajectories and metadata without overwriting a complete run."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'COMPLETE.json').exists():
        raise FileExistsError('A complete probe run already exists; load it or use a new run directory.')
    for name in ('times', 'xy'):
        if np.asarray(result[name]).dtype != np.dtype('float64') or not np.isfinite(result[name]).all():
            raise ValueError('Persist finite float64 trajectories only.')
    arrays = {name: result[name] for name in ('times', 'xy')}
    if contract.get('method') == 'BM4Midpoint':
        separation = np.asarray(result['copy_separation_norms'])
        if (separation.shape != (result['step_count'],) or not np.isfinite(separation).all()
                or np.any(separation < 0)):
            raise ValueError('BM4Midpoint requires finite nonnegative separation at every step.')
        arrays['copy_separation_norms'] = separation
    temporary = directory / 'trajectory.npz.tmp'
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(directory / 'trajectory.npz')
    metadata = dict(contract=contract, runtime_seconds=result['runtime_seconds'],
                    step_count=result['step_count'], coordinate_layout='(step, particle, xy), unwrapped',
                    initial_state_included=True, sample_scope='Every complete step, including initial state')
    if 'method_diagnostics' in result:
        metadata['method_diagnostics'] = result['method_diagnostics']
    (directory / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    complete = dict(sha256={name: _digest(directory / name) for name in ('trajectory.npz', 'metadata.json')})
    marker = directory / 'COMPLETE.json.tmp'
    marker.write_text(json.dumps(complete, indent=2) + '\n')
    marker.replace(directory / 'COMPLETE.json')
    return directory


# Keep existing RK4 notebooks and stored contracts usable without recomputation.
load_rk4_probe = load_poincare_probe
save_rk4_probe = save_poincare_probe
