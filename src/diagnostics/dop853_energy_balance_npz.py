"""Portable, pickle-free persistence for the DOP853 energy-balance study."""
import json
from pathlib import Path
import numpy as np


def save_energy_balance(path, arrays, metadata):
    """Atomically replace one complete study archive with explicit metadata."""
    path = Path(path)
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, metadata_json=json.dumps(metadata), **arrays)
    temporary.replace(path)


def load_energy_balance(path):
    """Read numeric arrays and JSON without executing serialized Python objects."""
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata_json']))
        arrays = {key: data[key].copy() for key in data.files if key != 'metadata_json'}
    if metadata.get('schema_version') != 1:
        raise ValueError('Unsupported energy-balance archive schema.')
    times = arrays['times']
    if times.ndim != 1 or times.size < 2 or not np.all(np.diff(times) > 0):
        raise ValueError('Invalid saved-time grid.')
    for method in ('DOP853', 'BM4Implicit', 'GaussLegendre4', 'RK4'):
        for field, count in [('states', 6), ('H', 3), ('kappa', 3), ('balance', 3), ('distance', 3)]:
            values = arrays[method+'.'+field]
            if values.shape != (count, times.size) or not np.all(np.isfinite(values)):
                raise ValueError(f'Invalid {method} {field} history.')
    for kind in ('endpoint', 'max', 'rms', 'mean'):
        values = arrays['BM4Implicit.mu_'+kind]
        if values.shape != (3, times.size) or not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError('Invalid multiplier history.')
    global_mu = arrays['BM4Implicit.mu_global_statistics']
    if global_mu.shape != (4, times.size) or not np.all(np.isfinite(global_mu)) or np.any(global_mu < 0):
        raise ValueError('Invalid global multiplier statistics.')
    return arrays, metadata
