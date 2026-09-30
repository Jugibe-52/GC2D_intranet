"""Read checksum-verified cycle returns independently of trajectory archives."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def load_poincare_cycle_csv(study_directory, run_id):
    """Return metadata, normalized (cycles, particles, xy) points and saved colors.

    Only metadata and cycle returns are required and verified. This loader does
    not claim to validate missing full trajectories or initial-state artifacts.
    """
    if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id or run_id in ('.', '..'):
        raise ValueError('run_id must be a single directory name.')
    directory = Path(study_directory) / 'resultados' / run_id
    manifest = json.loads((directory / 'COMPLETE.json').read_text())
    if manifest.get('schema_version') != 1 or manifest.get('run_id') != run_id:
        raise ValueError('Unsupported or mismatched result manifest.')
    for name in ('metadata.json', 'positions_after_each_cycle.csv'):
        with (directory / name).open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != manifest.get('sha256', {}).get(name):
            raise ValueError(f'Result checksum mismatch: {name}')
    metadata = json.loads((directory / 'metadata.json').read_text())
    ids = metadata['particle_ids']
    cycles = metadata['cycles']
    if (metadata['run_id'] != run_id or len(set(ids)) != len(ids)
            or len(ids) != metadata['particle_count'] or not ids or cycles < 1):
        raise ValueError('Invalid run, particle IDs or cycle count in metadata.')
    # CSV row order need not match the original particle order in metadata.
    indices = {pid: j for j, pid in enumerate(ids)}
    points = np.full((cycles, len(ids), 2), np.nan)
    seen = np.zeros((cycles, len(ids)), dtype=bool)
    colors = [None] * len(ids)
    with (directory / 'positions_after_each_cycle.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            pid, cycle = int(row['particle']), int(row['cycle'])
            if pid not in indices or not 1 <= cycle <= cycles:
                raise ValueError('Unexpected particle or cycle in saved returns.')
            j, k = indices[pid], cycle - 1
            if seen[k, j]:
                raise ValueError('Duplicate particle return in a saved cycle.')
            if colors[j] is not None and colors[j] != row['color']:
                raise ValueError('Particle color changes between saved cycles.')
            colors[j] = row['color']
            points[k, j] = float(row['x_over_L']), float(row['y_over_L'])
            seen[k, j] = True
    if not seen.all() or not np.isfinite(points).all() or np.any((points < 0) | (points > 1)):
        raise ValueError('Incomplete or invalid normalized Poincare returns.')
    return metadata, points, colors
