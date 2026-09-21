"""Export a self-contained particle selector for saved Poincare section points."""

import base64
import json
import re
from pathlib import Path

import numpy as np


def export_poincare_selector(path, positions, particle_ids, colors, *, title, subtitle):
    """Write an offline HTML figure with independent particle visibility.

    ``positions`` has shape (returns, particles, 2), already wrapped and
    normalized into [0, 1]. Every saved return is embedded as float64; no
    trajectory integration or point subsampling occurs in this function.
    """
    xy = np.asarray(positions, dtype='<f8')
    ids = np.asarray(particle_ids)
    colors = list(colors)
    if xy.ndim != 3 or xy.shape[2] != 2 or not xy.shape[0]:
        raise ValueError('Expected nonempty (returns, particles, 2) coordinates.')
    if not np.isfinite(xy).all() or np.any((xy < 0) | (xy > 1)):
        raise ValueError('Coordinates must be finite and normalized to [0, 1].')
    if (ids.ndim != 1 or len(ids) != xy.shape[1] or len(set(ids)) != len(ids)
            or not np.issubdtype(ids.dtype, np.integer)):
        raise ValueError('Provide a unique integer ID for every particle.')
    if len(colors) != len(ids) or any(not re.fullmatch(r'#[0-9a-fA-F]{6}', c) for c in colors):
        raise ValueError('Provide one hexadecimal RGB color per particle.')
    # Particle-major order permits drawing selected particles without scanning others.
    packed = np.ascontiguousarray(xy.transpose(1, 0, 2), dtype='<f8')
    payload = dict(ids=ids.tolist(), colors=colors, returns=xy.shape[0],
                   title=str(title), subtitle=str(subtitle),
                   coordinates=base64.b64encode(packed.tobytes()).decode('ascii'))
    serialized = json.dumps(payload).replace('<', '\\u003c')
    template = Path(__file__).with_name('_poincare_selector.html').read_text()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(template.replace('__POINCARE_DATA__', serialized), encoding='utf-8')
    return destination
