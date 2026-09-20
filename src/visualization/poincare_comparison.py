"""Offline, synchronized animation of saved Poincare sections."""
from pathlib import Path
import base64
import json
import numpy as np


def export_poincare_comparison(path, coordinates, particle_ids, colors, titles, cycles_per_frame=25):
    """Export (method, cycle, particle, xy) normalized returns as interactive HTML.

    Display coordinates use float32; source calculation files remain unchanged.
    Every saved return is retained, even when playback advances several cycles.
    """
    xy = np.asarray(coordinates, dtype="<f4")
    if xy.ndim != 4 or xy.shape[0] != 3 or xy.shape[-1] != 2:
        raise ValueError("Expected coordinates with shape (3, cycles, particles, 2).")
    if not np.isfinite(xy).all() or np.any((xy < 0) | (xy > 1)):
        raise ValueError("Expected finite periodic coordinates in [0, 1].")
    if xy.shape[2] != len(particle_ids) or len(colors) != len(particle_ids) or len(titles) != 3:
        raise ValueError("Particle labels, colors and panel titles must match the data.")
    if int(cycles_per_frame) < 1:
        raise ValueError("cycles_per_frame must be positive.")
    config = dict(shape=xy.shape, ids=list(map(int, particle_ids)), colors=list(colors),
                  titles=list(titles), step=int(cycles_per_frame),
                  data=base64.b64encode(xy.tobytes()).decode("ascii"))
    template = Path(__file__).with_name("_poincare_comparison.html").read_text()
    Path(path).write_text(template.replace("__CONFIG__", json.dumps(config)), encoding="utf-8")
    return Path(path)
