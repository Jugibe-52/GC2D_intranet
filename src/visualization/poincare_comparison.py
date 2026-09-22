"""Offline, synchronized animation of saved Poincare sections."""
from pathlib import Path
import base64
import json
import numpy as np


def export_poincare_panel_comparison(path, panels, cycles_per_frame=25, *,
                                    initial_view=None, initial_cycle=1,
                                    highlight_particle=None, title=None,
                                    regions=None, highlight_particles=None):
    """Export aligned panels, optionally focusing a square region and particle.

    ``initial_view`` is (x/L, y/L, span/L); both axes share the same span.
    ``highlight_particle`` adds a selection shortcut and draws opaque returns.
    """
    if not panels:
        raise ValueError("At least one comparison panel is required.")
    encoded_panels = []
    cycle_count = None
    for panel in panels:
        xy = np.asarray(panel["coordinates"], dtype="<f4")
        ids = list(map(int, panel["particle_ids"]))
        colors = list(panel["colors"])
        if xy.ndim != 3 or xy.shape[-1] != 2:
            raise ValueError("Each panel must have coordinates shaped (cycles, particles, 2).")
        if not np.isfinite(xy).all() or np.any((xy < 0) | (xy > 1)):
            raise ValueError("Expected finite periodic coordinates in [0, 1].")
        if xy.shape[1] != len(ids) or len(colors) != len(ids):
            raise ValueError("Each panel's particle labels and colors must match its data.")
        if len(set(ids)) != len(ids):
            raise ValueError("Particle identifiers must be unique within each panel.")
        if cycle_count is None:
            cycle_count = xy.shape[0]
        elif xy.shape[0] != cycle_count:
            raise ValueError("All panels must contain the same number of aligned cycles.")
        encoded_panels.append(
            dict(
                shape=xy.shape,
                ids=ids,
                colors=colors,
                title=str(panel["title"]),
                data=base64.b64encode(xy.tobytes()).decode("ascii"),
            )
        )
    if int(cycles_per_frame) < 1:
        raise ValueError("cycles_per_frame must be positive.")
    if isinstance(initial_cycle, bool) or int(initial_cycle) != initial_cycle or not 1 <= initial_cycle <= cycle_count:
        raise ValueError('initial_cycle must be an integer within the saved record.')
    config = dict(panels=encoded_panels, step=int(cycles_per_frame), initialCycle=int(initial_cycle))
    if initial_view is not None:
        view = np.asarray(initial_view, dtype=float)
        if (view.shape != (3,) or not np.isfinite(view).all() or view[2] <= 0
                or np.any(view[:2] < 0) or np.any(view[:2] + view[2] > 1)):
            raise ValueError('initial_view must be a positive square contained in the unit cell.')
        config['initialView'] = dict(x=float(view[0]), y=float(view[1]), span=float(view[2]))
    if highlight_particle is not None:
        if highlight_particle not in {pid for panel in encoded_panels for pid in panel['ids']}:
            raise ValueError('The highlighted particle must exist in a panel.')
        config['highlightParticle'] = int(highlight_particle)
    if title is not None:
        config['title'] = str(title)
    all_ids = {pid for panel in encoded_panels for pid in panel['ids']}
    if highlight_particles is not None:
        if not set(highlight_particles).issubset(all_ids):
            raise ValueError('All highlighted particles must exist in a panel.')
        config['highlightParticles'] = list(map(int, highlight_particles))
    if regions is not None:
        if not regions:
            raise ValueError('At least one named region is required.')
        config['regions'] = []
        for region in regions:
            view = np.asarray(region['view'], dtype=float)
            if (view.shape != (3,) or not np.isfinite(view).all() or view[2] <= 0
                    or np.any(view[:2] < 0) or np.any(view[:2] + view[2] > 1)
                    or region['particle_id'] not in all_ids):
                raise ValueError('Each region requires a valid square viewport and an existing particle.')
            config['regions'].append(dict(label=str(region['label']),
                view=dict(x=float(view[0]), y=float(view[1]), span=float(view[2])),
                particle=int(region['particle_id'])))
    template = Path(__file__).with_name("_poincare_comparison.html").read_text()
    payload = json.dumps(config).replace('<', '\\u003c')
    Path(path).write_text(template.replace("__CONFIG__", payload), encoding="utf-8")
    return Path(path)


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
    panels = [
        dict(
            title=titles[index],
            coordinates=xy[index],
            particle_ids=particle_ids,
            colors=colors,
        )
        for index in range(xy.shape[0])
    ]
    return export_poincare_panel_comparison(path, panels, cycles_per_frame)
