"""Offline, synchronized animation of saved Poincare sections."""
from pathlib import Path
import base64
import json
import numpy as np


def _encode_field(field):
    """Encode a periodic Hamiltonian raster and its collocated velocity grid."""
    potential = np.asarray(field['potential'], dtype=float)
    velocity = np.asarray(field['velocity'], dtype=float)
    phases = np.asarray(field['phases'], dtype=float)
    bounds = np.asarray(field['bounds'], dtype=float)
    if (potential.ndim != 3 or potential.shape[1] != potential.shape[2]
            or velocity.ndim != 4 or velocity.shape[1] != velocity.shape[2]
            or velocity.shape[-1] != 2 or potential.shape[0] != len(phases)
            or velocity.shape[0] != len(phases) or len(phases) < 2
            or bounds.shape != (3,) or bounds[2] <= 0
            or any(not np.isfinite(a).all() for a in (potential, velocity, phases, bounds))
            or not np.all(np.diff(phases) > 0)):
        raise ValueError('Invalid potential, velocity or phase grids.')
    return dict(shape=potential.shape, vectorShape=velocity.shape, phases=phases.tolist(),
                bounds=dict(x=float(bounds[0]), y=float(bounds[1]), span=float(bounds[2])),
                potential=base64.b64encode(potential.astype('<f4').tobytes()).decode('ascii'),
                velocity=base64.b64encode(velocity.astype('<f4').tobytes()).decode('ascii'),
                minimum=float(potential.min()), maximum=float(potential.max()),
                maxSpeed=float(np.linalg.norm(velocity, axis=-1).max()),
                units=str(field['units']))


def _encode_panels(panels, bounds):
    """Validate aligned samples and encode browser copies in their original units."""
    if not panels:
        raise ValueError("At least one comparison panel is required.")
    lower, upper = bounds[:2], bounds[:2] + bounds[2]
    encoded_panels = []
    cycle_count = None
    for panel in panels:
        xy = np.asarray(panel["coordinates"], dtype=float)
        static = bool(panel.get("static", False))
        ids = list(map(int, panel["particle_ids"]))
        colors = list(panel["colors"])
        if xy.ndim != 3 or xy.shape[-1] != 2:
            raise ValueError("Each panel must have coordinates shaped (cycles, particles, 2).")
        if not np.isfinite(xy).all() or np.any((xy < lower) | (xy > upper)):
            raise ValueError("Expected finite coordinates within coordinate_bounds.")
        if xy.shape[0] < 1 or xy.shape[1] < 1 or (static and xy.shape[0] != 1):
            raise ValueError('Panels require particles and saved samples; static panels require exactly one sample.')
        if xy.shape[1] != len(ids) or len(colors) != len(ids):
            raise ValueError("Each panel's particle labels and colors must match its data.")
        if len(set(ids)) != len(ids):
            raise ValueError("Particle identifiers must be unique within each panel.")
        if not static:
            if cycle_count is None:
                cycle_count = xy.shape[0]
            elif xy.shape[0] != cycle_count:
                raise ValueError("All animated panels must contain the same number of aligned cycles.")
        encoded_panels.append(
            dict(
                shape=xy.shape,
                ids=ids,
                colors=colors,
                title=str(panel["title"]),
                static=static,
                data=base64.b64encode(xy.astype('<f4').tobytes()).decode("ascii"),
            )
        )
        if 'field' in panel:
            encoded_panels[-1]['field'] = _encode_field(panel['field'])
    cycle_count = 1 if cycle_count is None else cycle_count
    return encoded_panels, cycle_count


def export_poincare_panel_comparison(path, panels, cycles_per_frame=25, *,
                                    initial_view=None, initial_cycle=1,
                                    highlight_particle=None, title=None,
                                    regions=None, highlight_particles=None,
                                    coordinate_bounds=None, axis_labels=None,
                                    description=None, particle_groups=None,
                                    datasets=None, dataset_label="Dataset",
                                    selected_dataset=None, particle_labels=None,
                                    first_cycle=1):
    """Export aligned panels with optional selection between completed datasets.

    Coordinates default to the unit cell. ``coordinate_bounds=(x0,y0,span)``
    accepts physical coordinates without changing their units. Static panels
    contain one fixed sample. Datasets share panel shapes, IDs and colors;
    a dataset without panels is shown as unavailable in the selector.
    ``first_cycle=0`` includes the original initial state. Hollow circles mark
    the selected interval's first sample. A static panel may carry a ``field``
    with phase-dependent potential and velocity grids instead of particles.
    """
    bounds = np.asarray((0., 0., 1.) if coordinate_bounds is None else coordinate_bounds, dtype=float)
    if bounds.shape != (3,) or not np.isfinite(bounds).all() or bounds[2] <= 0:
        raise ValueError('coordinate_bounds must describe a finite positive square.')
    lower, upper = bounds[:2], bounds[:2] + bounds[2]
    encoded_panels, cycle_count = _encode_panels(panels, bounds)
    if isinstance(cycles_per_frame, bool) or int(cycles_per_frame) != cycles_per_frame or int(cycles_per_frame) < 1:
        raise ValueError("cycles_per_frame must be positive.")
    if first_cycle not in (0, 1):
        raise ValueError('first_cycle must be zero or one.')
    if isinstance(initial_cycle, bool) or int(initial_cycle) != initial_cycle or not first_cycle <= initial_cycle < first_cycle + cycle_count:
        raise ValueError('initial_cycle must be an integer within the saved record.')
    config = dict(panels=encoded_panels, step=int(cycles_per_frame), initialCycle=int(initial_cycle),
                  firstCycle=int(first_cycle))
    config['coordinateBounds'] = dict(x=float(bounds[0]), y=float(bounds[1]), span=float(bounds[2]))
    if datasets is not None:
        encoded_datasets = []
        seen = set()
        for dataset in datasets:
            key = str(dataset['key'])
            if key in seen:
                raise ValueError('Dataset keys must be unique.')
            seen.add(key)
            item = dict(key=key, label=str(dataset['label']), note=str(dataset.get('note', '')))
            if dataset.get('panels') is not None:
                encoded, count = _encode_panels(dataset['panels'], bounds)
                if count != cycle_count or len(encoded) != len(encoded_panels) or any(
                    any(a[k] != b[k] for k in ('shape', 'ids', 'colors', 'static'))
                    for a, b in zip(encoded, encoded_panels)
                ):
                    raise ValueError('Datasets must share cycle counts, panel shapes, IDs and colors.')
                item['panels'] = encoded
            encoded_datasets.append(item)
        selected = str(selected_dataset)
        if not any(d['key'] == selected and 'panels' in d for d in encoded_datasets):
            raise ValueError('The selected dataset must contain completed panels.')
        config.update(datasets=encoded_datasets, datasetLabel=str(dataset_label), selectedDataset=selected)
    if particle_labels is not None:
        config['particleLabels'] = {str(key): str(value) for key, value in particle_labels.items()}
    if axis_labels is not None:
        if len(axis_labels) != 2:
            raise ValueError('axis_labels must contain an x and a y label.')
        config['axisLabels'] = list(map(str, axis_labels))
    if description is not None:
        config['description'] = str(description)
    if initial_view is not None:
        view = np.asarray(initial_view, dtype=float)
        if (view.shape != (3,) or not np.isfinite(view).all() or view[2] <= 0
                or np.any(view[:2] < lower) or np.any(view[:2] + view[2] > upper)):
            raise ValueError('initial_view must be a positive square contained in coordinate_bounds.')
        config['initialView'] = dict(x=float(view[0]), y=float(view[1]), span=float(view[2]))
    if highlight_particle is not None:
        if highlight_particle not in {pid for panel in encoded_panels for pid in panel['ids']}:
            raise ValueError('The highlighted particle must exist in a panel.')
        config['highlightParticle'] = int(highlight_particle)
    if title is not None:
        config['title'] = str(title)
    all_ids = {pid for panel in encoded_panels for pid in panel['ids']}
    if particle_groups is not None:
        config['particleGroups'] = []
        for group in particle_groups:
            ids = list(map(int, group['particle_ids']))
            if not ids or not set(ids).issubset(all_ids):
                raise ValueError('Particle groups must contain existing particle identifiers.')
            config['particleGroups'].append(dict(label=str(group['label']), ids=ids))
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
                    or np.any(view[:2] < lower) or np.any(view[:2] + view[2] > upper)
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
