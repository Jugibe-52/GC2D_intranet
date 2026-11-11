"""Attach full-integration horizontal transport to synchronized rho viewers."""

from collections.abc import Mapping
from copy import deepcopy
import base64
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np


_METHOD_LABELS: dict[str, str] = {
	'BM4Midpoint': 'BM4-M', 'BM4Implicit': 'BM4-I',
	'RK4': 'RK4', 'GaussLegendre4': 'GL4',
}


def _case_key(rho: Any, method: Any) -> tuple[float, str]:
	"""Normalize JSON rho values without relying on case order."""
	value: float = float(rho)
	if not np.isfinite(value) or value < 0 or str(method) not in _METHOD_LABELS:
		raise ValueError('Transport cases require a finite rho and a supported method.')
	return round(value, 12), str(method)


def _case_values(case: Mapping[str, Any], ids: list[int], cycles: int) -> dict[str, Any]:
	"""Validate particle identities, finite lengths and the reported normalization."""
	particle_ids: list[Any] = list(case['particle_ids'])
	if (len(particle_ids) != len(set(particle_ids)) or set(particle_ids) != set(ids)
			or any(isinstance(pid, bool) or not isinstance(pid, int) for pid in particle_ids)):
		raise ValueError('Transport particle IDs must match every displayed particle exactly.')
	if (isinstance(case['cycles'], bool) or case['cycles'] != cycles
			or isinstance(case['steps_per_cycle'], bool)
			or int(case['steps_per_cycle']) != case['steps_per_cycle']
			or case['steps_per_cycle'] < 1):
		raise ValueError('Transport cycles must match the full displayed trajectory interval.')
	values: dict[str, np.ndarray] = {}
	for name in ('travel_x', 'width_x', 'x_min', 'x_max'):
		array: np.ndarray = np.asarray(case[name], dtype=float)
		if array.shape != (len(ids),) or not np.isfinite(array).all():
			raise ValueError(f'Transport {name} must contain one finite value per particle.')
		values[name] = array
	if (np.any(values['travel_x'] < 0) or np.any(values['width_x'] < 0)
			or not np.allclose(values['width_x'], values['x_max'] - values['x_min'],
							 rtol=2e-12, atol=1e-14)):
		raise ValueError('Transport lengths must be nonnegative and agree with x extrema.')
	q_values: list[Any] = list(case['q_x'])
	if len(q_values) != len(ids):
		raise ValueError('Transport q_x must contain one value per particle.')
	particles: dict[str, Any] = {}
	for index, pid in enumerate(particle_ids):
		width: float = float(values['width_x'][index])
		q: Any = q_values[index]
		if width == 0:
			if q is not None:
				raise ValueError('Transport q_x must be null for a zero-width trajectory.')
		elif (q is None or isinstance(q, bool) or not np.isfinite(float(q))
				or not np.isclose(float(q), values['travel_x'][index] / (cycles * width),
								 rtol=2e-12, atol=1e-14)):
			raise ValueError('Transport q_x must equal travel_x / (cycles * width_x).')
		particles[str(pid)] = dict(q=None if q is None else float(q),
			travel=float(values['travel_x'][index]), width=width,
			xMin=float(values['x_min'][index]), xMax=float(values['x_max'][index]))
	return dict(method=str(case['method']), label=_METHOD_LABELS[str(case['method'])],
		particles=particles, stepsPerCycle=int(case['steps_per_cycle']))


def attach_horizontal_transport(
	config: Mapping[str, Any], results: Mapping[str, Any],
) -> dict[str, Any]:
	"""Return a viewer copy with complete, identity-checked per-dataset metrics.

	The metric uses physical unwrapped x over the entire saved interval. Null
	values denote zero width, never an omitted case. Dataset labels are numeric
	rho values; each nonstatic panel's title is its canonical method name.
	"""
	if results.get('schema_version') != 1:
		raise ValueError('Unsupported horizontal transport results schema.')
	metric: Mapping[str, Any] = results.get('metric', {})
	if (metric.get('coordinate') != 'physical_unwrapped_x'
			or metric.get('integral') != 'accepted_endpoint_trapezoid_abs_vx'):
		raise ValueError('Transport must use absolute physical x velocity at accepted step endpoints.')
	first_cycle: int = int(config.get('firstCycle', 1))
	if first_cycle != 0 or not config.get('datasets'):
		raise ValueError('Transport requires rho datasets with the initial state at cycle zero.')
	cases: dict[tuple[float, str], Mapping[str, Any]] = {}
	for case in results.get('cases', []):
		key: tuple[float, str] = _case_key(case['rho'], case['method'])
		if key in cases:
			raise ValueError('Duplicate rho-method transport case.')
		cases[key] = case
	output: dict[str, Any] = deepcopy(dict(config))
	used: set[tuple[float, str]] = set()
	for dataset in output['datasets']:
		if 'panels' not in dataset:
			continue
		methods: list[dict[str, Any]] = []
		cycles: int | None = None
		for panel in dataset['panels']:
			if panel.get('static'):
				continue
			key = _case_key(dataset['key'], panel['title'])
			if key not in cases:
				raise ValueError(f'Missing horizontal transport for rho={key[0]}, {key[1]}.')
			count: int = int(panel['shape'][0]) - 1
			if count <= 0 or (cycles is not None and cycles != count):
				raise ValueError('Transport panels must share a positive complete cycle interval.')
			cycles = count
			methods.append(_case_values(cases[key], panel['ids'], cycles))
			used.add(key)
		if {method['method'] for method in methods} != set(_METHOD_LABELS):
			raise ValueError('Horizontal transport requires all four method panels per rho.')
		dataset['horizontalTransport'] = dict(cycles=cycles, methods=methods,
			units=str(metric.get('units', 'm')))
	if set(cases) != used:
		raise ValueError('Transport cases must match the available viewer rho-method grid exactly.')
	output['horizontalTransportMetric'] = dict(metric)
	return output


def export_horizontal_transport_viewer(
	source_html: str | Path, results: Mapping[str, Any], output_html: str | Path, *,
	cycle_positions: Mapping[tuple[float, str], np.ndarray],
) -> Path:
	"""Reuse verified field backgrounds and replace every trajectory with its rerun.

	``cycle_positions`` contains folded cell fractions with shape (cycles+1,
	particles, 2), ordered by the baseline viewer's particle IDs. The source HTML
	fingerprint in metric metadata is checked before any output is written.
	"""
	source_bytes: bytes = Path(source_html).read_bytes()
	if hashlib.sha256(source_bytes).hexdigest() != results.get('metric', {}).get('source_html_sha256'):
		raise ValueError('The baseline viewer does not match the recorded source HTML fingerprint.')
	match: re.Match[str] | None = re.search(r'^const cfg=(.*);$', source_bytes.decode('utf-8'), re.MULTILINE)
	if match is None:
		raise ValueError('The baseline HTML must contain one self-contained viewer configuration.')
	config: dict[str, Any] = attach_horizontal_transport(json.loads(match.group(1)), results)
	positions: dict[tuple[float, str], np.ndarray] = {
		_case_key(rho, method): np.asarray(xy, dtype=float)
		for (rho, method), xy in cycle_positions.items()
	}
	used: set[tuple[float, str]] = set()
	for dataset in config['datasets']:
		for panel in dataset.get('panels', []):
			if panel.get('static'):
				continue
			key: tuple[float, str] = _case_key(dataset['key'], panel['title'])
			if key not in positions:
				raise ValueError(f'Missing rerun trajectory for rho={key[0]}, {key[1]}.')
			xy: np.ndarray = positions[key]
			if (xy.shape != tuple(panel['shape']) or not np.isfinite(xy).all()
					or np.any((xy < 0) | (xy > 1))):
				raise ValueError('Rerun positions must match panel shapes and be finite cell fractions.')
			old_initial: np.ndarray = np.frombuffer(base64.b64decode(panel['data']), dtype='<f4').reshape(panel['shape'])[0]
			if not np.allclose(xy[0], old_initial, rtol=0, atol=6e-8):
				raise ValueError('Rerun initial positions differ from the verified baseline experiment.')
			panel['data'] = base64.b64encode(xy.astype('<f4').tobytes()).decode('ascii')
			used.add(key)
	if set(positions) != used:
		raise ValueError('Rerun trajectories must match the complete viewer rho-method grid.')
	selected: dict[str, Any] = next(item for item in config['datasets']
		if item['key'] == config['selectedDataset'])
	config['panels'] = selected['panels']
	config['description'] = str(config.get('description', '')) + (
		' Horizontal transport Qx is calculated from the repeated full integration, '
		'using physical unwrapped x and all accepted steps. The displayed returns '
		'come from that same repetition. Particle values cover the entire experiment '
		'and do not change with the displayed cycle interval.')
	template: str = Path(__file__).with_name('_poincare_comparison.html').read_text()
	payload: str = json.dumps(config, allow_nan=False).replace('<', '\\u003c')
	path: Path = Path(output_html)
	path.write_text(template.replace('__CONFIG__', payload), encoding='utf-8')
	return path


__all__ = ['attach_horizontal_transport', 'export_horizontal_transport_viewer']
