"""French-labelled runtime figures from verified section 4.3 archives."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
import numpy as np

from diagnostics.io.persistence import load_solution
from studies.method_timings import METHODS

LABELS = ('RK2', 'Point milieu implicite', 'RK4', 'Gauss–Legendre',
	'ABBA2 · moyenne', 'ABBA2 · symétrique', 'BM4 · moyenne', 'BM4 · symétrique')
COLORS = ('#21856d', '#66a58b', '#3568aa', '#d16b24', '#be8b2c', '#d35d64', '#777777', '#9260a8')


def summarize(directory: Path) -> dict[str, Any]:
	"""Validate archive completeness and derive only runtime statistics."""
	record: dict[str, Any] = json.loads((directory / 'study.json').read_text())
	timings: dict[str, Any] = json.loads((directory / 'timings.json').read_text())
	if (record['methods'] != list(METHODS) or not timings['same_container']
			or len(timings['measurements']) != len(METHODS) * (record['repetitions'] + 1)):
		raise ValueError('Incomplete or incompatible timing protocol.')
	rows: list[dict[str, Any]] = []
	for name in METHODS:
		loaded: Any = load_solution(directory / 'runs' / name)
		if (loaded.solution.states.shape != (record['cycles'] * record['samples_per_cycle'] + 1, 80)
				or loaded.metadata['method'] != name
				or loaded.metadata['call_id'] != timings['call_id']
				or loaded.metadata['code_sha256'] != record['code_sha256']):
			raise ValueError('Saved solution or provenance differs from the timing study.')
		values: list[dict[str, Any]] = [m for m in timings['measurements'] if m['method'] == name]
		if sorted(m['repetition'] for m in values) != list(range(record['repetitions'] + 1)):
			raise ValueError('Missing or duplicated timing repetition.')
		for measurement in values:
			first: bool = measurement['repetition'] == 0
			if (measurement['phase'] != ('first' if first else 'warm')
					or measurement['integration_cache_misses'] != int(first)
					or measurement['integration_cache_hits'] != int(not first)
					or measurement['step_count'] != record['cycles'] * record['steps_per_cycle']
					or not np.isfinite(measurement['seconds']) or measurement['seconds'] <= 0):
				raise ValueError('Invalid elapsed time, compilation state or step count.')
		warm: list[float] = [m['seconds'] for m in values if m['repetition'] > 0]
		rows.append(dict(method=name, first_seconds=next(m['seconds'] for m in values if m['repetition'] == 0),
			warm_seconds=warm, median_seconds=float(np.median(warm)),
			minimum_seconds=min(warm), maximum_seconds=max(warm)))
	fastest: dict[str, Any] = min(rows, key=lambda r: r['median_seconds'])
	for row in rows:
		row['relative_time'] = row['median_seconds'] / fastest['median_seconds']
	return dict(study=record, rows=rows, environment=timings['environment'],
		call_id=timings['call_id'], fastest_method=fastest['method'],
		timings_sha256=sha256((directory / 'timings.json').read_bytes()).hexdigest(),
		repeated_states_identical=timings['repeated_states_identical'])


def export_figures(summary: dict[str, Any], destination: Path) -> list[Path]:
	"""Export one chart with absolute and relative time axes, without integration."""
	rows: list[dict[str, Any]] = summary['rows']
	if [r['method'] for r in rows] != list(METHODS):
		raise ValueError('The figure requires all eight methods in the agreed order.')
	for row in rows:
		if (not np.isfinite([row['first_seconds'], *row['warm_seconds']]).all()
				or not 0 < row['minimum_seconds'] <= row['median_seconds'] <= row['maximum_seconds']):
			raise ValueError('Expected positive, finite, ordered times.')
	destination.mkdir(parents=True, exist_ok=True)
	y: np.ndarray = np.arange(len(rows))
	median: np.ndarray = np.array([r['median_seconds'] for r in rows])
	low: np.ndarray = np.array([r['minimum_seconds'] for r in rows])
	high: np.ndarray = np.array([r['maximum_seconds'] for r in rows])
	paths: list[Path] = []
	stem: str = 'comparacion_tiempos_metodos_fr'
	fig: Any
	ax: Any
	fig, ax = plt.subplots(figsize=(8.2, 4.9), layout='constrained')
	factor: float = float(median.min())
	baseline: str = LABELS[list(METHODS).index(summary['fastest_method'])]
	for i, (row, color) in enumerate(zip(rows, COLORS)):
		ax.scatter(row['warm_seconds'], [i]*len(row['warm_seconds']), color=color, alpha=.4, s=18)
		ax.errorbar(median[i], i, xerr=[[float(median[i]-low[i])], [float(high[i]-median[i])]],
			fmt='o', color=color, capsize=5, markersize=7)
		ax.scatter(row['first_seconds'], i, marker='D', facecolors='none', edgecolors=color, s=62, linewidths=1.5)
	ax.set_yticks(y, LABELS)
	ax.invert_yaxis()
	ax.grid(True, axis='x', alpha=.2)
	ax.set_axisbelow(True)
	ax.spines[['top', 'right']].set_visible(False)
	fig.suptitle(f"JAX sur CPU · {summary['study']['cycles']} cycles · {summary['study']['particle_count']} particules", fontsize=12)
	ax.set(xscale='log', xlabel='Temps de calcul (s)')
	ax.set_xlim(float(low.min()) * .8, max(r['first_seconds'] for r in rows) * 1.25)
	ax.set_xticks([2, 5, 10, 20, 50, 100], ['2', '5', '10', '20', '50', '100'])
	ax.xaxis.set_minor_formatter(NullFormatter())
	# Both axes use the same measurements and one fixed cached-median denominator.
	relative_axis: Any = ax.secondary_xaxis('top', functions=(lambda seconds: seconds / factor, lambda ratio: ratio * factor))
	relative_axis.set_xlabel(f'Temps / médiane de {baseline}')
	relative_axis.set_xticks([1, 2, 5, 10, 20, 50], ['1×', '2×', '5×', '10×', '20×', '50×'])
	relative_axis.xaxis.set_minor_formatter(NullFormatter())
	ax.axvline(factor, color='#777777', lw=.8, linestyle='--', zorder=0)
	ax.legend(handles=[Line2D([], [], marker='o', color='#444444', linestyle='none', label='Médiane après compilation'),
		Line2D([], [], marker='D', color='#444444', markerfacecolor='none', linestyle='none', label='Première exécution')],
		loc='upper center', bbox_to_anchor=(.5, -.17), ncol=2, frameon=False, fontsize=10)
	figure_hashes: dict[str, str] = {}
	for suffix in ('.pdf', '.png'):
		path: Path = destination / (stem + suffix)
		fig.savefig(path, dpi=180, metadata={'Creator': 'GC2D method timing study'} if suffix == '.pdf' else None)
		figure_hashes[suffix] = sha256(path.read_bytes()).hexdigest()
		paths.append(path)
	plt.close(fig)
	path = destination / (stem + '.json')
	path.write_text(json.dumps({**summary, 'figure_sha256': figure_hashes,
		'generator_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
		'layout': 'One plot: seconds below, time divided by fastest cached median above; both axes logarithmic',
		'interval': 'Minimum to maximum of three cached repetitions; not a confidence interval',
		'relative_denominator': 'Fastest cached median, held fixed for all measurements and interval endpoints',
		'relative_denominator_seconds': factor}, indent=2, allow_nan=False) + '\n')
	paths.append(path)
	return paths


__all__ = ['summarize', 'export_figures']
