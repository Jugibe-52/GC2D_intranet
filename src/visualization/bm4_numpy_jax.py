"""Thesis figures from saved paired backend timings and trajectories."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from diagnostics.io.persistence import load_solution
from diagnostics.io.storage import ArtifactStore


_METHOD_STYLES: dict[str, tuple[str, str]] = {
	'BM4Midpoint': ('BM4, moyenne', '#777777'),
	'BM4Implicit': ('BM4 implicite', '#9260a8'),
	'GaussLegendre4': ('Gauss–Legendre', '#d16b24'),
	'RK2': ('RK2', '#21856d'),
}


def summarize(directory: Path, *, bucket: str | None = None,
		particle_groups: dict[str, Path] | None = None) -> dict[str, Any]:
	"""Verify saved solutions and recompute timings and unwrapped discrepancies."""
	directory.mkdir(parents=True, exist_ok=True)
	path: Path = directory / 'study.json'
	if not path.exists():
		if bucket is None:
			raise FileNotFoundError('Supply the saved study directory or its bucket location.')
		ArtifactStore(bucket).get('study.json', path)
	record: dict[str, Any] = json.loads(path.read_text())
	particle_count: int = record.get('particle_count', 40)
	selections: dict[str, dict[str, Any]] = {}
	group_records: dict[str, dict[str, Any]] = {}
	for group, source in (particle_groups or {}).items():
		selection: dict[str, Any] = json.loads(source.read_text())
		requested: list[int] = [p['id'] for p in selection['particles']]
		if not requested or len(set(requested)) != len(requested):
			raise ValueError('Particle selections must contain distinct particle IDs.')
		selections[group] = selection
		group_records[group] = dict(dataset_id=selection['dataset_id'],
			source_file=source.name, source_sha256=sha256(source.read_bytes()).hexdigest(),
			requested_particle_ids=requested, status=selection['status'],
			verified_for_reflected_potential=selection['selection']['verified_for_reflected_potential'])
	rows: list[dict[str, Any]] = []
	for count in record['cycles']:
		for name in record['methods']:
			key: str = f'{name}-{count}'
			loaded: dict[str, Any] = {}
			for backend in ('scipy', 'jax'):
				local: Path = directory / 'runs' / key / backend
				location: str | Path = local if (local / 'manifest.json').exists() else record['bucket'] + '/runs/' + key + '/' + backend
				loaded[backend] = load_solution(location)
				data: Any = loaded[backend]
				if (data.metadata['code_sha256'] != record['code_sha256']
						or data.metadata['rho'] != record['rho'] or data.metadata['method'] != name
						or data.metadata['config']['cycles'] != count or data.metadata['backend'] != backend
						or data.metadata['particle_count'] != particle_count
						or data.solution.states.shape != (10 * count + 1, 2 * particle_count)
						or data.solution.diagnostics['step_count'] != 40 * count):
					raise ValueError('Archive inputs or dimensions do not match the selected study.')
			left: Any = loaded['scipy']
			right: Any = loaded['jax']
			np.testing.assert_array_equal(left.solution.t, right.solution.t)
			np.testing.assert_array_equal(left.solution.states[0], right.solution.states[0])
			delta: np.ndarray = left.solution.states - right.solution.states
			distance: np.ndarray = np.hypot(delta[:, :particle_count], delta[:, particle_count:])
			particles: list[dict[str, Any]] = left.metadata['particle_record']['particles']
			ids: list[int] = [p['id'] for p in particles]
			if (particles != right.metadata['particle_record']['particles']
					or len(ids) != particle_count or len(set(ids)) != particle_count
					or ids != record.get('particle_ids', ids)):
				raise ValueError('Paired archives must retain the selected distinct particle records.')
			group_distances: dict[str, dict[str, Any]] = {}
			for group, selection in selections.items():
				available: list[int] = [p['id'] for p in selection['particles'] if p['id'] in ids]
				if not available:
					raise ValueError(f'No saved trajectory for particle group {group}.')
				if group_records[group].get('available_particle_ids', available) != available:
					raise ValueError('Particle group membership changes between saved cases.')
				group_records[group]['available_particle_ids'] = available
				group_records[group]['missing_particle_ids'] = [p['id'] for p in selection['particles'] if p['id'] not in ids]
				if record.get('particle_selection', {}).get('operation') == 'union':
					frozen: dict[str, Any] = next(s for s in record['particle_selection']['source_sets']
						if s['dataset_id'] == selection['dataset_id'])
					if (group_records[group]['missing_particle_ids']
							or frozen['sha256'] != group_records[group]['source_sha256']):
						raise ValueError('The saved union must contain the complete, unchanged candidate sets.')
				# Resolve columns from persisted IDs, and verify that their physical inputs match.
				for particle in selection['particles']:
					if particle['id'] in ids:
						np.testing.assert_array_equal(particle['position_m'], particles[ids.index(particle['id'])]['position_m'])
				indices: list[int] = [ids.index(particle_id) for particle_id in available]
				selected_distance: np.ndarray = distance[:, indices]
				group_distances[group] = dict(maximum_distance=float(selected_distance.max()),
					per_particle_maximum_distance=selected_distance.max(axis=0).tolist())
			pair: dict[str, Any] = left.metadata['pair']
			if pair != right.metadata['pair']:
				raise ValueError('Backends do not belong to the same paired measurement.')
			row: dict[str, Any] = dict(method=name, cycles=count,
				maximum_distance=float(distance.max()),
				group_distances=group_distances,
				rms_distance=float(np.sqrt(np.mean(distance**2))),
				final_maximum_distance=float(distance[-1].max()),
				maximum_distance_first_10_cycles=float(distance[:101].max()),
				environment=pair['environment'], call_id=left.metadata['call_id'], backends={})
			for backend in ('scipy', 'jax'):
				measurements: list[dict[str, Any]] = [m for m in pair['measurements'] if m['backend'] == backend]
				warm: list[float] = [m['seconds'] for m in measurements if m['phase'] == 'warm']
				first: list[float] = [m['seconds'] for m in measurements if m['phase'] == 'first']
				if len(warm) != record['repetitions'] or len(first) != 1 or min(warm + first) <= 0:
					raise ValueError('Missing or invalid timing repetitions.')
				row['backends'][backend] = dict(first_seconds=first[0], warm_seconds=warm,
					median_seconds=float(np.median(warm)), minimum_seconds=min(warm), maximum_seconds=max(warm))
			row['speedup'] = row['backends']['scipy']['median_seconds'] / row['backends']['jax']['median_seconds']
			rows.append(row)
	return dict(study=record, rows=rows, particle_groups=group_records,
		agreement_convention='Unwrapped Euclidean distance in base-cell units; agreement is not accuracy.')


def plot_comparison(summary: dict[str, Any]) -> Any:
	"""Show all selected methods and both libraries on one logarithmic chart."""
	styles: dict[str, tuple[str, str]] = _METHOD_STYLES
	methods: list[str] = summary['study']['methods']
	cycles: list[int] = summary['study']['cycles']
	# Reject incomplete or nonpositive series before creating a log-scale figure.
	for method in methods:
		if method not in styles:
			raise ValueError(f'No plot style for method {method}.')
		selected: list[dict[str, Any]] = [r for r in summary['rows'] if r['method'] == method]
		if sorted(r['cycles'] for r in selected) != sorted(cycles):
			raise ValueError('The figure requires every selected duration exactly once.')
		for row in selected:
			for backend in ('scipy', 'jax'):
				v: dict[str, Any] = row['backends'][backend]
				if not (0 < v['minimum_seconds'] <= v['median_seconds'] <= v['maximum_seconds']
						and np.isfinite([v['minimum_seconds'], v['median_seconds'], v['maximum_seconds'], v['first_seconds']]).all()
						and v['first_seconds'] > 0):
					raise ValueError('Figure timings must be finite, positive and ordered.')
	fig: Any
	ax: Any
	fig, ax = plt.subplots(figsize=(7.6, 5.6), layout='constrained')
	for method in methods:
		label: str
		color: str
		label, color = styles[method]
		rows: list[dict[str, Any]] = sorted((row for row in summary['rows'] if row['method'] == method), key=lambda r: r['cycles'])
		x: np.ndarray = np.asarray([row['cycles'] for row in rows])
		for backend, linestyle, marker, name in (('scipy', '-', 'o', 'NumPy/SciPy'), ('jax', '--', 's', 'JAX')):
			values: list[dict[str, Any]] = [row['backends'][backend] for row in rows]
			median: np.ndarray = np.asarray([v['median_seconds'] for v in values])
			error: np.ndarray = np.asarray([[v['median_seconds']-v['minimum_seconds'] for v in values],
				[v['maximum_seconds']-v['median_seconds'] for v in values]])
			ax.errorbar(x, median, yerr=error, color=color, linestyle=linestyle, marker=marker,
				linewidth=1.6, markersize=5, capsize=4, label=f'{label} · {name}')
			ax.scatter(x, [v['first_seconds'] for v in values], marker=marker, s=52,
				facecolors='none', edgecolors=color, linewidths=1.1, zorder=4)
	ax.set(xscale='log', yscale='log', xlabel='Durée intégrée (cycles)', ylabel='Temps de calcul (s)')
	ax.set_xticks(cycles, [str(c) for c in cycles])
	ax.minorticks_off()
	ax.grid(True, which='major', alpha=.2)
	ax.spines[['top', 'right']].set_visible(False)
	handles: Any
	legend_labels: Any
	handles, legend_labels = ax.get_legend_handles_labels()
	legend_order: list[int] = list(range(0, len(handles), 2)) + list(range(1, len(handles), 2))
	fig.legend([handles[i] for i in legend_order], [legend_labels[i] for i in legend_order], loc='upper center',
		bbox_to_anchor=(.54, .935), ncol=2, fontsize=9, frameon=False)
	particle_count: int = summary['study'].get('particle_count', 40)
	fig.suptitle(str(particle_count) + r' particules · $\widetilde{\rho}=0{,}015$ · CPU (2 cœurs)', fontsize=11, y=.98)
	fig.get_layout_engine().set(rect=(0, .07, 1, .74))
	fig.text(.5, .018, 'Symboles pleins : médiane de 3 répétitions · Barres : min–max\nSymboles vides : premier appel',
		ha='center', fontsize=8.5)
	return fig


def plot_agreement(summary: dict[str, Any]) -> Any:
	"""Compare group maxima on common axes; absent groups retain the all-particle view."""
	methods: list[str] = summary['study']['methods']
	cycles: list[int] = summary['study']['cycles']
	groups: dict[str, dict[str, Any]] = summary.get('particle_groups', {})
	group_labels: dict[str, str] = {'chaotic': 'Candidates chaotiques', 'periodic': 'Candidates périodiques'}
	if any(group not in group_labels for group in groups):
		raise ValueError('No French panel label for the selected particle group.')
	if not methods or not cycles or len(set(methods)) != len(methods) or len(set(cycles)) != len(cycles):
		raise ValueError('Select distinct methods and positive integration durations.')
	if not np.isfinite(cycles).all() or min(cycles) <= 0:
		raise ValueError('Logarithmic durations must be finite and positive.')
	series: dict[str, list[dict[str, Any]]] = {}
	# Validate every series before allocating a figure; never replace zero by epsilon.
	for method in methods:
		if method not in _METHOD_STYLES:
			raise ValueError(f'No plot style for method {method}.')
		rows: list[dict[str, Any]] = sorted(
			(row for row in summary['rows'] if row['method'] == method), key=lambda row: row['cycles'])
		if [row['cycles'] for row in rows] != sorted(cycles):
			raise ValueError('The figure requires every selected duration exactly once.')
		values: np.ndarray = np.asarray([
			row['group_distances'][group]['maximum_distance'] if group else row['maximum_distance']
			for group in (list(groups) or ['']) for row in rows])
		if not np.isfinite(values).all() or np.any(values <= 0):
			raise ValueError('Logarithmic distances must be finite and strictly positive.')
		series[method] = rows
	fig: Any
	axes: Any
	fig, axes = plt.subplots(1, max(1, len(groups)), figsize=(7.6, 4.8), sharey=True,
		squeeze=False, layout='constrained')
	markers: dict[str, str] = {'BM4Midpoint': 'D', 'BM4Implicit': 'o', 'GaussLegendre4': 's', 'RK2': '^'}
	for ax, group in zip(axes[0], list(groups) or ['']):
		for method in methods:
			label: str
			color: str
			label, color = _METHOD_STYLES[method]
			ax.plot([row['cycles'] for row in series[method]],
				[row['group_distances'][group]['maximum_distance'] if group else row['maximum_distance']
					for row in series[method]],
				color=color, marker=markers[method], markersize=6, linewidth=1.5, label=label)
		ax.set(xscale='log', yscale='log', xlabel='Durée intégrée (cycles)')
		ax.set_xticks(sorted(cycles), [str(c) for c in sorted(cycles)])
		ax.minorticks_off()
		ax.grid(True, which='major', alpha=.2)
		ax.spines[['top', 'right']].set_visible(False)
		if group:
			available: list[int] = groups[group]['available_particle_ids']
			ids_label: str = ', '.join(str(i) for i in available)
			ax.set_title(f'{group_labels[group]} ({len(available)})\nID : {ids_label}', fontsize=9.5)
	axes[0, 0].set_ylabel('Distance maximale (unités de cellule)')
	fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center',
		bbox_to_anchor=(.52, .92), ncol=len(methods), fontsize=9, frameon=False)
	rho: str = f"{summary['study']['rho']:g}".replace('.', '{,}')
	fig.suptitle('Écart NumPy/SciPy–JAX · '
		+ rf'$\widetilde{{\rho}}={rho}$', fontsize=11)
	fig.get_layout_engine().set(rect=(0, .13, 1, .75))
	footnote: str = ('Chaque point : maximum du groupe sur tous les instants enregistrés\n'
		'Lignes : guides entre des calculs distincts · Classification provisoire' if groups else
		f"Chaque point : maximum sur les {summary['study'].get('particle_count', 40)} particules et tous les instants enregistrés\n"
		'Lignes : guides visuels entre des calculs distincts')
	fig.text(.5, .025, footnote, ha='center', fontsize=8.5)
	return fig


def _export_figure(fig: Any, summary: dict[str, Any], destination: Path) -> Any:
	"""Write stable PDF/PNG exports and their measured provenance."""
	destination.parent.mkdir(parents=True, exist_ok=True)
	for suffix in ('.pdf', '.png'):
		# Stable metadata lets unchanged plots reuse immutable derived archives.
		metadata: dict[str, Any] | None = {'CreationDate': None, 'ModDate': None} if suffix == '.pdf' else None
		fig.savefig(destination.with_suffix(suffix), dpi=220, metadata=metadata)
	record: dict[str, Any] = {**summary, 'figure_sha256': {
		suffix: sha256(destination.with_suffix(suffix).read_bytes()).hexdigest() for suffix in ('.pdf', '.png')}}
	destination.with_suffix('.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
	return fig


def export_figure(summary: dict[str, Any], destination: Path) -> Any:
	"""Export the runtime figure as PDF/PNG plus measured provenance."""
	return _export_figure(plot_comparison(summary), summary, destination)


def export_agreement_figure(summary: dict[str, Any], destination: Path) -> Any:
	"""Export maximum backend discrepancy, preserving its aggregation convention."""
	record: dict[str, Any] = {**summary, 'figure_metric': {
		'name': 'maximum_distance',
		'aggregation': ('Maximum over available particles within each candidate group and all saved times in each independent case.'
			if summary.get('particle_groups') else 'Maximum over all selected particles and all saved times within each independent case.'),
		'distance': 'Euclidean distance between paired NumPy/SciPy and JAX positions, without periodic wrapping.',
		'units': 'base-cell side length',
		'x_axis': 'Total integration duration in cycles; distinct case for each duration.',
	}}
	return _export_figure(plot_agreement(summary), record, destination)


__all__ = ['summarize', 'plot_comparison', 'export_figure', 'plot_agreement', 'export_agreement_figure']
