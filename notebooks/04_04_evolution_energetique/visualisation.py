# %% [markdown]
# # 4.4 — Energy-balance residuals
#
# Run all cells to download the verified public diagnostics and redraw the
# four-panel figure retained in section 4.4. This notebook needs only NumPy and
# Matplotlib, available in Google Colab. It needs no repository, JAX, credentials
# or new trajectory integration. The data are the archived measurements of the
# independent online-diagnostic run, not a new calculation on this runtime.
#
# To repeat the calculation with your own Modal account, open the paired
# [calculation notebook](https://colab.research.google.com/drive/1U_zj1rM87b4gYk-hPJslB3CKr2PE2Ejo).
# Its new results remain separate from the archived diagnostics displayed here.
#
# Eight methods independently integrated all 84 original particles for 5000
# cycles, with normalized gyro-radius 0.015 and 40 fixed steps per cycle. The
# analysis selects eight periodic candidates and eight chaotic candidates from
# those same integrations. The public subset preserves their IDs and original
# float64 diagnostics at each whole-cycle endpoint. Cycle 0 is retained to
# verify initialization and is excluded from every RMS statistic.
#
# The Hamiltonian depends on time, so energy conservation is not the relevant
# identity. Along an exact trajectory, $dH/dt=\partial_t H$. On each numerical
# trajectory the balance residual is
#
# $$R^{(q)}(k)=H(k,y_k)-H(0,y_0)-I^{(q)}(k),\qquad q\in\{2,4\}.$$
#
# With $g_n=\partial_t H(t_n,y_n)$ and fixed step $h=1/40$, the two online
# quadratures use the same accepted states:
#
# $$I^{(2)}_{n+2}=I^{(2)}_n+\frac h3(g_n+4g_{n+1}+g_{n+2}),$$
# $$I^{(4)}_{n+4}=I^{(4)}_n+\frac {2h}3(g_n+4g_{n+2}+g_{n+4}).$$
#
# The implementation uses the archived accepted times and the exact quadratic
# interpolation weights for their floating-point spacing. It evaluates the
# physical Hamiltonian directly. No diagnostic feeds back into the dynamics.
# The public subset retains cycle diagnostics; stepwise derivative histories
# were not stored. This viewer recomputes residuals and statistics, not the
# stepwise quadratures.
# A small residual does not certify an accurate trajectory. Comparing the
# quadratures measures sensitivity, not an error bound. Particle group labels
# remain provisional; the independent rerun was not compared trajectory by
# trajectory with the parent campaign.

# %%
from hashlib import sha256
import io
import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np

PUBLIC_DATA_BASE_URL = 'https://f003.backblazeb2.com/file/gc2d-colab-public/04_04_evolution_energetique/main-20261117-rho-div20-online-balance-20261120/v1'
EXPECTED_MANIFEST_SHA256 = 'a76603ba09e80eda5b1af794b51af1a8b96aa188f554c51c7bc93aee6cb8115e'
STUDY_PREFIX = 'studies/energy_balance_online'
RUN_ID = 'main-20261117-rho-div20-online-balance-20261120'
METHODS = ('RK2', 'ImplicitMidpoint', 'RK4', 'GaussLegendre4', 'ABBA2Midpoint', 'ABBA2Implicit', 'BM4Midpoint', 'BM4Implicit')
GROUPS = {'periodic_candidates8': [1, 2, 60, 64, 65, 72, 76, 83], 'chaotic8': [12, 15, 18, 21, 24, 27, 30, 33]}
DIAGNOSTICS = ('hamiltonian', 'delta_h', 'integral_simpson', 'integral_coarse', 'residual', 'quadrature_gap')


def require(condition: bool, message: str) -> None:
	if not condition:
		raise ValueError(message)


def download_bytes(filename: str) -> bytes:
	"""Read a bounded public HTTPS object without credentials."""
	require(urlsplit(PUBLIC_DATA_BASE_URL).scheme == 'https', 'The public data URL must use HTTPS.')
	request = Request(PUBLIC_DATA_BASE_URL + '/' + filename, headers={'User-Agent': 'GC2D-Colab-visualisation/1'})
	try:
		with urlopen(request, timeout=60) as response:
			require(urlsplit(response.url).scheme == 'https', 'The download was redirected away from HTTPS.')
			data = response.read(64 * 1024 * 1024 + 1)
	except (HTTPError, URLError, TimeoutError) as error:
		raise RuntimeError(f'Could not download {filename}. Check the runtime connection and retry this cell.') from error
	require(0 < len(data) <= 64 * 1024 * 1024, f'Empty or unexpectedly large public object: {filename}.')
	return data


def load_public_data() -> tuple:
	"""Check the pinned manifest and all bytes before interpreting the arrays."""
	raw = download_bytes('manifest.json')
	require(sha256(raw).hexdigest() == EXPECTED_MANIFEST_SHA256, 'The public manifest differs from this notebook\'s pinned version.')
	manifest = json.loads(raw)
	require(manifest.get('schema_version') == 1 and manifest.get('study_prefix') == STUDY_PREFIX
		and manifest.get('run_id') == RUN_ID and set(manifest.get('files', {})) == {'study.json', 'diagnostics.npz'}, 'Incompatible dataset manifest.')
	contents = {}
	for name, expected in manifest['files'].items():
		data = download_bytes(name)
		require(len(data) == expected['bytes'] and sha256(data).hexdigest() == expected['sha256'], f'Size or SHA-256 mismatch: {name}.')
		contents[name] = data
	with np.load(io.BytesIO(contents['diagnostics.npz']), allow_pickle=False) as saved:
		arrays = {name: saved[name].copy() for name in saved.files}
	return json.loads(contents['study.json']), arrays, manifest


def metrics(values: np.ndarray) -> dict:
	"""Reduce individual residuals, never average previously computed RMS values."""
	return {'rms': float(np.sqrt(np.mean(values**2))),
		'maximum_absolute': float(np.abs(values).max()),
		'block_100_cycle_rms': np.sqrt(np.mean(values.reshape(50, 100, 8)**2, axis=(1, 2))).tolist()}


def summarize_saved(study: dict, arrays: dict) -> dict:
	"""Verify the cycle diagnostics and reproduce every retained thesis statistic."""
	protocol = dict(schema_version=1, study_prefix=STUDY_PREFIX, run_id=RUN_ID,
		parent_campaign_id='main-20261117-rho-div20', methods=list(METHODS), rho=.015,
		cycles=5000, step=.025, steps_per_cycle=40, samples_per_cycle=10,
		energy_samples_per_cycle=1, integrated_particle_count=84, groups=GROUPS)
	for key, expected in protocol.items():
		require(study.get(key) == expected, f'Incompatible study field: {key}.')
	ids = sorted(sum(GROUPS.values(), []))
	require(study['particle_ids'] == ids and [p['id'] for p in study['particles']] == ids, 'Unexpected particle membership or column order.')
	require(set(study['cases']) == set(METHODS), 'Incomplete case provenance.')
	expected_arrays = {'cycle', 'particle_ids'} | {method + '/' + key for method in METHODS for key in DIAGNOSTICS}
	require(set(arrays) == expected_arrays, 'Missing or unexpected diagnostic arrays.')
	np.testing.assert_array_equal(arrays['cycle'], np.arange(5001))
	np.testing.assert_array_equal(arrays['particle_ids'], ids)
	statistics = {}
	initial_h = None
	for method in METHODS:
		case = study['cases'][method]
		require(case['all_steps_converged'] is True and case['all_values_finite'] is True
			and case['energy_feedback'] is False and case['derivative_history_saved'] is False, 'Invalid original execution status.')
		data = {key: arrays[method + '/' + key] for key in DIAGNOSTICS}
		for key, values in data.items():
			require(values.dtype == np.float64 and values.shape == (5001, 16) and np.isfinite(values).all(), f'Invalid diagnostic {method}/{key}.')
			if key != 'hamiltonian':
				np.testing.assert_array_equal(values[0], np.zeros(16))
		if initial_h is None:
			initial_h = data['hamiltonian'][0].copy()
		np.testing.assert_array_equal(initial_h, data['hamiltonian'][0])
		np.testing.assert_array_equal(data['delta_h'], data['hamiltonian'] - data['hamiltonian'][0])
		np.testing.assert_array_equal(data['residual'], data['delta_h'] - data['integral_simpson'])
		np.testing.assert_array_equal(data['quadrature_gap'], data['integral_simpson'] - data['integral_coarse'])
		residuals = {2: data['delta_h'] - data['integral_simpson'], 4: data['delta_h'] - data['integral_coarse']}
		for group, group_ids in GROUPS.items():
			columns = [ids.index(particle_id) for particle_id in group_ids]
			statistics[method, group] = {q: metrics(values[1:, columns]) for q, values in residuals.items()}
	centers = (np.arange(50) * 100 + 50.5).tolist()
	panels = [('a', 'periodic_candidates8', 2), ('b', 'periodic_candidates8', 4),
		('c', 'chaotic8', 2), ('d', 'chaotic8', 4)]
	series = [dict(method=method, group=group, panel=letter, quadrature_update_steps=q,
		cycle_centers=centers, values=statistics[method, group][q]['block_100_cycle_rms'])
		for letter, group, q in panels for method in METHODS]
	global_metrics = [dict(method=method, group=group, balance_two_steps=statistics[method, group][2]['rms'],
		balance_four_steps=statistics[method, group][4]['rms'],
		four_over_two=statistics[method, group][4]['rms'] / statistics[method, group][2]['rms'])
		for group in GROUPS for method in METHODS]
	require(series == study['reference_series'] and global_metrics == study['reference_global_metrics'], 'Recomputed statistics differ from the retained thesis figure.')
	return dict(run_id=RUN_ID, series=series, global_metrics=global_metrics)

# %%
study, diagnostics, public_manifest = load_public_data()
summary = summarize_saved(study, diagnostics)
print(f'Verified dataset: {STUDY_PREFIX}/{RUN_ID}')
print('Eight complete original integrations; 16 selected particles; 5000 whole-cycle endpoints per particle.')
print('Verified all diagnostic identities, 32 block-RMS series and 16 global-RMS comparisons.')
print('Group IDs:', json.dumps(GROUPS))

# %% [markdown]
# ## The retained four-panel comparison
#
# Each point is the RMS of 800 residuals: eight particles over 100 successive
# cycle endpoints. Blocks cover cycles 1–100, 101–200, and so on; their centers
# are 50.5, 150.5, …, 4950.5. Periodic candidates occupy panels (a–b), chaotic
# candidates panels (c–d). Each pair shares logarithmic limits. No zero is
# replaced by an artificial positive floor.
#
# The data retain $H$ and $\Delta H$ to verify the balance identities.
# The four-panel balance figure below reproduces the section 4.4 comparison.

# %%
STYLES = {
	'RK2': ('RK2', '#c43b34', '-', 'o'),
	'ImplicitMidpoint': ('Point milieu implicite', '#7d54a7', '--', 'v'),
	'RK4': ('RK4', '#1b7898', '-.', 's'),
	'GaussLegendre4': ('Gauss–Legendre', '#c08b0c', ':', 'D'),
	'ABBA2Midpoint': ('ABBA2 (moyenne)', '#4b7c3d', (0, (5, 2)), '^'),
	'ABBA2Implicit': ('ABBA2 (symétrique)', '#4855a2', (0, (3, 1, 1, 1)), 'P'),
	'BM4Midpoint': ('BM4 (moyenne)', '#a45483', (0, (1, 1)), 'x'),
	'BM4Implicit': ('BM4 (symétrique)', '#333333', (0, (6, 1, 1, 1, 1, 1)), '+'),
}


def plot_balance(summary: dict):
	"""Draw the four retained panels with common logarithmic limits per group."""
	series = {(row['method'], row['group'], row['quadrature_update_steps']): row for row in summary['series']}
	limits = {}
	for group in GROUPS:
		values = np.concatenate([series[method, group, q]['values'] for method in METHODS for q in (2, 4)])
		require(np.isfinite(values).all() and np.all(values > 0), 'Logarithmic residuals must be finite and positive.')
		limits[group] = (float(values.min() / 1.4), float(values.max() * 1.4))
	panels = [('a', 'periodic_candidates8', 2, 'Périodiques'), ('b', 'periodic_candidates8', 4, 'Périodiques'),
		('c', 'chaotic8', 2, 'Chaotiques'), ('d', 'chaotic8', 4, 'Chaotiques')]
	style = {'font.size': 9.5, 'axes.titlesize': 10.5, 'axes.labelsize': 9.5,
		'xtick.labelsize': 9, 'ytick.labelsize': 9, 'pdf.fonttype': 42,
		'savefig.facecolor': 'white', 'axes.spines.top': False, 'axes.spines.right': False}
	with plt.rc_context(style):
		figure, axes = plt.subplots(1, 4, figsize=(10.2, 5.2))
		figure.subplots_adjust(left=.085, right=.975, bottom=.21, top=.70, wspace=.40)
		for axis, (letter, group, q, label) in zip(axes, panels):
			for index, (method, (name, color, line_style, marker)) in enumerate(STYLES.items()):
				row = series[method, group, q]
				axis.plot(row['cycle_centers'], row['values'], label=name, color=color, linestyle=line_style,
					marker=marker, markevery=(index % 5, 5), markersize=3.5,
					markerfacecolor='white', markeredgewidth=.75, linewidth=1.25)
			axis.set(yscale='log', ylim=limits[group], xlim=(0, 5000),
				xlabel='Nombre de cycles', title=f'({letter}) {label}\nSimpson · {q} pas')
			axis.set_xticks([0, 2500, 5000])
			axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1,)))
			axis.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 3, 4, 5, 6, 7, 8, 9)))
			axis.yaxis.set_minor_formatter(NullFormatter())
			if q == 4:
				axis.tick_params(axis='y', which='both', labelleft=False)
			axis.grid(which='major', color='#dce1e7', linewidth=.6)
			axis.grid(which='minor', axis='y', color='#edf0f3', linewidth=.4)
			axis.set_axisbelow(True)
		axes[0].set_ylabel(r'$\sqrt{\langle(\mathcal{R}^{(q)})^2\rangle}$')
		handles, labels = axes[0].get_legend_handles_labels()
		order = [0, 4, 1, 5, 2, 6, 3, 7]
		legend = figure.legend([handles[i] for i in order], [labels[i] for i in order], loc='upper center',
			bbox_to_anchor=(.52, .995), ncol=4, frameon=False, fontsize=9, columnspacing=1.1, handlelength=2.8)
		title = figure.text(.53, .84, r'$\widetilde\rho=0{,}015$ · Même trajectoire · 40 pas fixes par cycle', ha='center', fontsize=10)
		note = figure.text(.53, .065, 'Hamiltonien normalisé · Moyennes quadratiques par blocs de 100 cycles\n'
			'À phase fixe · Échelles logarithmiques identiques dans chaque paire', ha='center', va='center',
			fontsize=9.5, linespacing=1.5, color='#4b5563')
		figure.canvas.draw()
		renderer = figure.canvas.get_renderer()
		boxes = [axis.get_tightbbox(renderer) for axis in axes] + [item.get_window_extent(renderer) for item in (legend, title, note)]
		require(all(b.x0 >= 0 and b.y0 >= 0 and b.x1 <= figure.bbox.width and b.y1 <= figure.bbox.height for b in boxes), 'Figure text exceeds the canvas.')
		require(all(a.x1 < b.x0 for a, b in zip(boxes[:3], boxes[1:4])), 'Panel labels overlap.')
		require(boxes[4].y0 > boxes[5].y1 and boxes[5].y0 > max(b.y1 for b in boxes[:4])
			and boxes[6].y1 < min(b.y0 for b in boxes[:4]), 'Figure annotations overlap.')
	return figure


figure = plot_balance(summary)
plt.show()

# %% [markdown]
# ## Global RMS and quadrature sensitivity
#
# These are the RMS values across all 5000 endpoints and all eight particles of
# each group. The last column is $\mathrm{RMS}(R^{(4)})/\mathrm{RMS}(R^{(2)})$.
# It is not $\mathrm{RMS}(R^{(4)}-R^{(2)})/\mathrm{RMS}(R^{(2)})$.
# The periodic fourth-order residuals are strongly sensitive to the wider
# quadrature spacing. The comparisons do not separate integration error from
# quadrature error or establish a rigorous ranking of trajectory accuracy.

# %%
print(f"{'Method':20s} {'Group':22s} {'RMS, 2 steps':>14s} {'RMS, 4 steps':>14s} {'4 / 2':>10s}")
for row in summary['global_metrics']:
	print(f"{row['method']:20s} {row['group']:22s} {row['balance_two_steps']:14.7e} "
		f"{row['balance_four_steps']:14.7e} {row['four_over_two']:10.4f}")
