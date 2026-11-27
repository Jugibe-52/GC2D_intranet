# %% [markdown]
# # 4.5 — Projection correction and Newton iterations
#
# Run all cells to verify the included data snapshot and reproduce the two
# two-panel figures from section 4.5. Only NumPy and Matplotlib are required;
# both are available in Google Colab. No repository checkout, credentials or
# trajectory integrations are needed.
#
# The snapshot contains the verified statistical reductions and their audit
# provenance from `main-20261117-rho-div20`: 84 particles, 5000 cycles,
# 40 fixed steps per cycle and 10 saved samples per cycle. The original
# scientific archives remain in the configured Backblaze results bucket.
# Their 21 complete solution files total about 6.1 GB; embedding the small
# reductions makes this viewer independent of private storage credentials.
#
# The paired [calculation notebook](https://colab.research.google.com/drive/1dlmACJ_X4mgxi73jNzgoiD6VG926mh_Q) runs a new study
# using your own Modal account. Those new results remain separate from this
# archived reference snapshot, whose exact figures are reproduced here.
#
# The radius is already normalized by the base-cell width:
# $\widetilde\rho=i/400$, $i=0,\ldots,10$. No additional division by 20 is
# applied. ABBA2 at radius zero did not converge and remains unavailable.
#
# This viewer checks the pinned bytes, case inventory and all eight plotted
# series. The audit records the checks on all 21 original archives and all
# 147 diagnostic arrays. The small snapshot does **not** contain those arrays;
# it cannot independently recompute their medians or certify the original
# integrations merely by checking the audit file. `prepare_public_data.py`
# performs that full archive verification and reduction locally.

# %%
import base64
from hashlib import sha256
import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import zlib

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter
import numpy as np

STUDY_PREFIX = 'studies/projection_correction'
RUN_ID = 'main-20261117-rho-div20'
PUBLIC_DATA_BASE_URL = 'https://f003.backblazeb2.com/file/gc2d-colab-public/04_05_correction_projection/viewer/v1'
# The included snapshot is the default; public deployment is a separate action.
DATA_SOURCE = 'snapshot'
EXPECTED_MANIFEST_SHA256 = 'c88767c8bf12af1c4fd17775598f21857d4bcd5206cae568075c804aa2f83c8a'
EMBEDDED_SNAPSHOT_B64 = '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'
METHODS = ('ABBA2Implicit', 'BM4Implicit')
STATISTICS = ('sampled_median', 'sampled_maximum', 'newton_mean_iterations', 'newton_max_iterations')


def require(condition: bool, message: str) -> None:
	if not condition:
		raise ValueError(message)


def download_bytes(filename: str) -> bytes:
	"""Optional public HTTPS transport; errors never fall back to another dataset."""
	require(urlsplit(PUBLIC_DATA_BASE_URL).scheme == 'https', 'The public data URL must use HTTPS.')
	request = Request(PUBLIC_DATA_BASE_URL + '/' + filename, headers={'User-Agent': 'GC2D-Colab-visualisation/1'})
	try:
		with urlopen(request, timeout=60) as response:
			require(urlsplit(response.url).scheme == 'https', 'The download was redirected away from HTTPS.')
			data = response.read(1024 * 1024 + 1)
	except (HTTPError, URLError, TimeoutError) as error:
		raise RuntimeError(f'Could not download {filename}; public deployment may be unavailable. The included snapshot works with DATA_SOURCE="snapshot".') from error
	require(0 < len(data) <= 1024 * 1024, f'Empty or unexpectedly large public object: {filename}.')
	return data


def load_data() -> tuple:
	"""Verify the same pinned manifest and content hashes for either source."""
	require(DATA_SOURCE in ('snapshot', 'public_https'), 'Choose snapshot or public_https explicitly.')
	if DATA_SOURCE == 'snapshot':
		objects = json.loads(zlib.decompress(base64.b64decode(EMBEDDED_SNAPSHOT_B64)))
		read = lambda name: base64.b64decode(objects[name], validate=True)
	else:
		read = download_bytes
	raw = read('manifest.json')
	require(sha256(raw).hexdigest() == EXPECTED_MANIFEST_SHA256, 'The data manifest differs from this notebook\'s pinned version.')
	manifest = json.loads(raw)
	require(manifest.get('schema_version') == 1 and manifest.get('study_prefix') == STUDY_PREFIX
		and manifest.get('run_id') == RUN_ID and set(manifest.get('files', {})) == {'summary.json', 'provenance.json'}, 'Incompatible dataset manifest.')
	contents = {}
	for name, expected in manifest['files'].items():
		data = read(name)
		require(len(data) == expected['bytes'] and sha256(data).hexdigest() == expected['sha256'], f'Size or SHA-256 mismatch: {name}.')
		contents[name] = json.loads(data)
	return contents['summary.json'], contents['provenance.json'], manifest


def verify_summary(summary: dict, provenance: dict) -> dict:
	"""Check reductions, missing-case semantics and equality to retained plot series."""
	protocol = dict(schema_version=1, study_prefix=STUDY_PREFIX, run_id=RUN_ID, campaign_id=RUN_ID,
		config=dict(cycles=5000, steps_per_cycle=40, samples_per_cycle=10), particle_count=84,
		rho_values=(np.arange(11) / 400).tolist(), rho_convention='base_cell_width', representative_rho=.015,
		sampled_values_per_case=4200000, particle_steps_per_case=16800000)
	for key, expected in protocol.items():
		require(summary.get(key) == expected, f'Unexpected study field: {key}.')
	for key in ('schema_version', 'study_prefix', 'run_id'):
		require(provenance.get(key) == protocol[key], f'Unexpected provenance field: {key}.')
	cases = summary['cases']
	require(len(cases) == 22 and len({case['case_key'] for case in cases}) == 22, 'Incorrect case inventory.')
	require(set(case['method'] for case in cases) == set(METHODS), 'Unexpected methods.')
	missing = [case for case in cases if case['status'] != 'complete']
	require(missing == [dict(case_key='rho-0.00-ABBA2Implicit', rho=0.0, method='ABBA2Implicit',
		status='unavailable', reason='nonlinear_convergence_failure')], 'The unavailable case must remain explicit.')
	complete = [case for case in cases if case['status'] == 'complete']
	proofs = {case['case_key']: case for case in provenance['cases']}
	require(len(proofs) == len(complete) == 21 and set(proofs) == {case['case_key'] for case in complete}, 'Incomplete audit inventory.')
	for case in complete:
		values = [case[key] for key in (*STATISTICS, 'max_residual_to_tolerance')]
		require(np.isfinite(values).all() and 0 < case['sampled_median'] <= case['sampled_maximum'], 'Invalid correction statistics.')
		require(0 <= case['newton_mean_iterations'] <= case['newton_max_iterations'] <= 40
			and case['newton_max_iterations'] == int(case['newton_max_iterations'])
			and 0 <= case['max_residual_to_tolerance'] <= 1, 'Invalid Newton statistics.')
		proof = proofs[case['case_key']]
		require(proof['all_checked_identities_hold'] is True and proof['statistics_equal_thesis_summary'] is True
			and len(proof['arrays']) == 7, 'Incomplete source-array audit.')
		for name, digest in case['files_sha256'].items():
			require(proof['files'][name]['sha256'] == digest, 'Archive fingerprint differs between summary and audit.')
	series = {}
	for method in METHODS:
		rows = sorted((case for case in cases if case['method'] == method), key=lambda case: case['rho'])
		require([case['rho'] for case in rows] == summary['rho_values'], 'Incorrect radius inventory.')
		series[method] = {key: np.asarray([case.get(key, np.nan) for case in rows]) for key in STATISTICS}
	plot_rows = [row for plot in provenance['plots'].values() for row in plot['series']]
	require(len(plot_rows) == 8 and len({(r['method'], r['statistic']) for r in plot_rows}) == 8, 'Incomplete thesis-figure series.')
	for row in plot_rows:
		require(row['rho'] == summary['rho_values'], 'Thesis-figure radii differ.')
		np.testing.assert_array_equal(series[row['method']][row['statistic']], [np.nan if value is None else value for value in row['values']])
	return series

# %%
summary, provenance, manifest = load_data()
series = verify_summary(summary, provenance)
print(f'Verified {DATA_SOURCE} dataset: {STUDY_PREFIX}/{RUN_ID}')
print('21 complete archived cases; ABBA2 at radius zero remains unavailable.')
print('Verified pinned snapshot hashes, 22 case records and eight retained thesis-figure series.')
print('The provenance records full verification of 21 source archives and 147 arrays; those raw arrays are not embedded.')

# %% [markdown]
# ## Meaning and limits of the four panels
#
# The correction magnitude is $\|\mu\|_2=\sqrt{\mu_x^2+\mu_y^2}$ for one
# particle, in base-cell-width units. Its median and maximum use all 84
# particles and 50,000 saved corrections, corresponding to the last step of
# each four-step block: 4,200,000 values per complete case, excluding the
# initial state. These are **sampled** statistics. The Euclidean maximum over
# all 200,000 steps is unavailable because intermediate block records contain
# only infinity-norm maxima, without the two multiplier components.
#
# Newton means and maxima use all 200,000 steps. Each count belongs to a
# **joint solve for 84 particles**, not a separate solve for one particle and
# not only the saved steps. The correction norm is neither an algebraic
# residual nor a trajectory error. The methods follow their own trajectories;
# this single-step-size comparison does not determine convergence order.
# Lines connect only the eleven computed radii; they do not add measurements.
# Figure text is French to match the thesis.

# %%
NAMES = {'ABBA2Implicit': 'ABBA2', 'BM4Implicit': 'BM4'}
COLORS = {'ABBA2Implicit': '#2563a6', 'BM4Implicit': '#c45b24'}
MARKERS = {'ABBA2Implicit': 'o', 'BM4Implicit': 'D'}
PANELS = {
	'correction': (
		('sampled_median', '(a) Médiane de la correction', (7e-11, 6e-5), 'log'),
		('sampled_maximum', '(b) Maximum de la correction', (3e-6, 1.8e-3), 'log')),
	'newton': (
		('newton_mean_iterations', '(a) Moyenne des itérations', (.9, 2.75), 'linear'),
		('newton_max_iterations', '(b) Maximum des itérations', (.5, 8.6), 'linear')),
}


def plot_comparison(summary: dict, series: dict, name: str):
	"""Draw one retained two-panel comparison; preserve missing values as NaN."""
	radii = np.asarray(summary['rho_values'])
	style = {'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
		'xtick.labelsize': 9, 'ytick.labelsize': 9, 'pdf.fonttype': 42,
		'savefig.facecolor': 'white', 'axes.spines.top': False, 'axes.spines.right': False}
	with plt.rc_context(style):
		figure, axes = plt.subplots(1, 2, figsize=(8.3, 2.9))
		figure.subplots_adjust(left=.095, right=.98, top=.76, bottom=.24, wspace=.36)
		curves = {}
		for axis, (key, title, limits, scale) in zip(axes, PANELS[name]):
			for method in METHODS:
				curve, = axis.plot(radii, series[method][key], color=COLORS[method], marker=MARKERS[method],
					markersize=4.5, linewidth=1.6, label=NAMES[method])
				np.testing.assert_array_equal(curve.get_xdata(), radii)
				np.testing.assert_array_equal(curve.get_ydata(), series[method][key])
				if method == 'ABBA2Implicit':
					require(np.isnan(curve.get_ydata()[0]), 'The failed case acquired a plotted value.')
				curves[method] = curve
			axis.axvline(summary['representative_rho'], color='#747b85', linestyle=':', linewidth=1.2, zorder=0)
			axis.set(title=title, xlabel=r'Rayon normalisé $\widetilde\rho$',
				ylabel=r'$\|\mu\|_2$ (coordonnées normalisées)' if scale == 'log' else 'Itérations par pas',
				xlim=(-.0007, .0257), ylim=limits, yscale=scale)
			axis.set_xticks(np.arange(0, .0251, .005))
			axis.set_xticks(radii, minor=True)
			axis.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.3f}'.replace('.', ',')))
			if scale == 'log':
				axis.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
				axis.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
				axis.yaxis.set_minor_formatter(NullFormatter())
			else:
				axis.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=key == 'newton_max_iterations'))
				axis.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'.replace('.', ',')))
			axis.grid(which='major', color='#dce1e7', linewidth=.7)
			axis.grid(which='minor', axis='y', color='#edf0f3', linewidth=.5)
			axis.set_axisbelow(True)
		handles = [curves[method] for method in METHODS] + [Line2D([], [], color='#747b85', linestyle=':', linewidth=1.2,
			label=r'Référence : $\widetilde\rho=0{,}015$')]
		figure.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .99), ncol=3, frameon=False, fontsize=10)
		figure.text(.5, .035, r'ABBA2 : calcul non convergé à $\widetilde\rho=0$.', ha='center', va='center', fontsize=9, color='#4b5563')
		figure.canvas.draw()
	return figure


correction_figure = plot_comparison(summary, series, 'correction')
plt.show()
newton_figure = plot_comparison(summary, series, 'newton')
plt.show()

# %% [markdown]
# ## Numerical values and interpretation
#
# The table preserves the recorded precision; the maximum residual/tolerance
# ratio is a separate convergence diagnostic over all steps. It must not be
# confused with the correction norm. BM4 has smaller sampled median and
# maximum corrections at all ten shared positive radii, but their ratios
# differ and the radius dependence is not strictly monotone. Smaller
# corrections alone do not establish greater trajectory accuracy.

# %%
print(f"{'Method':15s} {'Radius':>8s} {'Median':>14s} {'Sampled max.':>14s} {'Newton mean':>12s} {'Newton max':>10s} {'Residual/tol.':>14s}")
for case in summary['cases']:
	if case['status'] != 'complete':
		print(f"{case['method']:15s} {case['rho']:8.4f}  unavailable: nonlinear convergence failure")
	else:
		print(f"{case['method']:15s} {case['rho']:8.4f} {case['sampled_median']:14.7e} {case['sampled_maximum']:14.7e} "
			f"{case['newton_mean_iterations']:12.6f} {case['newton_max_iterations']:10d} {case['max_residual_to_tolerance']:14.7f}")
representative = {case['method']: case for case in summary['cases'] if case['rho'] == summary['representative_rho']}
for key in ('sampled_median', 'sampled_maximum'):
	print(f"ABBA2/BM4 {key} ratio at radius 0.015: {representative['ABBA2Implicit'][key] / representative['BM4Implicit'][key]:.2f}")
