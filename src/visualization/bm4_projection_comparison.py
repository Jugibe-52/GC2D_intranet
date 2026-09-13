"""Scientific plots and three-panel particle animation for BM4 comparisons."""

from pathlib import Path
import html
import json
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from potential import Potential
from dynamics import GuidingCenterDynamics


_METHODS = ("BM4Midpoint", "BM4Implicit")
_COLORS = ("#1565c0", "#d84315", "#00897b")


def plot_bm4_comparison(
	arrays: dict[str, np.ndarray], metadata: dict[str, Any],
) -> dict[str, Any]:
	"""Show coverage, reference-audited errors, timing, mu and nonlinear work."""
	times, step_times = arrays["times"], arrays["step_times"]
	n = arrays["initial_state"].size // 2
	summary = metadata["summary"]
	figures = {}
	fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
	xmin, ymin, period = arrays["domain"]
	x, y = arrays["initial_state"].reshape(2, n)
	center = np.asarray([xmin + period / 2, ymin + period / 2])
	ax.plot([center[0], x[-1]], [center[1], y[-1]], color="0.65", zorder=1)
	ax.scatter(center[0], center[1], marker="+", color="black", s=100, label="Cell center")
	for j in range(n):
		ax.scatter(x[j], y[j], color=_COLORS[j % 3], s=55, label=f"P{j+1}: ({x[j]:.3f}, {y[j]:.3f})")
	ax.set(xlim=(xmin, xmin+period), ylim=(ymin, ymin+period), aspect="equal",
		xlabel="x (normalized)", ylabel="y (normalized)", title="Initial particles on one radius")
	ax.legend(fontsize=8)
	figures["initial_particles"] = fig
	fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
	for name, color in zip(_METHODS, _COLORS):
		distances = arrays[f"{name}.distance"]
		axes[0, 0].semilogy(times[1:], np.sqrt(np.mean(distances[:, 1:]**2, axis=0)), label=name, color=color)
		axes[1, 0].semilogy(times[1:], np.sqrt(np.mean(arrays[f"{name}.energy_error"][:, 1:]**2, axis=0)), label=name, color=color)
		row = summary["methods"][name]
		axes[0, 1].scatter(row["runtime_median_seconds"], row["trajectory_rms"], label=name, color=color, s=60)
		axes[1, 1].bar(name, row["trajectory_final_rms"], color=color)
	axes[0, 0].semilogy(times[1:], np.sqrt(np.mean(arrays["reference.distance"][:, 1:]**2, axis=0)), "k--", label="DOP853 / Radau discrepancy")
	axes[1, 0].semilogy(times[1:], np.sqrt(np.mean(arrays["reference.energy_error"][:, 1:]**2, axis=0)), "k--", label="Reference energy discrepancy")
	axes[0, 0].set(xlabel="Time", ylabel="Periodic trajectory RMS error", title="Accuracy versus DOP853")
	axes[1, 0].set(xlabel="Time", ylabel="Physical H RMS error", title="Energy-history agreement (not conservation)")
	axes[0, 1].set(xlabel="Median runtime (s)", ylabel="Space-time RMS distance", title="Accuracy / runtime", xscale="log", yscale="log")
	axes[1, 1].set(ylabel="Final periodic RMS distance", ylim=(0, None), title="Final trajectory accuracy")
	for ax in axes.flat:
		ax.grid(alpha=.2)
	for ax in (axes[0, 0], axes[1, 0], axes[0, 1]):
		ax.legend(fontsize=8)
	figures["accuracy"] = fig
	fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
	baseline = summary["methods"]["BM4Midpoint"]["runtime_median_seconds"]
	for i, (name, color) in enumerate(zip(_METHODS, _COLORS)):
		samples = arrays[f"{name}.runtime_samples"]
		q1, median, q3 = np.quantile(samples, [.25, .5, .75])
		axes[0].bar(name, median, yerr=np.asarray([[median-q1], [q3-median]]), capsize=6, color=color)
		axes[0].scatter(np.full(samples.size, i), samples, color="black", s=15, zorder=4)
		axes[1].bar(name, median/baseline, color=color)
		axes[1].text(i, median/baseline, f"{median/baseline:.2f}x", ha="center", va="bottom")
	axes[0].set(ylabel="Wall-clock seconds", title="Median, IQR and individual repeats")
	axes[1].set(ylabel="Runtime / BM4Midpoint runtime", title="Relative execution time")
	figures["runtime"] = fig
	fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
	mu = arrays["BM4Implicit.mu"].reshape(2, n, -1)
	for j in range(n):
		for component in range(2):
			axes[component, 0].plot(step_times, mu[component, j], lw=1, color=_COLORS[j % 3], label=f"P{j+1}")
		axes[0, 1].semilogy(step_times, np.max(np.abs(mu[:, j]), axis=0), color=_COLORS[j % 3], label=f"P{j+1}")
	axes[0, 1].semilogy(step_times, np.max(np.abs(mu), axis=(0, 1)), "k--", lw=1, label="All particles")
	axes[0, 0].set(ylabel=r"$\mu_x$", title="BM4Implicit: signed components")
	axes[1, 0].set(ylabel=r"$\mu_y$", xlabel="Step end time")
	axes[0, 1].set(ylabel=r"$\|\mu_i\|_\infty$", title="BM4Implicit: multiplier per particle")
	axes[1, 1].semilogy(step_times, arrays["BM4Midpoint.copy_separation_norms"], color=_COLORS[0])
	axes[1, 1].set(xlabel="Step end time", ylabel=r"$\|u_f-v_f\|_\infty$", title="BM4Midpoint: copy separation (not mu)")
	for ax in axes.flat:
		ax.grid(alpha=.2)
	for ax in (axes[0, 0], axes[1, 0], axes[0, 1]):
		ax.legend()
	figures["multipliers"] = fig
	fig, axes = plt.subplots(2, 1, figsize=(10, 5), layout="constrained", sharex=True)
	axes[0].step(step_times, arrays["BM4Implicit.nonlinear_iterations"], where="post", label="Newton corrections")
	axes[0].step(step_times, arrays["BM4Implicit.residual_evaluations"], where="post", label="Residual evaluations")
	axes[0].set(ylabel="Count per complete step", title="BM4Implicit nonlinear work; BM4Midpoint has no nonlinear solve")
	axes[0].legend()
	axes[1].plot(step_times, arrays["BM4Implicit.nonlinear_residual_norms"] / arrays["BM4Implicit.nonlinear_tolerances"])
	axes[1].axhline(1, color="red", ls="--")
	axes[1].set(xlabel="Step end time", ylabel="Residual / tolerance")
	figures["nonlinear_work"] = fig
	return figures


def animate_bm4_comparison(
	potential: Potential, arrays: dict[str, np.ndarray], *, frames: int = 201, fps: int = 10,
) -> FuncAnimation:
	"""Animate all three particles with stable colors and periodic path breaks."""
	names = ("DOP853", *_METHODS)
	times = arrays["times"]
	indices = np.unique(np.linspace(0, times.size-1, min(frames, times.size)).round().astype(int))
	xmin, ymin, period = arrays["domain"]
	# Evaluate the evolving measured potential only at displayed frame times.
	fields = np.asarray(potential.evaluate_grid(times[indices]))[::2, ::2]
	vmin, vmax = float(np.min(fields)), float(np.max(fields))
	fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained", sharex=True, sharey=True)
	paths, markers, images, positions = [], [], [], []
	for ax, name in zip(axes, names):
		images.append(ax.imshow(fields[:, :, 0].T, origin="lower", extent=(xmin, xmin+period, ymin, ymin+period),
			cmap="RdBu_r", vmin=vmin, vmax=vmax, alpha=.65))
		xy = arrays[f"{name}.states"].reshape(2, -1, times.size).copy()
		xy[0] = xmin + (xy[0] - xmin) % period
		xy[1] = ymin + (xy[1] - ymin) % period
		positions.append(xy)
		lines, dots = [], []
		for j in range(xy.shape[1]):
			line, = ax.plot([], [], color=_COLORS[j % 3], lw=1.25, label=f"P{j+1}")
			dot, = ax.plot([], [], "o", color=_COLORS[j % 3], ms=5)
			ax.scatter(*xy[:, j, 0], marker="x", color=_COLORS[j % 3], s=40)
			lines.append(line)
			dots.append(dot)
		paths.append(lines)
		markers.append(dots)
		ax.set(title=name, xlabel="x (normalized)", xlim=(xmin, xmin+period), ylim=(ymin, ymin+period), aspect="equal")
		ax.legend(loc="upper left", fontsize=8)
	axes[0].set_ylabel("y (normalized)")
	fig.colorbar(images[0], ax=axes, shrink=.7, label="Potential at current time")
	title = fig.suptitle("")

	def update(frame: int) -> list[Any]:
		index = indices[frame]
		title.set_text(f"Three radial starts | time = {times[index]:.2f} | crosses: initial positions")
		for panel, xy in enumerate(positions):
			images[panel].set_data(fields[:, :, frame].T)
			for j in range(xy.shape[1]):
				x, y = xy[:, j, :index+1].copy()
				breaks = np.flatnonzero((np.abs(np.diff(x)) > period/2) | (np.abs(np.diff(y)) > period/2)) + 1
				x[breaks], y[breaks] = np.nan, np.nan
				paths[panel][j].set_data(x, y)
				markers[panel][j].set_data([xy[0,j,index]], [xy[1,j,index]])
		return [title]

	return FuncAnimation(fig, update, frames=indices.size, interval=1000/fps, blit=False)


def render_bm4_comparison(
	potential: Potential, arrays: dict[str, np.ndarray], metadata: dict[str, Any],
	output_directory: str | Path, *, frames: int = 201, fps: int = 10,
) -> None:
	"""Export static scientific figures and an H.264 animation from stored data."""
	directory = Path(output_directory)
	directory.mkdir(parents=True, exist_ok=True)
	for name, figure in plot_bm4_comparison(arrays, metadata).items():
		figure.savefig(directory / f"{name}.png", dpi=160)
		plt.close(figure)
	effective = GuidingCenterDynamics(potential, rho=metadata["config"]["rho"]).effective_potential
	animation = animate_bm4_comparison(effective, arrays, frames=frames, fps=fps)
	animation.save(str(directory / "trajectories.mp4"), writer="ffmpeg", fps=fps, dpi=120,
		extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
	plt.close(animation._fig)  # type: ignore[attr-defined]
	write_bm4_report(arrays, metadata, directory)


def write_bm4_report(
	arrays: dict[str, np.ndarray], metadata: dict[str, Any], directory: str | Path,
) -> None:
	"""Write a reviewable numerical report with figures and a playable video."""
	target = Path(directory)
	summary = metadata["summary"]
	rows = summary["methods"]
	fastest = min(rows, key=lambda name: rows[name]["runtime_median_seconds"])
	accurate = min(rows, key=lambda name: rows[name]["trajectory_rms"])
	energy_best = min(rows, key=lambda name: rows[name]["energy_rms"])
	floor = summary["reference_rms_floor"]
	conclusion = (f"Fastest median: {fastest}. Smallest trajectory RMS: {accurate}. "
		f"Smallest physical-energy RMS error: {energy_best}. "
		f"DOP853/Radau trajectory RMS discrepancy: {floor:.6g}. "
		"The reference discrepancy estimates numerical resolution, not a rigorous error bound.")
	if min(row["trajectory_rms"] for row in rows.values()) < 10 * floor:
		conclusion += " At least one method is within ten times the reference discrepancy; its accuracy ranking needs a tighter reference."
	headers = ["Method", "Median s", "Q1 s", "Q3 s", "Trajectory RMS", "Final RMS", "Max distance", "Energy RMS", "Max energy error"]
	keys = ["runtime_median_seconds", "runtime_q1_seconds", "runtime_q3_seconds", "trajectory_rms", "trajectory_final_rms", "trajectory_max", "energy_rms", "energy_max"]
	data_rows = [[name] + [f"{rows[name][key]:.6g}" for key in keys] for name in _METHODS]
	def table(headings: list[str], values: list[list[str]]) -> str:
		return "<table><thead><tr>" + "".join(f"<th>{html.escape(v)}</th>" for v in headings) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{html.escape(v)}</td>" for v in row) + "</tr>" for row in values) + "</tbody></table>"
	positions = arrays["initial_state"].reshape(2, -1)
	initial_rows = [[f"P{j+1}", f"{positions[0,j]:.9g}", f"{positions[1,j]:.9g}"] for j in range(positions.shape[1])]
	mu = summary["mu"]
	mu_text = "; ".join(f"{key}: {value:.6g}" for key, value in mu.items())
	work = summary["nonlinear_work"]
	figures = "".join(f'<figure><img src="{name}.png" alt="{name.replace("_", " ")}"></figure>' for name in ("initial_particles", "accuracy", "runtime", "multipliers", "nonlinear_work"))
	page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>BM4 projection comparison</title>
<style>body{{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 24px;color:#172b3a}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:10px;border-bottom:1px solid #ccd5db;text-align:right}}th:first-child,td:first-child{{text-align:left}}img,video{{width:100%}}figure{{margin:30px 0}}pre{{white-space:pre-wrap;background:#f4f6f8;padding:20px}}p{{line-height:1.6}}</style>
<h1>BM4Midpoint versus BM4Implicit</h1>
<p>Three radial starts; interval {arrays['times'][0]:g} to {arrays['times'][-1]:g}; {summary['step_count']} complete steps. Physical-state formulations; identical data and save times.</p>
<p>{html.escape(conclusion)}</p>{table(headers, data_rows)}
<p>Errors use minimum-image periodic distances. Physical-energy error is disagreement with the DOP853 energy history, not energy conservation. Timings exclude reference generation and the separate multiplier replay.</p>
<h2>Particle evolution</h2><video controls preload="metadata" poster="initial_particles.png" src="trajectories.mp4"></video>
<p>Particle colors are consistent across panels. Crosses show initial positions. Trajectories wrap periodically; lines break at the cell boundary.</p>
<h2>Initial positions</h2>{table(['Particle', 'x', 'y'], initial_rows)}
<h2>Projection multiplier</h2><p>BM4Implicit global infinity-norm statistics: {mu_text}.</p>
<p>BM4Midpoint has no mu. Its discarded copy separation is displayed separately.</p>
<pre>{html.escape(json.dumps(work, indent=2))}</pre>{figures}
<details><summary>Reproducibility metadata and complete numerical summary</summary><pre>{html.escape(json.dumps(metadata, indent=2))}</pre></details></html>'''
	(target / "report.html").write_text(page, encoding="utf-8")
	(target / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
