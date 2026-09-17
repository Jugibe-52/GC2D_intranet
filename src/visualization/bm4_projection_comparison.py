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
from studies.bm4_projection_comparison import summarize_bm4_particles


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
	fig, axes = plt.subplots(n, 2, figsize=(12, 3*n), layout="constrained", squeeze=False)
	for j in range(n):
		for name, color in zip(_METHODS, _COLORS):
			axes[j, 0].semilogy(times[1:], np.maximum(arrays[f"{name}.distance"][j, 1:], 1e-18), label=name, color=color)
			axes[j, 1].semilogy(times[1:], np.maximum(np.abs(arrays[f"{name}.energy_error"][j, 1:]), 1e-18), label=name, color=color)
		axes[j, 0].set(title=f"P{j+1}: trajectory error vs DOP853", ylabel="Periodic distance")
		axes[j, 1].set(title=f"P{j+1}: physical energy error vs DOP853", ylabel="Absolute H error")
		for ax in axes[j]:
			ax.set_xlabel("Normalized time")
			ax.grid(alpha=.2)
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
	axes[0].set(ylabel="Wall-clock seconds", title="Joint three-particle runtime: median and IQR")
	axes[1].set(ylabel="Runtime / BM4Midpoint runtime", title="Relative execution time")
	figures["runtime"] = fig
	fig, axes = plt.subplots(n, 3, figsize=(13, 2.8*n), layout="constrained", squeeze=False)
	mu = arrays["BM4Implicit.mu"].reshape(2, n, -1)
	for j in range(n):
		for component, label in enumerate(("mu_x", "mu_y")):
			axes[j, component].plot(step_times, mu[component, j], lw=1, color=_COLORS[j % 3])
			axes[j, component].set(title=f"P{j+1}: {label}", ylabel=label)
		axes[j, 2].semilogy(step_times, np.maximum(np.max(np.abs(mu[:, j]), axis=0), 1e-18), color=_COLORS[j % 3])
		axes[j, 2].set(title=f"P{j+1}: multiplier infinity norm", ylabel="max(|mu_x|, |mu_y|)")
		for ax in axes[j]:
			ax.set_xlabel("Step end time")
			ax.grid(alpha=.2)
	fig.suptitle("BM4Implicit projection multipliers per particle; BM4Midpoint has no multiplier")
	figures["multipliers"] = fig
	fig, axes = plt.subplots(2, 1, figsize=(10, 5), layout="constrained", sharex=True)
	axes[0].step(step_times, arrays["BM4Implicit.nonlinear_iterations"], where="post", label="Newton corrections")
	axes[0].step(step_times, arrays["BM4Implicit.residual_evaluations"], where="post", label="Residual evaluations")
	axes[0].set(ylabel="Count per complete step", title="Joint BM4Implicit nonlinear work; BM4Midpoint has no nonlinear solve")
	axes[0].legend()
	axes[1].plot(step_times, arrays["BM4Implicit.nonlinear_residual_norms"] / arrays["BM4Implicit.nonlinear_tolerances"])
	axes[1].axhline(1, color="red", ls="--")
	axes[1].set(xlabel="Step end time", ylabel="Residual / tolerance")
	figures["nonlinear_work"] = fig
	if "reference.refinement_distance" in arrays:
		fig, axes = plt.subplots(1, n, figsize=(4.5*n, 4), layout="constrained", squeeze=False)
		full_times = arrays["reference.full_times"]
		for j, ax in enumerate(axes[0]):
			ax.semilogy(full_times[1:], np.maximum(arrays["reference.refinement_distance"][j, 1:], 1e-18), color=_COLORS[j % 3])
			ax.set(xlabel="Normalized time", ylabel="Periodic trajectory discrepancy", title=f"P{j+1}: DOP853 refinement")
			ax.grid(alpha=.2)
		figures["reference_validation"] = fig
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

	# Layout is constant across frames; solve it once before encoding the video.
	update(0)
	fig.canvas.draw()
	fig.set_layout_engine("none")
	return FuncAnimation(fig, update, frames=indices.size, interval=1000/fps, blit=False)


def render_bm4_comparison(
	potential: Potential, arrays: dict[str, np.ndarray], metadata: dict[str, Any],
	output_directory: str | Path, *, frames: int = 201, fps: int = 10,
	render_animation: bool = True,
) -> None:
	"""Export static scientific figures and an H.264 animation from stored data."""
	directory = Path(output_directory)
	directory.mkdir(parents=True, exist_ok=True)
	for name, figure in plot_bm4_comparison(arrays, metadata).items():
		figure.savefig(directory / f"{name}.png", dpi=160)
		plt.close(figure)
	if render_animation or not (directory / "trajectories.mp4").exists():
		effective = GuidingCenterDynamics(potential, rho=metadata["config"]["rho"]).effective_potential
		animation = animate_bm4_comparison(effective, arrays, frames=frames, fps=fps)
		animation.save(str(directory / "trajectories.mp4"), writer="ffmpeg", fps=fps, dpi=120,
			extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
		plt.close(animation._fig)  # type: ignore[attr-defined]
	write_bm4_report(arrays, metadata, directory)


def write_bm4_report(
	arrays: dict[str, np.ndarray], metadata: dict[str, Any], directory: str | Path,
) -> None:
	"""Write per-particle results; keep full numerical provenance in the NPZ."""
	target = Path(directory)
	particles = summarize_bm4_particles(arrays)
	def table(headings: list[str], values: list[list[str]]) -> str:
		return "<table><tr>" + "".join(f"<th>{html.escape(v)}</th>" for v in headings) + "</tr>" + "".join("<tr>" + "".join(f"<td>{html.escape(v)}</td>" for v in row) + "</tr>" for row in values) + "</table>"
	accuracy = []
	positions = []
	for p in particles:
		for name, r in p["methods"].items():
			accuracy.append([p["particle"], name] + [f"{r[k]:.6e}" for k in ("trajectory_rms", "trajectory_final", "trajectory_max", "energy_rms", "energy_max")])
		for name, xy in p["positions"].items():
			positions.append([p["particle"], name] + [f"{v:.9f}" for v in xy])
	mu = [[p["particle"]] + [f"{p['mu'][k]:.6e}" for k in ("mean", "rms", "maximum", "final")] for p in particles]
	runtime = [[name] + [f"{metadata['summary']['methods'][name][k]:.6f}" for k in ("runtime_median_seconds", "runtime_q1_seconds", "runtime_q3_seconds")] for name in _METHODS]
	refinement = [ [p["particle"]] + [f"{p['reference_refinement'][k]:.6e}" for k in ("rms", "maximum")] for p in particles if "reference_refinement" in p]
	public = {
		"config": {k: v for k, v in metadata["config"].items() if not k.startswith("audit_")},
		"potential_parameters": metadata.get("potential_parameters"),
		"particles": particles,
		"joint_runtime_seconds": {name: {k: v for k, v in metadata["summary"]["methods"][name].items() if k.startswith("runtime_")} for name in _METHODS},
		"joint_nonlinear_work": metadata["summary"]["nonlinear_work"],
	}
	figures = "".join(f'<figure><img src="{name}.png" alt="{name.replace("_", " ")}"></figure>' for name in ("initial_particles", "accuracy", "runtime", "multipliers", "nonlinear_work"))
	ref_section = ("<h2>DOP853 reference refinement per particle</h2>" + table(["Particle", "RMS discrepancy", "Maximum discrepancy"], refinement) + '<img src="reference_validation.png" alt="DOP853 refinement">') if refinement else ""
	page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>BM4 per-particle comparison</title>
<style>body{{font:16px system-ui;max-width:1300px;margin:40px auto;padding:0 24px;color:#172b3a}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:8px;border-bottom:1px solid #ccd5db;text-align:right}}img,video{{width:100%}}pre{{white-space:pre-wrap}}p{{line-height:1.6}}</style>
<h1>BM4Midpoint versus BM4Implicit: per-particle results</h1>
<p>Normalized time [{arrays["times"][0]:g}, {arrays["times"][-1]:g}]; {metadata["summary"]["step_count"]} complete steps. Three radial starts; no averaging across particles.</p>
<h2>Trajectory and physical-energy errors against DOP853</h2>
{table(["Particle", "Method", "Trajectory RMS", "Final distance", "Max distance", "Energy RMS", "Max absolute energy error"], accuracy)}
<p>Trajectory error uses minimum-image periodic distance. RMS is the trapezoidal time integral of squared error divided by elapsed time, then square-rooted. Energy error compares the physical Hamiltonian history, not conservation.</p>
<h2>Initial and final positions (normalized, unwrapped)</h2>{table(["Particle", "State / method at final time", "x", "y"], positions)}
<h2>BM4Implicit multiplier infinity norm per particle</h2>{table(["Particle", "Mean", "RMS", "Maximum", "Final"], mu)}
<p>Multiplier statistics use all accepted steps. BM4Midpoint has no multiplier. Projection occurs once after each complete twelve-stage BM4 map.</p>
<h2>Joint runtime: all three particles</h2>{table(["Method", "Median seconds", "Q1 seconds", "Q3 seconds"], runtime)}
<p>These are measured joint integrations, not per-particle timings; they are not divided by three. One warm-up, three alternating serial repetitions, one BLAS thread. Reference generation and diagnostic replay excluded.</p>
{ref_section}
<p>Float64 arithmetic (53-bit significand) is distinct from integration accuracy. DOP853 tolerances: rtol=5e-13, atol=5e-15, max_step=0.0025. Coarser reference: 1e-12, 1e-14, 0.005. Refinement uses all saved reference nodes. The measured discrepancy is not a rigorous error bound. No independent analytic trajectory is available.</p>
<h2>Three-particle evolution</h2><video controls preload="metadata" src="trajectories.mp4"></video>
{figures}<details><summary>Configuration and per-particle numerical data</summary><pre>{html.escape(json.dumps(public, indent=2))}</pre></details></html>'''
	(target / "report.html").write_text(page, encoding="utf-8")
	(target / "summary.json").write_text(json.dumps(public, indent=2), encoding="utf-8")
