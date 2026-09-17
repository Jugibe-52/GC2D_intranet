"""Plots of physical accuracy, extended balance, temporal envelopes and cost."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation

from studies.gc_energy_bound import GCEnergyBoundResult, METHODS
from .notebooks import display_records_table

COLORS = {"BM4Implicit": "#007F86", "RK4": "#C95132", "DOP853": "#333F55"}


def show_energy_table(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    """Display selected records with scientific notation for nonzero small errors."""
    prepared = [SimpleNamespace(**{key: row.get(key) for key in columns}) for row in rows]
    formatted = tuple((key, key.replace("_", " "),
                       ".5e" if all(isinstance(row.get(key), (float, np.floating)) for row in rows) else None)
                      for key in columns)
    display_records_table(prepared, columns=formatted)


def _positive(values: np.ndarray) -> np.ndarray:
    """Mask exact zeros only on logarithmic axes; retain original stored values."""
    return np.where(values > 0, values, np.nan)


def plot_energy_histories(result: GCEnergyBoundResult, *, level: int = 0) -> Figure:
    """Separate physical energy, reference error, extended drift and its envelope."""
    a = result.arrays
    t = a[f"h{level}/times"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    axes[0, 0].plot(a["reference/times"], a["reference/H"], color=COLORS["DOP853"], label="DOP853")
    for name in METHODS:
        p = f"h{level}/{name}"
        axes[0, 0].plot(t, a[p+"/H"], color=COLORS[name], alpha=.75, label=name)
        axes[0, 1].plot(a.get(f"h{level}/comparison_times", t), a[p+"/H_reference_error"], color=COLORS[name], label=name)
        axes[1, 0].plot(t, a[p+"/K_error"], color=COLORS[name], label=name)
        axes[1, 1].loglog((t-t[0])[1:], _positive(a[p+"/K_envelope"][1:]), color=COLORS[name], label=name)
    axes[0, 1].plot(a["reference/times"], a["reference/energy_discrepancy"],
                    color="0.5", ls="--", label="DOP853 - Radau")
    for ax, title, ylabel in zip(axes.ravel(),
            ("Physical Hamiltonian (not constant)", "Physical-energy error vs reference",
             "Extended energy-balance error", "Running maximum: is it still growing?"),
            ("H(t, z)", "H_num - H_DOP853", "K - K(0)", "max |K - K(0)|")):
        ax.set(xlabel="Normalized time", ylabel=ylabel, title=title)
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    return fig


def plot_energy_refinement(result: GCEnergyBoundResult) -> Figure:
    """Show every h-dependent temporal envelope and fixed-horizon error scaling."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    steps = result.metadata["config"]["steps"]
    stop = result.metadata["config"]["horizons"][-1]
    for j, name in enumerate(METHODS):
        for level, step in enumerate(steps):
            t = result.arrays[f"h{level}/times"]
            values = result.arrays[f"h{level}/{name}/K_envelope"]
            axes[0, j].loglog((t-t[0])[1:], _positive(values[1:]), label=f"h={step:g}")
        axes[0, j].set(title=name+": all step sizes", ylabel="max |K - K(0)|", xlabel="Horizon T")
        for metric, marker in (("K", "o"), ("H_reference", "s")):
            rows = [r for r in result.envelopes if r["method"] == name and r["metric"] == metric and r["horizon"] == stop]
            hs = np.array([r["step"] for r in rows]); es = np.array([r["maximum"] for r in rows])
            axes[1, j].loglog(hs, _positive(es), marker+"-", label=metric)
            if metric == "K" and es[-1] > 0:
                axes[1, j].loglog(hs, es[-1]*(hs/hs[-1])**4, "k:", label="h^4 guide (not a fitted bound)")
        axes[1, j].set(title=name+f": fixed horizon T={stop:g}", ylabel="Maximum absolute error", xlabel="Step h")
    for ax in axes.ravel():
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    return fig


def plot_energy_blocks(result: GCEnergyBoundResult, *, level: int = 0) -> Figure:
    """Display the signed K-error range and mean of consecutive time blocks."""
    step = result.metadata["config"]["steps"][level]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, name in zip(axes, METHODS):
        rows = [r for r in result.blocks if r["method"] == name and r["step"] == step]
        centers = [(r["start"]+r["stop"])/2 for r in rows]
        ax.fill_between(centers, [r["K_min"] for r in rows], [r["K_max"] for r in rows],
                         color=COLORS[name], alpha=.25, label="Block minimum to maximum")
        ax.plot(centers, [r["K_mean"] for r in rows], "o-", color=COLORS[name], label="Block mean")
        ax.set(title=name, xlabel="Block center time", ylabel="K - K(0)")
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    return fig


def plot_gc_accuracy_and_cost(result: GCEnergyBoundResult, *, level: int = 0) -> Figure:
    """Show the orbit, full periodic-distance history, and timing tradeoffs."""
    a = result.arrays
    t = a[f"h{level}/times"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(*a["reference/states"], color=COLORS["DOP853"], label="DOP853")
    axes[0, 0].scatter(*a["initial_state"], marker="*", color="black", s=80, label="Initial state")
    for name in METHODS:
        p = f"h{level}/{name}"
        axes[0, 0].plot(*a[p+"/states"], lw=1, color=COLORS[name], label=name)
        axes[0, 1].semilogy(a.get(f"h{level}/comparison_times", t), _positive(a[p+"/distance"]), color=COLORS[name], label=name)
        rows = [r for r in result.summary if r["method"] == name]
        xs = np.array([r["runtime_median"] for r in rows])
        low = xs-np.array([r["runtime_q25"] for r in rows])
        high = np.array([r["runtime_q75"] for r in rows])-xs
        axes[1, 0].errorbar(xs, [r["trajectory_rms"] for r in rows], xerr=np.array([low, high]),
                            fmt="o-", color=COLORS[name], label=name)
        axes[1, 1].plot([r["step"] for r in rows], xs, "o-", color=COLORS[name], label=name)
    axes[0, 1].semilogy(a["reference/times"], _positive(a["reference/periodic_discrepancy"]),
                        "k--", label="DOP853 - Radau audit")
    axes[0, 0].set(title="One GC orbit (unwrapped coordinates)", xlabel="x", ylabel="y", aspect="equal")
    # Equal coordinate spans keep short, almost vertical orbits from collapsing
    # the axes to a thin strip and obscuring tick labels and the legend.
    plotted_states = np.concatenate([a["reference/states"],
        *(a[f"h{level}/{name}/states"] for name in METHODS)], axis=1)
    lower, upper = plotted_states.min(axis=1), plotted_states.max(axis=1)
    center = (lower+upper)/2
    radius = max(float(np.max(upper-lower))*.6, .1)
    axes[0, 0].set_xlim(center[0]-radius, center[0]+radius)
    axes[0, 0].set_ylim(center[1]-radius, center[1]+radius)
    axes[0, 1].set(title="Trajectory error at shared saved nodes", xlabel="Normalized time", ylabel="Minimum-image distance")
    axes[1, 0].set(title="Accuracy vs physical integration time", xlabel="Median seconds (IQR)", ylabel="Time-RMS periodic distance",
                   xscale="log", yscale="log")
    axes[1, 1].set(title="Absolute runtime comparison", xlabel="Step h", ylabel="Median seconds", xscale="log", yscale="log")
    for ax in axes.ravel():
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    return fig


def plot_bm4_nonlinear_work(result: GCEnergyBoundResult, *, level: int = 0) -> Figure:
    """Display per-step Newton work and the complete projection-norm history."""
    a = result.arrays; p = f"h{level}/BM4Implicit"
    t = a[f"h{level}/times"][1:]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    axes[0].plot(t, a[p+"/nonlinear_iterations"], label="Newton corrections")
    axes[0].plot(t, a[p+"/residual_evaluations"], label="Residual evaluations", alpha=.7)
    axes[0].set(ylabel="Count per accepted step", title="BM4 nonlinear work")
    axes[0].legend(fontsize=8)
    axes[1].plot(t, a[p+"/projection_multiplier_norms"], color=COLORS["BM4Implicit"])
    axes[1].set(ylabel="Infinity norm of mu", title="Projection multiplier")
    axes[2].plot(t, a[p+"/nonlinear_residual_norms"]/a[p+"/nonlinear_tolerances"])
    axes[2].axhline(1, color="black", ls="--", label="Acceptance limit")
    axes[2].set(ylabel="Residual / tolerance", title="Nonlinear acceptance audit")
    for ax in axes:
        ax.set_xlabel("Normalized time"); ax.grid(alpha=.2)
    return fig


def animate_energy_orbit(result: GCEnergyBoundResult, *, level: int = 0,
                         frame_count: int = 201, fps: int = 10) -> FuncAnimation:
    """Animate aligned reference/BM4/RK4 states, retaining all states in the result."""
    a = result.arrays; times = a[f"h{level}/times"]
    if f"h{level}/comparison_times" in a:
        times = a[f"h{level}/comparison_times"]
        ri = a[f"h{level}/comparison_reference_indices"]
        mi = a[f"h{level}/comparison_method_indices"]
        states = {"DOP853": a["reference/states"][:, ri],
                  **{name: a[f"h{level}/{name}/states"][:, mi] for name in METHODS}}
    else:
        stride = (len(a["reference/times"])-1)//(len(times)-1)
        states = {"DOP853": a["reference/states"][:, ::stride],
                  **{name: a[f"h{level}/{name}/states"] for name in METHODS}}
    frames = np.unique(np.linspace(0, len(times)-1, min(frame_count, len(times)), dtype=int))
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    all_states = np.concatenate(list(states.values()), axis=1)
    lower, upper = all_states.min(axis=1), all_states.max(axis=1)
    center = (lower+upper)/2
    radius = max(float(np.max(upper-lower))*.6, .1)
    ax.set(xlim=(center[0]-radius, center[0]+radius),
           ylim=(center[1]-radius, center[1]+radius), xlabel="x", ylabel="y", aspect="equal")
    lines = {name: ax.plot([], [], color=COLORS[name], label=name)[0] for name in states}
    dots = {name: ax.plot([], [], "o", color=COLORS[name], ms=5)[0] for name in states}
    ax.legend(); ax.grid(alpha=.2)

    def update(index: int) -> list[Any]:
        for name, values in states.items():
            lines[name].set_data(values[0, :index+1], values[1, :index+1])
            dots[name].set_data([values[0, index]], [values[1, index]])
        ax.set_title(f"One HDF5 guiding-center orbit: t={times[index]:.4g}")
        return [*lines.values(), *dots.values()]

    return FuncAnimation(fig, update, frames=frames, interval=1000/fps, blit=False)
