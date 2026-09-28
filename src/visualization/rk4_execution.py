"""Display saved RK4 backend comparisons without running numerical methods."""

from typing import Any

import numpy as np

from studies.rk4_execution import RK4ExecutionComparison, periodic_discrepancy


def comparison_table(comparison: RK4ExecutionComparison) -> str:
    """Format measured first calls, steady timing and numerical discrepancies."""
    lines = ["Execution          First call   Median [Q25, Q75] (s)       Speedup   Max distance"]
    for key, row in comparison.metadata["runs"].items():
        lines.append(f"{key:18s} {row['first_call_seconds']:9.3f}s  "
                     f"{row['median_seconds']:.4f} [{row['q25_seconds']:.4f}, {row['q75_seconds']:.4f}]  "
                     f"{row['speedup_vs_scipy']:7.2f}x  {row['maximum_periodic_discrepancy']:.3e}")
    fastest = min(comparison.metadata["runs"], key=lambda key: comparison.metadata["runs"][key]["median_seconds"])
    lines.append(f"Fastest measured median: {fastest}. Distances compare with the same RK4 CPU map.")
    return "\n".join(lines)


def plot_rk4_execution_comparison(comparison: RK4ExecutionComparison) -> Any:
    """Show timing quartiles, full discrepancy history, geometry and energy."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    records = comparison.metadata["runs"]
    keys = list(records)
    median = np.array([records[key]["median_seconds"] for key in keys])
    low = np.array([records[key]["q25_seconds"] for key in keys])
    high = np.array([records[key]["q75_seconds"] for key in keys])
    axes[0, 0].bar(keys, median, yerr=np.stack((median-low, high-median)), capsize=5)
    axes[0, 0].set(title="Complete integration time after warm-up", ylabel="Seconds (median and IQR)")
    reference = comparison.solutions["scipy_cpu_0"]
    period = comparison.potential.grid.period
    for key, solution in comparison.solutions.items():
        if key != "scipy_cpu_0":
            discrepancy = periodic_discrepancy(reference, solution, period)
            axes[0, 1].plot(solution.t, discrepancy.max(axis=0), label=key)
        if "generalized_energy_error" in solution.diagnostics:
            energy = np.asarray(solution.diagnostics["generalized_energy_error"])
            axes[1, 1].plot(solution.t, np.abs(energy).max(axis=0), label=key)
    axes[0, 1].set(title="Agreement with the CPU RK4 trajectory", xlabel="Normalized time",
                   ylabel="Maximum periodic distance")
    axes[1, 1].set(title="Passive energy balance", xlabel="Normalized time", ylabel="max |H + kappa - H(initial)|")
    for ax in (axes[0, 1], axes[1, 1]):
        ax.set_yscale("symlog", linthresh=1e-15)
        if ax.lines:
            ax.legend()
    x, y = reference.positions()
    axes[1, 0].scatter(x[:, 0], y[:, 0], s=10, label="All initial particles")
    for index in range(min(8, x.shape[0])):
        axes[1, 0].plot(x[index], y[index], linewidth=.8)
    axes[1, 0].set(title="Initial positions and first eight trajectories", xlabel="x", ylabel="y")
    axes[1, 0].set_aspect("equal")
    axes[1, 0].legend()
    return figure


__all__ = ["comparison_table", "plot_rk4_execution_comparison"]
