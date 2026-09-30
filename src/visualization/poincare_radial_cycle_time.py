"""Plot archived radial cycle-time results without integrating trajectories."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from visualization.poincare_selector import export_poincare_selector


def render_radial_cycle(saved, directory):
    """Export B-style wrapped returns, initial positions and copy separation."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    meta, solution = saved.metadata, saved.solution
    c = meta["config"]
    x, y = solution.positions()
    stride = c["steps_per_cycle"]
    cycle_xy = np.stack((x[:, stride::stride].T, y[:, stride::stride].T), axis=-1)
    points = ((cycle_xy - meta["cell_origin"]) % meta["cell_period"]) / meta["cell_period"]
    ids, colors = meta["particle_ids"], meta["colors"]
    title = "BM4Midpoint | cycle-time stream H = 2*pi*Phi_hat"
    subtitle = f'{len(ids)} particles | {c["cycles"]} cycles | {stride} steps/cycle'
    initial = (np.asarray(meta["initial_positions"]) - meta["cell_origin"]) / meta["cell_period"]
    for name, xy in (("poincare_section", points), ("initial_positions", initial[None])):
        fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
        try:
            for j, (pid, color) in enumerate(zip(ids, colors)):
                ax.scatter(xy[:, j, 0], xy[:, j, 1], color=color,
                           s=3 if name == "poincare_section" else 25, label=str(pid))
            ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="x / L", ylabel="y / L",
                   aspect="equal", title=f"{title}\n{subtitle}")
            ax.grid(alpha=.16)
            fig.savefig(directory / f"{name}.png", dpi=150)
        finally:
            plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 3), layout="constrained")
    try:
        ax.plot(solution.t[1:], solution.diagnostics["copy_separation_norms"], lw=.5)
        ax.set(xlabel="Forcing cycles", ylabel="Copy separation infinity norm",
               title="Copy separation before arithmetic projection (all particles)")
        fig.savefig(directory / "diagnostics.png", dpi=150)
    finally:
        plt.close(fig)
    return export_poincare_selector(directory / "poincare_selector.html", points, ids, colors,
                                   title=title, subtitle=subtitle)


__all__ = ["render_radial_cycle"]
