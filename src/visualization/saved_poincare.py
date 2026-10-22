"""Render saved cycle CSVs without requiring full trajectories or integration."""

from html import escape
from pathlib import Path

import matplotlib.pyplot as plt

from diagnostics.studies.poincare_section import load_poincare_cycle_csv
from visualization.poincare_selector import export_poincare_selector

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy import ndarray
    from typing import Any


def render_saved_section(study_directory: str | Path, run_id: str) -> Path:
    """Save a static section and an offline selector, preserving IDs and colors."""
    metadata: dict[str, Any]
    points: ndarray
    colors: list[str | None]
    metadata, points, colors = load_poincare_cycle_csv(study_directory, run_id)
    output = Path(study_directory) / 'figuras' / run_id
    output.mkdir(parents=True, exist_ok=True)
    ids = metadata['particle_ids']
    title = f'Hamiltonian Poincare section | {metadata["method"]}'
    subtitle = (f'{len(ids)} particles · {len(points)} cycles · '
                f'{metadata["steps_per_cycle"]} steps/cycle')
    selector: Path = export_poincare_selector(
        output / 'poincare_selector.html', points, ids, colors,
        title=title, subtitle=subtitle,
    )
    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(10, 8), layout='constrained')
    try:
        for j, (pid, color) in enumerate(zip(ids, colors)):
            ax.scatter(points[:, j, 0], points[:, j, 1], color=color,
                       s=3, alpha=.85, linewidths=0, label=str(pid))
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='x / L', ylabel='y / L',
               title=f'{title}\n{subtitle}', aspect='equal')
        ax.grid(alpha=.16)
        ax.legend(title='Original particle ID', ncol=3, fontsize=7,
                  loc='upper left', bbox_to_anchor=(1.01, 1))
        fig.savefig(output / 'poincare_section.png', dpi=160)
        fig.savefig(output / 'poincare_section.svg')
    finally:
        plt.close(fig)
    return selector


def display_saved_section(selector: str | Path) -> None:
    """Embed the offline selector so notebook servers need no local-file URL."""
    from IPython.display import HTML, Image, display

    selector = Path(selector)
    display(Image(filename=str(selector.with_name('poincare_section.png'))))
    document: str = escape(selector.read_text(encoding='utf-8'), quote=True)
    display(HTML(f'<iframe srcdoc="{document}" title="Poincare particle selector" '
                 'width="100%" height="1100" style="border:0"></iframe>'))
