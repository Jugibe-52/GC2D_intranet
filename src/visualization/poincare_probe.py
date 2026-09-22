"""Gap context and portable viewer links for new Poincare probe particles."""

from html import escape
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
from threading import Thread
from urllib.parse import quote

import matplotlib.pyplot as plt
import numpy as np


_VIEWER_SERVERS: dict[Path, ThreadingHTTPServer] = {}


def serve_probe_viewer(path: Path) -> str:
    """Return a loopback HTTP URL, served for the lifetime of the notebook kernel.

    Reexecuting the cell reuses its server. Only the HTML directory is served,
    and the listener is restricted to this computer.
    """
    path = Path(path).resolve(strict=True)
    if path.parent not in _VIEWER_SERVERS:
        handler = partial(SimpleHTTPRequestHandler, directory=str(path.parent))
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
        Thread(target=server.serve_forever, daemon=True).start()
        _VIEWER_SERVERS[path.parent] = server
    port = _VIEWER_SERVERS[path.parent].server_port
    return f'http://127.0.0.1:{port}/{quote(path.name)}'


def plot_probe_context(background, settings, *, view: tuple[float, float, float],
                       positions: np.ndarray | None = None, method: str = 'RK4'):
    """Show original returns and the probe at the same square spatial scale."""
    figure, axis = plt.subplots(figsize=(7, 7), layout='constrained')
    x, y, span = view
    for j, color in enumerate(background.colors):
        points = background.positions[1:, j]
        inside = ((points[:, 0] >= x) & (points[:, 0] <= x + span)
                  & (points[:, 1] >= y) & (points[:, 1] <= y + span))
        axis.scatter(*points[inside].T, s=2, color=color, alpha=.35, rasterized=True)
    if positions is not None:
        axis.scatter(*positions[1:, 0].T, s=7, color=settings.color,
                     label=f'Particle {settings.particle_id}: cycle returns', zorder=3)
    axis.scatter(*settings.initial_xy_over_L, s=110, marker='*', color=settings.color,
                 edgecolors='black', linewidths=.7, zorder=4,
                 label=f'Particle {settings.particle_id}: initial state')
    axis.set(xlim=(x, x + span), ylim=(y, y + span), aspect='equal',
             xlabel='x / L', ylabel='y / L', title=f'{method}: probe inside the sampled gap')
    axis.grid(alpha=.15)
    axis.legend(loc='upper left', fontsize=9)
    return figure


def display_probe_viewer(path: Path, *, height: int = 1000, local_url: str | None = None,
                         method: str = 'RK4', particle_label: str = 'particle 49') -> None:
    """Show a local URL plus an offline preview of the same HTML.

    A relative link follows the notebook server's own origin and authentication;
    no public upload or additional HTTP server is needed.
    """
    from IPython.display import HTML, display

    path = Path(path).resolve()
    url = local_url or quote(os.path.relpath(path, Path.cwd()), safe='/')
    display(HTML(f'<p><a href="{escape(url)}" target="_blank" rel="noopener">'
                 f'Open the {escape(method)} {escape(particle_label)} viewer in a new tab</a></p>'))
    display(HTML(f'<iframe title="{escape(method)} probe viewer" width="100%" height="{int(height)}" '
                 f'srcdoc="{escape(path.read_text(encoding="utf-8"), quote=True)}"></iframe>'))


def plot_region_sections(backgrounds, probes, regions, *, panels=None):
    """Compare original samples and matching seeds in each requested region.

    Rows are regions; columns are methods. Optional panels use the same complete
    cycle returns exported to HTML, so figures never substitute dense trajectories
    for once-per-cycle Poincare samples.
    """
    methods = list(backgrounds)
    figure, axes = plt.subplots(len(regions), len(methods), squeeze=False,
                                figsize=(6 * len(methods), 5.5 * len(regions)), layout='constrained')
    for row, region in enumerate(regions):
        x, y, span = region['view']
        pid = region['particle_id']
        probe = probes[pid]
        for col, method in enumerate(methods):
            axis, background = axes[row, col], backgrounds[method]
            for j, color in enumerate(background.colors):
                points = background.positions[1:, j]
                inside = ((points[:, 0] >= x) & (points[:, 0] <= x + span)
                          & (points[:, 1] >= y) & (points[:, 1] <= y + span))
                axis.scatter(*points[inside].T, s=4, color=color, alpha=.4, rasterized=True)
            if panels is not None:
                panel = panels[method]
                points = panel['coordinates'][:, panel['particle_ids'].index(pid)]
                axis.scatter(*points.T, s=7, color=probe['color'], zorder=3,
                             label=f'Particle {pid}: cycle returns', rasterized=True)
            axis.scatter(*probe['initial_xy_over_L'], s=110, marker='*', color=probe['color'],
                         edgecolors='white' if probe['color'] == '#111111' else 'black',
                         linewidths=.7, zorder=4, label=f'Particle {pid}: initial state')
            axis.set(xlim=(x, x + span), ylim=(y, y + span), aspect='equal', xlabel='x / L',
                     ylabel='y / L', title=f'{method} — {region["label"]}')
            axis.ticklabel_format(useOffset=False)
            axis.grid(alpha=.15)
            axis.legend(loc='upper right', fontsize=8)
    return figure
