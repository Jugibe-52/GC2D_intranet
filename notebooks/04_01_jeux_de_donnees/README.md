# 4.1 — Particle datasets

Open `visualisation_ensembles.ipynb` with `GC2D_codigo/.venv/bin/python` and run all cells.
This single notebook displays the four sets of initial positions in a 2×2 figure: 40 original particles, 84 particles with neighbors, eight chaotic candidates and eight periodic candidates.
It preserves their positions, IDs and colors from `data/particles/`.
Set `show_ids = True` to display identifiers; `phase` selects the potential's phase.

The background is the reflected, gyroaveraged potential from `data/potential/V1/PHI_2.h5`.
The radius comes from the sibling thesis's `datos/campana_referencia.json` (currently `rho = 0.015` in base-cell widths).
Figure labels are in French, following the thesis convention.
The notebook needs the two sibling repositories and the local potential, but no trajectory archives.
It performs no integration, figure export, download or upload.
A separate calculation notebook is unnecessary for this viewing-only task, as explicitly requested.

The previous notebooks, including their figure-generation cells and saved outputs, are preserved in [TFM_redaccion/notebooks/archives_ensembles_particules](../../../TFM_redaccion/notebooks/archives_ensembles_particules/).
The scripts `segundo_conjunto_84.py`, `cuarto_conjunto_poincare.py` and the historical `conjuntos_particulas_rho030.ipynb` remain at their original locations in `TFM_redaccion/notebooks/`.
