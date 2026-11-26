# 4.9 — Geometric defects

- [calcul.ipynb](calcul.ipynb) calculates or recovers accumulated ABBA2Midpoint and BM4Midpoint defects for both eight-particle groups, then preserves matrices, states, individual timings and provenance.
- [visualisation.ipynb](visualisation.ipynb) loads verified results and produces area and reversal defect figures without repeating integrations.
- [geometria_midpoint_barrido.py](geometria_midpoint_barrido.py) is the generator called by the visualization notebook.
- [geometria_midpoint_graficas.py](geometria_midpoint_graficas.py) and [geometria_midpoint_grupos.py](geometria_midpoint_grupos.py) provide access to the existing complementary generators. The latter reevaluates one-step returns from the historical study, so it does more than read saved results.

This folder holds the canonical notebooks themselves, including their saved outputs.
The scripts are relative links to the generators in `TFM_redaccion/notebooks/`.
The code and thesis repositories must remain siblings named `GC2D_codigo` and `TFM_redaccion`.
Use the Python kernel from `GC2D_codigo/.venv`; the first cell finds the code repository from this section folder or its parents.
Notebook prose is English, and thesis figure text is French.
This is section 4.9; section 4.8 on Poincaré sections is excluded from the section-folder organization.

Both notebooks share prefix `studies/midpoint_geometry` and run `midpoint-jacobian-product-20261122-rho0015`.
The scientific destination remains `gc2d_data:gc2d-notebooks-data/studies/midpoint_geometry/midpoint-jacobian-product-20261122-rho0015`; `GC2D_codigo/outputs/midpoint_geometry/` holds local recovery copies.
Calculation remains disabled by default (`RUN_COMPUTATION=False`), and the historical `midpoint-geometry-20261118-rho0015` study retains its identity.
This reorganization moves no results and starts no calculation or transfer.
