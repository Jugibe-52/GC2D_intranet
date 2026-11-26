# 4.7 — Trajectory similarity

[visualisation.ipynb](visualisation.ipynb) loads the eight saved trajectories at `rho = 0.015`, computes distances at Poincare returns and plots similarity matrices and curves. These statistical reductions belong to visualization; no separate calculation notebook or new integration is needed.

[visualisation_details.ipynb](visualisation_details.ipynb) displays the saved particle 33 and particle 12 figures. `REGENERER_FIGURES = False` is the default; enabling it invokes the linked source scripts without running trajectory integrations. Both notebooks are regular files in this directory. Linked scripts preserve canonical thesis paths and saved provenance.

From the code root, regenerate the notebooks with:

```bash
.venv/bin/python -B ../TFM_redaccion/notebooks/semejanza_trayectorias.py --notebook
.venv/bin/python -B notebooks/04_07_similitude_trajectoires/creer_details.py
```

These commands write into this section and clear notebook outputs without running the analyses. The local `.gitignore` preserves the generated notebooks' prior ignored status. Existing scientific explanations and recorded outputs of the main notebook were retained during relocation. Figure labels remain French. Initialization works from the section folder or either repository root.
