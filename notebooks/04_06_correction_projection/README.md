# 4.6 — Projection correction

[visualisation.ipynb](visualisation.ipynb) is the section's regular notebook file. It plots ABBA2/BM4 projection corrections and Newton iterations from the archived campaign, checking the provenance of `TFM_redaccion/datos/correccion_proyeccion.json`. No new trajectory calculation is needed.

Linked scripts retain their canonical thesis locations. From the code root, regenerate the notebook with:

```bash
.venv/bin/python -B ../TFM_redaccion/notebooks/correccion_proyeccion_graficas.py --notebook
```

The command writes `visualisation.ipynb` here and clears its recorded outputs. Opening it does not require regeneration. The local `.gitignore` preserves the generated notebook's prior ignored status. Initialization finds the thesis from this folder or either repository root. Existing scientific explanations and execution outputs are retained.
