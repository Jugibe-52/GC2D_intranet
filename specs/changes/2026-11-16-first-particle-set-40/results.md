On 2026-11-16 the user authorized making IDs 1–40 the first dataset and updating the notebook and thesis. This change is implemented.

`original_40.json` preserves the first forty complete particle records from the historical set, including physical and normalized coordinates, IDs and colors. The notebook now reads the four active sets in the order 40, 84, 8 and 8. Its saved figure has been regenerated with French text. The dataset README documents the active files and the historical role of `original_48.json`; its Spanish reading copy is also saved.

Checks performed:

- Exact equality of the 40 records against both the historical 48-particle set and the first 40 entries of `expanded_84.json`; IDs are exactly 1–40.
- Agreement of the normalized coordinates with the eight-ray star formula, with absolute tolerance 2e-15.
- Unchanged SHA-256 hashes for the four existing datasets and the measured field. The pre-existing Git modification of `PHI_2.h5` was preserved.
- All three notebook code cells executed directly in one Python namespace using the project virtual environment. Only `plt.show` was replaced with PNG capture. The saved output contains four panels with 40, 84, 8 and 8 points; notebook format validation passed and the figure was inspected.
- The initial Jupyter-kernel attempt was unavailable because sandbox restrictions prevent opening local sockets. Direct execution of the same cell sources completed successfully and produced the saved figure; no trajectory integration or remote computation was performed.
- Notebook source syntax, `git diff --check` and `scripts/check_sdd.py` passed. Spanish Markdown reading copies are ignored by the existing Git rules.

The thesis implementation and successful PDF compilation are recorded in `TFM_redaccion/specs/changes/2026-11-16-primer-conjunto-40/results.md`. No commits, uploads or deployments were made.
