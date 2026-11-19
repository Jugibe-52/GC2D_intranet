On 2026-11-16 the user authorized replacing the fourth dataset with IDs 1, 2, 60, 64, 65, 72, 76 and 83 in the particle data, notebook and thesis. The change is implemented.

`periodic_candidates_8.json` preserves the eight complete records from `expanded_84.json`, in the requested order: two original particles and six neighbors. Its provenance records the user selection and explicitly leaves periodicity unverified for the reflected potential. The historical `periodic_8.json`, including all evidence for IDs 85–92, remains unchanged.

The notebook loads the new set in the fourth panel, checks its IDs and full records, uses a French candidate-periodicity title and explains the historical dataset separately. Dataset documentation and the periodic-selection guide, including their complete Spanish reading copies, agree with this behavior.

Checks performed:

- All four code cells executed directly in a shared Python namespace using the project virtual environment and Agg, with `plt.show` replaced only by output capture. The notebook contains two regenerated PNG outputs and passes `nbformat.validate`. Panel counts are 40, 84, 7 and 8; the individual first-set view contains 40. The overview PNG was inspected.
- Exact selected IDs and complete records match `expanded_84.json`. The new selection has no inherited periods or verification records. Physical-to-normalized coordinate checks and the potential cycle-closure check in the notebook passed.
- All pre-existing particle JSON files, the source field and the first-set thesis figure artifacts retain their baseline SHA-256 hashes. The optional thesis export remains disabled.
- The thesis table contains exactly the selected IDs and correctly rounded coordinates. Spanish reading copies remain excluded by existing Git rules. `git diff --check` and `scripts/check_sdd.py` passed.

No new trajectories were integrated. Direct cell execution retains the workaround for unavailable local Jupyter sockets established in the earlier task. The thesis compilation and concurrent source edits are recorded in `TFM_redaccion/specs/changes/2026-11-16-cuarto-conjunto-8/results.md`. No commits, uploads or deployments were performed.
