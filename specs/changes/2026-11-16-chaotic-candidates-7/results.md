On 2026-11-16 the user authorized replacing the third dataset with IDs 12, 15, 18, 21, 24, 27 and 30. They then explicitly authorized placing the first-set figure to the left of its thesis explanation. Both changes are implemented.

`chaotic_candidates_7.json` contains the seven complete original records, in the requested order, with user-selection provenance and no inherited numerical classification from the former eight-particle dataset. That older JSON is retained as historical data. The notebook and dataset README now identify the active sets as 40, 84, 7 and 8 particles. The Spanish reading copy is updated.

The notebook also displays an individual 40-particle figure with French labels. The optional `export_thesis_figure` flag is disabled by default. With the existing sibling thesis checkout, enabling it writes `primer_conjunto_40_fr.pdf`, its PNG and `primer_conjunto_40.json` to that repository's figures directory. These artifacts were generated for the authorized thesis layout.

Checks performed:

- The seven IDs, record order, counts and complete particle records match the exact selected entries of `original_40.json`. The selection metadata explicitly leaves chaotic behavior unverified for the reflected potential.
- All pre-existing particle JSON files and the measured field retain their baseline SHA-256 hashes. Other dataset coordinates, colors and IDs are unchanged.
- All four notebook code cells executed directly in one Python namespace using the project virtual environment, with only display capture replacing `plt.show`. The export flag was enabled in memory for this execution and remains disabled in the saved parameters. The four panels contain 40, 84, 7 and 8 points; the individual figure contains exactly 40. The notebook saves two regenerated PNG outputs and passes notebook-format validation.
- The individual PNG was inspected. Exported parameters confirm IDs 1–40, reflected spatial extension, rho 0.30, phase zero and zero trajectory integrations. PDF and PNG exports exist.
- `git diff --check` and `scripts/check_sdd.py` passed. Spanish reading copies are excluded by the existing Git rules.

The thesis change and compilation are recorded in `TFM_redaccion/specs/changes/2026-11-16-tercer-conjunto-7/results.md`. No numerical trajectories were recomputed, and no remote jobs, commits, uploads or deployments were performed. Notebook cells were executed directly because local Jupyter socket creation is unavailable in this sandbox, as established during the preceding task.
