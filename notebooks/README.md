# Thesis notebooks

The eight study folders live physically in this directory, `GC2D_codigo/notebooks`, and follow the current French thesis: sections 4.1–4.6 and 4.8, plus appendix C. Section 4.7, the visual analysis of Poincaré sections, has no separate folder. The main execution folder contains calculation only; its visualization remains on the reference website.

| Folder | Study | Notebooks |
| --- | --- | --- |
| [04_01_execution_principale](04_01_execution_principale/) | Main execution | Calculation only; visualization on the website |
| [04_02_cout_calcul](04_02_cout_calcul/) | Computational cost | Calculation and visualization |
| [04_03_numpy_jax](04_03_numpy_jax/) | NumPy/SciPy and JAX | Calculation and visualization |
| [04_04_evolution_energetique](04_04_evolution_energetique/) | Energy evolution | Calculation and visualization |
| [04_05_correction_projection](04_05_correction_projection/) | Projection correction | Calculation and visualization |
| [04_06_similitude_trajectoires](04_06_similitude_trajectoires/) | Trajectory similarity | Visualization |
| [04_08_defauts_geometriques](04_08_defauts_geometriques/) | Geometric defects | Calculation and visualization |
| [annexe_C_ensembles_particules](annexe_C_ensembles_particules/) | Particle datasets | One notebook; four initial-position panels |

Open notebooks with this repository's `.venv` environment. They locate `GC2D_codigo` and the sibling `TFM_redaccion` from either repository root or their section folder. Canonical notebooks are regular files in the section directories. The former `04_02_bm4_numpy_jax`, `04_03_method_timings`, `04_08_midpoint_geometry` folders and standalone `04_01_01_*.ipynb` entry points have been replaced; the duplicate section folders in the thesis repository have been removed.

Folder prefixes match the current French table of contents. Particle datasets belong to appendix C; main execution is section 4.1, computational cost 4.2 and NumPy/JAX 4.3. The remaining prefixes follow their current section numbers.

Some plotting and energy-study scripts remain in `TFM_redaccion/notebooks` because their paths and hashes belong to saved scientific provenance. Relative links named `sources` or individual `.py` files provide access to those canonical scripts. Data, result archives, run IDs and B2 destinations remain unchanged. `developements/` is outside this reorganization.

Appendix C contains a single viewing notebook. The previous dataset and third-set figure-generation notebooks, including saved outputs, are preserved in `TFM_redaccion/notebooks/archives_ensembles_particules/`. Thesis-local generated notebooks in folders 4.1, 4.4, 4.5 and 4.6 retain their existing ignored status through scoped `.gitignore` files; their Python sources and regeneration commands are documented in each folder. No files were staged and no tracking was forced. Saved outputs were preserved; they are historical execution outputs, not evidence of a new simulation during this move.

Visualization reads saved results and may compute reductions, statistics or plots without repeating integrations. The energy reconstruction subfolder preserves its distinct historical study. Regenerating a notebook replaces its cells and clears stored outputs; it is not required to open the current notebooks. No simulation, transfer, publication or figure export is part of this relocation.
