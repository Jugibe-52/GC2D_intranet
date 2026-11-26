# Thesis notebooks

The eight agreed chapter-4 folders live physically in this directory, `GC2D_codigo/notebooks`. Section 4.8 is excluded. The main execution folder contains calculation only; its visualization remains on the reference website.

| Folder | Study | Notebooks |
| --- | --- | --- |
| [04_01_jeux_de_donnees](04_01_jeux_de_donnees/) | Particle datasets | One notebook; four initial-position panels |
| [04_02_execution_principale](04_02_execution_principale/) | Main execution | Calculation only; visualization on the website |
| [04_03_numpy_jax](04_03_numpy_jax/) | NumPy/SciPy and JAX | Calculation and visualization |
| [04_04_cout_calcul](04_04_cout_calcul/) | Computational cost | Calculation and visualization |
| [04_05_evolution_energetique](04_05_evolution_energetique/) | Energy evolution | Calculation and visualization |
| [04_06_correction_projection](04_06_correction_projection/) | Projection correction | Visualization |
| [04_07_similitude_trajectoires](04_07_similitude_trajectoires/) | Trajectory similarity | Visualization |
| [04_09_defauts_geometriques](04_09_defauts_geometriques/) | Geometric defects | Calculation and visualization |

Open notebooks with this repository's `.venv` environment. They locate `GC2D_codigo` and the sibling `TFM_redaccion` from either repository root or their section folder. Canonical notebooks are regular files in the section directories. The former `04_02_bm4_numpy_jax`, `04_03_method_timings`, `04_08_midpoint_geometry` folders and standalone `04_01_01_*.ipynb` entry points have been replaced; the duplicate section folders in the thesis repository have been removed.

The folder names retain the explicitly agreed organization. The thesis text was edited concurrently and currently places computational cost before NumPy/JAX (sections 4.3 and 4.4 respectively); that order differs from these two agreed folder prefixes. This relocation does not change the thesis text or silently interchange the folder names.

Some plotting and energy-study scripts remain in `TFM_redaccion/notebooks` because their paths and hashes belong to saved scientific provenance. Relative links named `sources` or individual `.py` files provide access to those canonical scripts. Data, result archives, run IDs and B2 destinations remain unchanged. `developements/` is outside this reorganization.

Section 4.1 now contains a single viewing notebook. The previous dataset and third-set figure-generation notebooks, including saved outputs, are preserved in `TFM_redaccion/notebooks/archives_ensembles_particules/`. Thesis-local generated notebooks in folders 4.2, 4.5, 4.6 and 4.7 retain their existing ignored status through scoped `.gitignore` files; their Python sources and regeneration commands are documented in each folder. No files were staged and no tracking was forced. Saved outputs were preserved; they are historical execution outputs, not evidence of a new simulation during this move.

Visualization reads saved results and may compute reductions, statistics or plots without repeating integrations. The energy reconstruction subfolder preserves its distinct historical study. Regenerating a notebook replaces its cells and clears stored outputs; it is not required to open the current notebooks. No simulation, transfer, publication or figure export is part of this relocation.
