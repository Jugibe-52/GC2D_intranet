# 4.5 — Projection correction in Colab

Published notebooks: [visualization](https://colab.research.google.com/drive/1CIgaOG-slC8KnXpg0gGbzdvbwX_keNyh) and [calculation](https://colab.research.google.com/drive/1dlmACJ_X4mgxi73jNzgoiD6VG926mh_Q).

[visualisation.ipynb](visualisation.ipynb) reproduces the two figures in French section 4.5 from an embedded, checksum-verified reference snapshot. It needs no local repository, private credentials or trajectory integration. The original scientific results remain in B2; embedding the small statistical summary is a deliberate portability exception, avoiding a new public-data upload while browser attachment is unavailable. Optional public-HTTPS loading is not evidence that the new viewer bundle has been uploaded.

[calcul.ipynb](calcul.ipynb) prepares a fresh reproduction using the reader's personal Modal account and offers verified result ZIP downloads. Launch is explicitly disabled by default. Each case has its own receipt, so confirmed calls can be recovered without submitting them again. Reader downloads follow the existing exception to private-B2 result storage.

Both notebooks identify `studies/projection_correction` and reference run `main-20261117-rho-div20`: ABBA2Implicit and BM4Implicit, eleven normalized radii `i/400`, 84 original particles, 5000 cycles, 40 fixed steps per cycle and ten saved samples per cycle. The reference contains 21 complete cases; ABBA2 at radius zero is unavailable after nonlinear convergence failure. Fresh calculation status must not be substituted for that historical availability record.

The correction statistics use the Euclidean norm of the two signed multiplier components over 84 particles and 50000 sampled steps. The Euclidean maximum over every integration step cannot be recovered from these archives. Newton means and maxima use all 200000 joint particle steps. Correction size is neither a nonlinear residual nor a trajectory error.

## Reproduction

Build the viewer with `python3 notebooks/04_05_correction_projection/build_visualisation.py` and the calculation with `python3 notebooks/04_05_correction_projection/build_calculation.py`, from the code root. The historical thesis `correccion_proyeccion_graficas.py --notebook` delegates to the portable viewer builder; its ordinary historical figure export is retained. The original extraction script and scientific archives remain unchanged. The previous viewer is backed up under `outputs/colab-backups/04_05_correction_projection/`.

The calculation reuses the pinned public section-4.4 kit containing the original physical potential and frozen main-campaign numerical sources. Its exact initial states and time plan also match the 21 available projection cases. Numerical validation is reduced; it does not rerun the 5000-cycle campaign. Notebook prose is English and figure text French. The existing ignored Git status is retained.

Publication links, actual validation results and limitations are recorded in [results.md](../../specs/changes/2026-11-27-colab-projection-45/results.md). Local checks do not demonstrate execution in a Colab runtime or on Modal.
