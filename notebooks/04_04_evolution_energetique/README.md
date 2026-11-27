# 4.4 — Online energy balance in Colab

Published notebooks: [visualisation](https://colab.research.google.com/drive/14kK36hblOqiPnGgDMMTO8KId7gXwd7dH) and [calculation](https://colab.research.google.com/drive/1U_zj1rM87b4gYk-hPJslB3CKr2PE2Ejo). The public viewer data and the calculation kit are verified through anonymous HTTPS. The exact 600524808-byte kit is transported in three SHA256-checked parts; its original potential and numerical sources are preserved.

Validation covers four viewer cells over public HTTPS and a local eight-method, 84-particle, two-cycle calculation, including checkpoint recovery. The latter matches the first two archived cycles bit for bit. No Colab runtime or Modal calculation was executed for this validation.

[visualisation.ipynb](visualisation.ipynb) reads verified public data and reproduces the single four-panel figure retained in French section 4.4. The panels compare Simpson accumulated every two and four fixed steps for the periodic and chaotic candidate groups. It does not repeat trajectory integrations. The separate Hamiltonian-variation figure, broken-axis discussion and obsolete two-panel description have been removed.

[calcul.ipynb](calcul.ipynb) guides personal Modal authentication, an explicit calculation and download of a checked result ZIP to the reader's computer. Its summaries compare both balance residuals. Delta H remains an internal input to the residual, not a separate plotted result. Reader downloads are the agreed exception to the project's default private-B2 persistence policy.

The source study is `main-20261117-rho-div20-online-balance-20261120`: eight methods, 84 particles, normalized rho=0.015, 5000 cycles, 40 fixed steps and ten saved positions per cycle. Diagnostics are saved at each cycle end. The figure analyzes two groups of eight particles and uses 100-cycle RMS blocks; reported global RMS values use all 5000 cycle ends. Quadrature sensitivity is neither a rigorous error bound nor a certificate of trajectory accuracy.

The [sources](sources/) link preserves access to the historical thesis scripts. The numerical calculation script, reference archives and their hash-bound receipts remain unchanged. [reconstruction](reconstruction/) is a distinct historical Hermite/Gauss study and is not the source published for current section 4.4. Previous active notebooks are backed up under `outputs/colab-backups/04_04_evolution_energetique/` before replacement.

## Reproduction

The standalone viewer source is [visualisation.py](visualisation.py). Public-data preparation and focused validation are in [prepare_public_data.py](prepare_public_data.py) and [validate_visualisation.py](validate_visualisation.py). Calculation input preparation and checks are in [prepare_calculation_data.py](prepare_calculation_data.py) and [validate_calculation.py](validate_calculation.py).

From the code repository, rebuild with `python3 notebooks/04_04_evolution_energetique/build_visualisation.py` or `python3 notebooks/04_04_evolution_energetique/build_calculation.py`. The historical thesis `crear_cuadernos.py` delegates to these builders so regeneration cannot restore removed cells. Preserve meaningful saved outputs before rebuilding. Notebook prose is English; figure text is French. Their existing ignored Git status is unchanged.

Actual tests, publication links and remaining limitations are recorded in [results.md](../../specs/changes/2026-11-27-colab-energy-44/results.md). A local check or saved output is not evidence that a Colab runtime or remote Modal calculation has executed.
