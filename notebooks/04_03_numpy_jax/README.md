# 04_03 — NumPy/SciPy and JAX comparison

- [calcul.ipynb](calcul.ipynb) prepares, executes or recovers the BM4Implicit, GaussLegendre4 and RK2 comparison, then preserves individual timings, trajectories, parameters and provenance.
- [visualisation.ipynb](visualisation.ipynb) loads saved results and produces the runtime and trajectory-agreement figures without repeating integrations.

This folder holds the canonical notebooks themselves, including their saved outputs.
Use the Python kernel from `GC2D_codigo/.venv`; the first cell finds the code repository from this section folder or its parents.
Notebook prose is English, and thesis figure text is French.
The code and thesis repositories must remain siblings named `GC2D_codigo` and `TFM_redaccion` for shared thesis inputs and figure exports.

Both notebooks use prefix `studies/bm4_numpy_jax` and run `methods-20261119-union16-rho0015-c10-30-90`.
The scientific destination remains `gc2d_data:gc2d-notebooks-data/studies/bm4_numpy_jax/methods-20261119-union16-rho0015-c10-30-90`; `GC2D_codigo/outputs/` holds local recovery copies.
This reorganization moves no results and starts no calculation or transfer.
