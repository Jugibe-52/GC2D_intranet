# 04_04 — Method runtime

- [calcul.ipynb](calcul.ipynb) executes or recovers the eight-method JAX study and preserves individual timings, results, parameters and provenance.
- [visualisation.ipynb](visualisation.ipynb) loads saved results and exports the runtime comparison figure without repeating integrations.

This folder holds the canonical notebooks themselves, including their saved outputs.
Use the Python kernel from `GC2D_codigo/.venv`; the first cell finds the code repository from this section folder or its parents.
Notebook prose is English, and thesis figure text is French.
The code and thesis repositories must remain siblings named `GC2D_codigo` and `TFM_redaccion` for shared thesis inputs and figure exports.

Both notebooks use prefix `studies/method_timings` and run `jax-20261118-rho0015-c500`.
The scientific destination remains `gc2d_data:gc2d-notebooks-data/studies/method_timings/jax-20261118-rho0015-c500`; `GC2D_codigo/outputs/` holds local recovery copies.
This reorganization moves no results and starts no calculation or transfer.
