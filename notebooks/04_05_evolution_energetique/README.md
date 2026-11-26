# 4.5 — Energy evolution

The regular files [calcul.ipynb](calcul.ipynb) and [visualisation.ipynb](visualisation.ipynb) use `main-20261117-rho-div20-online-balance-20261120`, stored under `studies/energy_balance_online/` in `gc2d_data:gc2d-notebooks-data`. Calculation verifies and reuses completed results, or computes missing cases; visualization validates saved files and plots Hamiltonian variation and balance residual without integrating trajectories.

The [sources](sources/) link points to the canonical thesis scripts, whose paths and fingerprints stay unchanged. Local binaries and receipts remain linked at their existing locations. The notebooks retain their recorded outputs; these do not constitute a new run after relocation.

[reconstruction](reconstruction/) contains the two notebooks for the distinct historical study `main-20261117-rho-div20-balance-20261120`. Its calculation source retains the exact SHA-256 recorded in the scientific summary.

From the code repository root, regenerate notebooks with:

```bash
.venv/bin/python -B ../TFM_redaccion/notebooks/04_04_02_balance_energetico_en_linea/crear_cuadernos.py
.venv/bin/python -B ../TFM_redaccion/notebooks/04_04_02_balance_energetico/crear_cuadernos.py
```

Both commands write into this section (the historical pair under `reconstruction/`) and clear notebook outputs. The generated notebooks retain their prior ignored status via the local `.gitignore`. Existing scientific explanations and recorded outputs are preserved from the source studies. Open notebooks from this folder, the code root or the sibling thesis root with the GC2D environment.
