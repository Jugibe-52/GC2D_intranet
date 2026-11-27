# 4.1 — Main execution

This section contains only a calculation notebook: visualization is already available on the [reference campaign website](https://gc2d-main-reflected-rho-div20.pages.dev/).

- `calcul.ipynb` inspects existing results and can launch a separate campaign after explicit activation.
- `calcul.py` contains the checks, launcher and cells used to regenerate the notebook.
- The scientific selection remains in [`../../../TFM_redaccion/datos/campana_referencia.json`](../../../TFM_redaccion/datos/campana_referencia.json).

## Normal use

Open `calcul.ipynb` from this folder, the code repository root or the sibling thesis repository root and run the cells with `EXECUTER_NOUVELLE_CAMPAGNE = False`.
This mode checks records and inventories local manifests without writes, integrations or transfers.
It does not repeat verification of the B2 archive bytes.
Missing local copies are reported without automatic downloads.
The same inspection is available from the code repository root:

```bash
python3 notebooks/04_01_execution_principale/calcul.py
```

The reference `main-20261117-rho-div20` retains 86 results and two convergence failures: `ABBA2Implicit` and `ImplicitMidpoint` at `rho = 0`.
Its numerical status remains `failed`; unavailable cases are neither removed nor treated as successful.

## New campaign

Choose a distinct, stable `RUN_ID` and change `EXECUTER_NOUVELLE_CAMPAGNE` only when a new run is intended.
The cell displays the commands even when execution is disabled.
When enabled, the launcher prepares a local campaign, deploys the current worker and runs the cases on Modal/JAX CPU float64, preserving results in the configured bucket.
Deployment updates the Modal application `gc2d-main-reflected`: launch this step only when no other campaign is active on that application.

Preparation calls `prepare_campaign` with the configuration, method list and eleven radii read from the selection file.
It explicitly imposes `rho = i/400`, `i = 0,...,10`, without using the historical generic code grid.
The 84 positions, IDs and colors come from `TFM_redaccion/datos/conjuntos_particulas_rho030.json`; this historical filename does not select the simulation radius.
Each case uses 5000 cycles, 40 fixed steps per cycle and ten saved samples per cycle.
Coordinates remain unwrapped; the spatial extension has period 2.
Every scheme evaluates the same physical Hamiltonian directly, without temporal quadrature.

The proposed interpreter is `GC2D_codigo/.venv/bin/python`; `PYTHON_GC2D` can select another configured interpreter.
Launching requires the GC2D/JAX/Modal dependencies, Modal authentication and existing rclone access.
This notebook installs no programs and changes no credentials.
New archives are stored under:

```text
GC2D_codigo/notebooks/developements/main_reflected_campaign/<RUN_ID>/
gc2d_data:gc2d-notebooks-data/developements/main_reflected_campaign/<RUN_ID>
```

The local copy supports collection and resumption; it does not replace the B2 destination.
Current code may differ from the frozen reference sources: its fingerprints, parameters, Modal receipts and per-case measurements are recorded separately.
Transfers are excluded from worker calculation timings.
Numerical or transfer failures remain explicit and stop the process.
To resume a campaign, retain its `RUN_ID`, sources and receipts; do not remove receipts to force submission.
No command exports or publishes the website.
The reference is protected against preparation or execution through this entry point.

The [code protocol](../../docs/studies/main-reflected-campaign.md) describes the archives, diagnostics and resumption of runs.

## Regeneration and verification

After changing cells in `calcul.py`, regenerate the notebook from the code repository root:

```bash
python3 notebooks/04_01_execution_principale/calcul.py --notebook
```

From the thesis repository root, prefix the script path with `../GC2D_codigo/`; from this section, use `python3 calcul.py --notebook`.
Regeneration writes only `calcul.ipynb` beside its source in this section.
The generated notebook preserves its prior local, ignored status through this folder's `.gitignore`; `calcul.py` remains its reproducible source.
Inspection and preview use only the Python standard library.
Validation of this relocation covers the notebook structure, read-only execution, paths from both repository roots and the section folder, and command construction.
No simulation, publication or remote operation is part of this reorganization.
