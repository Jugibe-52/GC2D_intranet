# Periodic particle selection

The fourth dataset, `data/particles/periodic_8.json`, contains independent initial conditions for the reflected measured potential at rho = 0.30. It uses the same `PHI_2.h5` fingerprint, constant field 0, temporal mode 15, B = 1.5 T and physical-cycle normalization as the main campaign. Positions are unwrapped; the spatial period is two base-cell widths. Closed orbits must have zero net displacement, not merely agree after folding.

## Procedure

The preparation script verifies every reference archive hash and extracts integer-cycle returns from the 84-particle Gauss–Legendre run. Near returns inside the displayed base square provide seeds; subsequent integration is unrestricted in the reflected extension. Candidates are not assumed periodic because they look regular.

Batched damped Newton corrections solve `P_q(x) - x = 0`, where `P_q` advances q forcing cycles. Centered coordinate differences of size 2e-6 approximate its Jacobian; corrections are capped at 0.025 cell widths and reduced by a finite line search. Shooting tolerance is 2e-11. Every function evaluation uses a complete fixed-step JAX integration through the public simulation runner. Search uses RK4 with 80 steps per cycle.

Validation continues each root with Gauss–Legendre at 80, 160 and 320 steps per cycle, extending to 640 or 1280 when necessary. Newton stage tolerances are absolute 1e-12 and relative 1e-11, with at most 40 corrections. The final seed must change by less than 1e-7 between successive grids. Gauss–Legendre and independent RK4 integrations at twice the final resolution must each close within 1e-7 cell widths. Five consecutive full periods must close within 1e-6. These are measured numerical thresholds, not rigorous trajectory error bounds or existence proofs.

All earlier integer-cycle returns must remain more than 1e-4 from the initial point, establishing the minimum period at the resolved numerical scale. The cycle-to-cycle excursion must exceed 0.002, excluding stationary points for the periods studied here. Pairwise comparison of every saved return excludes phase duplicates and period-two spatial translations. Periodicity is verified only at rho = 0.30, not throughout the rho sweep.

## Reproduction

From the repository root, with the main reference archive available locally:

```bash
.venv/bin/python scripts/select_periodic_particles.py prepare
.venv/bin/modal run examples/modal_periodic_particles_app.py --periods 5,6,7,8,9,10,11,12,13,14,15,17
.venv/bin/modal run examples/modal_periodic_particles_app.py --mode verify --periods 5,6,7,8,9,10,11,12
.venv/bin/python scripts/select_periodic_particles.py finalize
```

Each job saves its Modal call ID before waiting and retains its result under `build/periodic_search_rho030`. Existing receipts recover the same call rather than submitting it again. Failed finite-resolution checks can continue from their last grid with `--mode verify --periods <list> --attempt 2`; accepted candidates are not repeated. Earlier attempts and their costs remain recorded. A fresh run of the current verifier continues finer grids within the original job when repeated closure needs improvement.

The finalizer refuses an incomplete or duplicate selection. The final JSON retains coordinates, minimum periods, refinements, cross-method closure, repeated returns, reference fingerprints, call IDs and measured resources. Compute charges are estimates for completed worker intervals; they exclude container startup, idle time, transfer and work lost to preemption, and are not invoices.

The notebook reads the final JSON locally. Neither this dataset nor the viewing notebook is uploaded to B2; both remain under Git. The thesis and the main campaign are unchanged.

## Selected period-11 candidate

The recorded run selected the second period-11 root after the first failed repeated closure. To reproduce that additional check, run `.venv/bin/modal run examples/modal_periodic_particles_app.py --mode verify --periods 11 --candidate-index 1` before finalization. The finalizer considers all accepted candidates for each selected period and checks pairwise distinctness again. Measured results are recorded in [the execution report](../../specs/changes/2026-11-15-periodic-particle-set/results.md).
