+++
schema_version = 2
kind = "change-tasks"
id = "2026-10-18-hdf5-frequency-units"
spec_revision = 2
plan_revision = 1
sources = ["src/potential/load.py", "src/potential/potential.py", "src/potential/_evaluation.py", "src/potential/jax_potential.py", "src/dynamics/gc.py", "src/studies/historical_h5.py", "src/studies/dimensional_h5_midpoint.py", "src/studies/poincare_rho_sweep.py", "src/studies/h5_provenance.py", "src/studies/_potential_snapshot.py", "src/studies/poincare_probe.py", "src/diagnostics/persistence.py", "tests/test_h5_rescaling.py", "tests/test_gc2d_h5_potential.py", "tests/test_potential_snapshot.py", "tests/test_solution_persistence.py"]
documents = ["specs/current/potential-import.md", "docs/dynamics/gc2d-h5-import.md", "specs/changes/2026-10-18-hdf5-independent-rescaling/spec.md", "specs/changes/2026-10-18-hdf5-independent-rescaling/plan.md", "specs/changes/2026-10-18-hdf5-independent-rescaling/tasks.md"]
open_questions = []

[[tasks]]
id = "T-001"
description = "Record review and acceptance of spec revision 2 and plan revision 1 before runtime implementation; Q1 is already resolved as cyclic public frequencies."
depends_on = []
done = true

[[tasks]]
id = "T-002"
description = "Inventory affected live consumers and capture historical fields, time arrays, metadata, fingerprints and reduced trajectories from the current working tree."
depends_on = ["T-001"]
done = true

[[tasks]]
id = "T-003"
description = "Add independent FFT-produced HDF5 fixtures and record the failing physical-phase/period baseline before correcting implementation."
depends_on = ["T-002"]
done = true

[[tasks]]
id = "T-004"
description = "Implement raw cyclic frequencies, physical T=1/f_ref and consistent NumPy/JAX/direct/random evaluation and derivative conventions."
depends_on = ["T-003"]
done = true

[[tasks]]
id = "T-005"
description = "Adapt processed-data readers and writers with separate source, runtime and time conventions; verify original hashes before conversion."
depends_on = ["T-004"]
done = true

[[tasks]]
id = "T-006"
description = "Preserve historical study equations, radial/cycle_time amplitudes, custom periods, source-axis sampling and viewer results."
depends_on = ["T-004", "T-005"]
done = true

[[tasks]]
id = "T-007"
description = "Migrate remaining Python, generators, examples and live notebooks, recording covered consumers and justified frozen exclusions."
depends_on = ["T-006"]
done = true

[[tasks]]
id = "T-008"
description = "Update current POT contract, affected physics/model/JAX documentation and confirmed memory, with full Spanish Markdown copies."
depends_on = ["T-007"]
done = true

[[tasks]]
id = "T-009"
description = "Run focused and full validation, record numerical evidence and limitations, reconcile all records and close only after every acceptance criterion passes."
depends_on = ["T-008"]
done = true

[[checks]]
id = "V-001"
criteria = ["AC-001", "AC-002"]
procedure = "Create a small deterministic FFT-derived HDF5 fixture as specified in plan.md; retain failure before the fix. Run .venv/bin/python -m unittest discover -s tests -p test_h5_frequency_units.py -v after creating that planned module. Compare original time samples, period and analytic first/second/mixed derivatives; record errors and source producer SHA-256."
tests = ["tests/test_h5_frequency_units.py", "tests/test_gc2d_h5_potential.py"]
result = "passed"
evidence = "Independent FFT fixture failed before correction (max absolute signal error 0.24722087); passes after correction at 1.665e-16 maximum error. Positive bins 2 and 6, N=64, dt=0.037 s, asymmetric spatial data and nonzero origins; first/second/mixed derivatives pass 1e-10 tolerances. Producer SHA-256 recorded in spec.md."
checked_on = "2026-10-18"

[[checks]]
id = "V-002"
criteria = ["AC-003", "AC-007"]
procedure = "Run .venv/bin/python -m unittest discover -s tests -p test_h5_rescaling.py -v with all flag combinations, non-unit B, nonzero rho, first-selected reference and mean-only cases. Audit dimensional trajectories against DOP853 and Radau references with tighter tolerances/reduced steps; record maximum discrepancy/L and convergence floor."
tests = ["tests/test_h5_rescaling.py"]
result = "passed"
evidence = "Final test_h5*.py run: 13 tests pass. All four rescaling maps, first-selected reference, mean-only policy, B=1.5 and rho in (0,0.025). Max transformed position/L=1.247e-15; DOP853/Radau floor/L=5.506e-14; analytic time-integrated shear error/L=1.443e-15. Two reference cycles, 41 samples, DOP853 rtol/atol 1e-11/1e-13 max step T/100; Radau 1e-12/1e-14 max step T/200."
checked_on = "2026-10-18"

[[checks]]
id = "V-003"
criteria = ["AC-005"]
procedure = "Run .venv/bin/python -m unittest discover -s tests -p test_potential_snapshot.py -v and the same command with test_solution_persistence.py. Add cases for known unmarked cyclic, explicit angular and new cyclic representations, unknown tags, double conversion, source provenance and original hash validation; include pickle/probe readers."
tests = ["tests/test_potential_snapshot.py", "tests/test_solution_persistence.py"]
result = "passed"
evidence = "12 solution persistence tests and 3 snapshot tests pass, plus the final H5 tests covering pickle angular/cyclic conversion, legacy metadata defaults and probe archive conventions/checksums. Original archive checksums remain verified before conversion; known unmarked, explicit angular and new cyclic data retain their actual phase, and unknown conventions fail."
checked_on = "2026-10-18"

[[checks]]
id = "V-004"
criteria = ["AC-004"]
procedure = "Run focused test_h5_study_selection.py, test_poincare_radial_cycle_time.py, test_poincare_rho_sweep.py and test_poincare_gap_viewers.py via unittest discovery. Compare both historical conventions and custom periods against T-002 fields, trajectories, clocks and metadata. Reproduce native and 32x32 fingerprints using the previous change procedure, then run a disposable reduced folded-radial viewer notebook smoke."
tests = ["tests/test_h5_study_selection.py", "tests/test_poincare_radial_cycle_time.py", "tests/test_poincare_rho_sweep.py", "tests/test_poincare_gap_viewers.py"]
result = "passed"
evidence = "Measured PHI_2 native and 32x32 fingerprints exactly match captured working-tree baseline; fields and first/second time derivatives, custom clock r=14, legacy clock, radial/cycle_time effective fields and short trajectories have max difference 0.0. 19 historical HDF5 tests and 55 Poincare tests pass. Actual folded-radial notebook export call passed using four methods, two particles, one cycle, 32 steps and temporary HTML."
checked_on = "2026-10-18"

[[checks]]
id = "V-005"
criteria = ["AC-002", "AC-006"]
procedure = "Run focused test_jax_potential.py, test_core.py and test_input_validation_contracts.py via unittest discovery. Cover direct/random construction, temporal/mixed derivatives, subclass behavior, spectral factor 2, B ownership, orientation and filtering. Compare available NumPy/JAX implementations in float64 and document unavailable-backend skips."
tests = ["tests/test_jax_potential.py", "tests/test_core.py", "tests/test_input_validation_contracts.py"]
result = "passed"
evidence = "Direct/random construction, subclasses, NumPy/JAX values and derivatives, GC/FC dynamics and backend validation passed in the full run. One stale angular expectation in test_potential_execution was corrected to omega=2*pi*f; the final affected module passed all 17 tests. Runtime code did not change after the full run. GPU-only cases are unavailable and remain explicitly skipped."
checked_on = "2026-10-18"

[[checks]]
id = "V-006"
criteria = ["AC-006"]
procedure = "Use rg --hidden --no-ignore to audit load/from_h5 calls, frequencies, angular/cyclic tags, exp phases and 2*pi reference periods within src, scripts, examples, tests and notebooks. Inspect notebook JSON code cells and generated-code strings. Record each live migration or justified exclusion. Run only affected setup/import/reconstruction cells and a short integration in disposable copies; record paths and reduced parameters."
tests = ["tests/test_potential_execution.py"]
result = "passed"
evidence = "Audited 98 candidate paths and all notebook code-cell frequency references, including ignored files and Python wrappers. Updated seven notebooks (eight code cells and one Markdown cell), parsed all changed code and 12 generator code blocks. Executed 27 notebook load/composition expressions with a synthetic 16x16 field, two particles, one cycle, two rho values and two saved frames. Executed actual folded-radial viewer export against measured source with reduced live runs. Rechecked exclusions: 80 conflict copies, two superseded notebooks and ten frozen recorded executions; frozen snapshot loaders retain their own verified evaluator."
checked_on = "2026-10-18"

[[checks]]
id = "V-007"
criteria = ["AC-001", "AC-002", "AC-003", "AC-004", "AC-005", "AC-006", "AC-007"]
procedure = "After focused checks pass, run .venv/bin/python -m mypy src; MPLBACKEND=Agg .venv/bin/python -m unittest discover -s tests -v; MPLBACKEND=Agg .venv/bin/python examples/gc_orbit.py; MPLBACKEND=Agg .venv/bin/python examples/projected_abba.py; .venv/bin/python -m build; .venv/bin/python scripts/check_sdd.py. Review contract/model documentation and Spanish parity against implemented behavior. Record actual outputs and numerical evidence before verified status."
tests = []
result = "passed"
evidence = "Full headless discovery executed 562 tests in 347.883 s: 558 passed, three GPU-only skips and one stale test expectation failed. Corrected only that expectation (omega=2*pi*f) and reran its entire module: 17/17 passed in 4.316 s; reused unchanged full-run passes rather than repeating expensive JAX compilation. The final test_h5*.py run passed 13/13, including an additional probe-archive regression added after initial discovery. No unresolved test failures remain. Strict mypy passed all 256 source files. Both smoke examples passed (20 GC steps; four reduced_multiplier ABBA steps). Final source/wheel build passed; SDD checker passed 18 records/12 requirements and git diff --check passed. Spanish code blocks match originals; MEMORY has 185 lines."
checked_on = "2026-10-18"
+++

# HDF5 frequency units: tasks and evidence

## Execution

Preparation on 2026-10-18 creates a corrective draft after the previous
`2026-10-18-hdf5-independent-rescaling` change. The user confirmed that public
`Potential.frequencies` keeps Hz without temporal scaling. Specification revision
2 resolves that choice; this plan and task list have been reconciled with it.
The user accepted implementation on 2026-10-18 through `$sdd implementa`.
Runtime migration and all acceptance checks are complete. Full-run passes are
reused after a test-only expectation correction and successful focused rerun;
the original full-run failure remains recorded in V-007.

### Ordered work

| Task | Deliverable |
| --- | --- |
| T-001 | Single review of spec and plan, with tasks available; record actual acceptance and implementation scope. |
| T-002 | Current working-tree baseline and complete migration inventory, preserving existing uncommitted work. |
| T-003 | Independent FFT fixture, initial failing phase/period result and analytic derivative reference. |
| T-004 | Physical loader and all evaluation paths using cyclic public rates, f*T, T=1/f_ref and T/s**2. |
| T-005 | Explicit archive decoding and separate source/runtime/time provenance, without rewriting stored results. |
| T-006 | Historical radial and cycle_time reproduction, including custom clocks and exact sample fingerprints. |
| T-007 | All supported consumer migrations and recorded reduced notebook execution. |
| T-008 | Corrected current contract, affected documentation and memory; complete Spanish reading copies. |
| T-009 | Observed numerical evidence, required checks and final lifecycle reconciliation. |

### Reproducible checks

The metadata above owns executable procedures, criterion mappings and outcomes.
`tests/test_h5_frequency_units.py` now contains the independent FFT and archive
regressions and is included in the evidence references. If the environment lacks the
recorded .venv interpreter, identify the configured project interpreter without
installing dependencies silently. Preserve full failing output when a criterion
fails and explain the correction before rerunning that check.

For V-001, use a temporary test HDF5 with the producer's fft/N and fftfreq
semantics, not the external script's unavailable data imports. The producer
SHA-256 is recorded in spec.md. FFT time zero is the first sample. Include raw
source values exactly and show that one physical reference period is 1/f_ref.

For V-002, compare positions in common physical coordinates at common physical
times; record both the transformed-path difference and independent integration
error, including the chosen solver tolerances, sample times and refinement.
The prior change's passing rescaling tests do not establish source-unit truth.

For V-004/V-006, use
`notebooks/developements/poincare_section/four_methods_star_40_rho_sweep_folded_radial_potential/visualisation.ipynb`
as the target local viewer. Reconstruct its potential and run a small representative
integration using at most two particles, 32 integration steps and two frames where
those parameters apply; preserve method/configuration paths and record any necessary
adjustment. Load archived output only as needed for comparison. Never save reduced
results into canonical notebooks or regenerate production sweeps. Reuse existing
local source data; no download or publication is part of this change.

### Preparation evidence

Read-only investigation identified source FFT semantics and the historical radial
amplitude cancellation. The only changes authorized by this prepare operation are
these proposal records and their Spanish copies. The SDD checker validates document
structure and references; its preparation result is reported separately below and
does not complete any runtime acceptance check.

Preparation validation on 2026-10-18: `python3 scripts/check_sdd.py` passed
(18 records, 12 requirements). SHA-256 comparison of 53 pre-existing files
confirmed no changes to that captured baseline. The three Spanish reading copies
are ignored by Git. The working branch remains `main`. No runtime tests were
executed for this prepare operation.

## Outcome and limitations

Implemented and verified on 2026-10-18. Q1 is resolved. All tasks and criteria
have evidence; V-007 explicitly preserves the initial full-run failure and its
focused resolution. The full command was not repeated after that test-only edit. The source loader's
imported time routine and original time series were not supplied; seconds follow
the producer's declared units. Historical clocks and saved production data remain
unchanged. Earlier passes are not reused as proof of source units.

### Migrated notebooks

- `notebooks/developements/potential/study_gc2d_h5_modes.ipynb` — 0, 4, 6.
- `notebooks/developements/potential/visualize_gc2d_h5_potential.ipynb` — 4.
- `notebooks/developements/recurrences/study_20_horizontal_bm4_recurrences/particle_2_bm4_160_periodicity.ipynb` — 4.
- `notebooks/developements/recurrences/study_20_horizontal_bm4_recurrences/particle_2_return_precision.ipynb` — 4.
- `notebooks/developements/energy/BM4_RK4/calculation.ipynb` — 5.
- `notebooks/experiments/comparison/compare_abba4_r6_r8_configurations_aws_executed.ipynb` — 6.
- `notebooks/experiments/comparison/compare_abba4_r6_r8_configurations.ipynb` — 5.

### Reduced notebook reconstruction checks

- `notebooks/developements/study_bm4_mu_vs_displacement_2p.ipynb` — 4.
- `notebooks/developements/compare_bm4_with_sdirk4_two_particles.ipynb` — 4.
- `notebooks/developements/bm4_simple_inspection.ipynb` — 3.
- `notebooks/developements/study_100_parallel_bm4_domain_escapes.ipynb` — 4.
- `notebooks/developements/accuracy/step_ratio_abba4_bm4.ipynb` — 4.
- `notebooks/developements/accuracy/compare_three_order4_r4_models.ipynb` — 4.
- `notebooks/developements/recurrences/study_trajectory_10_recurrence_bm4_tighter_newton.ipynb` — 4.
- `notebooks/developements/recurrences/study_trajectory_10_four_cycle_segments_bm4.ipynb` — 5.
- `notebooks/developements/recurrences/study_trajectory_10_recurrence_bm4.ipynb` — 4.
- `notebooks/developements/recurrences/study_40_parallel_bm4_recurrences.ipynb` — 4.
- `notebooks/developements/recurrences/study_near_center_particle_recurrence_bm4.ipynb` — 4.
- `notebooks/developements/bm4_h5_simple_inspection/calculation.ipynb` — 3.
- `notebooks/developements/bm4_projection_comparison/comparison.ipynb` — 3.
- `notebooks/developements/potential/study_gc2d_h5_modes.ipynb` — 6.
- `notebooks/developements/potential/visualize_gc2d_h5_potential.ipynb` — 4.
- `notebooks/developements/recurrences/study_20_horizontal_bm4_recurrences/particle_2_bm4_160_periodicity.ipynb` — 4.
- `notebooks/developements/recurrences/study_20_horizontal_bm4_recurrences/particle_2_return_precision.ipynb` — 4.
- `notebooks/developements/recurrences/study_20_horizontal_bm4_recurrences/calculation.ipynb` — 3.
- `notebooks/developements/recurrences/study_20_horizontal_bm4_recurrences/visualization.ipynb` — 3.
- `notebooks/developements/accuracy/h5_three_radial_reference_2x_precision/h5_three_radial_reference_2x_precision.ipynb` — 1.
- `notebooks/developements/accuracy/h5_three_radial_reference/h5_three_radial_reference.ipynb` — 1.
- `notebooks/developements/energy/BM4_RK4/calculation.ipynb` — 5.
- `notebooks/developements/energy/compare_five_order4_models_with_sdirk_rk4/calculation.ipynb` — 3.
- `notebooks/developements/energy/compare_five_order4_models_with_sdirk_rk4/visualization.ipynb` — 3.
- `notebooks/developements/energy/energy_balance_dop853_bm4_gl4_rk4_horizontal_3p_200steps_10saves/calculation.ipynb` — 3.
- `notebooks/experiments/comparison/compare_abba4_r6_r8_configurations_aws_executed.ipynb` — 6.
- `notebooks/experiments/comparison/compare_abba4_r6_r8_configurations.ipynb` — 5.

### Historical reproduction evidence

Before modification, captured current working-tree PHI_2 evaluations at t=0.371
(dt=0,1,2), native and 32x32 sample arrays, custom reference r=14 and recorded
clocks. The native hash is
`ba7fd7e8b846e90d7f1668c123a6052666e935d0076439f4711ea6247b8ef8e3`;
the 32x32 hash is
`a851e116bb6ebe3cf121bd47f823b46e1bc0bff2dceb555f4366e308dd98e273`.
Both remain exact, as do all captured field/derivative values. For each of radial
and cycle_time, two particles at rho_hat=0.3 were integrated over [0,0.01] at 33
saved times using DOP853, rtol=1e-11, atol=1e-13 and max_step=0.01/32. Maximum
before/after trajectory, effective-field and clock differences are 0.0.

The viewer smoke executes the actual export expression from cell 4 of the named
notebook with measured PHI_2, one rho, all four live numerical methods, two
particles, one cycle and 32 steps per cycle. Only the temporary HTML destination
and reduced controls are substituted; no production archives are rewritten.
It uses two saved trajectory frames and three field phases: phase_steps must be
at least two in the existing sampler. Notebook composition checks use synthetic
16x16 data. Frozen snapshot loaders and their Python wrappers use their own
verified code and are deliberately unchanged. The generator's 12 code blocks
and eight edited notebook code cells parse successfully. Superseded diagnostic
outputs in edited cells were cleared; production trajectory files were not changed.

Prepared state is retained above as history. Current POT contracts, physics/JAX
and BM4 documentation, README and memory now agree with cyclic frequencies;
Spanish reading copies were updated alongside them. The previous verified
change has a dated successor note rather than rewritten historical evidence.
