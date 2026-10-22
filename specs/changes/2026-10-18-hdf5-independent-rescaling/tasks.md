+++
schema_version = 2
kind = "change-tasks"
id = "2026-10-18-hdf5-independent-rescaling"
spec_revision = 2
plan_revision = 3
sources = ["src/potential/potential.py", "src/potential/load.py", "src/potential/_evaluation.py", "src/potential/jax_potential.py", "src/dynamics/gc.py", "src/dynamics/fc.py", "src/studies/dimensional_h5_midpoint.py", "src/studies/poincare_rho_sweep.py", "tests/test_gc2d_h5_potential.py", "tests/test_potential_snapshot.py", "src/potential/grid.py", "src/studies/h5_provenance.py", "src/studies/_potential_snapshot.py", "src/diagnostics/io/persistence.py", "src/studies/poincare_radial_cycle_time.py", "src/studies/poincare_probe.py", "src/studies/reference_trajectory.py", "src/visualization/poincare_method_comparison.py", "tests/test_h5_study_selection.py", "tests/test_solution_persistence.py", "tests/test_jax_potential.py", "tests/test_poincare_rho_sweep.py"]
documents = ["specs/changes/2026-10-18-hdf5-independent-rescaling/spec.md", "specs/changes/2026-10-18-hdf5-independent-rescaling/plan.md"]
open_questions = []

[[tasks]]
id = "T-001"
description = "Resolve the remaining radial-study physics choice, reconcile its accepted meaning and record implementation scope before code changes."
depends_on = []
done = true

[[tasks]]
id = "T-002"
description = "Inventory every active loader call and its physical/time convention, including aliases, forwarded kwargs, scripts, documentation and ignored notebook code cells; record historical exclusions and the project Python environment."
depends_on = ["T-001"]
done = true

[[tasks]]
id = "T-003"
description = "Extend independent square HDF5 fixtures with asymmetric fields, nonzero origins, angular frequencies and mean-only data; retain the current orientation and reject incompatible shapes."
depends_on = ["T-002"]
done = true

[[tasks]]
id = "T-004"
description = "Implement the accepted loader signature without B, source orientation without transposition, unscaled defaults and independent coordinate/time/amplitude transformations T/s**2; remove import resampling."
depends_on = ["T-003"]
done = true

[[tasks]]
id = "T-005"
description = "Implement explicit immutable provenance and version-aware persistence, snapshot and pickle handling; preserve known old processed arrays and reject unknown conventions without reinterpretation."
depends_on = ["T-004"]
done = true

[[tasks]]
id = "T-006"
description = "Migrate direct construction, random, NumPy/JAX, phase displays and evaluation to angular frequencies; verify derivatives, subclasses and gyroaveraging without cyclic conversions in the runtime path."
depends_on = ["T-004", "T-005"]
done = true

[[tasks]]
id = "T-007"
description = "Add B to GC dynamics, use the effective Hamiltonian P/B throughout equations and execution adapters, and migrate study composition without duplicate magnetic or coordinate scaling."
depends_on = ["T-004", "T-005"]
done = true

[[tasks]]
id = "T-008"
description = "Adapt source-based legacy reconstruction, custom archived periods, original-index recovery, fingerprints and viewers through the new reader without weakening provenance checks or rewriting saved trajectories."
depends_on = ["T-005", "T-007"]
done = true

[[tasks]]
id = "T-009"
description = "Migrate all remaining active Python callers, examples, scripts and tests, including inherited/aliased loaders and keyword dictionaries; account explicitly for callers formerly requesting resampling."
depends_on = ["T-006", "T-007", "T-008"]
done = true

[[tasks]]
id = "T-010"
description = "Migrate every active notebook caller in experiments, sympy and developements; validate representative cells in disposable copies, retain Git-ignore status and distinguish historical outputs from migrated code."
depends_on = ["T-009"]
done = true

[[tasks]]
id = "T-011"
description = "Update current import contracts and affected public/model documentation to accepted implemented behavior, with full Spanish copies; reconcile relevant README wording and complete the file-by-file migration audit."
depends_on = ["T-009", "T-010"]
done = true

[[tasks]]
id = "T-012"
description = "Execute the remaining checks and required repository validation, record measured evidence and limitations, and close only when every criterion and task is satisfied; keep incomplete validation visible."
depends_on = ["T-006", "T-008", "T-011"]
done = true

[[checks]]
id = "V-001"
criteria = ["AC-001", "AC-002", "AC-005"]
procedure = "Run C18 and C01 and C02 after adding the accepted cases: exact stored amplitudes after the documented axis mapping; original counts/origins/spacing; all flag combinations; invalid Boolean/scale keywords, source shapes, frequencies and indices; smoothing and immutable provenance. Record test counts and any coordinate discrepancies."
tests = ["tests/test_h5_rescaling.py", "tests/test_gc2d_h5_potential.py", "tests/test_input_validation_contracts.py"]
result = "passed"
evidence = "Full suite includes 19 historical HDF5 tests, input validation and 8 new rescaling tests; exact default arrays, original orientation, rectangular rejection, flags, selection and filtering pass."
checked_on = "2026-10-18"

[[checks]]
id = "V-002"
criteria = ["AC-003", "AC-004"]
procedure = "Run C18 and C01 with independent harmonic references at omega=3 and 3*sqrt(2), physical periods 2*pi/omega and first/second/mixed derivatives. Check both spatial flags and both time flags, wrapping and inverse maps. Require phase/time and coordinate rtol=atol=1e-12; amplitude-aware derivative-map rtol=atol=1e-10 in common units. Report off-grid interpolation error separately and demonstrate that a multimode field need not have period one."
tests = ["tests/test_h5_rescaling.py", "tests/test_gc2d_h5_potential.py"]
result = "passed"
evidence = "H5RescalingTests: omega=3 and 3*sqrt(2), all four flag combinations, dt=0/1/2 and mixed derivatives pass at stated 1e-12/1e-10 tolerances. Off-grid comparisons use the same spline field, not an analytic spatial approximation."
checked_on = "2026-10-18"

[[checks]]
id = "V-003"
criteria = ["AC-005", "AC-006"]
procedure = "Run C03, C04 and C05 with JAX float64 enabled. Compare values and supported derivatives at corresponding points with rtol=atol=1e-10, subclass preservation, fixed-seed random behavior and the migrated angular direct-constructor convention; compare gyroaveraged fields with rho scaled inversely with coordinates. Missing JAX or skipped affected cases cannot count as a pass."
tests = ["tests/test_h5_rescaling.py", "tests/test_jax_potential.py", "tests/test_potential_execution.py", "tests/test_core.py"]
result = "passed"
evidence = "JAX 0.7.1 CPU float64 passes focused 17-test execution suite and 8-test rescaling suite, including B=1.7 preparation; full suite covers random, direct construction and GC/FC shared equations. GPU unavailable, explicitly skipped in 3 hardware-only cases."
checked_on = "2026-10-18"

[[checks]]
id = "V-004"
criteria = ["AC-005", "AC-006", "AC-008"]
procedure = "Run C06 and C07 using local temporary archives: round-trip fields/derivatives/scales, worker reconstruction without source access, legacy processed amplitudes/grids unchanged, cyclic rates decoded once into angular rates, and explicit rejection of unknown provenance versions/orientations. Record which historical schema variants were actually covered; never test by rewriting production archives."
tests = ["tests/test_h5_rescaling.py", "tests/test_potential_snapshot.py", "tests/test_solution_persistence.py"]
result = "passed"
evidence = "Legacy cyclic pickle, worker snapshot and solution archives decode once; unknown frequency conventions and unsupported legacy spatial provenance reject explicitly. Round trips retain arrays, derivatives and metadata without source access. Production archives untouched."
checked_on = "2026-10-18"

[[checks]]
id = "V-005"
criteria = ["AC-004", "AC-007"]
procedure = "Run C18 and C01, C05 and C08 with the short trajectory cases described below. Compare all coordinate/time combinations at identical physical times, rho=0 and nonzero rho, and unwrapped positions; require max position discrepancy/L <=1e-8 and independently measured reference discrepancy/L <=1e-9. Record actual settings and maxima, plus unchanged FC equations and affected FC-consumer units."
tests = ["tests/test_h5_rescaling.py", "tests/test_gc2d_h5_potential.py", "tests/test_poincare_rho_sweep.py", "tests/test_core.py"]
result = "passed"
evidence = "Two physical reference cycles, L=0.5, 16x16, 3 particles, omega=(3,3*sqrt(2)), B=1.5, rho=(0,0.025): max position/L=7.495e-11; independent DOP853/Radau floor/L=7.348e-11; analytic shear max coordinate/L=8.882e-16. FC equations unchanged; random omega=1 preserves FC time dependence."
checked_on = "2026-10-18"

[[checks]]
id = "V-006"
criteria = ["AC-007", "AC-008"]
procedure = "Run C09, C10 and C11 against independently specified historical fixtures. Include a reference period different from the first selected mode, old rank selectors recovered from original indices, asymmetric legacy orientation, distinct radial/cycle_time drift at the same forcing period, original fingerprints, and expected errors for ambiguous provenance."
tests = ["tests/test_h5_study_selection.py", "tests/test_poincare_radial_cycle_time.py", "tests/test_poincare_method_comparison.py"]
result = "passed"
evidence = "Actual PHI_2.h5 compared with HEAD potential implementation in a temporary isolated package: arrays and dt=0/1/2 grid evaluations bitwise identical, maximum difference 0, fingerprints identical at native and 32x32 resolution. Historical custom-period and radial/cycle_time regressions pass. Dimensional raw round-trip removal differs by at most 1.776e-15 (mean) and 7.851e-17 (modes)."
checked_on = "2026-10-18"

[[checks]]
id = "V-007"
criteria = ["AC-002", "AC-008"]
procedure = "Perform the complete migration audit described below over Python files, notebook code cells and live documentation, including ignored developements. Follow aliases and forwarded kwargs. Record every active caller as migrated and checked, identify deliberate negative tests and justified historical exclusions, and require zero unaccounted removed-keyword calls or old-default assumptions."
tests = []
result = "passed"
evidence = "Reviewed Python source/examples/scripts, generator code strings, all notebook code cells including ignored development notebooks, aliases, forwarded settings and live docs. Fixed 27 notebook load cells, 2 scripts and angular period/assertion displays. Zero unaccounted live old-loader calls; exclusions documented below."
checked_on = "2026-10-18"

[[checks]]
id = "V-008"
criteria = ["AC-002", "AC-007", "AC-008"]
procedure = "Execute each distinct affected notebook consumer path in memory or a disposable copy with two cycles, three particles, two rho values and at most eleven frames. Record notebook path, cell IDs, overrides, source fixture, outcomes and output limitations. Do not overwrite canonical outputs, download buckets or execute the 5000-cycle study."
tests = []
result = "passed"
evidence = "All 27 affected notebook load/composition expressions execute in memory on a 16x16 HDF5 fixture (source mode 15, omega=7): 2 cycles, 3 particles, rho=0/0.3, 11 samples each. Real PHI_2 viewer preparation/background and RK4 paths pass for both rho and radial/cycle_time, 16x16 backgrounds, 11 phases. Generator validates/compiles in memory; script forwarded load executes on measured HDF5. No canonical outputs changed."
checked_on = "2026-10-18"

[[checks]]
id = "V-009"
criteria = ["AC-001", "AC-002", "AC-003", "AC-004", "AC-005", "AC-006", "AC-007", "AC-008"]
procedure = "Run C12 through C16 after focused checks: strict mypy, the headless full suite, both smoke examples and package build. Record environment, exit codes, test/skip counts and failures; a suite pass does not waive missing JAX or historical-data coverage. Reuse checks on unchanged code rather than repeating costly runs."
tests = []
result = "passed"
evidence = "Python 3.12.14 in .venv; NumPy 1.26.4, SciPy 1.17.1, h5py 3.16.0, JAX 0.7.1. mypy: 256 source files pass. Full suite: 561 tests, OK, 3 GPU-only skips, 363.734 s. Final focused regressions: HDF5 19, execution 17, rescaling 8 pass. Both smoke examples exit 0; source and wheel build succeed."
checked_on = "2026-10-18"

[[checks]]
id = "V-010"
criteria = ["AC-008"]
procedure = "Review the final diff against the accepted spec/plan: public signatures, units, formulas, all caller conversions, English docs and Spanish reading copies, and model-specific architecture. Reconcile current POT-001 through POT-005 with actual behavior without changing unrelated records. Record inspected files and any unresolved mismatch."
tests = []
result = "passed"
evidence = "Reviewed API/formulas, phase consumers, immutable source metadata, old/new persistence, historical arithmetic, GC B preparation and FC boundaries. Updated current POT contract, README, HDF5/JAX docs, BM4 architecture, diagram and 180-line MEMORY, with complete Spanish Markdown reading copies. git diff --check passes."
checked_on = "2026-10-18"

[[checks]]
id = "V-011"
criteria = ["AC-008"]
procedure = "At closure run C17; verify spec/plan/tasks revision alignment, valid file references, every criterion linked to passed evidence, every task complete and actual approval recorded. This final check remains pending even if a structural check passes while preparing these draft tasks."
tests = []
result = "passed"
evidence = "Final structural checker passes 15 records and 12 requirements; spec 2, plan 3 and tasks agree, all tasks complete and all checks have observed evidence. Approval records the user decision of 2026-10-18."
checked_on = "2026-10-18"
+++

# Independent HDF5 rescaling: tasks and evidence

## Execution

Implemented on `main` on 2026-10-18 after the explicit SDD implementation request
and confirmation to preserve radial viewer dynamics and historical results.
The structured tasks/checks above own completion and evidence. Spec revision 2
and plan revision 3 are reconciled. No commit, push, deployment or cloud job.

The first full run exposed stale cyclic-frequency assertions and missing pandas
in the new local environment. These were corrected; the final suite passes.
The measured-field audit found a 1.313e-10 old-sampling discrepancy caused by
reconstructing axes before the historical spline fit. Preserving the original
source axes restored exact arrays and fingerprints; the focused HDF5 and notebook
checks were rerun after that fix. No tolerances or fingerprints were weakened.

## Reproducible checks

Use the repository root and `.venv/bin/python` (Python 3.12.14). Full suite:
`MPLBACKEND=Agg JAX_ENABLE_X64=1 .venv/bin/python -m unittest discover -s tests -v`.
C01-C11 are the corresponding modules below, all included in that run.
C18 is `JAX_ENABLE_X64=1 .venv/bin/python -m unittest discover -s tests -p 'test_h5_rescaling.py' -v`.
Final targeted reruns cover historical sampling, B preparation and legacy decoding.

| Command | Module or command after `.venv/bin/python` |
| --- | --- |
| C01 | `-m unittest discover -s tests -p 'test_gc2d_h5_potential.py' -v` |
| C02 | `-m unittest discover -s tests -p 'test_input_validation_contracts.py' -v` |
| C03 | `-m unittest discover -s tests -p 'test_jax_potential.py' -v` |
| C04 | `-m unittest discover -s tests -p 'test_potential_execution.py' -v` |
| C05 | `-m unittest discover -s tests -p 'test_core.py' -v` |
| C06 | `-m unittest discover -s tests -p 'test_potential_snapshot.py' -v` |
| C07 | `-m unittest discover -s tests -p 'test_solution_persistence.py' -v` |
| C08 | `-m unittest discover -s tests -p 'test_poincare_rho_sweep.py' -v` |
| C09 | `-m unittest discover -s tests -p 'test_h5_study_selection.py' -v` |
| C10 | `-m unittest discover -s tests -p 'test_poincare_radial_cycle_time.py' -v` |
| C11 | `-m unittest discover -s tests -p 'test_poincare_method_comparison.py' -v` |
| C12 | `-m mypy src` |
| C13 | full suite above |
| C14 | `examples/gc_orbit.py` with `MPLBACKEND=Agg` |
| C15 | `examples/projected_abba.py` with `MPLBACKEND=Agg` |
| C16 | `-m build` |
| C17 | `scripts/check_sdd.py` |
| C18 | new rescaling suite above |

## Python migration inventory

The listed implementation/example/script files were reviewed against their diff.
Loading uses physical defaults; angular reconstruction and read-boundary legacy
conversion are explicit. Historical callers use study composition, not removed
loader arguments. Synthetic cyclic test fixtures multiply their original rates
by 2*pi once at construction. FC equations and random phases are unchanged.

| File | Evidence |
| --- | --- |
| `examples/jax_potential.py` | raw JAX loading regression and CLI review |
| `scripts/create_gc_energy_bound_notebook.py` | in-memory generation/compilation or measured load |
| `scripts/run_bm4_projection_comparison.py` | in-memory generation/compilation or measured load |
| `src/diagnostics/persistence.py` | suite / focused checks |
| `src/dynamics/gc.py` | suite / focused checks |
| `src/potential/_evaluation.py` | suite / focused checks |
| `src/potential/load.py` | suite / focused checks |
| `src/potential/potential.py` | suite / focused checks |
| `src/studies/_potential_snapshot.py` | suite / focused checks |
| `src/studies/dimensional_h5_midpoint.py` | suite / focused checks |
| `src/studies/h5_provenance.py` | suite / focused checks |
| `src/studies/local_jax_star.py` | suite / focused checks |
| `src/studies/modal_cpu_comparison.py` | suite / focused checks |
| `src/studies/poincare_probe.py` | suite / focused checks |
| `src/studies/poincare_radial_cycle_time.py` | suite / focused checks |
| `src/studies/poincare_rho_sweep.py` | suite / focused checks |
| `src/studies/poincare_star.py` | suite / focused checks |
| `src/studies/single_particle_recurrence.py` | suite / focused checks |
| `src/visualization/abba4_configuration_comparison.py` | suite / focused checks |
| `src/studies/historical_h5.py` | suite / focused checks |

## Notebook inventory

Cell numbers are zero-based. Every row passed extraction/execution of its actual
load/composition expression, including forwarded dictionaries, on a 16x16 source
with field 15 at omega=7. Temporary settings: B=1.5, length=0.06, cubic splines,
2 cycles, 3 particles, rho=0 and 0.3, 11 output/phase samples; requested plotting
sample sizes reduced to 16 (the explicit 32x32 inspection case retained 32).
The former B/characteristic arguments are now explicit historical composition;
former nx/ny requests use separate study sampling. All retain historical
Hamiltonian amplitudes, characteristic coordinates and cycle-time phase.
No notebook integration/publication cells or stored outputs were overwritten.

| Notebook | Cell | Check |
| --- | --- | --- |
| `notebooks/developements/study_bm4_mu_vs_displacement_2p.ipynb` | 4 | passed |
| `notebooks/developements/compare_bm4_with_sdirk4_two_particles.ipynb` | 4 | passed |
| `notebooks/developements/bm4_simple_inspection.ipynb` | 3 | passed |
| `notebooks/developements/study_100_parallel_bm4_domain_escapes.ipynb` | 4 | passed |
| `notebooks/developements/accuracy/step_ratio_abba4_bm4.ipynb` | 4 | passed |
| `notebooks/developements/accuracy/compare_three_order4_r4_models.ipynb` | 4 | passed |
| `notebooks/developements/recurrences/study_trajectory_10_recurrence_bm4_tighter_newton.ipynb` | 4 | passed |
| `notebooks/developements/recurrences/study_trajectory_10_four_cycle_segments_bm4.ipynb` | 5 | passed |
| `notebooks/developements/recurrences/study_trajectory_10_recurrence_bm4.ipynb` | 4 | passed |
| `notebooks/developements/recurrences/study_40_parallel_bm4_recurrences.ipynb` | 4 | passed |
| `notebooks/developements/recurrences/study_near_center_particle_recurrence_bm4.ipynb` | 4 | passed |
| `notebooks/developements/bm4_h5_simple_inspection/calculation.ipynb` | 3 | passed |
| `notebooks/developements/bm4_projection_comparison/comparison.ipynb` | 3 | passed |
| `notebooks/developements/potential/study_gc2d_h5_modes.ipynb` | 6 | passed |
| `notebooks/developements/potential/visualize_gc2d_h5_potential.ipynb` | 4 | passed |
| `notebooks/developements/recurrences/study_20_horizontal_bm4_recurrences/particle_2_bm4_160_periodicity.ipynb` | 4 | passed |
| `notebooks/developements/recurrences/study_20_horizontal_bm4_recurrences/particle_2_return_precision.ipynb` | 4 | passed |
| `notebooks/developements/recurrences/study_20_horizontal_bm4_recurrences/calculation.ipynb` | 3 | passed |
| `notebooks/developements/recurrences/study_20_horizontal_bm4_recurrences/visualization.ipynb` | 3 | passed |
| `notebooks/developements/accuracy/h5_three_radial_reference_2x_precision/h5_three_radial_reference_2x_precision.ipynb` | 1 | passed |
| `notebooks/developements/accuracy/h5_three_radial_reference/h5_three_radial_reference.ipynb` | 1 | passed |
| `notebooks/developements/energy/BM4_RK4/calculation.ipynb` | 5 | passed |
| `notebooks/developements/energy/compare_five_order4_models_with_sdirk_rk4/calculation.ipynb` | 3 | passed |
| `notebooks/developements/energy/compare_five_order4_models_with_sdirk_rk4/visualization.ipynb` | 3 | passed |
| `notebooks/developements/energy/energy_balance_dop853_bm4_gl4_rk4_horizontal_3p_200steps_10saves/calculation.ipynb` | 3 | passed |
| `notebooks/experiments/comparison/compare_abba4_r6_r8_configurations_aws_executed.ipynb` | 6 | passed |
| `notebooks/experiments/comparison/compare_abba4_r6_r8_configurations.ipynb` | 5 | passed |

The measured viewer path was checked separately through `load_dimensional_h5_field`,
`build_rho_star`, `sample_rho_dynamics` and RK4: 2 cycles, 3 particles, rho=0/0.3,
both radial/cycle_time conventions, 16x16 scalar mesh, 5x5 vector mesh and 11 phases.
All four executions produced finite states and periodic background closure.

## Explicit archival exclusions

- 80 `*.from-b2-*.ipynb` conflict copies remain historical synchronization evidence;
  active canonical notebooks are migrated. No current import or runner selects them.
- Two `legacy_previous_study/` notebooks are explicitly superseded, as documented
  in the energy-balance study README; current results must not load their outputs.
- Ten notebooks under `resultados/` are saved executions of verified frozen code.
  Their study_io/experiment modules and `src/studies/poincare_batch_runtime/`
  intentionally import checked old source from snapshots, not the current evaluator.
- `energy/energy_balance_dop853_bm4_gl4_rk4_horizontal_3p_200steps_10saves/calculation_executed.ipynb`
  is a generated execution record: its README and run_remote.py identify it as
  output. The runner reads the migrated calculation.ipynb; it does not execute
  this saved output. The old loader name remains only in that historical record.
- Historical SDD records and stored notebook outputs are not live API instructions.
  Rejection tests intentionally exercise removed keywords. Technical legacy
  metadata names are retained for reading previously produced artifacts.

## Measured historical identity and limitations

Compared against the pre-change `HEAD` potential package in an isolated temporary
package on the actual local PHI_2 file. Native fingerprint:
`ba7fd7e8b846e90d7f1668c123a6052666e935d0076439f4711ea6247b8ef8e3`.
32x32 fingerprint:
`a851e116bb6ebe3cf121bd47f823b46e1bc0bff2dceb555f4366e308dd98e273`.
Arrays and dt=0/1/2 evaluations at t=0.371 matched exactly. Dimensional loading
removes a normalization round trip, with measured sample differences below 2e-15.

## Outcome and limitations

CPU JAX was exercised; GPU hardware was absent. No long production notebook,
remote/cloud execution, deployment or download was performed. Existing outputs
and ignored tracking status remain intact. Local Spanish copies accompany all
modified Markdown documents. MEMORY records confirmed decisions in 180 lines.
