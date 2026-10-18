+++
schema_version = 2
kind = "change-tasks"
id = "2026-10-18-poincare-simulation-skill"
spec_revision = 1
plan_revision = 1
sources = ["src/studies/poincare_automation.py", "src/studies/poincare_automation_io.py", "src/visualization/poincare_automation.py", "scripts/run_poincare_automation.py", "tests/test_poincare_automation.py", "tests/test_poincare_automation_viewer.py", "src/studies/poincare_rho_sweep.py", "src/studies/poincare_gap_probes.py", "src/visualization/poincare_comparison.py", "tests/test_poincare_rho_sweep.py", "tests/test_poincare_gap_viewers.py", "tests/test_poincare_gap_probes.py", "tests/test_h5_rescaling.py", "tests/test_h5_frequency_units.py", "scripts/check_sdd.py"]
documents = ["specs/changes/2026-10-18-poincare-simulation-skill/spec.md", "specs/changes/2026-10-18-poincare-simulation-skill/plan.md", "specs/README.md", "AGENTS.md"]
open_questions = []

[[tasks]]
id = "T-001"
description = "Record accepted specification/plan and explicit authorization for creating the skill; capture the dirty working-tree and reference geometry/viewer baseline."
depends_on = []
done = true

[[tasks]]
id = "T-002"
description = "Implement immutable automation inputs, fixed 48-particle identity, validated case expansion and default schedules."
depends_on = ["T-001"]
done = true

[[tasks]]
id = "T-003"
description = "Compose normalized/raw GC problems through the current loader, consistent rho/time/coupling maps and canonical method factories; verify short numerical equivalence."
depends_on = ["T-002"]
done = true

[[tasks]]
id = "T-004"
description = "Implement prepare/run/view CLI orchestration, complete case manifests, persistence, validated reuse and failure recovery without overwriting mismatched results."
depends_on = ["T-003"]
done = true

[[tasks]]
id = "T-005"
description = "Adapt saved cases to one existing HTML viewer containing all 48 particles, exact identities and effective fields for every case."
depends_on = ["T-004"]
done = true

[[tasks]]
id = "T-006"
description = "Create the project-local simular-poincare skill and discovery metadata with tested operation routing, defaults, missing-input handling and authorization boundaries."
depends_on = ["T-005"]
done = true

[[tasks]]
id = "T-007"
description = "Run reduced end-to-end and browser checks, numerical/persistence regressions and the appropriate repository validation; fix failures within scope."
depends_on = ["T-006"]
done = true

[[tasks]]
id = "T-008"
description = "Reconcile current PNC contracts, usage/model documentation and durable memory with actual implementation, maintain Spanish copies and record verification evidence."
depends_on = ["T-007"]
done = true

[[checks]]
id = "V-001"
criteria = ["AC-001", "AC-002"]
procedure = "Add and run tests/test_poincare_automation.py for independently computed 40-star fractions plus the eight literal probe seeds, exact IDs/colors, input validation, Cartesian cases and the unexecuted 5000*50 default schedule."
tests = ["tests/test_poincare_rho_sweep.py::RhoSweepTests.test_distinct_radial_ranks_geometry_and_physical_rho", "tests/test_poincare_gap_probes.py::GapProbeTests.test_geometry_is_explicit_and_scaled_with_unchanged_physics"]
result = "passed"
evidence = "14 final focused tests passed. Independent geometry checks cover all 48 fractions and original colors; default preparation yields 250000 steps and 5001 times without integration; exact case hashes distinguish close rho values and changed controls."
checked_on = "2026-10-18"

[[checks]]
id = "V-002"
criteria = ["AC-003", "AC-006"]
procedure = "Run new normalization/reference tests and existing test_h5_rescaling.py/test_h5_frequency_units.py; compare all rescaling combinations and short historical-radial equivalence with the numeric tolerances and DOP853/Radau audit in plan.md; report measured errors and clocks."
tests = ["tests/test_h5_rescaling.py", "tests/test_h5_frequency_units.py", "tests/test_poincare_rho_sweep.py::RhoSweepTests.test_radial_hamiltonian_changes_drift_without_changing_forcing_or_geometry"]
result = "passed"
evidence = "Synthetic 48-particle tests over four methods and all four rescaling combinations: DOP853/Radau discrepancy 1.000e-13 cell, maximum method/reference error 1.012e-10 cell, coordinate equivalence 5.829e-15 cell. Measured-source short historical radial comparison passed separately. Existing HDF5 regressions passed in the 574-test suite recorded in V-007."
checked_on = "2026-10-18"

[[checks]]
id = "V-003"
criteria = ["AC-004"]
procedure = "Run 1-2 synthetic forcing cycles with 48 particles through each supported method; verify exact schedules/initial state, float64 states, complete step counts, solver metadata, archive round-trip and zero integrator calls for a matching rerun; reject changed inputs and incomplete archives."
tests = ["tests/test_poincare_rho_sweep.py::RhoSweepTests.test_cycle_only_roundtrip_and_no_resubmission", "tests/test_poincare_gap_probes.py::GapProbeTests.test_saved_background_mismatches_and_id_collisions_are_rejected"]
result = "passed"
evidence = "Focused tests passed exact state/time/diagnostic and persistence checks; matching reruns made zero simulate calls. A deliberate failed rho case preserved its completed companion, then retry integrated only the failed case. Source, B, rho, IDs and time-convention mismatches rejected. The final focused run also preserved an incomplete archive before retrying and validated the new complete replacement."
checked_on = "2026-10-18"

[[checks]]
id = "V-004"
criteria = ["AC-005", "AC-006"]
procedure = "Add tests/test_poincare_automation_viewer.py; decode a reduced multi-case HTML payload for all 48 IDs/colors, all case labels, cycle zero, complete folded returns and 51 phases; verify closure/scaling and prove export does not integrate. Inspect case/particle/phase/Start-cycle controls in the browser and retain a screenshot."
tests = ["tests/test_poincare_gap_viewers.py", "tests/test_poincare_rho_sweep.py::RhoSweepTests.test_phase_fields_match_gc_dynamics_and_close_at_one_cycle"]
result = "passed"
evidence = "Three viewer tests passed: eight cases, 48 persistent IDs/colors, cycle zero and 51 phases, missing cases disabled, export without integration, raw/normalized field agreement. Browser verified 48/48 selection, probe isolation, Start cycle, phase and rho/B/method switching; screenshot retained at /tmp/gc2d-poincare-skill-audit-drxli655/viewer-multicase.jpg."
checked_on = "2026-10-18"

[[checks]]
id = "V-005"
criteria = ["AC-002", "AC-004", "AC-007"]
procedure = "Run CLI prepare without integration, a temporary reduced run and view-only export; verify default configuration and case counts, missing inputs, explicit overrides and unavailable backend handling. Run the skill-creator quick_validate.py on the new skill and inspect its discovery metadata and prepare/run/view/read-only prompt routing."
tests = []
result = "passed"
evidence = "CLI prepare/run/view/status passed in temporary studies; required-method absence rejected. Manual skill routing review covered missing rho/method, explicit overrides, unsupported physical models and read-only questions. python3 skill-creator quick_validate.py passed. The project venv lacks PyYAML; the system validator interpreter provided it without installation."
checked_on = "2026-10-18"

[[checks]]
id = "V-006"
criteria = ["AC-003", "AC-005", "AC-008"]
procedure = "With the measured PHI_2.h5, run a disposable 48-particle one-cycle smoke at 50 steps and export its viewer; compare mapped historical radial fields and short results, record source hash/time provenance, and confirm reference notebooks/source/archives were not overwritten."
tests = ["tests/test_poincare_rho_sweep.py", "tests/test_poincare_gap_viewers.py", "tests/test_poincare_gap_probes.py"]
result = "passed"
evidence = "Measured PHI_2.h5 SHA256 4bbd873cd05992e0a615d66ebbf4d96b38c8b3c36be2ca62932e66af7e6d4ced; 48 particles, BM4Midpoint/JAX CPU, one cycle at 50 steps. Against the old radial 40+8 SciPy runs: max coordinate error 1.503e-12 cell, relative H error 1.437e-12, relative velocity error 2.861e-11. Physical period 0.0004159549201179788 s; historical clock 0.002613521842534343 s. Viewer exported in temporary audit directory; historical inputs were not written."
checked_on = "2026-10-18"

[[checks]]
id = "V-007"
criteria = ["AC-007", "AC-008"]
procedure = "Run focused new tests, python -m mypy src, MPLBACKEND=Agg python -m unittest discover -s tests -v, both documented smoke examples and python -m build in the project environment; inspect new current PNC requirements, applicable model docs, full Spanish copies and ignored output paths; finish with python scripts/check_sdd.py and git diff --check."
tests = []
result = "passed"
evidence = "Full suite: 574 tests in 402.534 s, 571 passed and 3 expected unavailable-GPU skips. Final focused modules: 14 tests passed in 15.193 s, including three checks added after full-suite discovery (partial archive preservation, coupling time map, unavailable-backend preflight). Implementation stayed unchanged between these runs. Strict mypy passed on 259 source files; GC smoke 20 steps and projected ABBA smoke 4 steps passed; source/wheel build passed. Skill validation and bilingual/ignored-path audits passed. Only MEMORY.md changed among pre-existing tracked files. Final SDD and whitespace checks passed."
checked_on = "2026-10-18"
+++

# Poincare simulation skill: tasks and evidence

## Execution

The user accepted specification revision 1 and plan revision 1 and requested
implementation with the skill-creation authorization prefix on 2026-10-18.
All eight tasks and seven checks are complete. PNC-001 through PNC-004 are
established in the current contract, with executable evidence references.

The pre-implementation tracked-file hashes and dirty diff are retained in
`/tmp/gc2d-poincare-skill-audit-drxli655/`. Validation logs, reduced measured and
synthetic studies, and the browser screenshot use that same temporary folder.
All new scientific tests use at most two cycles. The first focused run exposed
an overly broad check that treated BM4Midpoint's zero unknown-dimension
diagnostic as nonlinear work; the corrected check distinguishes zero metadata
from an actual nonlinear solver. Eleven new focused tests then passed, followed
by 14 in the final focused run after adding three targeted recovery/rate/backend
checks. The implementation did not change between the successful runs.

Use `.venv/bin/python` for repository validation and `python3` for the skill
validator, which needs PyYAML available in the system interpreter. No packages
were installed. Strict mypy passed on 259 source files; both documented smoke
examples and the source/wheel build passed. The full suite passed 574 tests
in 402.534 seconds, with three expected GPU skips (571 executed successes).
The three tests added after full-suite discovery passed in the final focused
run; no expensive full-suite repetition was needed for a test-only addition.

Preparation checks passed on 2026-10-18 (21 records, 12 current requirements).
Final structural validation passes with 22 records and 16 requirements.
Spanish copies retain technical metadata and code and remain ignored by Git.
Whitespace checks include the new untracked files. The before/after hash audit
shows that only MEMORY.md changed among pre-existing tracked files; its final
length is 200 lines (Spanish copy: 208). Historical code, notebooks and archives
were preserved. Metadata above owns task/check completion and measured evidence.

## Outcome and limitations

Implemented and verified: one discoverable project-local skill, one shared
48-particle study, a reproducible CLI and one existing HTML viewer for all
requested cases. Local SciPy and JAX CPU were exercised; GPU and actual remote
execution were not available/exercised and no deployment was performed.
No commits, branch switches, global skill installation or bucket transfers
were made. The project remains on main with the user's prior changes intact.

During implementation the user separately requested execution of this skill
with BM4Midpoint, rho=0.3 and B=1.5. That production run completed 5000 cycles
at 50 steps/cycle with the normalized potential and local JAX CPU after the
backend choice was stated. Its output is under
`notebooks/developements/poincare_section/bm4_midpoint_48_rho_0_30_B_1_5_normalized_2026_10_18/`.
This is a user-requested production result, not a routine full-length test.

Independent reload verified a complete manifest, exact source hash, 250000
steps, zero output interpolation, finite float64 states shaped (96,5001),
cycle times 0 through 5000 and the original initial identity. Physical duration
is 2.079774600589894 seconds for the source frequency. The 7695120-byte HTML
viewer was opened in the browser with all 48 particles and cycles 0-5000;
`viewer.jpg` records that view. The launcher session reported exit 143 after
printing the completed viewer path; its cause was not established. The saved
result and viewer were independently validated successfully, and no trajectory
restart was needed. Do not describe that launcher as having returned zero.
