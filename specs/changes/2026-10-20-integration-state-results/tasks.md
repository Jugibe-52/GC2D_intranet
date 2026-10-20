+++
schema_version = 2
kind = "change-tasks"
spec_revision = 1
plan_revision = 1
id = "2026-10-20-integration-state-results"
sources = ["src/methods/base.py", "src/execution/_host.py", "src/execution/state.py", "src/states/history.py", "src/formulations/layout.py", "src/request.py", "src/contracts/problem.py"]
documents = ["specs/current/states.md", "specs/current/results.md", "specs/current/simulation.md"]
open_questions = []

[[tasks]]
id = "T-001"
description = "Capture baseline and review accepted records."
depends_on = []
done = true

[[tasks]]
id = "T-002"
description = "Implement containers, problem/request/layout ownership and integration flow."
depends_on = ["T-001"]
done = true

[[tasks]]
id = "T-003"
description = "Migrate runtime consumers, notebooks, archive readers and tests."
depends_on = ["T-002"]
done = true

[[tasks]]
id = "T-004"
description = "Update current contracts, model documentation, memory, translations and TFM."
depends_on = ["T-003"]
done = true

[[tasks]]
id = "T-005"
description = "Run required verification and record observed evidence."
depends_on = ["T-004"]
done = true

[[checks]]
id = "V-001"
criteria = ["AC-001"]
procedure = "Compare deterministic baseline arrays and diagnostics, test fixed shadows and adaptive/JAX sampling."
tests = ["tests/test_method_integration.py", "tests/test_adaptive_integration.py", "tests/test_jax_methods.py", "tests/test_state_formulations.py"]
result = "passed"
evidence = "152 exact comparisons with pre-change commit 26807d9039af35b87fd17c48ef12e3c388ce7451 passed (maximum absolute difference 0): GC/FC histories, numeric diagnostics and four generators. Fixed shadow independence, adaptive dense output and CPU/JAX cases passed in the full suite. Six pre-change archives loaded unchanged."
checked_on = "2026-10-20"

[[checks]]
id = "V-002"
criteria = ["AC-002"]
procedure = "Test States lifecycle, copy isolation, result ownership and validation."
tests = ["tests/test_states.py", "tests/test_state_ownership.py", "tests/test_execution.py", "tests/test_solution_persistence.py", "tests/test_method_validation_contracts.py"]
result = "passed"
evidence = "Recording completeness, finite samples, copy isolation, readonly arrays and diagnostics, pickle, independent metric axes and same-object runner return passed. Integer seeds retain fractional recorded samples; complete-array construction preserves dtype. Incomplete results are rejected before persistence. Final focused runs: 11 state, 7 integration, 13 persistence and 9 method-validation tests passed."
checked_on = "2026-10-20"

[[checks]]
id = "V-003"
criteria = ["AC-003"]
procedure = "Compare generators and raw problem snapshots; test arbitrary initial time."
tests = ["tests/test_core.py", "tests/test_states.py", "tests/test_state_ownership.py", "tests/test_input_validation_contracts.py"]
result = "passed"
evidence = "Circle, square, radial_star and ranked_radial_star match baseline coordinates exactly. GC/FC problem/layout snapshots and arbitrary t0 passed; deterministic reference runs start at 0.3 and sample both main endpoints and interior times."
checked_on = "2026-10-20"

[[checks]]
id = "V-004"
criteria = ["AC-004"]
procedure = "Audit imports, build package, test layout NumPy/JAX round-trips and request validation."
tests = ["tests/test_package_layout.py", "tests/test_states.py", "tests/test_extensible_architecture.py"]
result = "passed"
evidence = "Python imports and 1180 notebook code cells audited without retired imports. Isolated sdist/wheel build passed; extracted-wheel imports and nonzero-time energy-tracked simulation passed with no editable-source imports. Retired modules are absent; importing formulations.layout loads no numerical methods. NumPy/JAX layout round-trips and request validation passed."
checked_on = "2026-10-20"

[[checks]]
id = "V-005"
criteria = ["AC-005"]
procedure = "Run full unittest, mypy, examples, build, notebook audit, SDD validator and thesis compilation."
tests = ["tests/test_core.py", "tests/test_states.py", "tests/test_solution_persistence.py", "tests/test_jax_methods.py"]
result = "passed"
evidence = "Final full suite: 590 tests in 462.124 s, OK with 3 GPU-only skips because no JAX GPU is available. Mypy passed for 257 source files; both examples, package build and extracted-wheel smoke passed. Parsed 1180 notebook cells; reduced GC/BM4 and FC/RK4 calculation cells passed in memory. All outputs, execution counts and metadata in 16 tracked notebooks are unchanged. English docs and ignored Spanish companions updated; MEMORY.md has 218 lines. TFM latexmk -pdf main.tex passed, producing the 28-page root main.pdf. Final SDD validation passed (29 records, 19 requirements), and git diff --check passed."
checked_on = "2026-10-20"
+++
# Integration states: completed tasks

## Execution

Implemented the accepted recording/result architecture and migrated runtime consumers, public imports, notebooks, archives, model guides, current contracts, project memory and the TFM. The same completed States object crosses the method/executor/runner boundary. Formulations remain reusable across methods. The historical architecture diagrams and their renderer are retained unchanged and explicitly labeled historical.

## Numerical reference and reproduction

The reference was captured before changes from commit 26807d9039af35b87fd17c48ef12e3c388ce7451. Use Potential.random(A=0.04, M=2, nx=8, ny=8, seed=123), GC rho=0.1 and FC rho=0.2, eta=-0.1; initial GC values [1,1.2,0.7,0.8], FC adds [0.1,0.2,-0.2,0.3]. The span is (0.3,0.34), max_step=0.01 and saved times [0.3,0.303,0.307,0.31,0.327,0.34]. Runs cover GC RK4/BM4Implicit/DOP853 with energy and FC RK4/DOP853 with energy plus FC RK4 without energy (the temporary baseline label fc_bm4 denotes this last RK4 case; BM4 is GC-only). Four geometric generators and six archived results were also compared. The measured maximum absolute difference is zero.

Run the repository commands from AGENTS.md for unittest, mypy, examples and build. The checks above also cover extracted-wheel execution and source-only notebook migration. During migration, retired imports, read-only-history mutation, source-identity assumptions and complete-array dtype conversion were exposed by tests and corrected; the final results supersede those failures. Temporary detailed logs and baseline files are under /tmp/gc2d-evolution-* and /tmp/gc2d-state-evolution-baseline on this machine; they are not committed artifacts.

## Outcome and limitations

No JAX GPU was available, so three GPU-only checks are skipped. CPU/JAX and local worker/serialization tests passed; no Modal deployments, cloud integrations, bucket uploads, commits or synchronization were performed. Notebook validation used reduced setup/calculation cells, not full scientific campaigns. The TFM PDF was rebuilt in its repository root, as used by TeXstudio.


Reference note (2026-10-20): source references now resolve to methods.base and execution after the package merger. Historical design and results in this record retain their original execution context.
