+++
schema_version = 2
kind = "change-tasks"
spec_revision = 2
plan_revision = 2
id = "2026-10-20-unified-states"
sources = ["src/states/history.py", "src/diagnostics/geometry.py", "src/states/generators.py", "src/formulations/layout.py", "src/contracts/configuration.py", "src/contracts/problem.py"]
documents = ["specs/current/simulation.md", "specs/current/results.md", "docs/simulation/integration-architecture.md"]
open_questions = []

[[tasks]]
id = "T-001"
description = "Inventory imports, subclassing, optional empty configurations and geometry metadata; capture deterministic GC/FC and generator baselines before edits."
depends_on = []
done = true

[[tasks]]
id = "T-002"
description = "Implement State and canonical layout exports; preserve structural protocols and ownership."
depends_on = ["T-001"]
done = true

[[tasks]]
id = "T-003"
description = "Move all radial and contour generators to states.generators and extract area calculation, preserving geometric contracts."
depends_on = ["T-002"]
done = true

[[tasks]]
id = "T-004"
description = "Migrate runtime consumers, tests, examples and supported notebooks to states and states.generators, remove initial_conditions, and update model docs, contracts and migration guidance."
depends_on = ["T-003"]
done = true

[[tasks]]
id = "T-005"
description = "Run focused and repository checks, audit compatibility and record actual evidence."
depends_on = ["T-004"]
done = true

[[checks]]
id = "V-001"
criteria = ["AC-001", "AC-002"]
procedure = "Run python -m unittest discover -s tests -p test_core.py -v and focused State round-trip, invalid-input, copy isolation and JAX layout tests."
tests = ["tests/test_states.py", "tests/test_core.py", "tests/test_shared_array_dynamics.py"]
result = "passed"
evidence = "Seven unified-State tests passed, including copy isolation, GC/FC named components, pickle, invalid snapshots, exact history round-trips and JAX JIT preservation. Core tests passed in the full run; mypy reports no issues in 257 source files."
checked_on = "2026-10-20"

[[checks]]
id = "V-002"
criteria = ["AC-003"]
procedure = "Run migrated generator/area tests: exact baseline coordinates and ordering; reversed signed area, batched histories, periodic seams and invalid inputs."
tests = ["tests/test_core.py", "tests/test_studies.py"]
result = "passed"
evidence = "Four deterministic generator arrays and both contour areas match pre-migration captures exactly; input-validation tests passed (6 tests). Existing area, periodic reconstruction and orientation tests passed."
checked_on = "2026-10-20"

[[checks]]
id = "V-003"
criteria = ["AC-004"]
procedure = "Run state ownership and persistence tests and compare fixed-step GC/FC trajectories to pre-change baselines with identical fields, methods, steps and times. Require exact array equality when arithmetic is unchanged; investigate differences rather than loosening tolerances."
tests = ["tests/test_state_ownership.py", "tests/test_solution_persistence.py"]
result = "passed"
evidence = "Deterministic GC and FC RK4 histories match pre-migration arrays exactly (maximum absolute difference 0); both pre-migration archives load unchanged. Ownership tests (8) and persistence tests (12) passed, including absent initial snapshots."
checked_on = "2026-10-20"

[[checks]]
id = "V-004"
criteria = ["AC-005"]
procedure = "Audit all old imports and names, supported notebook cells and metadata, saved archive round-trips, model docs and migration examples. Verify states.generators imports work and no supported runtime caller imports initial_conditions; verify the old package is removed. Execute only reduced notebook samples."
tests = ["tests/test_package_layout.py"]
result = "passed"
evidence = "Retired package files removed and wheel audited without initial_conditions or contracts/state_layout. New wheel imports, generators, area and simulation passed in an isolated directory. Notebook audit parsed 1180 code cells with no syntax errors; two GC/FC notebooks executed reduced setup/integration in memory. Eight contour notebooks now pass explicit geometry metadata. English docs and Spanish copies updated."
checked_on = "2026-10-20"

[[checks]]
id = "V-005"
criteria = ["AC-001", "AC-002", "AC-003", "AC-004", "AC-005"]
procedure = "Run python -m mypy src; MPLBACKEND=Agg python -m unittest discover -s tests -v; python examples/gc_orbit.py; python examples/projected_abba.py; python -m build; python scripts/check_sdd.py. Record missing optional backends explicitly."
tests = ["tests/test_core.py", "tests/test_state_ownership.py"]
result = "passed"
evidence = "584 tests completed in 419.114 s: all passed except 3 GPU-only skips because no JAX GPU is available. Mypy passed for 257 source files; both examples passed; isolated sdist/wheel build and extracted-wheel simulation passed; SDD and whitespace checks passed."
checked_on = "2026-10-20"
+++

# Unified states: tasks and evidence

## Execution

Tasks T-001 through T-005 run in dependency order. Baselines are captured before implementation. The structured checks cover every acceptance criterion; all five implementation tasks and all verification checks are complete. Use the project's Python environment for recorded commands.

Preparation inspected the current storage classes, layouts, problem protocol, ownership contracts and relevant generator/area consumers. That preparation phase changed no runtime code or thesis files; the accepted implementation now migrates runtime code and its consumers.

## Outcome and limitations

Revision 2 is implemented and verified. State, layouts and external generators now share states; area evaluation is independent and the retired package is removed. Supported consumers, persistence and documentation are migrated. The three GPU-only checks were skipped because this host has no GPU; CPU/JAX coverage passed. No remote Modal job was submitted. Existing overview diagrams remain explicitly documented as historical renderings.

## Revision 2 — 2026-10-20

At the user's request, generators move to states.generators and initial_conditions is removed after migrating callers. This supersedes the organization proposed in revision 1. Tasks and checks have been reconciled with specification and plan revision 2. All implementation remains pending; there were no verification outcomes to invalidate.

Source references were updated for the 2026-10-20 states migration; prior recorded outcomes are unchanged.

## Implementation evidence — 2026-10-20

Before edits, deterministic baselines were captured from commit 9e862e2987dd9c2b068f2f1ce6ead1b081b78748 using Potential.random(A=0.04, M=2, nx=8, ny=8, seed=123), GC rho=0.1 and FC rho=0.2, eta=-0.1. RK4 used span (0,0.02), maximum step 0.005 and five samples. Both two-particle histories matched exactly after migration. Four generators and two areas also matched exactly.

The first full run exercised 562 tests and exposed five migration failures; the affected callers/imports and mutable test providers were corrected, then focused checks passed. The final complete run subsequently passed. An initial build without isolation failed because setuptools was absent from the environment; the standard isolated build succeeded, including the wheel smoke check. No dependency was installed into the project environment for that build.

## Final verification — 2026-10-20

584 tests completed in 419.114 s: all passed except 3 GPU-only skips because no JAX GPU is available. The public facade identity test was also rerun successfully after clarifying its import route. Ten tracked notebooks retain exactly the same outputs, execution counters and cell metadata; reduced validation ran only in memory. The repository memory records the new API in 211 lines, with its Spanish copy updated.

Source references above follow the module moves in `2026-10-20-integration-state-results`. The earlier design and recorded verification remain historical evidence of the State-based API.
