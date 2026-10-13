+++
schema_version = 2
kind = "change-tasks"
id = "2026-10-13-sdd-bootstrap"
sources = ["scripts/check_sdd.py", "tests/test_sdd.py", ".agents/skills/sdd/SKILL.md", "AGENTS.md", ".github/workflows/ci.yml"]
documents = ["specs/README.md", "specs/current/potential-import.md", "specs/current/simulation.md", "specs/current/results.md"]
open_questions = []
spec_revision = 1
plan_revision = 1

[[tasks]]
id = "T-001"
description = "Write the guide, baseline contracts, templates, and pending README correction proposal."
done = true
depends_on = []

[[tasks]]
id = "T-002"
description = "Create the project-local SDD skill and integrate its routing in repository instructions."
done = true
depends_on = []

[[tasks]]
id = "T-003"
description = "Implement the structural checker and focused tests, and add the CI check."
done = true
depends_on = []

[[tasks]]
id = "T-004"
description = "Complete independent workflow review and validation, document actual results and reusable project knowledge."
done = true
depends_on = []

[[checks]]
id = "V-001"
criteria = ["AC-001"]
procedure = "Compare each baseline requirement with its source and referenced test definitions; run the six baseline test modules for HDF5 imports, input validation, execution, core behavior, state ownership, and extensible architecture."
tests = ["tests/test_gc2d_h5_potential.py", "tests/test_input_validation_contracts.py", "tests/test_execution.py", "tests/test_core.py", "tests/test_state_ownership.py", "tests/test_extensible_architecture.py"]
result = "passed"
evidence = "All 65 tests in the six baseline modules passed. Source inspection confirmed all 12 requirements; the structural checker resolved their file and test-method references."
checked_on = "2026-10-13"

[[checks]]
id = "V-002"
criteria = ["AC-002"]
procedure = "Run the skill-creator quick validator and independently exercise preparation, status, resumed implementation, material scope changes, and verification with failed numerical checks."
tests = []
result = "passed"
evidence = "The skill-creator validator passed. An independent read-only review exercised all five workflow scenarios. Its stale-draft-approval and nested-test-selector findings were fixed and covered by regression tests; the scope-discussion wording was clarified."
checked_on = "2026-10-13"

[[checks]]
id = "V-003"
criteria = ["AC-003"]
procedure = "Run python -m unittest discover -s tests -p 'test_sdd.py' -v, python scripts/check_sdd.py, and python -m mypy --explicit-package-bases scripts/check_sdd.py tests/test_sdd.py."
tests = ["tests/test_sdd.py"]
result = "passed"
evidence = "All 18 SDD tests passed, including read-only CLI behavior and both review regressions. The checker passed with 5 records and 12 requirements. Mypy passed for the checker and its tests using --explicit-package-bases."
checked_on = "2026-10-13"

[[checks]]
id = "V-004"
criteria = ["AC-004"]
procedure = "Run python -m mypy src and inspect the working-tree diff for source changes."
tests = []
result = "failed"
evidence = "The global type check reports 108 errors in 32 existing source files. The source tree has no changes from this SDD installation; these failures remain outside its scope."
checked_on = "2026-10-13"

[[checks]]
id = "V-005"
criteria = ["AC-004"]
procedure = "Run MPLBACKEND=Agg python -m unittest discover -s tests -v."
tests = []
result = "passed"
evidence = "The full suite completed 540 tests in 427.941 seconds with no failures and 3 GPU-dependent skips. Discovery included the initial 16 SDD tests; the final checker and both added regressions subsequently passed all 18 focused SDD tests. No canonical notebooks were executed end to end."
checked_on = "2026-10-13"

[[checks]]
id = "V-006"
criteria = ["AC-004"]
procedure = "Run examples/gc_orbit.py, examples/projected_abba.py, and python -m build; inspect the CI edit and unchanged README paragraph."
tests = []
result = "passed"
evidence = "Both examples completed (20 RK4 steps and 4 reduced-multiplier ABBA steps); source and wheel distributions built successfully. CI includes the read-only SDD check, and README.md has no edits. Local validation uses Python 3.12.3."
checked_on = "2026-10-13"
+++

# Establish specification-driven development for GC2D: tasks

## Execution

Migrated from the original combined record on 2026-10-13. Its approval,
criteria, task completion, check results, and evidence are preserved. No task
dependencies were recorded previously; empty dependency lists preserve that
history. This file reorganization does not authorize or revalidate the change.

## Outcome and limitations

Implementation and the planned validation runs are complete. The baseline,
skill, checker, examples, build, and full test suite passed their checks. The
existing global type-check failures prevent this record from reaching
`verified`; it remains `implementing` with V-004 explicitly failed. Resolving
those unrelated source defects requires a separate scoped change. SDD reference
conventions were recorded in the project memory (168 total lines). The README selection proposal is
separate and remains unapproved. The structural checker cannot authenticate
approvals or prove scientific correctness, and it does not require every code
change to have an associated specification.
