+++
schema_version = 2
id = "2026-10-13-sdd-phases"
sources = ["scripts/check_sdd.py", "tests/test_sdd.py", ".agents/skills/sdd/SKILL.md", "AGENTS.md"]
documents = ["specs/README.md"]
open_questions = []
kind = "change-tasks"
spec_revision = 1
plan_revision = 1

[[tasks]]
id = "T-001"
description = "Define the split schema, migrate existing changes, and implement structural validation with regression tests."
depends_on = []
done = true

[[tasks]]
id = "T-002"
description = "Update the skill operations, templates, discovery metadata, guide, repository routing, and durable memory."
depends_on = ["T-001"]
done = true

[[tasks]]
id = "T-003"
description = "Review workflow scenarios, run required checks, and record results and remaining limitations."
depends_on = ["T-001", "T-002"]
done = true

[[checks]]
id = "V-001"
criteria = ["AC-001"]
procedure = "Inspect both migrated change directories against the original record contents, confirm unchanged lifecycle and evidence, and inspect git diff for src/ and specs/current/ changes."
tests = []
result = "passed"
evidence = "The migration assigned every original Markdown section to exactly one document and copied all original criteria, task completion, checks, approval, and status. README correction remains draft; bootstrap remains implementing with its failed global type check. Current contracts and src/ were not edited."
checked_on = "2026-10-13"

[[checks]]
id = "V-002"
criteria = ["AC-002"]
procedure = "Run the skill-creator quick validator and independently exercise phased preparation, combined preparation, resumption, scope revision, status, and verification scenarios."
tests = []
result = "passed"
evidence = "The skill-creator validator passed. Independent read-only scenarios covered each preparation phase, combined prepare, accepted resumption, scope revision, read-only status, and failed verification. Temporary fixtures exercised their structural consequences. Local-ID format validation and the explanation of removed-criterion references were tightened after review and covered by regression tests."
checked_on = "2026-10-13"

[[checks]]
id = "V-003"
criteria = ["AC-003"]
procedure = "Run python -m unittest discover -s tests -p 'test_sdd.py' -v, python scripts/check_sdd.py, and python -m mypy --explicit-package-bases scripts/check_sdd.py tests/test_sdd.py."
tests = ["tests/test_sdd.py"]
result = "passed"
evidence = "All 25 SDD tests passed after the review changes. The repository checker passed with 12 documents and 12 current requirements. Strict mypy passed for scripts/check_sdd.py and tests/test_sdd.py with --explicit-package-bases."
checked_on = "2026-10-13"

[[checks]]
id = "V-004"
criteria = ["AC-003"]
procedure = "Run python -m mypy src and inspect the source diff to distinguish pre-existing issues."
tests = []
result = "failed"
evidence = "The required global mypy check still reports 108 errors in 32 of 255 source files, matching the pre-existing baseline. No src/ files were changed. Checker and test-module typing passed separately; these unrelated source errors are not waived."
checked_on = "2026-10-13"

[[checks]]
id = "V-005"
criteria = ["AC-003"]
procedure = "Run MPLBACKEND=Agg python -m unittest discover -s tests -v, both required smoke examples, and python -m build. Do not run full notebooks."
tests = []
result = "passed"
evidence = "The full suite completed 547 tests in 429.778 seconds without failures, with 3 GPU-dependent skips. Discovery preceded two final review regressions; the final focused run passed all 25 SDD tests. Both required smoke examples passed (20 RK4 steps and 4 projected ABBA steps), and source/wheel builds succeeded. No full notebooks were executed."
checked_on = "2026-10-13"
+++

# Separate SDD preparation phases: tasks and evidence

## Execution

The user accepted the concrete interface and document split before this work.
The two earlier changes were migrated without replacing their historical
outcomes. Their task documents identify the layout migration explicitly.

## Outcome and limitations

The three-document workflow, operations, templates, migration, and checker are
implemented. All tasks are complete. Focused SDD tests, record/skill validation,
the general test suite, both smoke examples, and the package build succeeded.
The two independent review findings were addressed and regression-tested.

The change remains implementing because the required global type check still
fails with the same 108 pre-existing errors. Its output matched the earlier
baseline byte for byte. Closing this record as verified would incorrectly waive
that check; repairing unrelated source annotations is outside this change.
Scientific behavior and current contracts were not changed. The README HDF5
correction remains a separate unapproved draft.
