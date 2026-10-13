+++
schema_version = 2
id = "2026-10-13-sdd-phases"
sources = ["scripts/check_sdd.py", "tests/test_sdd.py", ".agents/skills/sdd/SKILL.md", "AGENTS.md"]
documents = ["specs/README.md"]
open_questions = []
kind = "change-plan"
revision = 1
spec_revision = 1
+++

# Separate SDD preparation phases: implementation plan

## Design

Retain version 1 for current contracts and introduce version 2 for the three
change documents. Group them by change directory and validate identity, shared
criteria, revisions, ordered task dependencies, and lifecycle gates. Warn about
stale upstream revisions in drafts; reject them after approval. Keep one owner
for each metadata field and preserve recorded history during migration.

Extend the existing skill with spec, plan, and tasks operations, preserve prepare
as their composition, and rename the current-contract template to contract.md.
Update guide, discovery metadata, project routing, and memory consistently.

## Compatibility

The two local flat change records migrate together; legacy layouts produce an
explicit diagnostic. Current contracts, numerical APIs, test references, CI entry
point, and existing acceptance/evidence retain their meaning. Empty dependencies
on migrated tasks preserve the absence of previously recorded dependency data.
No intermediate user approvals are added between preparation phases.

## Verification plan

Use isolated fixture repositories to test staged preparation, malformed metadata,
missing/duplicate references, revision mismatches, dependencies, and closure
without evidence. Verify templates and read-only CLI behavior. Independently
exercise realistic skill operations and scope changes. Run required repository
checks and report pre-existing failures without changing unrelated source code.
Notebook execution is unnecessary for this documentation/tooling change.
