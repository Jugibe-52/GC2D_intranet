+++
schema_version = 2
id = "2026-10-13-sdd-phases"
sources = ["scripts/check_sdd.py", "tests/test_sdd.py", ".agents/skills/sdd/SKILL.md", "AGENTS.md"]
documents = ["specs/README.md"]
open_questions = []
kind = "change-spec"
revision = 1
status = "implementing"
approval = "2026-10-13: The user explicitly authorized the reviewed split into spec.md, plan.md, and tasks.md, individual spec/plan/tasks operations, and the combined prepare operation, with corresponding guide, template, checker, and record migration changes."
requirements = []
proposed_requirements = []

[[criteria]]
id = "AC-001"
description = "Each existing change is separated into a specification, implementation plan, and task/evidence record without altering historical approvals or results; current numerical contracts remain unchanged."
requirements = []

[[criteria]]
id = "AC-002"
description = "The discoverable skill supports spec, plan, tasks, and combined prepare, preserves implement/verify/status, and applies one joint review before implementation with appropriate scope and revision handling."
requirements = []

[[criteria]]
id = "AC-003"
description = "The read-only checker validates staged drafts, cross-document identity and revisions, criterion coverage, task dependencies, and evidence; required validation outcomes are recorded accurately."
requirements = []
+++

# Separate SDD preparation phases: specification

## Problem

The original workflow stores specification, design, tasks, and evidence in one
change document and exposes only combined preparation. The user wants explicit
specification, plan, and task phases with matching skill operations.

## Scope

Reorganize the local SDD records and skill interface, migrate existing records,
and update the schema checker, tests, templates, guide, and repository routing.
Exclude numerical APIs, algorithms, study code, personal configuration, and
implementation of the separate README HDF5 correction draft.

## Proposed behavior

Give each change a directory containing spec.md, plan.md, and tasks.md. Keep
state/approval and criteria in spec.md, design in plan.md, and actionable tasks,
checks, and observed evidence in tasks.md. Allow progressive draft preparation.
Use declared upstream revisions to identify documents that need reconciliation.
Retain prepare as the combined workflow and one joint review before implementation.
