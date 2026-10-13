+++
schema_version = 2
kind = "change-spec"
id = "2026-10-13-sdd-bootstrap"
sources = ["scripts/check_sdd.py", "tests/test_sdd.py", ".agents/skills/sdd/SKILL.md", "AGENTS.md", ".github/workflows/ci.yml"]
documents = ["specs/README.md", "specs/current/potential-import.md", "specs/current/simulation.md", "specs/current/results.md"]
open_questions = []
revision = 1
status = "implementing"
approval = "2026-10-13: The user explicitly requested implementation of the reviewed GC2D SDD specification and plan. This acceptance covers the SDD installation, not the separate README correction draft."
requirements = ["POT-001", "POT-002", "POT-003", "POT-004", "POT-005", "SIM-001", "SIM-002", "SIM-003", "SIM-004", "SOL-001", "SOL-002", "SOL-003"]
proposed_requirements = []

[[criteria]]
id = "AC-001"
description = "The baseline describes the existing potential import, simulation, and result contracts with stable requirements and checked test references."
requirements = ["POT-001", "POT-002", "POT-003", "POT-004", "POT-005", "SIM-001", "SIM-002", "SIM-003", "SIM-004", "SOL-001", "SOL-002", "SOL-003"]

[[criteria]]
id = "AC-002"
description = "The discoverable local skill supports prepare, implement, verify, and read-only status, preserving acceptance and scope boundaries."
requirements = []

[[criteria]]
id = "AC-003"
description = "The standard-library checker reports invalid records and references, enforces closure evidence, and never executes procedures or writes files."
requirements = []

[[criteria]]
id = "AC-004"
description = "The checker is integrated into CI, the README correction remains a draft, and required project validation passes without changing numerical APIs."
requirements = []
+++

# Establish specification-driven development for GC2D: spec

## Problem

The project has numerical documentation, agent instructions, and extensive tests,
but no explicit lifecycle connecting a reviewed specification and plan with
implementation and observed acceptance evidence. Existing documentation can
drift, as demonstrated by the README's HDF5 selection paragraph.

## Scope

Introduce a minimal current-contract baseline, change records, a local skill,
templates, structural validation, and CI integration. Exclude new numerical
features, a complete specification of every method, MCP services, changes to
personal configuration, and automatic correction of the separate README draft.

## Proposed behavior

Use SDD by default for substantial changes. Review the specification and plan
together before implementation, respect existing authorization, and permit
small corrections to retain the ordinary workflow. Record real evidence before
marking a change verified. Keep proposals separate from current contracts.
