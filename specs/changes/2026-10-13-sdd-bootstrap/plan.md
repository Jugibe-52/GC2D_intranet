+++
schema_version = 2
kind = "change-plan"
id = "2026-10-13-sdd-bootstrap"
sources = ["scripts/check_sdd.py", "tests/test_sdd.py", ".agents/skills/sdd/SKILL.md", "AGENTS.md", ".github/workflows/ci.yml"]
documents = ["specs/README.md", "specs/current/potential-import.md", "specs/current/simulation.md", "specs/current/results.md"]
open_questions = []
revision = 1
spec_revision = 1
+++

# Establish specification-driven development for GC2D: plan

## Design

Markdown documents with versioned TOML frontmatter are stored in the repository.
The local skill coordinates investigation, accepted implementation, verification,
and status. A standard-library Python checker validates metadata, references,
and lifecycle completion without importing project code or executing procedures.
The existing quality CI job invokes it; dedicated unittest cases exercise its
failure modes using disposable repositories.

## Compatibility

The numerical public API, algorithms, dependencies, and notebook tracking rules
remain unchanged. Existing detailed architecture and theory retain their current
locations. The new workflow follows the existing personal write-authorization
rules. It requires Python 3.11 or later, consistent with the project.

## Verification plan

Check baseline semantics against source and existing deterministic tests. Exercise
the record validator with valid and malformed repositories and validate the skill
both structurally and through independent workflow scenarios. Run required type,
test, example, and package checks. Record pre-existing failures as failures;
structural validity of this record does not imply full verification.
