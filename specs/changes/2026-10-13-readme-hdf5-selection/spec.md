+++
schema_version = 2
kind = "change-spec"
id = "2026-10-13-readme-hdf5-selection"
sources = ["README.md", "src/potential/potential.py", "src/potential/load.py"]
documents = ["specs/current/potential-import.md", "docs/dynamics/gc2d-h5-import.md"]
open_questions = ["User review of this documentation-only specification and plan is pending."]
revision = 1
status = "draft"
approval = ""
requirements = ["POT-001", "POT-002", "POT-003", "POT-004"]
proposed_requirements = []

[[criteria]]
id = "AC-001"
description = "The HDF5 section describes source field 0 as constant and indx=(15,) as the original variable-field index, without amplitude ranking."
requirements = ["POT-001", "POT-002"]

[[criteria]]
id = "AC-002"
description = "The text requires an explicit characteristic_frequency and limits the period-one statement to a selected mode with that source frequency."
requirements = ["POT-003", "POT-004"]
+++

# Correct the README HDF5 selection description: spec

## Problem

The README's HDF5 paragraph still states `indx=(0, 1)` and describes a dominant
mode. The loader uses the original variable-field index `(15,)`, includes the
mean independently, and requires an explicit characteristic frequency. Its
unconditional period-one description is also too broad for arbitrary scales.
This discrepancy was observed during SDD setup; no runtime defect is asserted.

## Scope

Correct the outdated descriptive paragraph in the README HDF5 section. Exclude
loader changes, index reinterpretation, archive migration, numerical-method
changes, notebook execution, and unrelated README edits.

## Proposed behavior

Describe `indx=(15,)` with source field 0 always contributing the constant term.
Explain that the supplied angular-frequency scale controls normalization and
that a selected mode has period one when its source frequency equals that scale.
Retain the established default magnetic field and characteristic length. Link
to the existing detailed HDF5 contract instead of duplicating its derivation.
