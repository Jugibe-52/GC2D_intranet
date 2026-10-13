+++
schema_version = 2
kind = "change-plan"
id = "2026-10-13-readme-hdf5-selection"
sources = ["README.md", "src/potential/potential.py", "src/potential/load.py"]
documents = ["specs/current/potential-import.md", "docs/dynamics/gc2d-h5-import.md"]
open_questions = []
revision = 1
spec_revision = 1
+++

# Correct the README HDF5 selection description: plan

## Design

After the user accepts this specification and plan, edit only the affected
paragraph. Compare terminology with the implementation and current POT
requirements, run the planned checks, then record the result. The existing
code example already supplies the frequency and should remain intact.

## Compatibility

Documentation-only correction. Public names, loader defaults, saved files,
tracked notebook status, and numerical results remain unchanged.

## Verification plan

The manual comparison establishes that prose and example match the intended
contract. Existing deterministic loader tests confirm the referenced behavior;
the structural check validates this record's references. Additional numerical
tests are unnecessary for the text correction. A test result from SDD setup
does not by itself verify the future README edit.
