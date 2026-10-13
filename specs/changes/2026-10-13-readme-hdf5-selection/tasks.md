+++
schema_version = 2
kind = "change-tasks"
id = "2026-10-13-readme-hdf5-selection"
sources = ["README.md", "src/potential/potential.py", "src/potential/load.py"]
documents = ["specs/current/potential-import.md", "docs/dynamics/gc2d-h5-import.md"]
open_questions = []
spec_revision = 1
plan_revision = 1

[[tasks]]
id = "T-001"
description = "After acceptance, correct the outdated paragraph in the GC2D HDF5 potential section of README.md."
done = false
depends_on = []

[[tasks]]
id = "T-002"
description = "Review the corrected paragraph against the current import contract and perform the planned checks."
done = false
depends_on = []

[[checks]]
id = "V-001"
criteria = ["AC-001", "AC-002"]
procedure = "Compare the README HDF5 paragraph with POT-001 through POT-004 and Potential.load; verify that the existing example still passes its frequency explicitly."
tests = []
result = "pending"
evidence = ""
checked_on = ""

[[checks]]
id = "V-002"
criteria = ["AC-001", "AC-002"]
procedure = "Run python -m unittest discover -s tests -p 'test_gc2d_h5_potential.py' -v and python -m unittest discover -s tests -p 'test_input_validation_contracts.py' -v, then python scripts/check_sdd.py."
tests = ["tests/test_gc2d_h5_potential.py", "tests/test_input_validation_contracts.py"]
result = "pending"
evidence = ""
checked_on = ""
+++

# Correct the README HDF5 selection description: tasks

## Execution

Migrated from the original combined record on 2026-10-13. Its approval,
criteria, task completion, check results, and evidence are preserved. No task
dependencies were recorded previously; empty dependency lists preserve that
history. This file reorganization does not authorize or revalidate the change.

## Outcome and limitations

Not implemented or approved. This record is the first real SDD proposal and
must remain a draft until the user reviews it. The initial SDD installation
does not authorize or complete this separate correction.
