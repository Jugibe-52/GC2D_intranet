# Affected parts

Methods, execution, simulation, studies, diagnostics, tests and examples must
import the moved classes from their new owners. Package metadata must ship
`problem.py`. Import order must remain safe without optional JAX or Modal.

Comparison summaries and energy/recurrence plots must call the extracted
diagnostics. Migrate affected notebook source cells and local runtime copies
without full study execution or replacing saved outputs. Check representative
imports/calculations in memory. No remote jobs or transfers are included.

Update current specifications, migration/integration guides, affected model API
documents and Spanish reading copies. Preserve historical change records and
historical diagrams. Record confirmed ownership in memory within 300 lines.

CSV, NPZ and HDF5 records retain constructors, fields, shapes and schemas.
Existing Python calls to the extracted properties/method need the documented
function migration. Old module imports and pickles embedding those module paths
are not supported by aliases. New-object serialization must work.
