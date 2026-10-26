# Affected boundaries

Restore request imports, packaging and Modal source lists. Adapt simulation and
executor entry points, study calls, tests, examples and affected notebook cells.
Keep geometry generation, states, method preparation and numerical algorithms
unchanged. Update current documentation, model guides, memory and their Spanish
copies; preserve historical change records and backup notebooks.

Modal payload version 3 carries the request only inside the problem. Reject older
payloads explicitly; clients and workers need matching code before deployment.
Recreate saved configured problems and receipts from earlier versions. Numerical
NPZ/HDF5/CSV archive schemas are unchanged. No remote deployment is requested.

Verify request ownership, independent reuse, nonzero start times, custom executor
dispatch and local worker serialization, then run typing, the full test suite,
smoke examples, build and SDD checks. Execute only affected setup cells or small
representative notebook cases, without overwriting stored study results.
