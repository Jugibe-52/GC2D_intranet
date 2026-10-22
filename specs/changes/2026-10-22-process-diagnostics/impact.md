# Affected uses

Diagnostic implementations and imports in studies, visualization, tests,
examples and notebook-generating scripts move together. Update executable
imports in affected tracked and local development notebooks without rerunning
long studies or replacing stored results. Keep their tracking status unchanged;
historical `.from-b2-*` conflict copies retain their original source references.

Update current documentation and matching Spanish copies. Older change records
retain their acceptance and evidence; only source references that the document
checker requires to exist are redirected to the relocated files. Update stale
module references in project memory without adding duplicate policy.

Existing `from diagnostics import ...` names remain available. Direct imports
from relocated submodules must use their new paths. Numerical records, CSV/NPZ
schemas, output paths, units and precision are unchanged; existing saved data
remain readable. Packaging must include the new subpackages.

Verification covers import boundaries, all public exports, representative
observer and archive tests, the full suite, source typing, supported examples,
distribution builds and small notebook import checks.
