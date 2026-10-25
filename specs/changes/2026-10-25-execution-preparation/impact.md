# Affected parts

The implementation change is confined to `src/execution/execution.py`. Existing
host and compiled workers, methods, physics, observers and the public runner
keep their responsibilities. `Execution_Modal` continues to use the same local
coordinator in its worker.

Reuse execution, observation, method and compiled-step contract tests. Add a
focused regression only where the extraction exposes an uncovered lifecycle
failure. Check source types, the complete test suite, supported examples and
package building in the existing environment, including optional JAX on CPU.

Update the execution guide and `specs/current/simulation.md` with the preparation
boundaries and maintain their Spanish copies. Notebooks, public examples,
numerical settings and saved-data formats need no migration. No remote Modal
job or GPU availability is required for this internal refactoring.
