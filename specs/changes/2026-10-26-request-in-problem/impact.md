# Affected uses

Update consumers in methods, execution, contracts, studies and tests, plus
packaging and Modal application source lists. Check notebooks for direct imports;
only affected cells would need migration, without running complete studies.
Update current contracts, architecture and migration guidance, project ownership
instructions and memory, preserving historical diagrams and change records.

New serialized requests use problem.SimulationRequest. Recreate old configured
object pickles and redeploy remote workers from the same revision as clients.
Saved numerical archives retain their schemas. No remote deployment is requested.
Existing request validation, import isolation, worker serialization and integration
tests will verify the move, followed by typing, the full suite, smoke examples,
a package build and SDD validation. No physical or numerical algorithm changes.
