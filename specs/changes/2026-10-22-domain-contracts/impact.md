# Affected boundaries

Contract imports change across numerical methods, execution, formulations,
diagnostics, studies, visualization, tests, examples, and notebooks. Keep the
existing simulation facade and other explicit public exports stable. Avoid
runtime cycles with type-only references to concrete implementations.

Standalone Poincare workers must include their relocated lightweight contracts,
including the cloud transfer environment that has no scientific dependencies.
Packaging must discover every new package. Current Python pickle paths change;
new round trips must work, while old module-qualified pickles are not promised.
Recorded numeric archives and their schemas remain unchanged.

Update architecture and model documentation, current specifications, and project
memory. Preserve historical acceptance/results; only repair stale source
references needed by documentation checks. Development notebooks stay ignored
and are checked with reduced, disposable execution when relevant.
