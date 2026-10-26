# Group simulation inputs

Temporal configuration currently occupies a separate top-level request module,
although it describes the same simulation input boundary as the physical problem.
Group these definitions so their ownership is clear without changing equations,
initial data, sampling, or numerical results.
