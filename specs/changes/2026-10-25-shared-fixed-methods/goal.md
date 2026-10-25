# One fixed-step algorithm for NumPy and JAX

Fixed-step methods currently combine their numerical rules with NumPy-specific
operations, a JAX execution mode, separate JAX steps and compiled configuration
records. Understanding execution therefore requires following two versions of
several algorithms and conversion details owned by the coordinator.

Keep one implementation of each fixed-step algorithm, independent of the chosen
array library and CPU/GPU device. Potential, dynamics and algebraic solvers
should prepare their own compatible representations; execution should connect
them and coordinate the run. Methods should retain their equations and executable
algorithms without importing JAX or choosing a device.

This follows the completed
[execution preparation](../2026-10-25-execution-preparation/proposal.md) and
[JAX coordinates](../2026-10-25-jax-physical-coordinates/proposal.md) changes.
It is a new proposal, not a reopening of their recorded checks.

Preserve trajectories, projection choices, numerical controls, observation and
saved-data contracts. DOP853 and Radau are explicitly outside the refactoring:
their implementations and existing SciPy/JAX behavior must remain unchanged.
Moving duplicate algorithms to another package would not achieve this goal.
