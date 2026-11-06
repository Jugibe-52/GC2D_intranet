# Proposed behavior

The user accepted this scope on 2026-11-06 by requesting preparation and execution of the proposal presented in the conversation. Keep all three Jacobian modes. In auto, select an available derivative without evaluating it; otherwise select finite differences. For Newton, analytic requires a supplied derivative, and finite_difference explicitly selects approximation. Broyden retains its projection-supplied initial matrix and residual-only updates. Validate evaluated derivatives when needed and propagate failures.

Remove request from method.new_run and initialize and remove method.request. Execution passes the request directly to scheduling and recording. Fixed methods keep their nominal step and receive the actual time, state and duration in advance. Keep fresh instances, solvers, projections and passive energy observations.

Move the SciPy controller into execution. It prepares each live solver for the requested interval and decides whether requested samples need dense output; numerical advancement retains the live solver and its numerical state. Preserve accepted intervals, observer independence, interpolation work counts and reusable configurations. Adjust JAX preparation to use the same request-free initialization.

Unify derivative selection across Gauss, SDIRK, HBVM and projected compositions. Preserve HBVM’s existing first-residual validation boundary and the numerical equations. Update chapter 3 after implementing the behavior.
