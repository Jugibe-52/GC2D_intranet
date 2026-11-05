# Proposed interface

Prepare one `ProjectionProblem` for each step, binding the initial state, time,
step size, composition, formulation and derivative strategy. Its `residual(x)`
returns the residual vector after computing the copies; `jacobian(x)` returns
its derivative with respect to the same unknown. Reduced equations use
`x = mu` (2N entries), simultaneous equations use `x = (U, V, mu)` (6N).

Rename the formulation helpers to `assemble_residual`; they retain the
formulation-specific algebra. Keep the solver adapters responsible for carrying
the evaluated composition trace. Use an immutable, short-lived evaluation
context to reuse that trace when evaluating the corresponding Jacobian, without
a mutable cache shared between candidates or runs. Independent particle
Jacobians retain their batched 2x2 or 6x6 representation and block solves;
dense derivatives retain all cross-particle terms.

Newton continues to solve linear systems; Broyden keeps its existing initial
Jacobian approximation and updates. Preserve guesses, tolerances, convergence
counters, observations and reconstruction. Document the complete interface in
the TFM and keep the detailed Newton/Broyden explanation in chapter 3.

This implements the interface accepted in the conversation. No numerical
method, solver algorithm or saved-data format is being changed.
