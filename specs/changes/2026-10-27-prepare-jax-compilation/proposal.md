# Proposed change

Keep `prepare_step` responsible for the callable numerical map. Add
`prepare_integration` in `execution._jax_fixed`, called by `Execution.run`
inside its existing lifecycle guard, before recorder allocation and `run_steps`.
It prepares the fixed schedule and sampling arrays, places inputs on the chosen
device, and explicitly lowers and compiles the complete integrator with JAX.
Lowering translates the traced calculation into the compiler's input form.

Return a typed `PreparedIntegration` containing the executable, per-run inputs,
and recording information. The JAX `run_steps` executes this object, synchronizes
outputs, validates convergence and records results. Compile the step and loop
together. Keep a bounded cache of executables keyed by the prepared map, energy
functions, input shapes, dtypes and placement; never cache per-run initial values.

Public calls, numerical formulas, tolerances, physical coordinate conventions,
main/last/shadow steps, energy observation, result layout and archive formats
stay compatible. CPU and hybrid routes keep their existing behavior. Preserve
lazy JAX imports and finish consumed runs after preparation failures. Update the
related thesis subsection to describe the implemented sequence.
