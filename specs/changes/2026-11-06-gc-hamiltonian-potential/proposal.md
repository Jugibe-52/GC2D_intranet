# Proposed change

Rename the prepared `GuidingCenterDynamics.effective_potential` object to `hamiltonian_potential`, including study results that expose the same object. Keep its type `Potential`, construction, signed B scaling and gyro-radius convention. Remove the old attribute rather than adding an alias.

Evaluate spatial derivatives directly in `vector_field` and pack `(-dH_dy, dH_dx)` in the existing component-major order. Keep `hamiltonian(t, state)`, the particle Jacobians and the time-conjugate derivative on that same prepared object. JAX preparation must continue sharing fitted splines without repeating gyroaveraging.

Keep `Potential.electric_field` and its full-cyclotron use unchanged. Migrate the guiding-center diagnostic that currently calls it on the Hamiltonian to explicit derivatives, preserving its recorded negative-gradient values and diagnostic names. The public energy evaluation signature and numerical equations remain unchanged.
