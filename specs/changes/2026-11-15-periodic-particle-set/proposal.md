# Search and verification

Use the checksum-verified Gauss–Legendre campaign returns as candidate seeds. Refine zeros of the integer-cycle return map minus the initial position, with fixed-step JAX integrators and Modal execution. Keep unwrapped zero-winding closure; spatial period is two, not one. Exclude stationary points and duplicate phase points on the same orbit.

Refine the step at least twice and cross-check with another fourth-order method. Require measured closure below 1e-6 of the base cell width, stable initial conditions under refinement and separation from earlier returns. Record actual residuals, periods, integration controls and provenance; these are numerical evidence, not a mathematical existence proof. Store final records in data/particles/periodic_8.json and update the simple viewing notebook. Do not upload the dataset to B2.
