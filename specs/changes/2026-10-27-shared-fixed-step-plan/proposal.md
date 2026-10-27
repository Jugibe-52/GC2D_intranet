# Proposed change

Add an immutable NumPy plan containing main-step starts, ends and durations, output times, per-step sample offsets, endpoint/shadow modes and shadow durations. A single Python builder combines the validated request and nominal step without importing JAX or evaluating the field.

Execution creates this plan once before result allocation or JAX preparation. The ordinary fixed controller and compiled worker consume the same decisions. Custom controllers retain their own interval and sampling behavior. Direct standalone fixed-controller use can build its plan on demand.

Preserve endpoint tolerance and end-before-start priority, independent shadow states and energy, final-step clipping, observer order, diagnostics and compiled-cache reuse. Public simulation calls and saved-data formats do not change. Solver preparation is outside this change.
