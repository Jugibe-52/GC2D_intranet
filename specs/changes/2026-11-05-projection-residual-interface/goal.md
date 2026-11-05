# Projection residual interface

Make the complete projection equations available as `residual(x)` and
`jacobian(x)`, with only the current nonlinear unknown as their argument.
The current `residual(unknown, mapped, size)` merely assembles previously
computed copies and has caused confusion in the TFM explanation.

Keep the existing reduced and simultaneous mathematics, solver controls,
fixed-step trajectories, NumPy/JAX support and numerical work unchanged.
