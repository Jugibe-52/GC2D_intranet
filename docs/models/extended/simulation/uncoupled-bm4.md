# Uncoupled BM4 and independent C

BM4Implicit and BM4Midpoint have no coupling_frequency constructor argument.
Both bind GCDoubledMaps with coupling_frequency=None, also in JAX. Their signed
coefficients, endpoint-time convention and projection placement are unchanged.
The former implicit default (zero coupling) is preserved; midpoint now uses
no coupling instead of its former pi/8 default. Observer and archive diagnostics
retain a fixed zero rate for compatibility with recorded metadata.

Use `from formulations import HarmonicCoupling` for the independent C operation.
`HarmonicCoupling(frequency).apply(doubled_state, duration)` accepts a finite
component-major vector (u_x, u_y, v_x, v_y) of length 4N and returns a new array.
It preserves the copy mean and rotates their difference by 2*frequency*duration.
It leaves the caller's clock and passive momentum untouched. BM4 never calls C.

Study schemas that retain a legacy rate accept only zero for new calculations.
Archived coupled results and frozen historical bundles retain their original
meaning; current methods cannot resume those coupled trajectories. Historical
campaign descriptions are records of those earlier experiments.
