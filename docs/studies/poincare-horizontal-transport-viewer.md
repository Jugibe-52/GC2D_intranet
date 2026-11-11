# Horizontal transport beside Poincare particles

The four-method rho viewer optionally shows the full-integration horizontal
transport of every particle beside its selection checkbox:

\[
Q_x = \frac{\int_0^N |\dot{x}(t)|\,dt}{N(x_{\max}-x_{\min})}.
\]

The coordinate is physical unwrapped x, with one runtime unit per forcing cycle.
The numerical integral uses the trapezoidal rule on absolute physical velocity
at accepted complete-step endpoints. Extrema include the initial state and those
endpoints. The value is undefined when the width is zero. This is a finite-step
approximation of the continuous integral, not a sum of Poincare displacements
or signed internal splitting-stage lengths.

`visualization.poincare_transport.attach_horizontal_transport` accepts the
encoded viewer configuration and a schema-1 results mapping. Its `metric` must
identify `coordinate="physical_unwrapped_x"` and
`integral="accepted_endpoint_trapezoid_abs_vx"`. Each `cases` entry provides
`rho`, canonical `method`, `particle_ids`, `cycles`, `steps_per_cycle`, and
per-particle arrays `q_x`, `travel_x`, `width_x`, `x_min`, `x_max`. Use JSON null
for Qx only when width is zero. Length units are supplied by `metric.units`.

Validation rejects missing or duplicate rho-method cases, changed IDs, invalid
lengths, incompatible cycle counts and inconsistent normalization. Values are
matched by explicit particle ID, never by case ordering. The four method names
are `BM4Midpoint`, `BM4Implicit`, `RK4` and `GaussLegendre4`. The optional
`horizontal_transport` keyword on `export_rho_method_comparison` forwards this
validation; omitting it keeps the existing viewer behavior.

For the historical repetition, `export_horizontal_transport_viewer` takes a
baseline self-contained HTML, results, output path and `cycle_positions` mapping
from `(rho, method)` to folded arrays shaped `(cycles+1, particles, 2)`. It verifies
the baseline SHA-256 recorded in `metric.source_html_sha256`, the complete case
grid, trajectory shapes and original initial positions. The latter comparison
allows only the baseline browser's float32 rounding. It replaces every trajectory
and the selected initial panel payload, while retaining the verified field
backgrounds and particle presentation. No partial result is written on validation
failure. Historical rerun archives exported directly through the method exporter
reconstruct their field with `prepare_horizontal_reproduction` to preserve the
original geometry and normalization arithmetic.

The interface shows BM4-M, BM4-I, RK4 and GL4 in each particle card. The selected
rho updates all values together. Changing Start cycle or Current cycle does not
change metrics, which explicitly describe the entire experiment. Each value has
an accessible description with its full method name, particle ID, integral,
width, cycle count and complete steps per cycle. The JSON retains float64 values;
only the displayed text is rounded to five significant digits.

`scripts/run_horizontal_transport.py --mode publish-data` assembles matching
new trajectories and metrics from completed local archives. The development
notebook `four_methods_star_40_rho_sweep_folded_radial_potential/visualisation.ipynb`
exposes the separate preparation, benchmark, Modal calculation and local export
steps. Its paid-execution and local-rebuild switches default to false. Cloudflare
publication remains a separate explicit operation.
