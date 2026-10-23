# Proposed behavior

Fixed methods accept `step=0.1` by default, in the problem's time units. The value
belongs to reusable method configuration and is copied into each run. `advance`
retains its name and receives each effective duration. Final and shadow steps
never overwrite the configured `step`.

For [0, 1] with `step=0.3`, accepted endpoints are 0.3, 0.6, 0.9, 1.0, with
durations 0.3, 0.3, 0.3, 0.1 within floating-point round-off. A step larger than
the interval gives one shortened step. Reject Boolean, nonpositive or nonfinite
steps and grids that cannot advance representable time. Host and JAX use the
same schedule; shadow samples remain independent of the main trajectory.

`SimulationRequest` retains only `t_span` and `output_times`; `uniform` keeps
its output-density defaults. Move its former `max_step` argument to fixed
method `step` or adaptive method `max_step`, without request aliases.
DOP853/Radau retain their adaptive controller, tolerances and first-step behavior;
their bound now belongs to the method, with the existing default 0.1.

Example: `simulate(problem, RK4(step=0.03),
SimulationRequest.uniform(t_span=(0.0, 0.1), sample_count=6))`.

Preserve equations, projection choices, tolerances, energy normalization,
physical layouts, archive schemas and `advance` APIs. Non-divisible runs
intentionally change their main grid; divisible runs should agree within
round-off. Specialized formulation maps remain outside this change.
