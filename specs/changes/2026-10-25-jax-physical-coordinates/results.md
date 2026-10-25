# Results

On 2026-10-25 the user accepted the coordinate-based separation and requested
SDD preparation and execution together. The four documents were presented before
implementation. The final design incorporates the subsequent clarification:
the array module is a fixed class attribute, without a property or per-operation
backend recognition.

## Implemented

`PhysicalCoordinates` declares `array_namespace: ClassVar[ModuleType] = np`;
`JaxPhysicalCoordinates` only substitutes `jnp`. Shape validation, named views,
packing and role access are inherited. The JAX module loads through explicit
imports; importing the base dynamics or running NumPy does not require JAX.

`JaxGuidingCenterDynamics` and `JaxFullCyclotronDynamics` only declare JAX
coordinates. They inherit constructors and every physical method. Shared
methods use `self.coordinates` and its module. Execution creates the appropriate
class with JAX potentials while sharing the original samples and fitted splines,
without repeated gyroaveraging or source mutation. Compatible prepared objects
are reused; arbitrary subclass overrides remain rejected.

Direct compiled calls use the new classes from `dynamics.jax_dynamics` with
`JaxPotential`. Existing simulation calls and saved numerical formats remain
unchanged. Dynamics guides, model architecture, current contracts, migration
notes and Spanish copies are aligned. Memory records the fixed-module design
in 299 lines, with its Spanish reading copy updated.

## Checks

All commands used `.venv/bin/python` (Python 3.11.10). Numerical comparisons
used JAX CPU float64; the state-layout tracing test also covered float32.

- `PYTHONPATH=src:tests MPLBACKEND=Agg python -m unittest test_shared_array_dynamics test_potential_execution test_physical_workspaces test_independent_formulations test_compiled_step_contract test_package_layout test_jax_rk4.OptionalExecutionTests test_jax_potential.OptionalJaxImportTests -q`: 60 tests passed in 46.762 seconds.
- `python -m mypy src`: passed for 306 source files.
- `MPLBACKEND=Agg python -m unittest discover -s tests -v`: ran 703 tests in 735.764 seconds, with 699 passing, three hardware skips and one error. The error was an unmigrated tracing test in `test_states.py` using base NumPy coordinates; it now uses the explicit JAX dynamics coordinates. Production code did not change after this run.
- `MPLBACKEND=Agg python -m unittest discover -s tests -p 'test_states.py' -v`: all 13 tests passed in 0.444 seconds after migrating that test. Validation combines the full run with this focused rerun; the complete suite was not repeated. An earlier interrupted run is not counted as complete validation.
- Both smoke examples passed: `examples/gc_orbit.py` completed 20 steps; `examples/projected_abba.py` completed four steps with one Newton correction each.
- `python -m build`: built the source distribution and wheel. The wheel contains both new modules and the final fixed module attribute.
- The direct JAX example in `docs/dynamics/protocols.md` executed successfully.
- All 28 affected Markdown documents have Spanish copies with corresponding headings and identical code blocks. Spanish files remain ignored by Git. `git diff --check` passed.
- SDD preparation validation passed with 108 records and 22 requirements; validation including results passed with 109 records and 22 requirements.

## Numerical comparison

Compared the final implementation against coordinate/GC/FC modules loaded in
memory from commit `3e03b4b`, without changing the working source. All 54
comparisons passed at rtol `2e-11`, atol `2e-12`. Both NumPy and compiled JAX
results matched their pre-change counterparts exactly in the evaluated cases.
The largest NumPy/JAX absolute difference was `9.992007221626409e-16`; for
FC split flows it was `1.1102230246251565e-16`.

The periodic 12-by-14 field used mean `sin(x)*cos(y)`, one mode
`0.2*cos(x+y) + 0.1j*sin(x-y)` and cyclic frequency `0.7`, with cubic interpolation.
Two-particle component blocks were `[[.23,6.5],[1.14,-.4],[.3,-.2],[-.1,.5]]`.
GC used the first two blocks, rho `.1` and B `+/-1.7`; FC used rho `.3` and
eta `+/- .4`. Histories added `.03*k` for k=0,1,2 at times `[.1,.23,.4]`.
Checks covered fields, Hamiltonians, passive momentum derivatives and GC
Jacobians. FC flow durations were `-.13`, `0`, `.13` at time `.17`.

A separate four-case check used GC/FC with RK4 and DOP853, step/max_step `.05`,
interval `[0,.2]`, output times `[0,.03,.1,.2]` and energy observation. Explicit
JAX objects round-tripped through pickle produced exactly the same trajectories
and energy histories as automatically prepared objects; their initial layouts
remained structural NumPy snapshots.

The implementation and validation are complete using the full-run results and
the focused rerun above. GPU execution and live remote Modal jobs were not
exercised. No performance improvement was measured. The fixed module declaration
removes input-based backend detection; Python still reads the class attribute
when executing the shared methods.
