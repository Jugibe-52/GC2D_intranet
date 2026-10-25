# Affected code and uses

Potential and dynamics gain internal conversion ownership. Their optional JAX
modules and caches must preserve source identity, spline sharing, GC effective
fields, explicit JAX use and lazy imports. `InitialValueProblem` retains its
current public contract.

Solvers gain optional JAX classes and a shared solve contract under
`contracts/solvers/` where needed. Migrate algebraic validation, derivative and
linear-algebra helpers without making solvers aware of a particular projection.
The current exact-type solver validation cannot accept prepared alternatives
unchanged; compatibility checks must preserve supported configurations.

Fixed numerical steps, maps, projections and their shared base require
array-compatible operations and records. Remove their parallel JAX steps and
configuration containers. Update `contracts/methods/`, array helpers and
compiled-step consumers where required. The adaptive implementation is excluded.

Execution preparation delegates conversions, binds a common fixed step and
preserves compiled recording, observer binding and lifecycle boundaries.
The SciPy/JAX adapter remains in execution. Audit local Modal serialization and
worker dependencies without launching remote jobs. Keep prepared device resources
out of portable job configuration and result archives.

## Existing evidence and future checks

Relevant suites include `test_solver_configuration`,
`test_initial_solvers_formulations`, `test_function_methods`,
`test_method_validation_contracts`, `test_compiled_step_contract`,
`test_execution`, `test_execution_modal`, `test_shared_array_dynamics`,
`test_potential_execution`, `test_jax_potential`, `test_jax_rk4`,
`test_jax_methods` and the energy, projection and palindromic tests. Some tests
import private conversion/dispatch helpers; migrate those expectations while
retaining independent numerical checks. Old recorded successes are context,
not evidence that this proposed implementation works.

Keep public constructors, projection strings/classes, solver defaults, CPU/JAX
selection and NumPy result arrays. No physical data, archive schema or notebook
migration is intended. Inspect direct private consumers; only update affected
notebook cells, validating reduced disposable copies rather than full studies.
Do not change dependency versions or Git tracking.

After implementation, update the execution, solver/formulation, dynamics/JAX and
contract guides, affected fixed-model architecture documents and diagrams,
`specs/current/simulation.md` and any other genuinely affected current contract.
Preserve the model theory entry points and PDFs; mathematics is unchanged.
Reconcile obsolete memory entries about method-owned JAX adapters after the new
behavior is verified. Maintain Spanish reading copies. Adaptive model documents
and historical change records remain intact.
