# Contracts by source domain

`src/contracts/` mirrors source domains and their relevant subdirectories. It contains pure interfaces, shared records, aliases and contract validation. Numerical, plotting, persistence and orchestration implementations remain in their owning packages. `methods.functions` declares the supplied vector field and derivative structure; `methods.quadrature` declares weighted stage data and accepted interpolants.

```text
src/contracts/
├── arrays.py
├── dynamics/
│   ├── protocols.py
│   └── coordinates.py
├── execution/
│   ├── controllers.py
│   └── plan.py
├── methods/
│   ├── maps.py
│   ├── projection.py
│   ├── base.py
│   ├── compiled.py
│   ├── functions.py
│   ├── quadrature.py
│   ├── observation.py
│   ├── step.py
│   ├── adaptive/scipy.py
│   ├── classical/gauss_legendre.py
│   └── extended/
│       ├── abba.py
│       ├── bm4.py
│       └── core/records.py
├── solvers/nonlinear.py
├── states/history.py
├── studies/
│   ├── _trajectory_accuracy.py
│   ├── _validation.py
│   ├── bm4_parallel_recurrence.py
│   ├── comparison.py
│   ├── gc_energy_bound.py
│   └── poincare_batch_runtime/
│       ├── cloud_job.py
│       └── newton_diagnostics.py
├── diagnostics/studies/five_method_comparison_csv.py
└── visualization/
    ├── energy.py
    ├── gauss_legendre4.py
    ├── hbvm42.py
    ├── implicit_method_accuracy.py
    ├── trajectory_accuracy.py
    └── trajectory_symplecticity.py
```

Each directory is an explicit Python package. Empty package initializers avoid
loading implementations or optional backends when a contract is imported. Import
internal types from the defining submodule; public facades keep their existing
exports. References from signatures to implementation types use `TYPE_CHECKING`
to avoid runtime cycles.

| Domain | Content |
|---|---|
| `dynamics` | Physical, Hamiltonian, Jacobian, and cyclotron capability interfaces. |
| `execution` | `StepController` and the read-only `FixedStepPlan`; planning and controllers stay in `execution`. |
| `methods` | `NumericalMethod`, function types, `CompiledStep`, quadrature, step records and family-specific observations. |
| `solvers` | `NewtonIteration` and `NewtonObserver`. |
| `states` | The `DiagnosticValue` alias used by physical histories. |
| `studies` | Comparison and archive records, validation, reference-configuration views, and worker dictionaries/interfaces. |
| `diagnostics` | Structural views required by comparison CSV persistence. |
| `visualization` | Existing structural views consumed by plots, without importing plotting libraries. |

`arrays.py` remains the shared NumPy/JAX representation boundary. It does not
select devices or prepare execution. Existing record validation is unchanged;
RMS, energy, and recurrence calculations remain in `diagnostics.studies`.
Implementations such as `IntegrationMethod`, `InitialValueProblem`, `States`,
and projection equations remain in their own domains.

Common observation records live in `methods/observation.py`; adaptive observations
live in `methods/adaptive/scipy.py`, Gauss observations in
`methods/classical/gauss_legendre.py`, and ABBA/BM4 observations in their
`methods/extended/` modules. The old flat `observation.py` and `study_results.py`
modules are removed. See [API migration](api-migration.md#contracts-organized-by-source-domain)
for removed paths and archive compatibility.

Standalone cloud bundles include the complete contracts package to avoid
shadowing scientific modules with a partial package. The transfer environment
imports only the lightweight worker contracts and can load them without the
installed project, NumPy, or plotting packages. Frozen study snapshots retain
their historical source and are not migrated in place.

Map and projection contracts belong to `contracts.methods`; `StateLayout` belongs to `contracts.dynamics.coordinates`.

## Shared numerical arrays

`NumericalMethod` requires numerical preparation, without a backend declaration.
`StepResult` carries native physical arrays, work, quadrature and convergence.
`contracts.arrays` selects compatible array operations and preserves finite-value
guards while tracing; it does not choose a device or physical algorithm.
Execution registers numerical payload records at its optional JAX boundary and
compiles the same fixed `advance` function. Host observation adapters and saved
results retain NumPy arrays. See [JAX execution](jax-execution.md).

Native mapping, ordered folds and branch selection also share numerical functions without unrolling long compositions during compilation.
