# Contracts by source domain

`src/contracts/` mirrors the source domains and their relevant subdirectories.
It contains all 40 existing Protocol interfaces and two TypedDict declarations,
along with the shared records, aliases, and validation already owned by contracts.
The reorganization relocated 37 protocols and two typed dictionaries from other
packages. Their numerical, plotting, persistence, and orchestration code remains
in those packages.

```text
src/contracts/
├── arrays.py
├── dynamics/protocols.py
├── execution/controllers.py
├── formulations/
│   ├── base.py
│   ├── layout.py
│   └── projection.py
├── methods/
│   ├── base.py
│   ├── compiled.py
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
│   ├── poincare_automation.py
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
| `execution` | `StepController`; fixed and adaptive controllers stay with their implementations. |
| `formulations` | Direct/adjoint and projection interfaces, plus `StateLayout`. |
| `methods` | `NumericalMethod`, `CompiledStep`, step records, and family-specific observations. |
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
