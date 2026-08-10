# Numerical diagnostics

This package contains opt-in numerical diagnostics used by reproducible studies.
Production methods emit neutral integration-stage or complete-step
observations; Jacobian calculations, symplecticity metrics, and output
persistence live here so they cannot affect simulations unless an observer is
explicitly passed. Generic complete-step maps use centered differences.
Implicit ABBA step observations additionally expose their converged stages, so
diagnostics can select either the implicit-function factorization or the
equivalent stage-increment factorization of the ideal-root tangent.

`diagnostics.symplecticity.SymplecticityObserver` studies the two-copy GC
state with `track_energy=False`. It writes indexed blocks below:

```text
outputs/<notebook folder>/<notebook stem>/<YYYY-MM-DD>/
```

Each block consists of a scalar CSV summary, compressed arrays in NPZ format,
and versioned JSON metadata. `diagnostics.output.write_diagnostic_block`
provides the shared synchronized writer. Existing blocks are never overwritten.

`diagnostics.projection.ProjectedSymplecticityAreaObserver` propagates the
tangent of the complete physical map ``P Phi E`` from the initial GC boundary.
At every selected complete BM4 step it records the projected Jacobian, its
symplectic defect, the separation between internal copies, and the transported
polygon area.

`diagnostics.symplecticity.GCAreaSymplecticityObserver` differentiates
complete physical GC steps. It records both the local symplecticity defect of
each numerical step and the accumulated defect, determinant drift, and
transported area of the discrete flow. Its `jacobian_method` is one of
`finite_difference`, `implicit_function`, or `stage_increment`; the two
analytic choices require an implicit ABBA step observation.
