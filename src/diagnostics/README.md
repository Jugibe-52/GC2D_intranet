# Numerical diagnostics

This package contains opt-in numerical diagnostics used by reproducible studies.
Production methods emit neutral integration-stage or complete-step
observations; Jacobian calculations, symplecticity metrics, and output
persistence live here so they cannot affect simulations unless an observer is
explicitly passed. Generic complete-step maps use centered differences.
Implicit ABBA step observations additionally expose their converged stages, so
diagnostics can select either the implicit-function factorization or the
equivalent stage-increment factorization of the ideal-root tangent. These step
observations also expose the selected nonlinear solver, correction and residual-
evaluation counts, and accepted residual so solver-work observers do not repeat
the nonlinear solve.

`diagnostics.ImplicitABBAIterationObserver` records nonlinear iterations,
explicit residual evaluations, final infinity-norm residual, effective stopping
tolerance, residual-to-tolerance ratio, and projection-multiplier norm for
selected accepted steps. It supports both implicit ABBA formulations, including
Newton and Broyden, and defaults to sampling every complete step. Compatibility
fields retain the former `newton_*` spelling while solver-neutral properties
and arrays use `nonlinear_*` names.

`diagnostics.ImplicitBM4IterationObserver` uses the same persisted record
schema for `BM4Implicit1` and `BM4Implicit2`. Each observation represents the
single Hairer projection solve surrounding one complete twelve-stage BM4
cycle. The observer does not perform additional BM4 maps or finite-difference
Jacobians.

`diagnostics.abba_jacobian.ImplicitABBAJacobianObserver` is independent of the
symplecticity studies. It records the local physical Jacobian of selected
complete implicit ABBA steps, extracts one `2 x 2` block per independent GC
particle, classifies the characteristic discriminant, and persists eigenvalue,
eigenvector, and singular-value decompositions. Complex eigenvectors are kept
in the NPZ arrays but are not assigned real line angles. The observer does not
accumulate tangent maps and does not calculate area or symplecticity metrics.

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
