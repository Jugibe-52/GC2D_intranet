# Research diagnostics

This directory contains opt-in numerical diagnostics used by development
notebooks. Production systems only emit neutral integration-stage observations;
finite-difference Jacobians, symplecticity metrics and output persistence live
here so they cannot affect simulations unless an observer is explicitly passed.

`research.symplecticity.SymplecticityObserver` studies the two-copy GC state
with `check_energy=False`. It writes indexed blocks below:

```text
outputs/<notebook folder>/<notebook stem>/<YYYY-MM-DD>/
```

Each block consists of a scalar CSV summary, compressed full Jacobians in NPZ
format and JSON metadata. Existing blocks are never overwritten.

`research.projection.ProjectedSymplecticityAreaObserver` propagates the tangent
of the complete physical map ``P Phi E`` from the initial GC boundary. At every
selected complete BM4 step it records the projected Jacobian, its symplectic
defect, the separation between internal copies and the transported polygon area.
