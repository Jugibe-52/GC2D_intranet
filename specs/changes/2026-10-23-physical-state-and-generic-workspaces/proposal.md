# Proposed behavior

## Physical coordinates belong to dynamics

Guiding-center dynamics declares `(x, y)`; full-cyclotron dynamics declares
`(x, y, vx, vy)`. These declarations are available without constructing a
potential or a dynamics instance. Generic operations validate, pack and split
component-major arrays using that description. Remove the separate GC/FC layout
classes rather than relocating or aliasing them.

Keep a lightweight immutable coordinate description, including component order
and position/velocity roles, in initial conditions and completed results. It is
data, not another subclass for each model. Reading a saved state must not require
a live dynamics, potential, numerical method or formulation. Existing physical
accessors and GC/FC initial-condition factories retain their behavior.

The problem obtains the physical description from dynamics. External systems
may declare their coordinates or supply an explicit structural description;
inheritance from built-in dynamics is unnecessary. Preserve the existing
`layout=` input and `.layout` accessors for this physical description, migrating
their implementations and imports. Validate declared order as well as dimensions;
never infer GC/FC identity from vector length or dimensions 2/4 alone.

## One physical state between steps

For d physical components and N particles, the accepted vector Z always has
shape `(d*N,)`. Histories retain `(d*N, *sample_axes)`. GC keeps all x entries
followed by all y entries; FC appends the vx and vy blocks. No unit conversion
or interleaving change is introduced.

The integration snapshot owns scalar time, Z and optional passive momentum
kappa with shape `(N,)`. Kappa starts at zero and remains persistent between
steps. Time and kappa are separate from Z and from temporary physical copies.
Use the same snapshot convention for single and double formulations.

Keep `advance` with explicit time and effective duration. Its state input and
`StepResult.state` become physical Z; carry momentum explicitly in a separate
optional input/output, and in the corresponding initial state and step records.
Controllers sample complete physical/auxiliary snapshots. Never hide momentum
in mutable run state: independent shadow steps must start with their own copy
of the accepted left endpoint, including its momentum.

| Model | Persistent Z, either formulation | Single workspace | Double workspace |
| --- | ---: | ---: | ---: |
| GC | 2N | 2N | 4N |
| FC | 4N | 4N | 8N |
| External d-component model | dN | dN | 2dN |

These sizes exclude the separate scalar time and optional N momentum values.
Energy tracking must not change physical solver scales or adaptive error control.

## Formulations construct temporary copies

`SingleFormulation(physical_size)` and `DoubleFormulation(physical_size)` retain
only the positive size of one physical copy. They do not accept a dynamics,
layout, particle count, tracking flag, initial values, time, projection or solver.
`lift(Z)` creates one or two equal copies; `join_copies` handles explicit copies,
including `(Z+mu, Z-mu)`; `split_copies` reads their blocks. Validate sizes and
sample axes while preserving NumPy/JAX semantics and independent mutable work.
An already doubled vector must not be accepted as physical input accidentally.

The method controls the complete operation:

```text
accepted Z -> temporary copies -> numerical composition -> projection -> new Z
```

Implicit symmetric projection prepares shifted copies for each candidate. Use
one outer projection around the established composition; midpoint variants
retain their arithmetic projection. Projection and acceptance produce the next
physical vector without retaining a second identical accepted copy. Formulation
helpers never average implicitly or select a numerical algorithm.

Remove auxiliary packing, physical-energy evaluation and history diagnostics
from the formulation classes. Methods retain passive quadrature and provide H
on saved physical states; output helpers combine H and kappa. Preserve
`d kappa/dt = -partial_t H`, physical Hamiltonian normalization and accepted-stage
reuse. Any summed doubled-map momentum convention remains explicit inside the
relevant method; it must not rescale persistent physical kappa.

## Complete the ownership change

Move GC/FC split-map operations, harmonic coupling and projection equations to
method-owned modules. Remove the model-specific state containers and the old
`GCExtendedFormulation`, `GCStageProjectedFormulation` and `FCSplitFormulation`
configuration wrappers. Provide their direct/adjoint operations through the
method-owned map interfaces, including existing coupling and energy capabilities.
Maps receive supplied work states and do not construct a second run initial state.
The `formulations` package must import neither physical model implementations
nor methods, execution or solvers; coordinate contracts belong with dynamics,
and numerical map/projection contracts with methods.

Keep public numerical method names, `SingleFormulation`/`DoubleFormulation`,
Newton/Broyden and `ReducedProjection`/`SimultaneousProjection` through their
canonical modules and the `simulation` facade. Document changed constructors
and retired `formulations` routes; migrate consumers without compatibility trees.
Existing `solver=None` defaults, coefficients, field evaluation, method-owned
step sizes, clipped final steps and sampling rules retain their meanings.

Saved physical arrays, archive schemas and energy/work histories stay compatible.
Dimension metadata must describe actual storage: the accepted vector has dN
entries, workspace has c*dN, and separate auxiliaries are reported separately.
Historical archives retain their recorded dimensions and labels. Generic FC
duplication does not add FC support to GC-only integrators. Methods that persist
unequal copies between steps are outside this proposal; current projected
methods accept equal copies and need no such state.
