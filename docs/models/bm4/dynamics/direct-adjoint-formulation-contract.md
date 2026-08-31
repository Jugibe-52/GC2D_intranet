# Physical-system and formulation contract for BM4

## Scope

BM4 is a fourth-order composition of a first-order map and its adjoint. The
composition engine does not construct those maps from a bare vector field;
instead, a `DirectAdjointFormulation` prepares them for one
`InitialValueProblem`. This boundary keeps the physical equations in
`dynamics`, the packed state convention in `initial_conditions`, and the
solvable direct/adjoint maps in `simulation.formulations`.

The generic public method

```python
BM4Composition(formulation, track_energy=False)
```

therefore accepts any formulation satisfying that protocol. The repository
provides two physical bindings:

| Physical system | Packed physical state | Formulation | BM4 internal state |
|---|---|---|---|
| Guiding centre (GC) | `[x_1, ..., x_N, y_1, ..., y_N]` | `GCExtendedFormulation` | Two physical copies `(u, v)`, plus optional momentum |
| Full cyclotron (FC) | `[x_1, ..., x_N, y_1, ..., y_N, vx_1, ..., vx_N, vy_1, ..., vy_N]` | `FCSplitFormulation` | One FC state, plus optional momentum |

`GCStageProjectedFormulation` is the uncoupled GC binding required by
`ProjectedBM4Composition` and used internally by `MidpointBM4`.

## Guiding-centre dynamics

For one particle, `GuidingCenterDynamics` uses the gyroaveraged Hamiltonian
`H(t, x, y)` and the convention

\[
f(t,x,y)=
\begin{pmatrix}-\partial_yH(t,x,y)\\ \partial_xH(t,x,y)\end{pmatrix}.
\]

The physical particles are independent under the prescribed potential. BM4
lifts the packed state `z in R^(2N)` to two copies `(u,v) in R^(4N)`. For a
fixed stage evaluation time `tau`, the uncoupled direct map of duration `s` is

\[
\widetilde v=v+s f(\tau,u),\qquad
\widetilde u=u+s f(\tau,\widetilde v),
\]

and the adjoint reverses the two shears,

\[
\widetilde u=u+s f(\tau,v),\qquad
\widetilde v=v+s f(\tau,\widetilde u).
\]

`GCExtendedFormulation(coupling_frequency=omega)` also applies the exact
harmonic binding flow. In per-particle order `(u_x,u_y,v_x,v_y)`, its matrix is

\[
C(s,\omega)=\frac12\left(B+
\cos(2\omega s)C_c+\sin(2\omega s)C_s\right),
\]

with the constant matrices implemented in `simulation/formulations/gc.py`.
Equivalently, the copy mean is invariant while the half-difference is rotated
through angle `2 omega s`:

\[
m=\frac{u+v}{2},\quad d=\frac{u-v}{2},\qquad
(u^+,v^+)=(m+R_{2\omega s}d,\;m-R_{2\omega s}d).
\]

The direct map applies this coupling after both shears; the adjoint applies it
before them. `omega=0` leaves the copies uncoupled. The coupling is absent from
`GCStageProjectedFormulation`, whose projection replaces both copies by their
arithmetic mean.

The public projection of a GC internal history is

\[
z=P(u,v)=\frac{u+v}{2}.
\]

For ordinary `BM4Composition`, this operation creates physical output but does
not re-embed the average between complete steps. Projection placement is owned
by the selected BM4 variant and is documented in the companion simulation
document.

## Full-cyclotron dynamics

`FullCyclotronDynamics` separates the exactly solvable field-free cyclotron
motion from the electric acceleration. `FCSplitFormulation` defines

- a direct map: exact cyclotron rotation followed by an electric kick; and
- an adjoint map: electric kick followed by exact cyclotron rotation.

Unlike the GC formulation, it does not duplicate the physical FC state. The
specialized projected BM4 classes are GC-only; FC is supported through
`BM4Composition(FCSplitFormulation())`.

## Energy extension

When `BM4Composition(..., track_energy=True)` is selected, the dynamics must
implement `ExtendedHamiltonianSystem`:

\[
K(z,t,k)=H(t,z)+k,\qquad \dot k=-\partial_tH(t,z).
\]

Each formulation advances the triangular momentum without feeding it back into
the physical variables. GC accumulates contributions from both copies and
publishes their half-normalized `extended_momentum`; FC publishes the direct
momentum. `energy_error` is the maximum saved-time drift of `H+k`. Enabling
this diagnostic leaves the physical trajectory unchanged.

`BM4_implicit2` instead accepts the complete one-particle extended state
`Z=(x,y,t,k)` internally and duplicates it to `(Z_1,Z_2) in R^8`. Its time and
momentum are part of the numerical map, so it is a different state extension,
not the energy-tracking mode of `BM4Composition`.

## Derivative capabilities

`BM4Implicit1` and `BM4Implicit2` support two Jacobian strategies:

| Value | Requirement and behavior |
|---|---|
| `"analytic"` | Requires `GuidingCenterDynamics` with spatial Hessians (`interpolation_order >= 3`); forms exact independent-particle stage products. |
| `"finite_difference"` | Differentiates the complete doubled BM4 map with centered differences. |

There is no automatic fallback: `newton_jacobian_method` explicitly selects
one of these paths. The fully extended `BM4_implicit2` differentiates
`(x,y,t,k)` shears analytically and additionally needs mixed space--time and
second-time potential derivatives. It currently supports exactly one GC
particle.

## Responsibility boundary

The dynamics and formulation layer owns:

- physical parameters and equations;
- packed state layouts;
- direct, adjoint, coupling, and projection maps;
- exact physical derivatives; and
- optional Hamiltonian and extended-momentum evaluations.

It does not own BM4 coefficients, projection timing, nonlinear tolerances,
fixed grids, saved output times, observers, studies, or plots. Those belong to
the numerical-method and experiment layers.

The companion source diagram is
[`direct-adjoint-formulation-contract.puml`](direct-adjoint-formulation-contract.puml).
