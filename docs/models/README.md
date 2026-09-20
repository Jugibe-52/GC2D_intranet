# Numerical model documentation

All 13 methods share `IntegrationMethod.integrate`, method-specific preparation
and the [generic six-phase architecture](../simulation/integration-architecture.md).
Eleven methods use fixed control; DOP853 and Radau supply adaptive control.
Every model retains its own numerical theory and detailed simulation diagram.
Physical dynamics and potential contracts live under [dynamics](../dynamics/).

| Public method | Canonical theory | Simulation architecture | Numerical scope |
|---|---|---|---|
| `BM4Implicit` | [source](bm4-implicit/tex/theory.tex) · [PDF](bm4-implicit/tex/theory.pdf) | [guide and diagram](bm4-implicit/simulation/bm4-simulation-architecture.md) | Fixed; twelve signed stages, one symmetric projection |
| `BM4Midpoint` | [source](bm4-midpoint/tex/theory.tex) · [PDF](bm4-midpoint/tex/theory.pdf) | [guide and diagram](bm4-midpoint/simulation/bm4-midpoint-simulation-architecture.md) | Fixed; twelve signed stages, one arithmetic mean |
| `ABBA2Implicit` | [source](abba2-implicit/tex/theory.tex) · [PDF](abba2-implicit/tex/theory.pdf) | [guide and diagram](abba2-implicit/simulation/abba2-implicit-simulation-architecture.md) | Fixed; one implicit projected ABBA map |
| `ABBA2Midpoint` | [source](abba2-midpoint/tex/theory.tex) · [PDF](abba2-midpoint/tex/theory.pdf) | [guide and diagram](abba2-midpoint/simulation/abba2-midpoint-simulation-architecture.md) | Fixed; one ABBA map and arithmetic mean |
| `ABBA4Implicit` | [source](abba4-implicit/tex/theory.tex) · [PDF](abba4-implicit/tex/theory.pdf) | [guide and diagram](abba4-implicit/simulation/abba4-implicit-simulation-architecture.md) | Fixed; three unprojected factors, one outer projection |
| `ABBA6Implicit` | [source](abba6-implicit/tex/theory.tex) · [PDF](abba6-implicit/tex/theory.pdf) | [guide and diagram](abba6-implicit/simulation/abba6-implicit-simulation-architecture.md) | Fixed; seven projected factors |
| `ExplicitEuler` | [source](explicit-euler/tex/theory.tex) · [PDF](explicit-euler/tex/theory.pdf) | [guide and diagram](explicit-euler/simulation/explicit-euler-simulation-architecture.md) | Fixed; first-order method |
| `RK4` | [source](rk4/tex/theory.tex) · [PDF](rk4/tex/theory.pdf) | [guide and diagram](rk4/simulation/rk4-simulation-architecture.md) | Fixed; 4th-order method |
| `GaussLegendre4` | [source](gauss-legendre4/tex/theory.tex) · [PDF](gauss-legendre4/tex/theory.pdf) | [guide and diagram](gauss-legendre4/simulation/gauss-legendre4-simulation-architecture.md) | Fixed; 4th-order method |
| `SDIRK4` | [source](sdirk4/tex/theory.tex) · [PDF](sdirk4/tex/theory.pdf) | [guide and diagram](sdirk4/simulation/sdirk4-simulation-architecture.md) | Fixed; 4th-order method |
| `HBVM42` | [source](hbvm42/tex/theory.tex) · [PDF](hbvm42/tex/theory.pdf) | [guide and diagram](hbvm42/simulation/hbvm42-simulation-architecture.md) | Fixed; 4th-order method |
| `DOP853` | [source](dop853/tex/theory.tex) · [PDF](dop853/tex/theory.pdf) | [guide and diagram](dop853/simulation/dop853-simulation-architecture.md) | Adaptive; SciPy session and dense output |
| `Radau` | [source](radau/tex/theory.tex) · [PDF](radau/tex/theory.pdf) | [guide and diagram](radau/simulation/radau-simulation-architecture.md) | Adaptive; SciPy session and dense output |

The [ABBA family guide](abba/simulation/abba-numerical-architecture.md) and its
[theory](abba/tex/theory.tex) describe four public classes and **39 normalized
configurations**. ABBA4 has one outer-projection implementation and 12 state,
formulation and solver configurations. The energy-enabled comparison uses eight.
The deprecated `ABBA4ImplicitSingleProjection` name is a compatibility factory
for the current `ABBA4Implicit`; it is not an additional numerical method.

The per-map ABBA4 projection implementation has been removed. Historical
per-map theory companions and proposed diagrams are labeled accordingly and
are not the current architecture. Saved study keys retain their original
scientific identities; obsolete comparison execution is explicitly rejected.

Every model's editable theoretical entry point is `tex/theory.tex`, with its
compiled `tex/theory.pdf`. Simulation diagrams are editable `.puml` DOT sources
with rendered `.svg` and `.png` counterparts. The generic diagram exists because
this cross-model architecture was explicitly requested.
